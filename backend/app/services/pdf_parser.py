"""
services/pdf_parser.py — Production-grade PDF parsing service.

Converts a raw PDF file into a structured ParsedPaper object that downstream
services (chunker, embedder, claim extractor) can consume without ever
touching the PDF file again.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY pymupdf4llm OVER RAW PyMuPDF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw PyMuPDF (fitz) extracts text in reading order per the PDF's internal
coordinate stream. For two-column academic papers, this means column 1 of
page 2 is interleaved with column 2 of page 1 — completely destroying
sentence continuity across columns.

pymupdf4llm uses a layout analysis pass before text extraction. It detects
column boundaries, sorts text blocks by visual reading order (left column
top→bottom, then right column top→bottom), and outputs clean markdown with
heading levels inferred from font sizes. The result is text a language model
can actually reason over — not the character soup raw PyMuPDF produces for
academic papers.

Concrete differences on a typical 2-column NLP paper:
  raw PyMuPDF:      "...attention mechanism [COLUMN BREAK] where dk is the..."
  pymupdf4llm:      "...attention mechanism where dk is the..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  parse_pdf(path, paper_id)
      │
      ├─► _run_pymupdf4llm(path)          blocking → run in ThreadPoolExecutor
      │       └─► pymupdf4llm.to_markdown()
      │
      ├─► _extract_metadata(path)         blocking → run in ThreadPoolExecutor
      │       └─► fitz.open() metadata + text scan
      │
      ├─► _clean_markdown(raw_md)
      │
      ├─► _split_into_sections(md)
      │       ├─► split on ## headings
      │       ├─► classify each heading → SectionType
      │       └─► build PaperSection objects
      │
      └─► build ParsedPaper(metadata, sections, raw_markdown)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWN LIMITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Scanned PDFs (image-only, no text layer) — pymupdf4llm falls back to
     Tesseract OCR. Quality depends on scan resolution. Equations will fail.
  2. Mathematical notation — LaTeX equations don't survive PDF→text extraction.
     "∇L(θ)" becomes garbled Unicode or is dropped.
  3. Tables — simple tables survive as aligned text. Complex spanning tables
     lose structure entirely.
  4. Figures and captions — figure content is lost; captions are usually kept.
  5. PDFs with custom fonts — character mapping may be wrong if the PDF embeds
     a non-standard encoding. Symptom: mojibake in output.
  6. Very long papers (>100 pages) — pymupdf4llm holds the full document in
     memory. May hit OOM on large books. Academic papers are fine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUTURE IMPROVEMENT PATHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Nougat (Meta) for math-heavy papers — produces LaTeX from scanned PDFs
  - Unstructured.io for better table extraction
  - GROBID for structured header/reference parsing (author disambiguation)
  - arXiv source XML when available — lossless alternative to PDF parsing
"""

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF — used for metadata + page count
import pymupdf4llm  # layout-aware markdown extraction

from app.core.exceptions import ProcessingError
from app.core.logging import get_logger
from app.models.parsed_paper import (
    DocumentMetadata,
    ParsedPaper,
    PaperSection,
    SectionType,
)

logger = get_logger(__name__)

# Module-level thread pool — pymupdf4llm is CPU-bound and not async-native.
# One pool, shared across all parse requests. max_workers=2 avoids OOM on
# concurrent large PDF uploads while still parallelising reasonably.
_PARSE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pdf_parser")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC API — one function, called by the Celery worker and tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def parse_pdf(file_path: str | Path, paper_id: str) -> ParsedPaper:
    """
    Parse a PDF file into a structured ParsedPaper object.

    This is the only function downstream code should call. All helper
    functions below are implementation details and are not part of the
    public API of this module.

    Args:
        file_path:  Absolute path to the PDF on disk.
        paper_id:   UUID from the upload endpoint — used for log correlation
                    and provenance tracking in the output.

    Returns:
        ParsedPaper with extracted metadata, ordered sections, and raw markdown.

    Raises:
        ProcessingError:  If the PDF cannot be parsed for any reason.
                          The original exception is chained (from exc).
        FileNotFoundError: If file_path does not exist.
    """
    path = Path(file_path).resolve()

    # ── Guard: file must exist before we touch the thread pool ───────────────
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at path: {path}")
    if not path.is_file():
        raise ProcessingError(f"Path exists but is not a file: {path}")

    logger.info(
        "pdf_parser.started",
        paper_id=paper_id,
        path=str(path),
        size_mb=round(path.stat().st_size / (1024 * 1024), 2),
    )
    t_start = time.monotonic()

    try:
        # ── Step 1: Run pymupdf4llm in thread pool (blocking C extension) ────
        # asyncio.get_event_loop().run_in_executor() offloads the blocking call
        # to a thread, yielding control back to the event loop while PyMuPDF
        # does CPU-intensive layout analysis. Without this, one large PDF
        # would block all other requests for the full parse duration.
        loop = asyncio.get_event_loop()
        raw_markdown, metadata = await asyncio.gather(
            loop.run_in_executor(
                _PARSE_EXECUTOR,
                _run_pymupdf4llm,  # function reference
                str(path),  # argument
            ),
            loop.run_in_executor(
                _PARSE_EXECUTOR,
                _extract_metadata,
                str(path),
            ),
        )
        # Both blocking calls run concurrently in the thread pool.
        # Total wall time ≈ max(parse_time, metadata_time), not their sum.

        # ── Step 2: Clean the raw markdown ───────────────────────────────────
        # pymupdf4llm output has inconsistent whitespace and occasional
        # artefacts from column detection. This pass normalises it.
        cleaned_markdown = _clean_markdown(raw_markdown)

        # ── Step 3: Split into sections ───────────────────────────────────────
        sections = _split_into_sections(cleaned_markdown)

        if not sections:
            raise ProcessingError(
                "No sections could be extracted from PDF. "
                "The document may be image-only or have no detectable headings.",
            )

        # ── Step 4: Populate abstract in metadata if found ───────────────────
        # Convenience: pull abstract text up into DocumentMetadata so callers
        # don't have to search through sections for it.
        abstract_section = next(
            (s for s in sections if s.section_type == SectionType.ABSTRACT),
            None,
        )
        if abstract_section:
            metadata.abstract = abstract_section.content

        # ── Step 5: Fill total word count into metadata ───────────────────────
        metadata.word_count = sum(s.word_count for s in sections)

        # ── Step 6: Try to backfill title from first section if metadata empty
        if not metadata.title or metadata.title == "Untitled Paper":
            first = sections[0]
            # Title is usually the first heading with heading_level == 1 or 2
            if first.section_type == SectionType.TITLE:
                metadata.title = first.heading

        elapsed = round((time.monotonic() - t_start) * 1000, 0)
        logger.info(
            "pdf_parser.completed",
            paper_id=paper_id,
            section_count=len(sections),
            total_words=metadata.word_count,
            has_abstract=abstract_section is not None,
            title=metadata.title,
            elapsed_ms=elapsed,
        )

        return ParsedPaper(
            paper_id=paper_id,
            file_path=str(path),
            metadata=metadata,
            sections=sections,
            raw_markdown=cleaned_markdown,
        )

    except (FileNotFoundError, ProcessingError):
        raise  # re-raise without wrapping — already descriptive

    except Exception as exc:
        # Wrap unexpected errors from pymupdf/fitz in our typed exception
        # so the Celery task and API layer get a consistent error shape.
        elapsed = round((time.monotonic() - t_start) * 1000, 0)
        logger.exception(
            "pdf_parser.failed",
            paper_id=paper_id,
            path=str(path),
            error=str(exc),
            elapsed_ms=elapsed,
        )
        raise ProcessingError(f"Failed to parse PDF '{path.name}': {exc}") from exc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 HELPERS — extraction (run in thread pool)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _run_pymupdf4llm(file_path: str) -> str:
    """
    Call pymupdf4llm.to_markdown() synchronously.

    This is intentionally a plain (non-async) function so it can be passed
    to run_in_executor. Never call this directly from async code — always
    use await loop.run_in_executor(_PARSE_EXECUTOR, _run_pymupdf4llm, path).

    Why to_markdown() over to_text() or raw fitz page.get_text():
      - to_markdown() runs layout analysis per page, detecting columns
      - Heading levels are inferred from font sizes (larger = higher level)
      - Two-column papers are read left→right, top→bottom per column
      - Output is clean markdown that LLMs process well
    """
    return pymupdf4llm.to_markdown(file_path)


def _extract_metadata(file_path: str) -> DocumentMetadata:
    """
    Extract document-level metadata using fitz (raw PyMuPDF).

    Uses fitz directly (not pymupdf4llm) because metadata lives in the
    PDF's info dictionary, which pymupdf4llm doesn't expose cleanly.

    Extraction strategy (each field tried in order, first success wins):
      title:    PDF metadata "title" → first page large-font text
      authors:  PDF metadata "author" → empty list
      year:     PDF metadata creation date → year pattern in first 500 chars
      arxiv_id: Search first 2 pages for arXiv ID pattern
    """
    doc = fitz.open(file_path)
    try:
        meta = doc.metadata or {}
        page_count = doc.page_count

        # Extract first-page text for fallback extraction (title, year, arxiv)
        # Limit to 1500 chars — enough to cover title, authors, abstract start
        first_page_text = ""
        if page_count > 0:
            first_page_text = doc[0].get_text()[:1500]

        # Second page for arXiv ID (sometimes on page 2 in footer)
        second_page_text = ""
        if page_count > 1:
            second_page_text = doc[1].get_text()[:500]

        search_text = first_page_text + " " + second_page_text

    finally:
        doc.close()  # always release the file handle

    return DocumentMetadata(
        title=_extract_title(meta, first_page_text),
        authors=_extract_authors(meta, first_page_text),
        year=_extract_year(meta, first_page_text),
        arxiv_id=_extract_arxiv_id(search_text),
        page_count=page_count,
        word_count=0,  # populated later after section splitting
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 HELPER — markdown cleaning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _clean_markdown(text: str) -> str:
    """
    Normalise pymupdf4llm output for consistent downstream processing.

    Problems this addresses:
      1. Excess blank lines (3+ in a row) from column boundary artefacts
      2. Trailing whitespace on lines (causes phantom word counts)
      3. Whitespace-only lines (look empty but aren't)
      4. Unicode "soft hyphen" (U+00AD) used for word wrapping in PDFs
         — invisible in editors but breaks tokenisation and NLI models
      5. Zero-width spaces (U+200B) from PDF copy protection layers
      6. Ligature characters (ﬁ, ﬂ, ﬀ) that tokenisers split incorrectly
    """
    # Remove PDF ligature artefacts before any other processing
    # ﬁ (U+FB01) → fi,  ﬂ (U+FB02) → fl,  ﬀ (U+FB00) → ff
    ligature_map = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u00ad": "",  # soft hyphen — just remove
        "\u200b": "",  # zero-width space — just remove
        "\u200c": "",  # zero-width non-joiner
        "\u200d": "",  # zero-width joiner
    }
    for char, replacement in ligature_map.items():
        text = text.replace(char, replacement)

    # Normalise line endings (PDFs sometimes produce \r\n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace from every line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Collapse 3+ blank lines into exactly 2 (one blank line between paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are only whitespace
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)

    return text.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 HELPER — section splitting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Compiled once at module load — not inside the function.
# Matches lines that start with 1–6 # characters followed by a space.
# The (?m) flag makes ^ match start-of-line, not just start-of-string.
_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)(?:\s+#+\s*)?$")

# Lookahead split: split the markdown at each heading start, keeping the
# heading as part of the following section (not a trailing part of the previous).
_SECTION_SPLIT_RE = re.compile(r"(?=^#{1,6}\s)", re.MULTILINE)


def _split_into_sections(markdown: str) -> list[PaperSection]:
    """
    Split full-document markdown into a list of PaperSection objects.

    Algorithm:
      1. Split markdown on heading boundaries using a lookahead regex.
         Lookahead (not lookbehind) means the heading line stays with the
         content that follows it, not the content that preceded it.
      2. For each chunk: extract the heading, classify it, count words.
      3. Build a PaperSection for each non-empty chunk.

    Edge cases handled:
      - Text before the first heading (no #) → single UNKNOWN section
      - Headings with no content (next heading immediately follows) → skipped
      - Deeply nested headings (###, ####) → captured as sub-sections
    """
    # Split on heading boundaries
    raw_chunks = _SECTION_SPLIT_RE.split(markdown)

    sections: list[PaperSection] = []

    for idx, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        # Try to extract a heading line from the start of this chunk
        heading_match = _HEADING_RE.match(chunk)

        if heading_match:
            # chunk starts with a heading
            hashes = heading_match.group(1)  # e.g. "##"
            heading = heading_match.group(2).strip()  # e.g. "3.1  Methods"
            level = len(hashes)  # 2

            # Content is everything after the first heading line
            content_start = heading_match.end()
            content = chunk[content_start:].strip()
        else:
            # Text before first heading — usually empty or page header artefacts
            heading = "Preamble"
            level = 1
            content = chunk

        # Skip sections with no usable content after extracting the heading
        if not content:
            continue

        section_type = _classify_section(heading)
        word_count = len(content.split())

        try:
            section = PaperSection(
                heading=heading,
                content=content,
                section_type=section_type,
                section_index=len(sections),  # 0-based, tracks insertion order
                heading_level=level,
                word_count=word_count,
            )
            sections.append(section)
        except Exception as e:
            # Log and skip malformed sections — don't fail the whole document
            logger.warning(
                "pdf_parser.section_skipped",
                heading=heading[:50],
                reason=str(e),
            )
            continue

    return sections


def _classify_section(heading: str) -> SectionType:
    """
    Map a raw heading string to a canonical SectionType.

    Matching is case-insensitive. Numbered headings are handled by also
    matching patterns like "1 Introduction", "1. Introduction", "1  Introduction".

    The order of checks matters — more specific patterns first.
    "1 Introduction" should match INTRODUCTION, not UNKNOWN.

    This is a heuristic. It will misclassify unusual headings like
    "Results and Discussion" (→ RESULTS, missing the Discussion aspect).
    For this project, imperfect classification is acceptable — the claim
    extractor uses section_type as a soft priority signal, not a hard filter.
    """
    # Normalise: lowercase, strip leading section numbers and punctuation
    # "3.1  Encoder and Decoder" → "encoder and decoder"
    t = heading.lower().strip()
    t = re.sub(r"^([a-z]|\d+(\.\d+)*)[\s\.\-]+", "", t)  # strip "3.1 ", "3.", "A "
    t = t.strip()

    # Ordered from most specific to most general
    if re.match(r"^abstract", t):
        return SectionType.ABSTRACT

    if re.match(r"^(introduction|motivation|overview)", t):
        return SectionType.INTRODUCTION

    if re.match(r"^(background|related work|prior work|literature review)", t):
        return SectionType.BACKGROUND

    if re.match(
        r"^(method|approach|model|architecture|framework|system|"
        r"proposed|our (method|model|approach)|technical)",
        t,
    ):
        return SectionType.METHODS

    if re.match(
        r"^(experiment|result|evaluation|benchmark|performance|"
        r"empirical|ablation|analysis|quantitative)",
        t,
    ):
        return SectionType.RESULTS

    if re.match(
        r"^(discussion|interpretation|implication|limitation|"
        r"failure|error analysis)",
        t,
    ):
        return SectionType.DISCUSSION

    if re.match(r"^(conclusion|summary|future work|closing)", t):
        return SectionType.CONCLUSION

    if re.match(r"^(reference|bibliography|citation)", t):
        return SectionType.REFERENCES

    if re.match(r"^(appendix|supplement|additional|extended)", t):
        return SectionType.APPENDIX

    return SectionType.UNKNOWN


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METADATA EXTRACTION HELPERS — all pure functions (no I/O)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _extract_title(meta: dict, first_page_text: str) -> str | None:
    """
    Extract paper title robustly from metadata or first-page text.

    Strategy:
      1. Use PDF metadata title if trustworthy
      2. Otherwise scan first-page lines
      3. Score title candidates intelligently
      4. Return highest-scoring candidate
    """

    # ─────────────────────────────────────────────────────────
    # Strategy 1: PDF metadata
    # ─────────────────────────────────────────────────────────
    pdf_title = (meta.get("title") or "").strip()

    bad_meta_patterns = [
        "untitled",
        "microsoft word",
        "acrobat",
        "default",
        "journal",
        "springer",
        "elsevier",
        "doi",
        "copyright",
    ]

    # Metadata always wins if it looks valid
    if pdf_title:

        if len(pdf_title) >= 10 and not any(
            p in pdf_title.lower() for p in bad_meta_patterns
        ):
            return pdf_title[:512]

    # ─────────────────────────────────────────────────────────
    # Strategy 2: First-page heuristic extraction
    # ─────────────────────────────────────────────────────────
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]

    blacklist_patterns = [
        r"copyright",
        r"all rights reserved",
        r"received:",
        r"accepted:",
        r"published",
        r"available online",
        r"correspondence",
        r"doi",
        r"issn",
        r"university",
        r"department",
        r"journal",
        r"vol\.",
        r"issue",
        r"abstract",
        r"keywords",
        r"introduction",
        r"www\.",
        r"http",
        r"@\w+",
        r"bioRxiv",
        r"medRxiv",
    ]

    candidates = []

    for idx, line in enumerate(lines[:25]):

        lower = line.lower()

        # Skip blacklisted lines
        if any(re.search(pattern, lower) for pattern in blacklist_patterns):
            continue

        # Length constraints
        if len(line) < 15 or len(line) > 200:
            continue

        # Skip paragraph-like sentences
        if line.endswith("."):
            continue

        # Skip lines with too many commas
        if line.count(",") > 4:
            continue

        # Skip mostly numeric lines
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.6:
            continue

        # Title scoring
        score = 0

        # Earlier lines are more likely titles
        score += max(0, 25 - idx)

        # Good title length
        if 30 <= len(line) <= 120:
            score += 15

        # Title-case words
        words = line.split()

        capitalized_words = sum(1 for w in words if len(w) > 2 and w[0].isupper())

        capitalization_ratio = capitalized_words / max(len(words), 1)

        if capitalization_ratio > 0.5:
            score += 20

        # Penalize weird symbols
        weird_chars = sum(1 for c in line if not (c.isalnum() or c in " -:,()/"))

        score -= weird_chars * 5

        candidates.append((score, line))

    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])

        best_title = candidates[0][1][:512]

        logger.info(
            "pdf_parser.title_detected",
            detected_title=best_title,
        )

        return best_title

    # ─────────────────────────────────────────────────────
    # FALLBACK TITLE EXTRACTION
    # ─────────────────────────────────────────────────────

    fallback_lines = [
        line.strip() for line in first_page_text.splitlines() if line.strip()
    ]

    for line in fallback_lines[:15]:

        # Skip tiny lines
        if len(line) < 10:
            continue

        # Skip obvious metadata
        lower = line.lower()

        if any(
            x in lower
            for x in [
                "abstract",
                "introduction",
                "doi",
                "copyright",
                "keywords",
            ]
        ):
            continue

        logger.warning(
            "pdf_parser.fallback_title_used",
            fallback_title=line[:120],
        )

        return line[:512]

    logger.error("pdf_parser.title_extraction_failed")

    return None


def _extract_authors(meta: dict, first_page_text: str = "") -> list[str]:
    """
    Extract authors from metadata or first page text.
    """

    author_str = (meta.get("author") or "").strip()

    # ─────────────────────────────────────────
    # Strategy 1: PDF metadata
    # ─────────────────────────────────────────
    if author_str:

        if ";" in author_str:
            parts = [a.strip() for a in author_str.split(";")]

        elif " and " in author_str.lower():
            parts = [a.strip() for a in re.split(r"\s+and\s+", author_str, flags=re.I)]

        elif author_str.count(",") >= 2:
            parts = [a.strip() for a in author_str.split(",")]

        else:
            parts = [author_str]

        cleaned = [p for p in parts if p and len(p) > 1]

        if cleaned:
            return cleaned

    # ─────────────────────────────────────────
    # Strategy 2: First-page heuristic
    # ─────────────────────────────────────────

    possible_authors = []

    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]

    for line in lines[:20]:

        # Skip long lines
        if len(line) > 120:
            continue

        # Skip section headings
        lower = line.lower()

        if any(
            x in lower
            for x in [
                "abstract",
                "introduction",
                "keywords",
                "doi",
                "journal",
            ]
        ):
            continue

        # Detect probable author line
        if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", line):

            # Avoid title-like long lines
            if len(line.split()) <= 10:
                possible_authors.append(line)

    if possible_authors:

        logger.warning(
            "pdf_parser.fallback_authors_used",
            authors=possible_authors[:3],
        )

        return possible_authors[:3]

    return []


def _extract_year(meta: dict, first_page_text: str) -> int | None:
    """
    Extract publication year.

    Priority:
      1. PDF creation date metadata (format: "D:YYYYMMDDHHmmSS")
      2. Year pattern (4 digits, 1990-2099) in first-page text

    Why not use modification date? PDFs are often modified after publication
    (for watermarks, corrections). Creation date is more likely to reflect
    the actual year the paper was written.
    """
    current_year = 2035

    # Strategy 1: PDF creation date header
    creation_date = meta.get("creationDate") or meta.get("modDate") or ""
    if creation_date:
        # PDF date format: D:20230615120000+00'00'
        m = re.search(r"D:(\d{4})", creation_date)
        if m:
            year = int(m.group(1))
            if 1990 <= year <= current_year:
                return year

    # Strategy 2: Four-digit year in first page text
    # Match years in realistic range only — avoid matching DOIs, page numbers
    # ─────────────────────────────────────────────────────
    # Strategy 2: find years in first page text
    # ─────────────────────────────────────────────────────
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", first_page_text)

    valid_years = []

    for y in years:
        year = int(y)

        if 1990 <= year <= current_year:
            valid_years.append(year)

    if not valid_years:
        return None

    # Prefer most common year
    freq = {}

    for y in valid_years:
        freq[y] = freq.get(y, 0) + 1

    sorted_years = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    return sorted_years[0][0]


def _extract_arxiv_id(text: str) -> str | None:
    """
    Search for an arXiv identifier in the provided text.

    arXiv ID formats:
      New (post-April 2007):  YYMM.NNNNN  (e.g. 2305.14314)
      New with version:        YYMM.NNNNNvN (e.g. 2305.14314v2)
      Old (pre-2007):          cat/NNNNNNN  (not handled — too old for this use case)

    We look in the first ~2000 characters of the PDF (header area where
    arXiv IDs typically appear as "arXiv:2305.14314" or in footer watermarks).
    """
    # Pattern matches the YYMM.NNNNN[vN] format
    # Require word boundary on both sides to avoid matching partial DOIs
    pattern = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")
    match = pattern.search(text)
    if match:
        candidate = match.group(1)
        # Sanity check: first 4 digits should be a plausible YYMM
        year_month = candidate[:4]
        if "9000" > year_month >= "0701":  # arXiv new format started 2007-04
            return candidate
    return None
