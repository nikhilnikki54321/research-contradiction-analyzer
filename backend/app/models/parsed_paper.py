"""
models/parsed_paper.py — Pydantic models for the PDF parsing pipeline output.

These models live at the boundary between the PDF parser and everything
downstream (chunker, embedder, claim extractor). They are the contract
that all three services agree on.

Separation from models/paper.py (upload/API models) is intentional:
  - paper.py   → API layer: what clients see (upload response, paper status)
  - parsed_paper.py → service layer: what the parsing pipeline produces

A PaperSection is the atomic unit of parsed content. A ParsedPaper
aggregates all sections for one document plus document-level metadata.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Section type taxonomy ─────────────────────────────────────────────────────


class SectionType(str, Enum):
    """
    Canonical section categories for academic research papers.

    These map to the standard IMRaD structure plus common extras.
    Used by the claim extractor to prioritize which sections to scan:
      - ABSTRACT, RESULTS, CONCLUSION → highest claim density
      - METHODS, INTRODUCTION         → medium claim density
      - REFERENCES, APPENDIX          → skip for claim extraction

    UNKNOWN is used when a heading cannot be confidently categorized.
    It is NOT an error — many real sections (e.g. "Limitations",
    "Related Work") don't map cleanly to the main categories.
    """

    TITLE = "title"  # document title heading
    ABSTRACT = "abstract"  # paper summary
    INTRODUCTION = "introduction"  # problem statement, motivation
    BACKGROUND = "background"  # related work, prior art
    METHODS = "methods"  # model, approach, architecture
    RESULTS = "results"  # experiments, evaluations, benchmarks
    DISCUSSION = "discussion"  # interpretation of results
    CONCLUSION = "conclusion"  # summary, future work
    REFERENCES = "references"  # bibliography
    APPENDIX = "appendix"  # supplementary material
    UNKNOWN = "unknown"  # could not be classified


# ── Core section model ────────────────────────────────────────────────────────


class PaperSection(BaseModel):
    """
    One logical section of a parsed research paper.

    A section corresponds to a heading (## Abstract, ## 3 Methods, etc.)
    and all the text beneath it until the next heading of the same or
    higher level.

    section_index is 0-based and represents the order the section appeared
    in the document. The chunker uses this for provenance tracking.
    """

    heading: str = Field(
        description="The raw heading text as extracted from the PDF. "
        "May include numbering (e.g. '3.1  Encoder Stacks').",
        max_length=512,
    )
    content: str = Field(
        description="Full text content of this section in cleaned markdown. "
        "Includes sub-headings if they were merged into this section.",
    )
    section_type: SectionType = Field(
        description="Canonical category. Used to prioritize sections for "
        "claim extraction.",
    )
    section_index: int = Field(
        description="0-based position in the document. Preserves reading order.",
        ge=0,
    )
    heading_level: int = Field(
        description="Markdown heading level (1=# 2=## 3=### etc.).",
        ge=1,
        le=6,
    )
    word_count: int = Field(
        description="Number of whitespace-separated tokens in content.",
        ge=0,
    )
    page_start: int | None = Field(
        default=None,
        description="1-based page number where this section starts. "
        "None if page tracking was not requested.",
        ge=1,
    )

    @field_validator("content")
    @classmethod
    def content_not_empty_after_strip(cls, v: str) -> str:
        """Whitespace-only sections are a parsing artifact — reject them."""
        if not v.strip():
            raise ValueError("Section content cannot be empty or whitespace-only")
        return v.strip()

    @model_validator(mode="after")
    def word_count_matches_content(self) -> "PaperSection":
        """Keep word_count consistent with actual content."""
        actual = len(self.content.split())
        if self.word_count != actual:
            self.word_count = actual
        return self

    @property
    def is_useful_for_claims(self) -> bool:
        """
        True if this section is worth scanning for empirical claims.
        Skips references and appendices — they don't contain primary claims.
        Also skips very short sections (< 30 words) which are usually
        just headings with no substantive content.
        """
        skip_types = {SectionType.REFERENCES, SectionType.APPENDIX, SectionType.TITLE}
        return self.section_type not in skip_types and self.word_count >= 30

    def short_preview(self, chars: int = 200) -> str:
        """Return first N characters of content for logging."""
        if len(self.content) <= chars:
            return self.content
        return self.content[:chars] + "..."


# ── Document metadata model ───────────────────────────────────────────────────


class DocumentMetadata(BaseModel):
    """
    Document-level metadata extracted from PDF headers and text.

    All fields are optional because academic PDFs are notoriously
    inconsistent about what metadata they embed. The parser makes a
    best-effort extraction and leaves fields None when unavailable.
    """

    title: str | None = Field(
        default=None,
        description="Paper title. Extracted from PDF metadata or inferred "
        "from first large-font text.",
        max_length=512,
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author names. May be empty if not in PDF metadata.",
    )
    year: int | None = Field(
        default=None,
        description="Publication year. Extracted from PDF creation date "
        "or year pattern in text.",
        ge=1900,
        le=2100,
    )
    arxiv_id: str | None = Field(
        default=None,
        description="arXiv identifier if found in metadata or text "
        "(e.g. '2305.14314' or '2305.14314v2').",
        pattern=r"^\d{4}\.\d{4,5}(v\d+)?$",
    )
    abstract: str | None = Field(
        default=None,
        description="Full abstract text. Extracted from the ABSTRACT section "
        "for convenient access without scanning all sections.",
    )
    page_count: int = Field(
        description="Total number of pages in the PDF.",
        ge=1,
    )
    word_count: int = Field(
        description="Total word count across all sections.",
        ge=0,
    )


# ── Top-level result ──────────────────────────────────────────────────────────


class ParsedPaper(BaseModel):
    """
    Complete output of the PDF parsing service for one document.

    This is the object that flows into the chunker. Every field
    downstream services need is here — they should never re-read
    the PDF file.

    paper_id matches the UUID from the upload endpoint and is used
    throughout the pipeline for provenance tracking.
    """

    paper_id: str = Field(
        description="UUID from the upload endpoint. Ties this parsed result "
        "to the original upload record.",
    )
    file_path: str = Field(
        description="Absolute path to the source PDF on disk.",
    )
    metadata: DocumentMetadata
    sections: list[PaperSection] = Field(
        description="Ordered list of sections. Index 0 is first in document.",
        min_length=1,
    )
    raw_markdown: str = Field(
        description="Complete extracted markdown before section splitting. "
        "Preserved for debugging and fallback processing.",
    )
    parser_version: str = Field(
        default="pymupdf4llm",
        description="Parser backend identifier for reproducibility.",
    )

    @property
    def abstract_section(self) -> PaperSection | None:
        """Return the ABSTRACT section if one was found."""
        for s in self.sections:
            if s.section_type == SectionType.ABSTRACT:
                return s
        return None

    @property
    def claim_sections(self) -> list[PaperSection]:
        """Return sections that are useful for claim extraction, in order."""
        return [s for s in self.sections if s.is_useful_for_claims]

    @property
    def total_word_count(self) -> int:
        return sum(s.word_count for s in self.sections)

    def section_by_type(self, section_type: SectionType) -> list[PaperSection]:
        """Return all sections of a given type."""
        return [s for s in self.sections if s.section_type == section_type]
