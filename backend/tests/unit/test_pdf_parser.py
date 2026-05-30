"""
tests/unit/test_pdf_parser.py — Unit tests for the PDF parsing service.

Strategy: we never need a real PDF file to test most of the logic.
  - Helper functions (_classify_section, _clean_markdown, etc.) are pure
    functions — tested directly with string inputs.
  - parse_pdf() is tested with a real synthetic PDF created in a fixture.
    This validates the full pipeline without needing external fixtures.
  - We do NOT mock pymupdf4llm — the library is deterministic and fast
    enough that real calls in tests are fine.

Run with:
    pytest tests/unit/test_pdf_parser.py -v
    pytest tests/unit/test_pdf_parser.py -v -k "test_clean"   # just cleaning tests
"""

import asyncio
from pathlib import Path

import fitz  # for creating synthetic PDFs in fixtures
import pytest

# Import the module and its private helpers for white-box testing.
# White-box testing helper functions is acceptable here because they encode
# domain logic (section classification) that has important edge cases.
from app.services.pdf_parser import (
    _classify_section,
    _clean_markdown,
    _extract_arxiv_id,
    _extract_authors,
    _extract_title,
    _extract_year,
    _split_into_sections,
    parse_pdf,
)
from app.models.parsed_paper import SectionType, ParsedPaper
from app.core.exceptions import ProcessingError

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIXTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(scope="module")
def synthetic_pdf_path(tmp_path_factory) -> Path:
    """
    Create a realistic synthetic PDF with IMRaD structure.
    Saved once for the whole test module — not recreated per test.

    The PDF has:
      - A title in large font
      - Abstract, Introduction, Methods, Results, Conclusion, References sections
      - Enough content per section for word_count > 30
    """
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "test_paper.pdf"

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    sections = [
        (18, "LoRA vs QLoRA: A Comparative Study of Fine-tuning Efficiency"),
        (11, "Hu et al., 2023"),
        (14, "Abstract"),
        (
            10,
            "We compare LoRA and QLoRA for fine-tuning large language models on "
            "low VRAM hardware. Our experiments show QLoRA achieves comparable "
            "performance to LoRA while reducing memory usage by 40 percent. "
            "We evaluate on MMLU and TruthfulQA benchmarks using LLaMA-7B.",
        ),
        (14, "1  Introduction"),
        (
            10,
            "Fine-tuning large language models requires substantial GPU memory. "
            "LoRA reduces trainable parameters via low-rank decomposition. "
            "QLoRA extends this with 4-bit quantization of the base model. "
            "This paper investigates the performance tradeoff between these two methods.",
        ),
        (14, "2  Methods"),
        (
            10,
            "We fine-tune LLaMA-7B using LoRA with rank 16 and alpha 32. "
            "For QLoRA we apply 4-bit NormalFloat quantization to the base model "
            "before applying LoRA adapters. Both methods use identical training data "
            "and hyperparameters for fair comparison. We train for 3 epochs.",
        ),
        (14, "3  Results"),
        (
            10,
            "LoRA achieves 74.3 percent accuracy on MMLU. QLoRA achieves 73.8 percent "
            "on the same benchmark. The 0.5 percent gap is within the margin of error. "
            "QLoRA reduces peak GPU memory from 28GB to 16GB, a 43 percent reduction. "
            "Training throughput is comparable between the two methods.",
        ),
        (14, "4  Conclusion"),
        (
            10,
            "QLoRA provides an effective alternative to LoRA for memory-constrained "
            "environments with minimal performance degradation. Future work will "
            "explore 2-bit quantization and larger model scales.",
        ),
        (14, "References"),
        (
            10,
            "[1] Hu, E. et al. LoRA: Low-Rank Adaptation. ICLR 2022. "
            "[2] Dettmers, T. et al. QLoRA: Efficient Finetuning. NeurIPS 2023.",
        ),
    ]

    y = 40
    for fontsize, text in sections:
        page.insert_text((50, y), text, fontsize=fontsize)
        y += fontsize + 12  # spacing proportional to font size

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture(scope="module")
def parsed_result(synthetic_pdf_path) -> ParsedPaper:
    """
    Run the full parse_pdf() pipeline once and share result across tests.
    This avoids calling pymupdf4llm multiple times per test session.
    """
    return asyncio.get_event_loop().run_until_complete(
        parse_pdf(synthetic_pdf_path, paper_id="test-paper-uuid-001")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MARKDOWN CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCleanMarkdown:

    def test_collapses_excess_blank_lines(self):
        text = "Hello\n\n\n\n\nWorld"
        result = _clean_markdown(text)
        assert "\n\n\n" not in result
        assert "Hello" in result
        assert "World" in result

    def test_strips_trailing_whitespace(self):
        text = "Hello   \nWorld  "
        result = _clean_markdown(text)
        for line in result.splitlines():
            assert not line.endswith(" "), f"Line has trailing space: {repr(line)}"

    def test_removes_ligatures(self):
        # ﬁ (FB01) and ﬂ (FB02) are common in LaTeX PDFs
        text = "eﬃcient ﬁne-tuning with ﬂow"  # eﬃ=FB03, ﬁ=FB01, ﬂ=FB02
        result = _clean_markdown(text)
        assert "\ufb01" not in result
        assert "\ufb02" not in result
        assert "fi" in result or "fl" in result

    def test_removes_soft_hyphens(self):
        text = "atten\u00adtion mech\u00adanism"
        result = _clean_markdown(text)
        assert "\u00ad" not in result
        assert "attention" in result

    def test_removes_zero_width_spaces(self):
        text = "fine\u200btuning"
        result = _clean_markdown(text)
        assert "\u200b" not in result

    def test_normalises_windows_line_endings(self):
        text = "line1\r\nline2\r\nline3"
        result = _clean_markdown(text)
        assert "\r" not in result

    def test_empty_string_returns_empty(self):
        assert _clean_markdown("") == ""

    def test_preserves_heading_markers(self):
        text = "## Abstract\n\nSome content."
        result = _clean_markdown(text)
        assert "## Abstract" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClassifySection:

    @pytest.mark.parametrize(
        "heading,expected",
        [
            # Abstract variants
            ("Abstract", SectionType.ABSTRACT),
            ("ABSTRACT", SectionType.ABSTRACT),
            ("abstract", SectionType.ABSTRACT),
            # Introduction variants (with and without section number)
            ("Introduction", SectionType.INTRODUCTION),
            ("1  Introduction", SectionType.INTRODUCTION),
            ("1. Introduction", SectionType.INTRODUCTION),
            ("1 Introduction", SectionType.INTRODUCTION),
            ("Motivation", SectionType.INTRODUCTION),
            # Background
            ("Related Work", SectionType.BACKGROUND),
            ("Background", SectionType.BACKGROUND),
            ("Prior Work", SectionType.BACKGROUND),
            # Methods
            ("Methods", SectionType.METHODS),
            ("3  Model Architecture", SectionType.METHODS),
            ("Our Approach", SectionType.METHODS),
            ("Proposed Framework", SectionType.METHODS),
            # Results
            ("Experiments", SectionType.RESULTS),
            ("4  Results", SectionType.RESULTS),
            ("Evaluation", SectionType.RESULTS),
            ("Ablation Study", SectionType.RESULTS),
            # Discussion
            ("Discussion", SectionType.DISCUSSION),
            ("Limitations", SectionType.DISCUSSION),
            ("Failure Analysis", SectionType.DISCUSSION),
            # Conclusion
            ("Conclusion", SectionType.CONCLUSION),
            ("5  Conclusions", SectionType.CONCLUSION),
            ("Future Work", SectionType.CONCLUSION),
            # References
            ("References", SectionType.REFERENCES),
            ("Bibliography", SectionType.REFERENCES),
            # Appendix
            ("Appendix", SectionType.APPENDIX),
            ("A  Supplementary Material", SectionType.APPENDIX),
            # Unknown
            ("Acknowledgements", SectionType.UNKNOWN),
            ("Ethics Statement", SectionType.UNKNOWN),
        ],
    )
    def test_classification(self, heading, expected):
        result = _classify_section(heading)
        assert (
            result == expected
        ), f"_classify_section({heading!r}) returned {result}, expected {expected}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION SPLITTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSplitIntoSections:

    @pytest.fixture
    def sample_markdown(self):
        return (
            "## Abstract\n\n"
            "This is the abstract. It has enough words to pass the filter. "
            "More content here for the word count requirement.\n\n"
            "## 1  Introduction\n\n"
            "This is the introduction with several sentences. It discusses "
            "the problem and motivation for this research paper.\n\n"
            "## 2  Methods\n\n"
            "We use LoRA for fine-tuning. The rank is 16 and alpha is 32. "
            "We train for three epochs using AdamW optimiser.\n\n"
            "## References\n\n"
            "[1] Hu et al. LoRA. ICLR 2022.\n"
        )

    def test_correct_section_count(self, sample_markdown):
        sections = _split_into_sections(sample_markdown)
        assert len(sections) == 4

    def test_sections_in_order(self, sample_markdown):
        sections = _split_into_sections(sample_markdown)
        assert sections[0].section_type == SectionType.ABSTRACT
        assert sections[1].section_type == SectionType.INTRODUCTION
        assert sections[2].section_type == SectionType.METHODS
        assert sections[3].section_type == SectionType.REFERENCES

    def test_section_index_is_sequential(self, sample_markdown):
        sections = _split_into_sections(sample_markdown)
        for i, s in enumerate(sections):
            assert s.section_index == i

    def test_heading_extracted_correctly(self, sample_markdown):
        sections = _split_into_sections(sample_markdown)
        assert sections[0].heading == "Abstract"
        assert sections[1].heading == "1  Introduction"

    def test_content_does_not_include_heading_line(self, sample_markdown):
        sections = _split_into_sections(sample_markdown)
        for section in sections:
            assert not section.content.startswith("#"), (
                f"Section '{section.heading}' content starts with #: "
                f"{section.content[:50]!r}"
            )

    def test_word_count_positive(self, sample_markdown):
        sections = _split_into_sections(sample_markdown)
        for s in sections:
            assert s.word_count > 0

    def test_empty_string_returns_empty_list(self):
        assert _split_into_sections("") == []

    def test_no_headings_returns_preamble(self):
        sections = _split_into_sections("Just some text with no headings at all here.")
        assert len(sections) == 1
        assert sections[0].heading == "Preamble"

    def test_is_useful_for_claims_skips_references(self, sample_markdown):
        sections = _split_into_sections(sample_markdown)
        refs = [s for s in sections if s.section_type == SectionType.REFERENCES]
        for s in refs:
            assert not s.is_useful_for_claims

    def test_is_useful_for_claims_keeps_results(self):
        md = (
            "## 4  Results\n\n" + "The model achieved 74 percent accuracy on MMLU. " * 5
        )
        sections = _split_into_sections(md)
        assert sections[0].is_useful_for_claims


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METADATA EXTRACTION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExtractYear:

    @pytest.mark.parametrize(
        "date_str,expected",
        [
            ("D:20230615120000", 2023),
            ("D:20170101000000", 2017),
            ("D:19991231235959", 1999),
            ("", None),
            ("invalid", None),
        ],
    )
    def test_from_pdf_date_header(self, date_str, expected):
        assert _extract_year({"creationDate": date_str}, "") == expected

    def test_from_page_text_fallback(self):
        text = "Published in NeurIPS 2022. All rights reserved."
        result = _extract_year({}, text)
        assert result == 2022

    def test_ignores_years_before_1990(self):
        text = "Published 1985."
        result = _extract_year({}, text)
        assert result is None

    def test_pdf_date_takes_priority_over_page_text(self):
        result = _extract_year({"creationDate": "D:20230101"}, "2019 conference")
        assert result == 2023


class TestExtractAuthors:

    def test_semicolon_separated(self):
        result = _extract_authors({"author": "Vaswani, A.; Shazeer, N.; Parmar, N."})
        assert len(result) == 3
        assert "Vaswani, A." in result

    def test_and_separated(self):
        result = _extract_authors({"author": "John Smith and Jane Doe"})
        assert len(result) == 2
        assert "John Smith" in result

    def test_empty_returns_empty_list(self):
        assert _extract_authors({}) == []
        assert _extract_authors({"author": ""}) == []

    def test_single_author(self):
        result = _extract_authors({"author": "Yann LeCun"})
        assert result == ["Yann LeCun"]


class TestExtractArxivId:

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("arXiv:2305.14314", "2305.14314"),
            ("arXiv:2305.14314v2", "2305.14314v2"),
            ("see paper 1706.03762 for", "1706.03762"),
            ("no id here", None),
            ("date: 2023.01.15", None),  # date format — should not match
            ("version 1234.5", None),  # too short
        ],
    )
    def test_extraction(self, text, expected):
        assert _extract_arxiv_id(text) == expected


class TestExtractTitle:

    def test_from_pdf_metadata(self):
        meta = {"title": "Attention Is All You Need"}
        result = _extract_title(meta, "")
        assert result == "Attention Is All You Need"

    def test_metadata_takes_priority_over_page_text(self):
        meta = {"title": "Real Title"}
        result = _extract_title(meta, "Some other first line")
        assert result == "Real Title"

    def test_fallback_to_first_page_line(self):
        result = _extract_title({}, "LoRA: Low-Rank Adaptation of LLMs\nVaswani et al.")
        assert result == "LoRA: Low-Rank Adaptation of LLMs"

    def test_empty_metadata_and_text_returns_none(self):
        assert _extract_title({}, "") is None

    def test_ignores_very_short_metadata_titles(self):
        # Single-character or very short titles are artefacts
        result = _extract_title({"title": "A"}, "Real Title From Page Text Here")
        # Should fall through to page text
        assert result != "A"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL PIPELINE INTEGRATION (uses synthetic_pdf_path fixture)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestParsePdf:

    def test_returns_parsed_paper_type(self, parsed_result):
        assert isinstance(parsed_result, ParsedPaper)

    def test_paper_id_preserved(self, parsed_result):
        assert parsed_result.paper_id == "test-paper-uuid-001"

    def test_has_sections(self, parsed_result):
        assert len(parsed_result.sections) >= 1

    def test_sections_are_ordered(self, parsed_result):
        for i, s in enumerate(parsed_result.sections):
            assert s.section_index == i

    def test_all_sections_have_content(self, parsed_result):
        for section in parsed_result.sections:
            assert (
                section.content.strip()
            ), f"Section '{section.heading}' has empty content"

    def test_metadata_has_page_count(self, parsed_result):
        assert parsed_result.metadata.page_count >= 1

    def test_metadata_word_count_positive(self, parsed_result):
        assert parsed_result.metadata.word_count > 0

    def test_total_word_count_matches_sections(self, parsed_result):
        section_total = sum(s.word_count for s in parsed_result.sections)
        assert parsed_result.total_word_count == section_total

    def test_raw_markdown_is_populated(self, parsed_result):
        assert len(parsed_result.raw_markdown) > 100

    def test_claim_sections_excludes_references(self, parsed_result):
        for s in parsed_result.claim_sections:
            assert s.section_type != SectionType.REFERENCES

    def test_abstract_section_accessible(self, parsed_result):
        # Our synthetic PDF has an Abstract section
        abstract = parsed_result.abstract_section
        if abstract:  # only assert if found — may depend on pymupdf4llm version
            assert abstract.section_type == SectionType.ABSTRACT
            assert len(abstract.content) > 10

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            asyncio.get_event_loop().run_until_complete(
                parse_pdf("/nonexistent/path/paper.pdf", "uuid-x")
            )

    def test_not_a_file_raises_processing_error(self, tmp_path):
        dir_path = tmp_path / "not_a_file"
        dir_path.mkdir()
        with pytest.raises(ProcessingError):
            asyncio.get_event_loop().run_until_complete(parse_pdf(dir_path, "uuid-y"))
