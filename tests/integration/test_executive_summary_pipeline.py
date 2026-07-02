"""
Integration tests for executive summary synthesis (issue #10).
All tests use FakeLLMClient — no real API calls.

The FakeLLMClient's final text response (no tool_use blocks) is extracted by
report_assembly and used as the executive summary. These tests verify that the
summary section is populated with that text and that it references specific
findings present in other sections.
"""
from pathlib import Path

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FEAT_CSV = str(FIXTURES_DIR / "feature_suggestion_data.csv")

# Canned executive summary that references specific findings from the fixture
_EXEC_SUMMARY_TEXT = (
    "This 20-row dataset contains 4 columns: price, category, a, and b. "
    "The 'price' column is highly right-skewed and would benefit from a log transform. "
    "The 'category' column has 20 unique values and requires an encoding strategy before modeling. "
    "Columns 'a' and 'b' are strongly correlated and may introduce multicollinearity."
)

# Full run: schema → missing → distribution → feature_suggestion → exec summary
_FULL_RESPONSES = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running schema inference."},
            {"type": "tool_use", "id": "tu_001", "name": "schema_inference", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking missing values."},
            {"type": "tool_use", "id": "tu_002", "name": "missing_value_analysis", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Computing distributions."},
            {"type": "tool_use", "id": "tu_003", "name": "distribution_analysis", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Generating feature engineering suggestions."},
            {"type": "tool_use", "id": "tu_004", "name": "feature_suggestion", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": _EXEC_SUMMARY_TEXT}],
    },
]


def _make_full_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_FULL_RESPONSES)


class TestExecutiveSummaryPopulated:
    def test_exec_summary_not_placeholder_in_full_run(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_full_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        exec_section = content.split("## Executive Summary")[1].split("## Data Quality")[0]
        assert "_Not yet analyzed in this run._" not in exec_section

    def test_exec_summary_contains_specific_column_names(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_full_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        exec_section = content.split("## Executive Summary")[1].split("## Data Quality")[0]
        assert "price" in exec_section
        assert "category" in exec_section

    def test_exec_summary_references_findings_present_in_other_sections(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_full_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        exec_section = content.split("## Executive Summary")[1].split("## Data Quality")[0]
        feat_section = content.split("## Feature Engineering Recommendations")[1]

        # "price" appears in both exec summary and feature engineering section
        assert "price" in exec_section
        assert "price" in feat_section

        # "category" appears in both exec summary and feature engineering section
        assert "category" in exec_section
        assert "category" in feat_section

    def test_exec_summary_contains_llm_text_verbatim(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_full_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # The canned LLM text should appear in the exec summary section
        assert "right-skewed" in content
        assert "multicollinearity" in content

    def test_all_five_sections_present_in_full_run(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_full_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        for section in [
            "## Executive Summary",
            "## Data Quality Scorecard",
            "## Distributions",
            "## Correlations",
            "## Feature Engineering Recommendations",
        ]:
            assert section in content


class TestPlaceholderReplacement:
    def test_no_placeholder_in_exec_summary_after_full_run(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_full_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        exec_section = content.split("## Executive Summary")[1].split("## Data Quality")[0]
        assert "_Not yet analyzed in this run._" not in exec_section

    def test_placeholder_replaced_by_llm_text(self, tmp_path):
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_full_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        exec_section = content.split("## Executive Summary")[1].split("## Data Quality")[0]
        # LLM text (not placeholder) is present in exec summary
        assert "skewed" in exec_section or "log transform" in exec_section


class TestLLMCallCount:
    def test_no_extra_llm_calls_for_exec_summary(self, tmp_path):
        """Executive summary reuses the final LLM text response; no extra API call needed."""
        llm = FakeLLMClient(responses=_FULL_RESPONSES)
        run_agent(
            csv_path=FEAT_CSV,
            output_dir=str(tmp_path),
            llm_client=llm,
            interactive=False,
        )
        # 4 tool calls + 1 final text = 5 total LLM calls (no extra call for exec summary)
        assert llm.call_count == 5
