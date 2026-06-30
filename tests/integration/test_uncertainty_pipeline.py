"""
Integration tests for the uncertainty mechanism (issue #2).

All tests use FakeLLMClient — no real API calls.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
AMBIGUOUS_CSV = str(FIXTURES_DIR / "ambiguous_id.csv")
TINY_CSV = str(FIXTURES_DIR / "tiny_dataset.csv")

_SCHEMA_RESPONSES = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running schema inference to understand column types."},
            {"type": "tool_use", "id": "tu_001", "name": "schema_inference", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Schema analysis complete."}],
    },
]


def _make_fake_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_SCHEMA_RESPONSES)


class TestNonInteractivePath:
    def test_ambiguous_column_scorecard_has_flagged_assumptions(self, tmp_path):
        run_agent(
            csv_path=AMBIGUOUS_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_fake_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Flagged assumptions" in content

    def test_flagged_assumptions_name_the_ambiguous_column(self, tmp_path):
        run_agent(
            csv_path=AMBIGUOUS_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_fake_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # ambiguous_id.csv has customer_id and product_id as high-cardinality identifier columns
        assert "customer_id" in content or "product_id" in content

    def test_flagged_assumptions_record_the_assumed_type(self, tmp_path):
        run_agent(
            csv_path=AMBIGUOUS_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_fake_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # The default assumption for ambiguous identifiers is "identifier"
        assert "identifier" in content

    def test_tiny_dataset_produces_flagged_assumption(self, tmp_path):
        run_agent(
            csv_path=TINY_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_fake_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "tiny_dataset" in content

    def test_tiny_dataset_mentions_row_count(self, tmp_path):
        run_agent(
            csv_path=TINY_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_fake_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "10 rows" in content

    def test_non_interactive_run_does_not_block(self, tmp_path):
        # If this test completes, the run didn't block waiting for stdin
        run_agent(
            csv_path=AMBIGUOUS_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_fake_llm(),
            interactive=False,
        )
        assert (tmp_path / "report.md").exists()


class TestInteractivePath:
    def test_interactive_run_prompts_and_completes(self, tmp_path):
        with patch("builtins.input", return_value="numeric"):
            run_agent(
                csv_path=AMBIGUOUS_CSV,
                output_dir=str(tmp_path),
                llm_client=_make_fake_llm(),
                interactive=True,
            )
        assert (tmp_path / "report.md").exists()

    def test_interactive_run_produces_no_flagged_assumptions(self, tmp_path):
        with patch("builtins.input", return_value="numeric"):
            run_agent(
                csv_path=AMBIGUOUS_CSV,
                output_dir=str(tmp_path),
                llm_client=_make_fake_llm(),
                interactive=True,
            )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Flagged assumptions" not in content

    def test_interactive_type_override_removes_ambiguous_marker(self, tmp_path):
        with patch("builtins.input", return_value="numeric"):
            run_agent(
                csv_path=AMBIGUOUS_CSV,
                output_dir=str(tmp_path),
                llm_client=_make_fake_llm(),
                interactive=True,
            )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # After user clarifies types, [!] markers should be gone
        assert "[!]" not in content

    def test_all_five_sections_present_in_uncertainty_run(self, tmp_path):
        run_agent(
            csv_path=AMBIGUOUS_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_fake_llm(),
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
