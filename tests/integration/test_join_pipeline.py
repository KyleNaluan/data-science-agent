"""
Integration tests for multi-CSV join key inference (issue #8).
All tests use FakeLLMClient — no real API calls.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
LEFT_CSV = str(FIXTURES_DIR / "join_left.csv")
RIGHT_CSV = str(FIXTURES_DIR / "join_right.csv")
AMBIG_A = str(FIXTURES_DIR / "join_ambiguous_a.csv")
AMBIG_B = str(FIXTURES_DIR / "join_ambiguous_b.csv")
NO_MATCH_A = str(FIXTURES_DIR / "join_no_match_a.csv")
SIMPLE_CSV = str(FIXTURES_DIR / "simple.csv")

_SCHEMA_RESPONSES = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running schema inference."},
            {"type": "tool_use", "id": "tu_001", "name": "schema_inference", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Analysis complete."}],
    },
]


def _make_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_SCHEMA_RESPONSES)


class TestSingleCSVUnaffected:
    def test_single_csv_runs_normally(self, tmp_path):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        assert (tmp_path / "report.md").exists()

    def test_single_csv_produces_no_join_assumption(self, tmp_path):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "ambiguous_join_key" not in content


class TestHighConfidenceJoin:
    def test_auto_join_completes(self, tmp_path):
        run_agent(
            csv_path=LEFT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            extra_csv_paths=[RIGHT_CSV],
        )
        assert (tmp_path / "report.md").exists()

    def test_auto_join_produces_unified_report(self, tmp_path):
        run_agent(
            csv_path=LEFT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            extra_csv_paths=[RIGHT_CSV],
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "## Data Quality Scorecard" in content

    def test_auto_join_no_flagged_assumption(self, tmp_path):
        # High-confidence join (user_id match) should not flag assumptions
        run_agent(
            csv_path=LEFT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            extra_csv_paths=[RIGHT_CSV],
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "ambiguous_join_key" not in content

    def test_joined_report_contains_columns_from_both_files(self, tmp_path):
        run_agent(
            csv_path=LEFT_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            extra_csv_paths=[RIGHT_CSV],
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # left has 'value', right has 'score'
        assert "value" in content
        assert "score" in content


class TestAmbiguousJoin:
    def test_ambiguous_join_produces_flagged_assumption(self, tmp_path):
        # Both id and order_id are candidate join keys
        run_agent(
            csv_path=AMBIG_A,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            extra_csv_paths=[AMBIG_B],
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "ambiguous_join_key" in content

    def test_ambiguous_join_names_candidate_columns(self, tmp_path):
        run_agent(
            csv_path=AMBIG_A,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            extra_csv_paths=[AMBIG_B],
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "id" in content or "order_id" in content

    def test_interactive_ambiguous_join_completes(self, tmp_path):
        with patch("builtins.input", return_value="proceed"):
            run_agent(
                csv_path=AMBIG_A,
                output_dir=str(tmp_path),
                llm_client=_make_llm(),
                interactive=True,
                extra_csv_paths=[AMBIG_B],
            )
        assert (tmp_path / "report.md").exists()

    def test_all_five_sections_present_after_join(self, tmp_path):
        run_agent(
            csv_path=AMBIG_A,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            extra_csv_paths=[AMBIG_B],
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
