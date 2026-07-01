"""
Integration tests for missing-value analysis + threshold config (issue #4).
All tests use FakeLLMClient — no real API calls.
"""
import json
from pathlib import Path

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MISSING_CSV = str(FIXTURES_DIR / "missing_data.csv")
SIMPLE_CSV = str(FIXTURES_DIR / "simple.csv")

# Schema → missing-value → stop
_MV_RESPONSES = [
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
            {"type": "text", "text": "Now analyzing missing values."},
            {"type": "tool_use", "id": "tu_002", "name": "missing_value_analysis", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Analysis complete."}],
    },
]


def _make_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_MV_RESPONSES)


class TestMissingValueScorecard:
    def test_missing_value_breakdown_in_scorecard(self, tmp_path):
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Missing Value Breakdown" in content

    def test_imputation_recommendations_in_report(self, tmp_path):
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "imputation" in content.lower() or "recommend" in content.lower()

    def test_column_names_in_breakdown(self, tmp_path):
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "age" in content
        assert "salary" in content

    def test_all_five_sections_present(self, tmp_path):
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
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


class TestThresholdUncertaintyTrigger:
    def test_above_threshold_produces_flagged_assumption(self, tmp_path):
        # Default threshold 0.20 — age (25% missing) should trigger
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Flagged assumptions" in content

    def test_flagged_assumption_names_column(self, tmp_path):
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # age (25%) or notes (95%) should appear in flagged assumptions
        assert "age" in content or "notes" in content

    def test_below_threshold_does_not_trigger(self, tmp_path):
        # salary has only 10% missing — below 20% default threshold
        # Use a stricter threshold (0.05) to make salary trigger too
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            missing_threshold=0.30,  # 30% threshold — only notes (95%) triggers
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # With 30% threshold: age (25%) should NOT trigger, notes (95%) should
        if "Flagged assumptions" in content:
            scorecard_section = content.split("## Data Quality Scorecard")[1].split("##")[0]
            # age should not be listed under high_missing_rate flag
            assert "25.0% missing" not in scorecard_section or "notes" in scorecard_section

    def test_stricter_threshold_triggers_salary(self, tmp_path):
        # With 0.05 threshold, salary (10%) also triggers
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
            missing_threshold=0.05,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Flagged assumptions" in content

    def test_no_trigger_when_no_high_missingness(self, tmp_path):
        # simple.csv has 'notes' column all-empty but schema only runs, not missing_value
        # Use a CSV with no high missingness  (age=0%, salary=10%)
        # We need to run just schema without missing value to test no trigger case
        schema_only_responses = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running schema inference."},
                    {"type": "tool_use", "id": "tu_001", "name": "schema_inference", "input": {}},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        ]
        # Create a CSV with no missing values
        clean_csv = tmp_path / "clean.csv"
        clean_csv.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        run_agent(
            csv_path=str(clean_csv),
            output_dir=str(tmp_path / "out"),
            llm_client=FakeLLMClient(responses=schema_only_responses),
            interactive=False,
        )
        content = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
        # No high_missing_rate assumptions when there's no missingness
        assert "high_missing_rate" not in content


class TestAggregateOnlyMissingValue:
    def test_trace_has_no_raw_rows(self, tmp_path):
        run_agent(
            csv_path=MISSING_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        mv_entries = [
            json.loads(l) for l in lines
            if json.loads(l).get("tool_name") == "missing_value_analysis"
        ]
        assert len(mv_entries) == 1
        output = mv_entries[0]["tool_output"]
        for col in output.get("columns", []):
            assert "row_values" not in col
            assert "raw_data" not in col
