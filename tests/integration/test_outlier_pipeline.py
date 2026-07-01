"""
Integration tests for outlier detection + chart output (issue #5).
All tests use FakeLLMClient — no real API calls.
"""
import json
import re
from pathlib import Path

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
OUTLIER_CSV = str(FIXTURES_DIR / "outlier_data.csv")

_OUTLIER_RESPONSES = [
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
            {"type": "text", "text": "Computing distributions."},
            {"type": "tool_use", "id": "tu_002", "name": "distribution_analysis", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running outlier detection."},
            {"type": "tool_use", "id": "tu_003", "name": "outlier_detection", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Analysis complete."}],
    },
]


def _make_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_OUTLIER_RESPONSES)


class TestOutlierChartGeneration:
    def test_outlier_png_created(self, tmp_path):
        run_agent(
            csv_path=OUTLIER_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        pngs = list((tmp_path / "charts").glob("*_outliers.png"))
        assert len(pngs) > 0

    def test_outlier_chart_referenced_in_report(self, tmp_path):
        run_agent(
            csv_path=OUTLIER_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "_outliers.png" in content

    def test_all_referenced_outlier_charts_exist(self, tmp_path):
        run_agent(
            csv_path=OUTLIER_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        refs = re.findall(r"!\[.*?\]\(charts/(.*?_outliers\.png)\)", content)
        assert len(refs) > 0, "Report should reference at least one outlier chart"
        for ref in refs:
            assert (tmp_path / "charts" / ref).exists(), f"Missing chart: {ref}"


class TestOutlierDistributionsSection:
    def test_outlier_narrative_in_distributions_section(self, tmp_path):
        run_agent(
            csv_path=OUTLIER_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        dist_section = content.split("## Distributions")[1].split("## Correlations")[0]
        assert any(
            word in dist_section.lower()
            for word in ["outlier", "iqr", "z-score"]
        )

    def test_both_method_counts_reported(self, tmp_path):
        run_agent(
            csv_path=OUTLIER_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        # Both IQR and z-score counts should appear in the distributions section
        dist_section = content.split("## Distributions")[1].split("## Correlations")[0]
        assert "IQR" in dist_section
        assert ("Z-score" in dist_section or "z-score" in dist_section)

    def test_all_five_sections_present(self, tmp_path):
        run_agent(
            csv_path=OUTLIER_CSV,
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


class TestAggregateOnlyOutlier:
    def test_trace_has_no_raw_rows(self, tmp_path):
        run_agent(
            csv_path=OUTLIER_CSV,
            output_dir=str(tmp_path),
            llm_client=_make_llm(),
            interactive=False,
        )
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        outlier_entries = [
            json.loads(l) for l in lines
            if json.loads(l).get("tool_name") == "outlier_detection"
        ]
        assert len(outlier_entries) == 1
        output = outlier_entries[0]["tool_output"]
        for col in output.get("columns", []):
            assert "row_values" not in col
            assert "raw_data" not in col
