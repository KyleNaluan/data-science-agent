"""
Integration tests for the distribution analysis tool + chart output (issue #3).

All tests use FakeLLMClient — no real API calls.
"""
import re
from pathlib import Path

import pytest

from ds_agent.cli import run_agent
from ds_agent.llm.fake import FakeLLMClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SIMPLE_CSV = str(FIXTURES_DIR / "simple.csv")

# LLM calls schema inference then distribution analysis, then stops.
_DISTRIBUTION_RESPONSES = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running schema inference first."},
            {"type": "tool_use", "id": "tu_001", "name": "schema_inference", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Now computing distributions for numeric columns."},
            {"type": "tool_use", "id": "tu_002", "name": "distribution_analysis", "input": {}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Analysis complete."}],
    },
]


@pytest.fixture
def fake_dist_llm() -> FakeLLMClient:
    return FakeLLMClient(responses=_DISTRIBUTION_RESPONSES)


class TestChartGeneration:
    def test_charts_directory_is_created(self, tmp_path, fake_dist_llm):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
            interactive=False,
        )
        assert (tmp_path / "charts").is_dir()

    def test_histogram_pngs_exist_on_disk(self, tmp_path, fake_dist_llm):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
            interactive=False,
        )
        pngs = list((tmp_path / "charts").glob("*.png"))
        assert len(pngs) > 0

    def test_all_referenced_chart_files_exist_on_disk(self, tmp_path, fake_dist_llm):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        refs = re.findall(r"!\[.*?\]\(charts/(.*?\.png)\)", content)
        assert len(refs) > 0, "Report should contain at least one chart reference"
        for ref in refs:
            assert (tmp_path / "charts" / ref).exists(), (
                f"Referenced chart '{ref}' not found on disk"
            )


class TestDistributionsSection:
    def test_distributions_section_is_populated(self, tmp_path, fake_dist_llm):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "## Distributions" in content
        assert "_Not yet analyzed in this run._" not in content.split("## Distributions")[1].split("##")[0]

    def test_report_contains_chart_image_links(self, tmp_path, fake_dist_llm):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "charts/" in content
        assert ".png" in content

    def test_distributions_section_has_narrative(self, tmp_path, fake_dist_llm):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        dist_section = content.split("## Distributions")[1].split("## Correlations")[0]
        # Narrative uses "skewed" or "symmetric" language
        assert any(word in dist_section.lower() for word in ["skew", "symmetric", "kurtosis"])

    def test_skew_and_kurtosis_values_in_report(self, tmp_path, fake_dist_llm):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
            interactive=False,
        )
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Skew:" in content
        assert "kurtosis" in content.lower()

    def test_all_five_sections_still_present(self, tmp_path, fake_dist_llm):
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
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


class TestAggregateOnlyInTrace:
    def test_distribution_trace_entry_has_no_raw_rows(self, tmp_path, fake_dist_llm):
        import json
        run_agent(
            csv_path=SIMPLE_CSV,
            output_dir=str(tmp_path),
            llm_client=fake_dist_llm,
            interactive=False,
        )
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        dist_entries = [
            json.loads(l) for l in lines
            if json.loads(l).get("tool_name") == "distribution_analysis"
        ]
        assert len(dist_entries) == 1
        output = dist_entries[0]["tool_output"]
        for col in output.get("columns", []):
            assert "row_values" not in col
            assert "raw_data" not in col
