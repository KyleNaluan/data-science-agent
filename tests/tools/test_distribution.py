import numpy as np
import pandas as pd
import pytest

from ds_agent.tools.distribution import DistributionTool, _describe_distribution


def _df(**kwargs: list) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestDistributionToolOutput:
    def test_returns_dict_with_columns_key(self):
        df = _df(x=[1.0, 2.0, 3.0, 4.0, 5.0])
        tool = DistributionTool()
        result = tool.run(df)
        assert isinstance(result, dict)
        assert "columns" in result

    def test_analyzes_all_numeric_columns_by_default(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[10.0, 20.0, 30.0, 40.0, 50.0], c=["x", "y", "x", "y", "x"])
        tool = DistributionTool()
        result = tool.run(df)
        analyzed = {col["column"] for col in result["columns"]}
        assert "a" in analyzed
        assert "b" in analyzed
        assert "c" not in analyzed  # categorical, not numeric

    def test_analyzes_only_requested_columns(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[10.0, 20.0, 30.0, 40.0, 50.0])
        tool = DistributionTool()
        result = tool.run(df, columns=["a"])
        assert len(result["columns"]) == 1
        assert result["columns"][0]["column"] == "a"

    def test_ignores_non_numeric_in_column_list(self):
        df = _df(a=[1, 2, 3], b=["x", "y", "z"])
        tool = DistributionTool()
        result = tool.run(df, columns=["a", "b"])
        analyzed = {col["column"] for col in result["columns"]}
        assert "a" in analyzed
        assert "b" not in analyzed

    def test_skips_fully_null_column(self):
        df = pd.DataFrame({"a": pd.array([None, None, None], dtype="Float64")})
        tool = DistributionTool()
        result = tool.run(df)
        assert result["columns"] == []


class TestAggregateOnlyOutput:
    def test_no_raw_row_values_in_output(self):
        df = _df(value=[10, 20, 30, 40, 50])
        tool = DistributionTool()
        result = tool.run(df)
        col = result["columns"][0]
        assert "row_values" not in col
        assert "raw_data" not in col
        assert "samples" not in col

    def test_bin_counts_are_aggregates(self):
        df = _df(value=[1, 2, 3, 4, 5] * 10)
        tool = DistributionTool()
        result = tool.run(df)
        col = result["columns"][0]
        assert "bin_counts" in col
        assert "bin_edges" in col
        assert isinstance(col["bin_counts"], list)
        assert all(isinstance(v, int) for v in col["bin_counts"])

    def test_bin_counts_sum_to_sample_count(self):
        values = list(range(1, 51))
        df = _df(v=values)
        tool = DistributionTool()
        result = tool.run(df)
        col = result["columns"][0]
        assert sum(col["bin_counts"]) == col["sample_count"]

    def test_bin_edges_length_is_counts_plus_one(self):
        df = _df(v=list(range(20)))
        tool = DistributionTool()
        result = tool.run(df)
        col = result["columns"][0]
        assert len(col["bin_edges"]) == len(col["bin_counts"]) + 1


class TestSkewAndKurtosis:
    def test_symmetric_data_near_zero_skew(self):
        # Normal-ish data should have near-zero skew
        rng = np.random.default_rng(42)
        values = rng.normal(loc=0, scale=1, size=500).tolist()
        df = _df(v=values)
        tool = DistributionTool()
        result = tool.run(df)
        col = result["columns"][0]
        assert abs(col["skew"]) < 0.5

    def test_right_skewed_data_positive_skew(self):
        # Exponential distribution is right-skewed (skew ~ 2)
        rng = np.random.default_rng(42)
        values = rng.exponential(scale=1, size=500).tolist()
        df = _df(v=values)
        tool = DistributionTool()
        result = tool.run(df)
        col = result["columns"][0]
        assert col["skew"] > 0.5

    def test_output_has_narrative(self):
        df = _df(v=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        tool = DistributionTool()
        result = tool.run(df)
        col = result["columns"][0]
        assert "narrative" in col
        assert len(col["narrative"]) > 0

    def test_sample_count_excludes_nulls(self):
        df = pd.DataFrame({"v": [1.0, 2.0, None, 4.0, 5.0]})
        tool = DistributionTool()
        result = tool.run(df)
        assert result["columns"][0]["sample_count"] == 4


class TestNarrativeDescription:
    def test_symmetric_narrative(self):
        narrative = _describe_distribution("price", skew=0.1, kurt=0.0, n=100)
        assert "symmetric" in narrative.lower()

    def test_right_skewed_narrative(self):
        narrative = _describe_distribution("price", skew=2.0, kurt=0.0, n=100)
        assert "right-skewed" in narrative.lower()

    def test_left_skewed_narrative(self):
        narrative = _describe_distribution("price", skew=-2.0, kurt=0.0, n=100)
        assert "left-skewed" in narrative.lower()

    def test_heavy_tails_narrative(self):
        narrative = _describe_distribution("price", skew=0.0, kurt=2.0, n=100)
        assert "leptokurtic" in narrative.lower() or "heavy" in narrative.lower()

    def test_light_tails_narrative(self):
        narrative = _describe_distribution("price", skew=0.0, kurt=-2.0, n=100)
        assert "platykurtic" in narrative.lower() or "light" in narrative.lower()

    def test_narrative_includes_column_name(self):
        narrative = _describe_distribution("revenue", skew=0.0, kurt=0.0, n=50)
        assert "revenue" in narrative

    def test_narrative_includes_sample_count(self):
        narrative = _describe_distribution("revenue", skew=0.0, kurt=0.0, n=99)
        assert "99" in narrative
