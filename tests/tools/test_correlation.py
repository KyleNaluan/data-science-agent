"""Unit tests for CorrelationTool (issue #6)."""
import pandas as pd
import pytest

from ds_agent.tools.correlation import CorrelationTool, _describe_correlation


def _df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestCorrelationToolOutput:
    def test_returns_dict_with_required_keys(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[2, 4, 6, 8, 10])
        result = CorrelationTool().run(df)
        assert "columns" in result
        assert "matrix" in result
        assert "ranked_pairs" in result
        assert "narrative_pairs" in result

    def test_fewer_than_two_numeric_columns_returns_empty(self):
        df = _df(a=[1, 2, 3])
        result = CorrelationTool().run(df)
        assert result["columns"] == []
        assert result["matrix"] == {}
        assert result["ranked_pairs"] == []

    def test_perfectly_correlated_columns(self):
        # b = 2*a → r = 1.0
        a = list(range(1, 11))
        b = [x * 2 for x in a]
        df = _df(a=a, b=b)
        result = CorrelationTool().run(df)
        pair = result["ranked_pairs"][0]
        assert pair["abs_correlation"] == pytest.approx(1.0, abs=1e-4)
        assert pair["correlation"] == pytest.approx(1.0, abs=1e-4)

    def test_perfectly_negatively_correlated(self):
        a = list(range(1, 11))
        c = list(range(10, 0, -1))  # reverse → r = -1.0
        df = _df(a=a, c=c)
        result = CorrelationTool().run(df)
        pair = result["ranked_pairs"][0]
        assert pair["correlation"] == pytest.approx(-1.0, abs=1e-4)

    def test_uncorrelated_columns_not_in_ranked_pairs(self):
        import numpy as np
        rng = np.random.default_rng(0)
        a = rng.standard_normal(100)
        d = rng.standard_normal(100)
        df = pd.DataFrame({"a": a, "d": d})
        result = CorrelationTool().run(df, min_abs_correlation=0.5)
        # With random uncorrelated data, no pair should exceed 0.5 threshold for n=100
        for pair in result["ranked_pairs"]:
            assert pair["abs_correlation"] >= 0.5

    def test_matrix_is_symmetric(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[5, 4, 3, 2, 1], c=[1, 1, 2, 2, 3])
        result = CorrelationTool().run(df)
        matrix = result["matrix"]
        for col_a in matrix:
            for col_b in matrix[col_a]:
                assert matrix[col_a][col_b] == pytest.approx(matrix[col_b][col_a], abs=1e-4)

    def test_diagonal_is_one(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[2, 3, 4, 5, 6])
        result = CorrelationTool().run(df)
        for col in result["columns"]:
            assert result["matrix"][col][col] == pytest.approx(1.0, abs=1e-4)

    def test_upper_triangle_only_in_ranked_pairs(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[2, 4, 6, 8, 10], c=[5, 4, 3, 2, 1])
        result = CorrelationTool().run(df)
        seen = set()
        for pair in result["ranked_pairs"]:
            key = tuple(sorted([pair["col_a"], pair["col_b"]]))
            assert key not in seen, "duplicate pair found"
            seen.add(key)

    def test_top_n_limits_ranked_pairs(self):
        df = _df(
            a=[1, 2, 3, 4, 5],
            b=[2, 4, 6, 8, 10],
            c=[5, 4, 3, 2, 1],
            d=[1, 3, 5, 7, 9],
        )
        result = CorrelationTool().run(df, top_n=2)
        assert len(result["ranked_pairs"]) <= 2

    def test_ranked_pairs_sorted_by_abs_correlation(self):
        a = list(range(1, 11))
        df = _df(a=a, b=[x * 2 for x in a], c=[x + 0.5 * i for i, x in enumerate(a)])
        result = CorrelationTool().run(df)
        scores = [p["abs_correlation"] for p in result["ranked_pairs"]]
        assert scores == sorted(scores, reverse=True)

    def test_non_numeric_columns_excluded(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[2, 4, 6, 8, 10], cat=["x", "y", "x", "y", "x"])
        result = CorrelationTool().run(df)
        assert "cat" not in result["columns"]


class TestAggregateOnly:
    def test_no_row_level_data(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[5, 4, 3, 2, 1])
        result = CorrelationTool().run(df)
        assert "row_values" not in result
        assert "raw_data" not in result
        for pair in result["ranked_pairs"]:
            assert "row_values" not in pair

    def test_matrix_values_are_scalars(self):
        df = _df(a=[1, 2, 3, 4, 5], b=[2, 4, 6, 8, 10])
        result = CorrelationTool().run(df)
        for col_a, row in result["matrix"].items():
            for col_b, val in row.items():
                assert isinstance(val, float)


class TestNarrative:
    def test_very_strong_positive(self):
        desc = _describe_correlation("a", "b", 0.95)
        assert "very strong" in desc.lower()
        assert "positively" in desc.lower()

    def test_strong_negative(self):
        desc = _describe_correlation("a", "c", -0.75)
        assert "strong" in desc.lower()
        assert "negatively" in desc.lower()

    def test_moderate_positive(self):
        desc = _describe_correlation("x", "y", 0.55)
        assert "moderate" in desc.lower()

    def test_weak_positive(self):
        desc = _describe_correlation("x", "y", 0.35)
        assert "weak" in desc.lower()

    def test_narrative_includes_r_value(self):
        desc = _describe_correlation("a", "b", 0.85)
        assert "0.850" in desc

    def test_narrative_includes_column_names(self):
        desc = _describe_correlation("income", "expense", 0.7)
        assert "income" in desc
        assert "expense" in desc

    def test_narrative_pairs_present_in_tool_output(self):
        a = list(range(1, 11))
        df = _df(a=a, b=[x * 2 for x in a])
        result = CorrelationTool().run(df)
        assert len(result["narrative_pairs"]) > 0
        for narrative in result["narrative_pairs"]:
            assert isinstance(narrative, str)
            assert len(narrative) > 0
