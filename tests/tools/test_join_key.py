"""Unit tests for JoinKeyInferenceTool (issue #8)."""
import pandas as pd
import pytest

from ds_agent.tools.join_key import (
    JoinKeyInferenceTool,
    _dtype_compatible,
    _name_score,
    _overlap_score,
    _uniqueness_score,
    score_candidates,
)


class TestNameScore:
    def test_exact_match_returns_one(self):
        assert _name_score("user_id", "user_id") == pytest.approx(1.0)

    def test_case_insensitive_exact(self):
        assert _name_score("UserID", "userid") == pytest.approx(1.0)

    def test_id_suffix_stripped(self):
        # "user_id" stripped of "_id" == "user" → base match with "user"
        score = _name_score("user_id", "user")
        assert score == pytest.approx(0.75)

    def test_no_similarity_returns_zero(self):
        assert _name_score("product_code", "customer_name") == pytest.approx(0.0)

    def test_same_base_different_suffix(self):
        # "order_key" stripped of "_key" == "order" → base match with "order"
        score = _name_score("order_key", "order")
        assert score == pytest.approx(0.75)


class TestDtypeCompatible:
    def test_both_numeric_compatible(self):
        assert _dtype_compatible("int64", "float64") is True

    def test_both_string_compatible(self):
        assert _dtype_compatible("object", "string") is True

    def test_numeric_vs_string_incompatible(self):
        assert _dtype_compatible("int64", "object") is False


class TestOverlapScore:
    def test_identical_sets_returns_one(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert _overlap_score(s, s) == pytest.approx(1.0)

    def test_disjoint_sets_returns_zero(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([4, 5, 6])
        assert _overlap_score(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = pd.Series([1, 2, 3, 4])
        b = pd.Series([3, 4, 5, 6])
        score = _overlap_score(a, b)
        assert 0.0 < score < 1.0


class TestUniquenessScore:
    def test_all_unique_returns_one(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert _uniqueness_score(s) == pytest.approx(1.0)

    def test_all_same_returns_near_zero(self):
        s = pd.Series([1, 1, 1, 1, 1])
        assert _uniqueness_score(s) == pytest.approx(1 / 5, abs=1e-4)

    def test_empty_series_returns_zero(self):
        s = pd.Series([], dtype=float)
        assert _uniqueness_score(s) == pytest.approx(0.0)


class TestScoreCandidates:
    def test_exact_name_match_gets_high_score(self):
        df_a = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "value": [10, 20, 30, 40, 50]})
        df_b = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "score": [90, 85, 92, 78, 88]})
        candidates = score_candidates(df_a, df_b)
        assert len(candidates) > 0
        top = candidates[0]
        assert top["col_a"] == "user_id"
        assert top["col_b"] == "user_id"
        assert top["score"] > 0.7

    def test_no_name_match_returns_empty(self):
        df_a = pd.DataFrame({"product_code": ["P1", "P2"], "price": [10, 20]})
        df_b = pd.DataFrame({"customer_id": [1, 2], "region": ["N", "S"]})
        candidates = score_candidates(df_a, df_b)
        assert candidates == []

    def test_dtype_incompatible_skipped(self):
        df_a = pd.DataFrame({"id": [1, 2, 3]})
        df_b = pd.DataFrame({"id": ["a", "b", "c"]})
        candidates = score_candidates(df_a, df_b)
        # int vs object — incompatible, so no candidates
        assert candidates == []

    def test_candidates_sorted_by_score_descending(self):
        df_a = pd.DataFrame({"id": [1, 2, 3], "order_id": [10, 20, 30]})
        df_b = pd.DataFrame({"id": [1, 2, 3], "order_id": [10, 20, 30]})
        candidates = score_candidates(df_a, df_b)
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)


class TestJoinKeyInferenceTool:
    def test_no_extra_df_returns_no_second_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = JoinKeyInferenceTool().run(df)
        assert result["confidence"] == "no_second_dataframe"

    def test_high_confidence_on_clear_match(self):
        df_a = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
        df_b = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "score": [90, 85, 92, 78, 88]})
        result = JoinKeyInferenceTool().run(df_a, extra_df=df_b)
        assert result["confidence"] == "high"
        assert result["best_candidate"] is not None
        assert result["best_candidate"]["col_a"] == "user_id"

    def test_no_candidates_confidence(self):
        df_a = pd.DataFrame({"product_code": ["P1", "P2", "P3"]})
        df_b = pd.DataFrame({"customer_id": [1, 2, 3]})
        result = JoinKeyInferenceTool().run(df_a, extra_df=df_b)
        assert result["confidence"] == "no_candidates"
        assert result["best_candidate"] is None

    def test_returns_top_candidates_list(self):
        df_a = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
        df_b = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "score": [90, 85, 92, 78, 88]})
        result = JoinKeyInferenceTool().run(df_a, extra_df=df_b)
        assert isinstance(result["candidates"], list)
        assert len(result["candidates"]) <= 10

    def test_aggregate_only_no_row_data(self):
        df_a = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
        df_b = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "score": [90, 85, 92, 78, 88]})
        result = JoinKeyInferenceTool().run(df_a, extra_df=df_b)
        assert "row_values" not in result
        for c in result["candidates"]:
            assert "row_values" not in c

    def test_ambiguous_case_produces_multiple_candidates(self):
        # Both id and order_id match exactly in both tables
        df_a = pd.DataFrame({"id": [1, 2, 3, 4, 5], "order_id": [10, 20, 30, 40, 50], "x": [1, 2, 3, 4, 5]})
        df_b = pd.DataFrame({"id": [1, 2, 3, 4, 5], "order_id": [10, 20, 30, 40, 50], "y": [9, 8, 7, 6, 5]})
        result = JoinKeyInferenceTool().run(df_a, extra_df=df_b)
        assert len(result["candidates"]) > 1
