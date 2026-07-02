"""Unit tests for FeatureSuggestionTool (issue #9)."""
import math

import pandas as pd
import pytest

from ds_agent.tools.feature_suggestion import FeatureSuggestionTool


def _df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_returns_dict_with_suggestions_key(self):
        df = _df(a=[1, 2, 3])
        result = FeatureSuggestionTool().run(df)
        assert "suggestions" in result

    def test_suggestions_is_list(self):
        df = _df(a=[1, 2, 3])
        result = FeatureSuggestionTool().run(df)
        assert isinstance(result["suggestions"], list)

    def test_each_suggestion_has_required_keys(self):
        # Use a skewed column to guarantee at least one suggestion
        df = _df(price=[1, 1, 1, 1, 1, 100, 1000, 10000])
        result = FeatureSuggestionTool().run(df)
        for s in result["suggestions"]:
            assert "columns" in s
            assert "suggestion_type" in s
            assert "rationale" in s
            assert "priority" in s

    def test_columns_field_is_list_of_strings(self):
        df = _df(price=[1, 1, 1, 1, 1, 100, 1000, 10000])
        result = FeatureSuggestionTool().run(df)
        for s in result["suggestions"]:
            assert isinstance(s["columns"], list)
            assert all(isinstance(c, str) for c in s["columns"])

    def test_no_raw_row_data(self):
        df = _df(price=[1, 1, 1, 1, 1, 100, 1000, 10000])
        result = FeatureSuggestionTool().run(df)
        assert "row_values" not in result
        assert "raw_data" not in result
        for s in result["suggestions"]:
            assert "row_values" not in s


# ---------------------------------------------------------------------------
# Skew-based suggestions
# ---------------------------------------------------------------------------

class TestSkewSuggestions:
    def _skewed_df(self):
        # Extreme right skew: small base values + one very large outlier
        return _df(price=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10000])

    def test_highly_skewed_column_produces_log_transform_suggestion(self):
        result = FeatureSuggestionTool().run(self._skewed_df())
        types = [s["suggestion_type"] for s in result["suggestions"]]
        assert "log_transform" in types

    def test_log_transform_suggestion_names_the_column(self):
        result = FeatureSuggestionTool().run(self._skewed_df())
        log_suggestions = [s for s in result["suggestions"] if s["suggestion_type"] == "log_transform"]
        assert any("price" in s["columns"] for s in log_suggestions)

    def test_log_transform_rationale_cites_skew_value(self):
        result = FeatureSuggestionTool().run(self._skewed_df())
        log_suggestions = [s for s in result["suggestions"] if s["suggestion_type"] == "log_transform"]
        assert log_suggestions, "expected at least one log_transform suggestion"
        rationale = log_suggestions[0]["rationale"]
        assert "skew=" in rationale

    def test_log_transform_rationale_cites_column_name(self):
        result = FeatureSuggestionTool().run(self._skewed_df())
        log_suggestions = [s for s in result["suggestions"] if s["suggestion_type"] == "log_transform"]
        rationale = log_suggestions[0]["rationale"]
        assert "price" in rationale

    def test_low_skew_column_does_not_trigger_log_transform(self):
        df = _df(x=list(range(1, 21)))  # uniform → near-zero skew
        result = FeatureSuggestionTool().run(df)
        log_suggestions = [s for s in result["suggestions"] if s["suggestion_type"] == "log_transform"]
        assert len(log_suggestions) == 0

    def test_custom_skew_threshold_respected(self):
        # Moderate skew should not trigger with high threshold
        df = _df(x=[1, 1, 2, 3, 5, 8, 10, 10, 10, 10])
        result_high = FeatureSuggestionTool().run(df, skew_threshold=5.0)
        result_low = FeatureSuggestionTool().run(df, skew_threshold=0.5)
        high_count = sum(1 for s in result_high["suggestions"] if s["suggestion_type"] == "log_transform")
        low_count = sum(1 for s in result_low["suggestions"] if s["suggestion_type"] == "log_transform")
        assert low_count >= high_count

    def test_priority_is_high_for_very_high_skew(self):
        result = FeatureSuggestionTool().run(self._skewed_df())
        log_suggestions = [s for s in result["suggestions"] if s["suggestion_type"] == "log_transform"]
        assert any(s["priority"] == "high" for s in log_suggestions)


# ---------------------------------------------------------------------------
# High-cardinality encoding suggestions
# ---------------------------------------------------------------------------

class TestEncodingSuggestions:
    def _high_card_df(self):
        categories = [f"cat_{i:02d}" for i in range(1, 21)]  # 20 unique values
        return _df(category=categories)

    def test_high_cardinality_categorical_produces_encoding_suggestion(self):
        result = FeatureSuggestionTool().run(self._high_card_df())
        types = [s["suggestion_type"] for s in result["suggestions"]]
        assert "encoding" in types

    def test_encoding_suggestion_names_column(self):
        result = FeatureSuggestionTool().run(self._high_card_df())
        enc = [s for s in result["suggestions"] if s["suggestion_type"] == "encoding"]
        assert any("category" in s["columns"] for s in enc)

    def test_high_cardinality_rationale_cites_unique_count(self):
        result = FeatureSuggestionTool().run(self._high_card_df())
        enc = [s for s in result["suggestions"] if s["suggestion_type"] == "encoding"]
        assert enc, "expected at least one encoding suggestion"
        rationale = enc[0]["rationale"]
        assert "20" in rationale

    def test_high_cardinality_rationale_cites_column_name(self):
        result = FeatureSuggestionTool().run(self._high_card_df())
        enc = [s for s in result["suggestions"] if s["suggestion_type"] == "encoding"]
        assert any("category" in s["rationale"] for s in enc)

    def test_high_cardinality_suggestion_has_high_priority(self):
        result = FeatureSuggestionTool().run(self._high_card_df())
        enc = [s for s in result["suggestions"] if s["suggestion_type"] == "encoding"]
        high_card = [s for s in enc if "20" in s["rationale"]]
        assert any(s["priority"] == "high" for s in high_card)

    def test_custom_cardinality_threshold_respected(self):
        df = _df(cat=["a", "b", "c", "d", "e"])  # 5 unique
        result_low = FeatureSuggestionTool().run(df, high_cardinality_threshold=3)
        result_high = FeatureSuggestionTool().run(df, high_cardinality_threshold=10)
        low_high = [s for s in result_low["suggestions"] if "high cardinality" in s["rationale"]]
        high_high = [s for s in result_high["suggestions"] if "high cardinality" in s["rationale"]]
        assert len(low_high) > len(high_high)

    def test_low_cardinality_categorical_produces_one_hot_suggestion(self):
        df = _df(status=["active", "inactive", "pending"])
        result = FeatureSuggestionTool().run(df)
        enc = [s for s in result["suggestions"] if s["suggestion_type"] == "encoding"]
        assert any("one-hot" in s["rationale"].lower() for s in enc)


# ---------------------------------------------------------------------------
# Correlation-based interaction term suggestions
# ---------------------------------------------------------------------------

class TestInteractionTermSuggestions:
    def _correlated_df(self):
        a = list(range(1, 21))
        b = [x * 2 + 1 for x in a]  # perfectly correlated with a
        return _df(a=a, b=b)

    def test_correlated_pair_produces_interaction_term_suggestion(self):
        result = FeatureSuggestionTool().run(self._correlated_df())
        types = [s["suggestion_type"] for s in result["suggestions"]]
        assert "interaction_term" in types

    def test_interaction_term_suggestion_names_both_columns(self):
        result = FeatureSuggestionTool().run(self._correlated_df())
        it = [s for s in result["suggestions"] if s["suggestion_type"] == "interaction_term"]
        assert it, "expected at least one interaction_term suggestion"
        cols = it[0]["columns"]
        assert "a" in cols and "b" in cols

    def test_interaction_term_rationale_cites_correlation_value(self):
        result = FeatureSuggestionTool().run(self._correlated_df())
        it = [s for s in result["suggestions"] if s["suggestion_type"] == "interaction_term"]
        rationale = it[0]["rationale"]
        assert "r=" in rationale

    def test_interaction_term_rationale_cites_column_names(self):
        result = FeatureSuggestionTool().run(self._correlated_df())
        it = [s for s in result["suggestions"] if s["suggestion_type"] == "interaction_term"]
        rationale = it[0]["rationale"]
        assert "a" in rationale and "b" in rationale

    def test_uncorrelated_pair_does_not_produce_interaction_suggestion(self):
        import numpy as np
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.standard_normal(100), "y": rng.standard_normal(100)})
        result = FeatureSuggestionTool().run(df, correlation_threshold=0.5)
        it = [s for s in result["suggestions"] if s["suggestion_type"] == "interaction_term"]
        assert len(it) == 0

    def test_custom_correlation_threshold_respected(self):
        a = list(range(1, 11))
        b = [x * 2 for x in a]
        df = _df(a=a, b=b)
        result_strict = FeatureSuggestionTool().run(df, correlation_threshold=0.99)
        result_loose = FeatureSuggestionTool().run(df, correlation_threshold=0.5)
        strict_it = [s for s in result_strict["suggestions"] if s["suggestion_type"] == "interaction_term"]
        loose_it = [s for s in result_loose["suggestions"] if s["suggestion_type"] == "interaction_term"]
        # perfectly correlated pair appears in both; both should find it
        assert len(loose_it) >= len(strict_it)

    def test_each_pair_appears_only_once(self):
        a = list(range(1, 11))
        df = _df(a=a, b=[x * 2 for x in a], c=[x * 3 for x in a])
        result = FeatureSuggestionTool().run(df)
        it = [s for s in result["suggestions"] if s["suggestion_type"] == "interaction_term"]
        seen = set()
        for s in it:
            key = tuple(sorted(s["columns"]))
            assert key not in seen, f"duplicate pair: {key}"
            seen.add(key)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframe_returns_no_suggestions(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = FeatureSuggestionTool().run(df)
        assert result["suggestions"] == []

    def test_single_numeric_column_no_interaction_term(self):
        df = _df(x=[1, 2, 3, 4, 5])
        result = FeatureSuggestionTool().run(df)
        it = [s for s in result["suggestions"] if s["suggestion_type"] == "interaction_term"]
        assert len(it) == 0

    def test_all_numeric_no_encoding_suggestions(self):
        df = _df(a=[1, 2, 3], b=[4, 5, 6])
        result = FeatureSuggestionTool().run(df)
        enc = [s for s in result["suggestions"] if s["suggestion_type"] == "encoding"]
        assert len(enc) == 0

    def test_all_same_value_categorical_no_suggestion(self):
        # Single unique value → no meaningful encoding needed
        df = _df(cat=["same", "same", "same"])
        result = FeatureSuggestionTool().run(df)
        enc = [s for s in result["suggestions"] if s["suggestion_type"] == "encoding"]
        assert len(enc) == 0

    def test_mixed_df_produces_multiple_suggestion_types(self):
        a = list(range(1, 21))
        df = pd.DataFrame({
            "price": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765],
            "category": [f"cat_{i:02d}" for i in range(1, 21)],
            "x": a,
            "y": [v * 2 for v in a],
        })
        result = FeatureSuggestionTool().run(df)
        types = {s["suggestion_type"] for s in result["suggestions"]}
        assert "log_transform" in types
        assert "encoding" in types
        assert "interaction_term" in types
