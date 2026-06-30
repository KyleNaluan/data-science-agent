from unittest.mock import patch

import pytest

from ds_agent.uncertainty import (
    TINY_DATASET_THRESHOLD,
    FlaggedAssumption,
    UncertaintyTrigger,
    handle_uncertainty,
)


def _make_trigger(
    trigger_type: str = "ambiguous_column_type",
    question: str = "Is this column an identifier?",
    default: str = "identifier",
    context: dict | None = None,
) -> UncertaintyTrigger:
    return UncertaintyTrigger(
        trigger_type=trigger_type,
        question=question,
        default=default,
        context=context or {"column": "id"},
    )


class TestNonInteractivePath:
    def test_returns_default_value(self):
        trigger = _make_trigger(default="identifier")
        resolved, _ = handle_uncertainty(trigger, interactive=False)
        assert resolved == "identifier"

    def test_returns_flagged_assumption(self):
        trigger = _make_trigger()
        _, assumption = handle_uncertainty(trigger, interactive=False)
        assert isinstance(assumption, FlaggedAssumption)

    def test_flagged_assumption_trigger_type(self):
        trigger = _make_trigger(trigger_type="ambiguous_column_type")
        _, assumption = handle_uncertainty(trigger, interactive=False)
        assert assumption.trigger_type == "ambiguous_column_type"

    def test_flagged_assumption_records_default(self):
        trigger = _make_trigger(default="numeric")
        _, assumption = handle_uncertainty(trigger, interactive=False)
        assert assumption.assumption == "numeric"

    def test_flagged_assumption_preserves_context(self):
        ctx = {"column": "customer_id", "ambiguity_reason": "high-cardinality int"}
        trigger = _make_trigger(context=ctx)
        _, assumption = handle_uncertainty(trigger, interactive=False)
        assert assumption.context["column"] == "customer_id"

    def test_tiny_dataset_trigger_non_interactive(self):
        trigger = UncertaintyTrigger(
            trigger_type="tiny_dataset",
            question="Only 10 rows — proceed?",
            default="proceed",
            context={"trigger_id": "__tiny_dataset__", "row_count": 10},
        )
        resolved, assumption = handle_uncertainty(trigger, interactive=False)
        assert resolved == "proceed"
        assert assumption is not None
        assert assumption.trigger_type == "tiny_dataset"

    def test_flagged_assumption_to_dict(self):
        trigger = _make_trigger()
        _, assumption = handle_uncertainty(trigger, interactive=False)
        d = assumption.to_dict()
        assert "trigger_type" in d
        assert "assumption" in d
        assert "context" in d


class TestInteractivePath:
    def test_accepts_user_input(self):
        trigger = _make_trigger(default="identifier")
        with patch("builtins.input", return_value="numeric"):
            resolved, assumption = handle_uncertainty(trigger, interactive=True)
        assert resolved == "numeric"
        assert assumption is None

    def test_empty_input_uses_default(self):
        trigger = _make_trigger(default="identifier")
        with patch("builtins.input", return_value=""):
            resolved, assumption = handle_uncertainty(trigger, interactive=True)
        assert resolved == "identifier"
        assert assumption is None

    def test_no_flagged_assumption_in_interactive_mode(self):
        trigger = _make_trigger()
        with patch("builtins.input", return_value="numeric"):
            _, assumption = handle_uncertainty(trigger, interactive=True)
        assert assumption is None


class TestConstants:
    def test_tiny_dataset_threshold_is_positive(self):
        assert TINY_DATASET_THRESHOLD > 0

    def test_tiny_dataset_threshold_value(self):
        assert TINY_DATASET_THRESHOLD == 100
