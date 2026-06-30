from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import Any

TINY_DATASET_THRESHOLD = 100


@dataclass
class UncertaintyTrigger:
    trigger_type: str
    question: str
    default: Any
    context: dict = field(default_factory=dict)


@dataclass
class FlaggedAssumption:
    trigger_type: str
    question: str
    assumption: Any
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def is_interactive() -> bool:
    return sys.stdin.isatty()


def handle_uncertainty(
    trigger: UncertaintyTrigger,
    interactive: bool | None = None,
) -> tuple[Any, FlaggedAssumption | None]:
    """
    Handle an uncertainty trigger.

    Returns (resolved_value, flagged_assumption_or_None).
    Interactive: prompts the user and returns their answer; no FlaggedAssumption recorded.
    Non-interactive: returns the documented default and records a FlaggedAssumption.
    """
    if interactive is None:
        interactive = is_interactive()

    if interactive:
        print(f"\n[Uncertainty] {trigger.question}")
        print(f"  Default: {trigger.default!r}")
        answer = input("  Your answer (or press Enter to accept default): ").strip()
        resolved = answer if answer else trigger.default
        return resolved, None

    return trigger.default, FlaggedAssumption(
        trigger_type=trigger.trigger_type,
        question=trigger.question,
        assumption=trigger.default,
        context=trigger.context,
    )
