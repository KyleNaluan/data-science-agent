from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG_FILE = _REPO_ROOT / "ds_agent.toml"

_BUILT_IN_DEFAULTS: dict[str, float] = {
    "missing_value_rate": 0.20,
}


@dataclass
class ThresholdConfig:
    missing_value_rate: float = 0.20


def load_thresholds(
    config_path: Path | None = None,
    **overrides: float,
) -> ThresholdConfig:
    """
    Load threshold config from ds_agent.toml, falling back to built-in defaults.

    Priority (highest to lowest):
      1. Keyword overrides passed as float (e.g. from CLI flags — only applied when not None)
      2. Values in config_path (default: ds_agent.toml at repo root)
      3. Built-in defaults
    """
    values = dict(_BUILT_IN_DEFAULTS)

    path = config_path if config_path is not None else _CONFIG_FILE
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        values.update(data.get("thresholds", {}))

    for k, v in overrides.items():
        if v is not None:
            values[k] = float(v)

    return ThresholdConfig(missing_value_rate=float(values["missing_value_rate"]))
