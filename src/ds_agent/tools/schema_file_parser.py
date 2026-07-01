from __future__ import annotations

import json
from pathlib import Path

_VALID_TYPES = frozenset([
    "numeric", "categorical", "datetime", "identifier", "boolean", "text",
])


def parse_schema_file(path: str | Path) -> dict[str, str]:
    """
    Parse a JSON schema file containing column type hints.

    Expected format::

        {
          "columns": {
            "column_name": "type",
            ...
          }
        }

    Valid types: numeric, categorical, datetime, identifier, boolean, text.

    Returns: mapping of column_name → type string.
    Raises FileNotFoundError if the file does not exist.
    Raises ValueError on JSON structure or unknown type errors.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Schema file not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "columns" not in data:
        raise ValueError(
            f"Schema file must be a JSON object with a 'columns' key: {p}"
        )

    cols = data["columns"]
    if not isinstance(cols, dict):
        raise ValueError(
            f"'columns' must be a JSON object mapping column names to types: {p}"
        )

    result: dict[str, str] = {}
    for col_name, col_type in cols.items():
        if col_type not in _VALID_TYPES:
            raise ValueError(
                f"Unknown type '{col_type}' for column '{col_name}'. "
                f"Valid types: {sorted(_VALID_TYPES)}"
            )
        result[str(col_name)] = str(col_type)

    return result
