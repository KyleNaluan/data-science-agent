import json
from pathlib import Path


class TraceLogger:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("")  # truncate / create

    def append(self, entry: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
