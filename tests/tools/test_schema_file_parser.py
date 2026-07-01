"""Unit tests for parse_schema_file (issue #7)."""
import json
from pathlib import Path

import pytest

from ds_agent.tools.schema_file_parser import parse_schema_file


@pytest.fixture
def schema_dir(tmp_path) -> Path:
    return tmp_path


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestParseSchemaFile:
    def test_returns_dict(self, schema_dir):
        f = _write(schema_dir / "s.json", {"columns": {"age": "numeric"}})
        result = parse_schema_file(f)
        assert isinstance(result, dict)

    def test_parses_column_types(self, schema_dir):
        f = _write(schema_dir / "s.json", {"columns": {"age": "numeric", "dept": "categorical"}})
        result = parse_schema_file(f)
        assert result["age"] == "numeric"
        assert result["dept"] == "categorical"

    def test_all_valid_types_accepted(self, schema_dir):
        valid = {t: t for t in ["numeric", "categorical", "datetime", "identifier", "boolean", "text"]}
        f = _write(schema_dir / "s.json", {"columns": valid})
        result = parse_schema_file(f)
        assert set(result.values()) == set(valid.values())

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_schema_file(Path("/nonexistent/schema.json"))

    def test_missing_columns_key_raises_value_error(self, schema_dir):
        f = _write(schema_dir / "s.json", {"types": {"age": "numeric"}})
        with pytest.raises(ValueError, match="'columns'"):
            parse_schema_file(f)

    def test_unknown_type_raises_value_error(self, schema_dir):
        f = _write(schema_dir / "s.json", {"columns": {"age": "integer"}})
        with pytest.raises(ValueError, match="integer"):
            parse_schema_file(f)

    def test_columns_not_dict_raises_value_error(self, schema_dir):
        f = _write(schema_dir / "s.json", {"columns": ["age", "numeric"]})
        with pytest.raises(ValueError):
            parse_schema_file(f)

    def test_empty_columns_returns_empty_dict(self, schema_dir):
        f = _write(schema_dir / "s.json", {"columns": {}})
        result = parse_schema_file(f)
        assert result == {}

    def test_agree_fixture_parses(self):
        fixture = Path(__file__).parent.parent / "fixtures" / "schema_hint_agree.json"
        result = parse_schema_file(fixture)
        assert "age" in result
        assert result["age"] == "numeric"

    def test_conflict_fixture_parses(self):
        fixture = Path(__file__).parent.parent / "fixtures" / "schema_hint_conflict.json"
        result = parse_schema_file(fixture)
        assert result["age"] == "categorical"
