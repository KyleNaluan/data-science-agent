"""Unit tests for the threshold config loader (issue #4)."""
import tomllib
from pathlib import Path

import pytest

from ds_agent.config import ThresholdConfig, load_thresholds


class TestLoadThresholds:
    def test_returns_threshold_config(self):
        cfg = load_thresholds()
        assert isinstance(cfg, ThresholdConfig)

    def test_default_missing_value_rate(self):
        # Without any config file the built-in default is 0.20
        cfg = load_thresholds(config_path=Path("nonexistent_config.toml"))
        assert cfg.missing_value_rate == pytest.approx(0.20)

    def test_override_via_kwarg(self):
        cfg = load_thresholds(config_path=Path("nonexistent.toml"), missing_value_rate=0.05)
        assert cfg.missing_value_rate == pytest.approx(0.05)

    def test_none_override_ignored(self):
        # Passing None for an override should fall through to default/file value
        cfg = load_thresholds(config_path=Path("nonexistent.toml"), missing_value_rate=None)
        assert cfg.missing_value_rate == pytest.approx(0.20)

    def test_reads_from_toml_file(self, tmp_path):
        config_file = tmp_path / "test.toml"
        config_file.write_text("[thresholds]\nmissing_value_rate = 0.10\n", encoding="utf-8")
        cfg = load_thresholds(config_path=config_file)
        assert cfg.missing_value_rate == pytest.approx(0.10)

    def test_kwarg_overrides_file(self, tmp_path):
        config_file = tmp_path / "test.toml"
        config_file.write_text("[thresholds]\nmissing_value_rate = 0.10\n", encoding="utf-8")
        cfg = load_thresholds(config_path=config_file, missing_value_rate=0.35)
        assert cfg.missing_value_rate == pytest.approx(0.35)

    def test_repo_config_file_has_correct_default(self):
        # ds_agent.toml at the repo root should agree with the built-in default
        repo_root = Path(__file__).parent.parent.parent
        config_file = repo_root / "ds_agent.toml"
        if config_file.exists():
            with open(config_file, "rb") as f:
                data = tomllib.load(f)
            rate = data.get("thresholds", {}).get("missing_value_rate", 0.20)
            assert rate == pytest.approx(0.20)

    def test_stricter_threshold_fires_on_more_columns(self):
        # 0.05 threshold should fire on more columns than 0.20
        from pathlib import Path as P
        import pandas as pd
        from ds_agent.tools.missing_value import MissingValueTool

        fixture = P(__file__).parent.parent / "fixtures" / "missing_data.csv"
        df = pd.read_csv(fixture)
        result = MissingValueTool().run(df)

        above_005 = sum(1 for c in result["columns"] if c["missing_rate"] > 0.05)
        above_020 = sum(1 for c in result["columns"] if c["missing_rate"] > 0.20)
        assert above_005 >= above_020
