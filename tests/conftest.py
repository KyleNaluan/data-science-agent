from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_csv() -> Path:
    return FIXTURES_DIR / "simple.csv"


@pytest.fixture
def ambiguous_id_csv() -> Path:
    return FIXTURES_DIR / "ambiguous_id.csv"


@pytest.fixture
def tiny_csv() -> Path:
    return FIXTURES_DIR / "tiny_dataset.csv"
