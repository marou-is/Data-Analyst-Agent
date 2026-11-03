import pytest

from analyst.data.clean import clean
from analyst.synthetic import make_synthetic_series


@pytest.fixture
def synthetic():
    """``(raw_frame, spec, anomaly_windows)`` with known defects."""
    return make_synthetic_series()


@pytest.fixture
def clean_frame(synthetic):
    raw, spec, _ = synthetic
    return clean(raw, spec).data
