import pytest
from streamlit.testing.v1 import AppTest

from analyst.config import PROJECT_ROOT
from analyst.dashboard import state

APP = PROJECT_ROOT / "src" / "analyst" / "dashboard" / "app.py"
TITLE = "Autonomous Data Analyst Agent"


@pytest.fixture(scope="module")
def loaded():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.button(key="load_sample").click().run()
    return at


def test_default_page_asks_for_data():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == TITLE
    assert at.header[0].value == "Data & profile"
    assert at.info and not at.metric


def test_sample_button_loads_superstore(loaded):
    at = loaded
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Rows"] == "9,994"
    assert metrics["Columns"] == "21"
    assert metrics["Quality score"] == "80/100"
    assert metrics["Date range"] == "2014-01-03 to 2017-12-30"
    assert at.session_state[state.FILENAME] == "superstore.csv"
    assert at.session_state[state.RAW].shape == (9994, 21)

    quality, columns = at.dataframe[0].value, at.dataframe[1].value
    assert quality["severity"].iloc[0].endswith("warning")
    assert {"lost_leading_zeros", "constant_column", "near_duplicate_rows", "outliers"} <= set(quality["check"])
    assert len(columns) == 21 and "key stats" in columns
    assert any(b.label == "Download profile JSON" for b in at.get("download_button"))


@pytest.mark.parametrize(("column", "has_chart"), [("Sales", True), ("Order Date", True), ("Region", True), ("Order ID", False)])
def test_column_explorer(loaded, column, has_chart):
    at = loaded
    at.selectbox(key="explore_column").select(column).run()
    assert not at.exception, at.exception
    # Superstore has no missing values, so the only other chart is the role breakdown.
    assert len(at.get("plotly_chart")) == 1 + has_chart


def test_switch_to_timeseries_page_keeps_loaded_table(loaded):
    at = loaded
    at.switch_page("app_pages/timeseries.py").run()
    assert not at.exception, at.exception
    assert at.title[0].value == TITLE
    assert at.header[0].value == "Time-series analysis"
    assert len(at.get("plotly_chart")) >= 2
    assert at.session_state[state.PROFILE].n_rows == 9994
