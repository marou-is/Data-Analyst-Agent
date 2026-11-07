import pandas as pd
import pytest

from analyst.agent import run_analysis
from analyst.agent.analyst import (
    SUGGESTED_QUESTIONS, answer_question, classify, find_dimension, find_measure, period_word, prepare_table,
)
from analyst.data.ingest import load_dataset
from analyst.data.profile import profile_dataset
from analyst.datasets import SUPERSTORE

TINY = pd.DataFrame({
    "Order Date": ["1/5/2023", "2/7/2023", "3/9/2023", "1/5/2024", "2/7/2024", "3/9/2024"],
    "Region": ["West", "East", "West", "East", "West", "West"],
    "Sub-Category": ["Chairs", "Paper", "Paper", "Chairs", "Phones", "Phones"],
    "Revenue": [100.0, 50.0, 30.0, 80.0, 200.0, 40.0],
    "Profit": [10.0, -5.0, 3.0, 8.0, 40.0, 4.0],
})


@pytest.fixture(scope="module")
def tiny():
    profile = profile_dataset(TINY)
    return prepare_table(TINY, profile), profile


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Which category generates the most revenue?", "breakdown"),
        ("Profit by region", "breakdown"),
        ("How has revenue changed over time?", "trend"),
        ("Which month had the highest sales?", "trend"),
        ("Are there unusual values in my dataset?", "anomalies"),
        ("What are the main data quality issues?", "quality"),
        ("Forecast sales for next month", "forecast"),
        ("hello", "overview"),
    ],
)
def test_classify(question, intent):
    assert classify(question) == intent


def test_column_resolution(tiny):
    _, profile = tiny
    assert find_measure("what about profit?", profile) == "Profit"
    assert find_measure("top regions", profile) == "Revenue"
    assert find_dimension("profit by sub category", profile) == "Sub-Category"
    assert find_dimension("which sub-categories sell best", profile) == "Sub-Category"
    assert find_dimension("top values", profile) == "Sub-Category"
    assert find_dimension("which region is best", profile) == "Region"


def test_breakdown_numbers(tiny):
    table, profile = tiny
    answer = answer_question("Which region has the most revenue?", table, profile)
    frame = answer.table.set_index("Region")
    assert frame.loc["West", "Revenue"] == 370.0 and frame.loc["East", "Revenue"] == 130.0
    assert frame.loc["West", "margin"] == pytest.approx(57 / 370)
    assert answer.summary.startswith("West has the highest total Revenue: $370")
    assert answer.chart == "bar" and answer.metric == "Revenue"

    lowest = answer_question("Which region has the lowest profit?", table, profile)
    assert lowest.table["Region"].iloc[0] == "East" and "followed by West" in lowest.summary


def test_trend_numbers(tiny):
    table, profile = tiny
    answer = answer_question("How has revenue changed over time?", table, profile)
    assert "+77.8% from 2023 ($180) to 2024 (partial year) ($320)" in answer.summary
    assert answer.chart == "line" and len(answer.table) == 15


def test_quality_and_overview_without_report(tiny):
    table, profile = tiny
    assert answer_question("any data quality problems?", table, profile).intent == "quality"
    overview = answer_question("hi", table, profile)
    assert "6 rows and 5 columns" in overview.summary and overview.table is None


@pytest.mark.parametrize(("freq", "n", "word"), [("D", 3, "days"), ("W-MON", 1, "week"), ("MS", 2, "months"),
                                                  ("h", 2, "hours"), ("30min", 4, "30min periods"), (None, 2, "periods")])
def test_period_word(freq, n, word):
    assert period_word(freq, n) == word


@pytest.fixture(scope="module")
def setup():
    raw = load_dataset(SUPERSTORE.path)
    profile = profile_dataset(raw, name="superstore")
    return prepare_table(raw, profile), profile, run_analysis(raw, filename="superstore.csv")


@pytest.mark.skipif(not SUPERSTORE.path.exists(), reason="Superstore data not downloaded")
class TestSuperstore:
    def test_every_suggested_question_has_evidence(self, setup):
        table, profile, report = setup
        for question in SUGGESTED_QUESTIONS:
            answer = answer_question(question, table, profile, report)
            assert answer.intent != "overview" and answer.evidence and answer.table is not None

    def test_category_revenue(self, setup):
        answer = answer_question(SUGGESTED_QUESTIONS[0], *setup)
        frame = answer.table.set_index("Category")
        assert frame["Sales"].idxmax() == "Technology"
        assert frame.loc["Technology", "Sales"] == pytest.approx(836154.03, abs=0.01)
        assert frame["Sales"].sum() == pytest.approx(2296919.4883)
        assert frame.loc["Furniture", "margin"] < 0.03

    def test_yearly_trend(self, setup):
        answer = answer_question(SUGGESTED_QUESTIONS[1], *setup)
        assert len(answer.table) == 48 and "+20.4% from 2016" in answer.summary

    def test_anomalies_use_report_rows(self, setup):
        answer = answer_question(SUGGESTED_QUESTIONS[2], *setup)
        assert "flagged 100 of 9,993 rows" in answer.summary
        assert {"Row ID", "reason"} <= set(answer.table.columns) and len(answer.table) == 15

    def test_forecast_uses_report(self, setup):
        answer = answer_question("Forecast sales", *setup)
        assert answer.intent == "forecast" and "next 48 days" in answer.summary
