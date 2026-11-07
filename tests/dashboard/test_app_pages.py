import pytest
from streamlit.testing.v1 import AppTest

from analyst.config import PROJECT_ROOT
from analyst.dashboard import state
from analyst.dashboard.views.overview import greeting

APP = PROJECT_ROOT / "src" / "analyst" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def go(at: AppTest, page: str) -> AppTest:
    at.button(key=f"nav_{page.lower().replace(' ', '_')}").click().run()
    assert not at.exception, at.exception
    assert at.session_state[state.PAGE] == page
    return at


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_greeting_by_hour():
    assert [greeting(h) for h in (8, 13, 21)] == ["Good morning", "Good afternoon", "Good evening"]


def test_overview_shows_superstore_kpis(app):
    values = metrics(app)
    assert values["Total sales"] == "$2,296,919"
    assert values["Total profit"] == "$286,409"
    assert values["Records"] == "9,994"
    assert values["Data quality"] == "80/100"
    assert len(app.get("plotly_chart")) == 2
    assert app.button(key="nav_overview").proto.type == "primary"


def test_overview_ask_button_opens_ai_analyst(app):
    app.button(key="overview_ask").click().run()
    assert not app.exception, app.exception
    assert app.session_state[state.PAGE] == "AI Analyst"


def test_explorer_profiles_dataset(app):
    go(app, "Data Explorer")
    values = metrics(app)
    assert values["Rows"] == "9,994"
    assert values["Columns"] == "21"
    assert values["Quality score"] == "80/100"
    preview, columns, quality = (d.value for d in app.dataframe)
    assert preview.shape == (100, 21)
    assert len(columns) == 21 and "key stats" in columns
    assert {"lost_leading_zeros", "constant_column", "near_duplicate_rows", "outliers"} <= set(quality["check"])


def test_ai_suggestion_answers_with_evidence(app):
    go(app, "AI Analyst")
    app.button(key="suggest_0").click().run()
    assert not app.exception, app.exception
    answer = app.session_state[state.ANSWER]
    assert answer.intent == "breakdown" and "Technology" in answer.summary
    assert any("Technology" in m.value for m in app.markdown)
    assert app.dataframe[0].value.shape[0] == 3


def test_ai_free_text_question(app):
    app.text_area(key=state.QUESTION).input("Are there unusual values in my dataset?")
    app.button(key="ask_run").click().run()
    assert not app.exception, app.exception
    assert app.session_state[state.ANSWER].intent == "anomalies"


def test_automation_create_and_run(app):
    go(app, "Automations")
    app.text_input(key="automation_name").input("Weekly quality")
    app.selectbox(key="automation_workflow").select("Data quality check")
    app.button[[b.label for b in app.button].index("Create automation")].click().run()
    assert not app.exception, app.exception
    items = app.session_state[state.AUTOMATIONS]
    assert [i.name for i in items] == ["Weekly quality"]

    app.button(key="automation_run_0").click().run()
    assert not app.exception, app.exception
    item = app.session_state[state.AUTOMATIONS][0]
    assert item.last_run is not None and item.last_result.startswith("Data quality score 80/100")
