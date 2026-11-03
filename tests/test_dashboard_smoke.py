from streamlit.testing.v1 import AppTest

from analyst.config import PROJECT_ROOT

APP = PROJECT_ROOT / "src" / "analyst" / "dashboard" / "app.py"


def test_app_renders_default_dataset_without_errors():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Autonomous Data Analyst Agent"
