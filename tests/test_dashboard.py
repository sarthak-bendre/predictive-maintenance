from streamlit.testing.v1 import AppTest


def run_app():
    return AppTest.from_file("../dashboard/app.py", default_timeout=30).run()


def decision_text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_dashboard_renders_and_flags_risky_preset(models_dir):
    at = run_app()
    assert not at.exception
    assert at.metric[0].label == "Failure probability"
    # Default preset is the overstrained worn tool: wear x torque > 9000 in the fixture model.
    assert "Schedule inspection" in decision_text(at)


def test_switching_to_healthy_preset_clears_alarm(models_dir):
    at = run_app()
    at.selectbox(key="preset").set_value("Healthy machine").run()
    assert not at.exception
    assert "No action needed" in decision_text(at)
    assert at.slider(key="tool_wear_min").value == 108


def test_missing_model_shows_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    at = run_app()
    assert at.error and "No model found" in at.error[0].value
