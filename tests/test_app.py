"""Headless UI test: every section renders, demo data loads, schema is detected, profiling runs."""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_every_section_renders_without_data():
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=60).run()
    _no_errors(at)
    for page in ["Dashboard", "Data Upload", "Profiling", "Quality Analysis", "Processing", "Model", "Experiments", "Results"]:
        at.sidebar.radio[0].set_value(page).run()
        _no_errors(at)
    at.sidebar.radio[0].set_value("Results").run()
    assert any("No saved experiments" in str(i.value) for i in at.info)


def test_demo_flow_upload_schema_profile():
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Data Upload").run()
    at.button(key="load_demo").click().run()
    _no_errors(at)
    roles = at.session_state["roles"]
    assert roles.target == "is_fraud" and roles.task == "binary_classification"
    assert roles.entity == "cc_num" and roles.datetime == "trans_date_trans_time"
    at.sidebar.radio[0].set_value("Profiling").run()
    at.button(key="run_profile").click().run()
    _no_errors(at)
    assert at.session_state["profile"] is not None
    assert at.session_state["profile_info"]["full_duplicate_rows"] == 0
    at.sidebar.radio[0].set_value("Dashboard").run()
    _no_errors(at)
    assert any(m.label == "Target" and m.value == "is_fraud" for m in at.metric)


def test_demo_flow_processing():
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Data Upload").run()
    at.button(key="load_demo").click().run()
    at.sidebar.radio[0].set_value("Processing").run()
    _no_errors(at)

    at.button(key="prepare_split").click().run()
    _no_errors(at)
    preparer = at.session_state["preparer"]
    assert preparer is not None and preparer.split_info

    for level in ["E0", "E1", "E2"]:
        at.button(key=f"build_{level}").click().run()
        _no_errors(at)
    levels = at.session_state["levels"]
    assert set(levels) == {"E0", "E1", "E2"}
    assert levels["E0"].features and levels["E1"].features and levels["E2"].features
    assert levels["E2"].info["point_in_time"]["passed"]

    at.sidebar.radio[0].set_value("Dashboard").run()
    _no_errors(at)


def test_demo_flow_model_training():
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Data Upload").run()
    at.button(key="load_demo").click().run()
    at.sidebar.radio[0].set_value("Processing").run()
    at.button(key="prepare_split").click().run()
    at.button(key="build_E1").click().run()
    _no_errors(at)

    at.sidebar.radio[0].set_value("Model").run()
    _no_errors(at)
    at.number_input[0].set_value(2).run()          # epochs: keep the headless run fast
    at.button(key="train_model").click().run()
    _no_errors(at)

    trained = at.session_state["model"]
    assert trained is not None and trained["level"] == "E1"
    assert trained["summary"]["epochs_run"] <= 2
    assert all(c["status"] != "fail" for c in trained["checks"])
    assert 0.0 <= trained["results"]["threshold_from_validation"] <= 1.0

    at.sidebar.radio[0].set_value("Dashboard").run()
    _no_errors(at)


def test_demo_flow_experiments():
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value("Data Upload").run()
    at.button(key="load_demo").click().run()
    at.sidebar.radio[0].set_value("Processing").run()
    at.button(key="prepare_split").click().run()
    _no_errors(at)

    at.sidebar.radio[0].set_value("Experiments").run()
    _no_errors(at)
    at.number_input[0].set_value(2).run()          # epochs: keep the headless run fast
    at.button(key="run_experiment").click().run()
    _no_errors(at)

    results = at.session_state["experiment"]
    assert results is not None and {r.level for r in results} == {"E0", "E1", "E2"}
    assert all(r.epochs_run <= 2 for r in results)

    at.sidebar.radio[0].set_value("Results").run()      # not run+saved here: covered by test_experiment.py with tmp_path
    _no_errors(at)


def test_demo_flow_quality_and_basic_preprocessing():
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Data Upload").run()
    at.button(key="load_demo").click().run()
    at.sidebar.radio[0].set_value("Quality Analysis").run()
    _no_errors(at)

    at.button(key="run_quality").click().run()
    _no_errors(at)
    quality = at.session_state["quality"]
    assert quality is not None and quality.scores["overall"] is not None
    assert quality.scores["completeness"]["score"] == 100.0

    at.button(key="run_preprocessing").click().run()
    _no_errors(at)
    prep = at.session_state["preprocessing"]
    assert prep is not None
    assert "unix_time" in prep.summary.dropped_columns
    assert prep.summary.rows_before == prep.summary.rows_after   # no duplicates in the demo data

    at.sidebar.radio[0].set_value("Dashboard").run()
    _no_errors(at)
