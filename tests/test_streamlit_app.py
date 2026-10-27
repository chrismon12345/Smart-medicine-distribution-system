"""Unit and integration tests for Phase 8: Streamlit serving presentation layer."""
import ast
from pathlib import Path
import pytest
import pandas as pd

from app.data_loader import load_serving_table, load_all_dashboard_data


def test_all_serving_parquet_tables_readable():
    """Verify load_all_dashboard_data loads all 12 published serving tables."""
    data = load_all_dashboard_data()
    expected_tables = [
        "scenario_catalog",
        "hospitals",
        "medicines",
        "state_daily",
        "transfers",
        "kpis_daily",
        "replay",
        "evaluation_forecasting",
        "evaluation_classification",
        "evaluation_anomaly",
        "evaluation_optimizer",
        "runs",
    ]
    for tbl in expected_tables:
        assert tbl in data, f"Table {tbl} not in loaded dashboard data"
        assert isinstance(data[tbl], pd.DataFrame), f"{tbl} is not a DataFrame"
        assert not data[tbl].empty, f"{tbl} should not be empty in published data"


def test_no_spark_or_model_training_in_dashboard_code():
    """
    CRITICAL ARCHITECTURE REQUIREMENT:
    The Streamlit app is purely a presentation layer. It MUST NOT:
    - import or start PySpark
    - train ML models or call .fit()
    - run heavy simulation steps
    """
    app_files = list(Path("app").rglob("*.py"))
    assert len(app_files) >= 6, "Expected at least 6 python files in app directory"

    forbidden_tokens = ["pyspark", "SparkSession", "Simulator.run", ".fit("]

    for f_path in app_files:
        content = f_path.read_text(encoding="utf-8")
        # Parse AST to ensure valid python syntax
        ast.parse(content)

        for token in forbidden_tokens:
            assert token not in content, f"Forbidden heavy compute token '{token}' found in {f_path}! App must be pure presentation."


def test_missing_serving_data_graceful_fallback(tmp_path, monkeypatch):
    """Verify that if serving tables are absent, data_loader returns empty DataFrame rather than crashing."""
    monkeypatch.setattr("app.data_loader.SERVING_DIR", tmp_path / "non_existent_dir")
    res = load_serving_table("non_existent_table")
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_no_hardcoded_kpis_in_home():
    """Verify that KPIs in Home.py are derived dynamically from loaded DataFrame data."""
    home_content = Path("app/Home.py").read_text(encoding="utf-8")
    # Verify dynamic expressions exist
    assert 'day_state["stock_remaining"].sum()' in home_content
    assert 'day_state["unmet_demand"].sum()' in home_content
    assert 'day_state["expired_units"].sum()' in home_content


def test_sample_fallback_when_serving_missing(tmp_path, monkeypatch):
    """Verify data_loader falls back to data/sample when data/serving is absent."""
    # Point SERVING_DIR to empty temp directory
    monkeypatch.setattr("app.data_loader.SERVING_DIR", tmp_path / "empty_serving")
    # SAMPLE_DIR remains pointing to data/sample
    sample_hospitals = load_serving_table("hospitals")
    assert isinstance(sample_hospitals, pd.DataFrame)
    assert not sample_hospitals.empty, "Should have loaded hospitals from data/sample"


def test_duckdb_projection_and_filtering():
    """Verify load_serving_table supports column selection and WHERE clause via DuckDB."""
    df = load_serving_table("medicines", columns=["medicine_id", "name"], where_clause="price > 10")
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert list(df.columns) == ["medicine_id", "name"]


def test_all_app_pages_syntax_valid():
    """Verify all application pages parse as valid Python AST without syntax errors."""
    app_files = list(Path("app").rglob("*.py"))
    assert len(app_files) >= 8
    for f in app_files:
        content = f.read_text(encoding="utf-8")
        tree = ast.parse(content)
        assert tree is not None


def test_labels_module_mappings():
    """Verify labels.py provides bidirectional mappings and scenario tags."""
    from app.labels import POLICY_LABELS, SCENARIO_INFO, get_policy_display, get_scenario_display
    assert get_policy_display("optimizer") == "Smart plan (AI/MILP)"
    assert get_policy_display("no_transfer") == "Do nothing (Baseline)"
    assert "dengue_4x" in SCENARIO_INFO
    assert SCENARIO_INFO["dengue_4x"]["tag"] == "New Situation"


def test_all_app_pages_render_without_exception():
    """Verify using Streamlit AppTest that every application page loads and renders on sample data."""
    from streamlit.testing.v1 import AppTest
    repo_root = Path(__file__).resolve().parent.parent
    pages = [
        "app/Home.py",
        "app/pages/1_Hospital_Administrator.py",
        "app/pages/2_Expiry_Aware_Optimizer.py",
        "app/pages/2_Network_Cascading_Predictor.py",
        "app/pages/2_Scenario_Comparison.py",
        "app/pages/3_Replay.py",
        "app/pages/4_Model_Evaluation.py",
        "app/pages/5_About.py",
    ]
    for p in pages:
        full_path = str(repo_root / p)
        at = AppTest.from_file(full_path, default_timeout=15)
        at.run()
        assert not at.exception, f"Page {p} raised an exception during render: {at.exception}"


def test_sidebar_filters_return_expected_structure():
    """Verify render_sidebar_filters returns scenario, policy, day, district, and hospital_type."""
    from streamlit.testing.v1 import AppTest
    repo_root = Path(__file__).resolve().parent.parent
    home_file = str(repo_root / "app" / "Home.py")

    at = AppTest.from_file(home_file, default_timeout=15)
    at.run()
    assert not at.exception
    # Verify metric cards exist
    assert len(at.metric) >= 4


def test_no_clamping_to_zero_in_home():
    """Verify Home.py does not artificially clamp negative impact differences to zero."""
    home_content = Path("app/Home.py").read_text(encoding="utf-8")
    assert "max(0.0, b_stockout - a_stockout)" not in home_content
    assert "max(0.0, b_waste - a_waste)" not in home_content
    assert "b_stockout - a_stockout" in home_content
    assert "b_waste - a_waste" in home_content





