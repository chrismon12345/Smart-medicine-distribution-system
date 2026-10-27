"""Unit tests for pipeline CLI and doctor diagnostics."""
import pytest
from pipeline.doctor import (
    check_python_version,
    check_packages,
    check_writable_dirs,
    check_etl_backend,
    run_doctor,
)
from pipeline.cli import get_default_profile, _run_export


def test_doctor_python_version():
    """Verify check_python_version returns expected status."""
    ok, msg, hint = check_python_version()
    assert isinstance(ok, bool)
    assert "Python" in msg


def test_doctor_packages_check():
    """Verify check_packages detects critical packages."""
    ok, passed, failed = check_packages()
    assert "pandas" in [p.split()[0] for p in passed]
    assert "duckdb" in [p.split()[0] for p in passed]
    assert "streamlit" in [p.split()[0] for p in passed]
    assert ok, f"Expected all required packages to be present, but missing: {failed}"


def test_doctor_writable_directories():
    """Verify check_writable_dirs succeeds on workspace directories."""
    ok, msg, hint = check_writable_dirs()
    assert ok, f"Directories should be writable: {msg}"


def test_doctor_etl_backend():
    """Verify check_etl_backend reports valid backend setting."""
    ok, msg, hint = check_etl_backend()
    assert ok, f"ETL backend check should succeed: {msg}"


def test_run_doctor_exit_code():
    """Verify run_doctor executes and returns 0 on healthy environment."""
    exit_code = run_doctor(verbose=False)
    assert exit_code == 0, "Doctor should return 0 exit code on healthy environment"


def test_default_profile_resolution(monkeypatch):
    """Verify default profile respects MEDSYS_PROFILE env var."""
    monkeypatch.delenv("MEDSYS_PROFILE", raising=False)
    assert get_default_profile() == "dev"

    monkeypatch.setenv("MEDSYS_PROFILE", "demo")
    assert get_default_profile() == "demo"


def test_cli_export_subcommand_runs(tmp_path, monkeypatch):
    """Verify export sub-stage executes and publishes parquet tables."""
    out_dir = tmp_path / "serving_test"
    from pipeline.export import export_serving_data
    export_serving_data(profile="dev", target_dir=str(out_dir))
    tables = list(out_dir.glob("*.parquet"))
    assert len(tables) >= 12, f"Expected at least 12 exported tables, got {len(tables)}"
