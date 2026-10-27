"""Environment and system diagnostic tool (doctor) for the Smart Medicine Redistribution System."""
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str, str]:
    """Verify Python is >= 3.11."""
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return True, f"Python {ver_str}", ""
    return False, f"Python {ver_str}", "Python >= 3.11 is required. Please install Python 3.11 or later."


def check_java() -> Tuple[bool, str, str]:
    """Check for Java runtime / JAVA_HOME."""
    java_cmd = shutil.which("java")
    java_home = os.environ.get("JAVA_HOME")
    if java_cmd:
        try:
            res = subprocess.run([java_cmd, "-version"], capture_output=True, text=True, timeout=5)
            version_line = (res.stderr or res.stdout).splitlines()[0] if (res.stderr or res.stdout) else "Installed"
            return True, f"Java detected ({version_line})", ""
        except Exception:
            return True, "Java detected", ""
    
    # Java is optional locally because DuckDB fallback is available
    hint = "Java not found on PATH. PySpark requires Java 17; ETL will use the built-in DuckDB/Arrow fallback engine. (Install OpenJDK 17 or set ETL_BACKEND=duckdb)."
    return False, "Java not found (Optional for local dev)", hint


def check_packages() -> Tuple[bool, List[str], List[str]]:
    """Check required serving and pipeline packages."""
    required = [
        "pandas",
        "duckdb",
        "pyarrow",
        "streamlit",
        "plotly",
        "pydeck",
        "numpy",
        "scipy",
        "sklearn",
        "statsmodels",
        "ortools",
        "mlflow",
        "yaml",
        "pytest",
    ]
    passed = []
    failed = []
    for pkg in required:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "installed")
            passed.append(f"{pkg} ({ver})")
        except ImportError:
            failed.append(pkg)
    return len(failed) == 0, passed, failed


def check_writable_dirs() -> Tuple[bool, str, str]:
    """Verify project data directories are writable."""
    test_dirs = [Path("data"), Path("data/serving"), Path("data/sample")]
    for d in test_dirs:
        d.mkdir(parents=True, exist_ok=True)
        try:
            test_file = d / ".write_test.tmp"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
        except Exception as e:
            return False, f"Cannot write to {d}: {e}", f"Ensure permissions for {d}"
    return True, "Project data directories are writable", ""


def check_etl_backend() -> Tuple[bool, str, str]:
    """Report active ETL backend."""
    backend = os.environ.get("ETL_BACKEND", "spark").lower()
    if backend in ["spark", "duckdb"]:
        return True, f"ETL_BACKEND='{backend}'", ""
    return False, f"Unknown ETL_BACKEND='{backend}'", "Set ETL_BACKEND to 'spark' or 'duckdb'."


def run_doctor(verbose: bool = False) -> int:
    """Run all health checks and print structured report."""
    print("=" * 60)
    print(" Smart Medicine Redistribution System - System Doctor")
    print("=" * 60)

    has_critical_failure = False

    # 1. Python
    py_ok, py_msg, py_hint = check_python_version()
    status = "[PASS]" if py_ok else "[FAIL]"
    print(f" {status:<6} {py_msg}")
    if not py_ok:
        print(f"        Hint: {py_hint}")
        has_critical_failure = True

    # 2. Java
    java_ok, java_msg, java_hint = check_java()
    java_status = "[PASS]" if java_ok else "[WARN]"
    print(f" {java_status:<6} {java_msg}")
    if not java_ok and verbose:
        print(f"        Hint: {java_hint}")

    # 3. Packages
    pkg_ok, passed_pkgs, failed_pkgs = check_packages()
    pkg_status = "[PASS]" if pkg_ok else "[FAIL]"
    print(f" {pkg_status:<6} Required packages ({len(passed_pkgs)} installed)")
    if not pkg_ok:
        print(f"        Missing packages: {', '.join(failed_pkgs)}")
        print("        Hint: Run 'pip install -r requirements-pipeline.txt'")
        has_critical_failure = True
    elif verbose:
        print(f"        Installed: {', '.join(passed_pkgs[:8])}...")

    # 4. Writable Dirs
    dir_ok, dir_msg, dir_hint = check_writable_dirs()
    dir_status = "[PASS]" if dir_ok else "[FAIL]"
    print(f" {dir_status:<6} {dir_msg}")
    if not dir_ok:
        print(f"        Hint: {dir_hint}")
        has_critical_failure = True

    # 5. ETL Backend
    etl_ok, etl_msg, etl_hint = check_etl_backend()
    etl_status = "[PASS]" if etl_ok else "[FAIL]"
    print(f" {etl_status:<6} {etl_msg}")
    if not etl_ok:
        print(f"        Hint: {etl_hint}")
        has_critical_failure = True

    print("-" * 60)
    if has_critical_failure:
        print(" System Doctor found issues that need attention.")
        return 1
    else:
        print(" System Doctor: Environment is ready to execute pipeline and app.")
        return 0


if __name__ == "__main__":
    sys.exit(run_doctor(verbose=True))
