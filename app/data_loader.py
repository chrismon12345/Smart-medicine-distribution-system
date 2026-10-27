"""Data loader utility for Streamlit dashboard with intelligent caching, DuckDB queries, and sample fallback."""
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import duckdb
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_DIR = REPO_ROOT / "data" / "serving"
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


def get_data_source() -> Optional[str]:
    """Detect whether serving, sample, or no data is available."""
    if SERVING_DIR.exists() and any(SERVING_DIR.glob("*.parquet")):
        return "serving"
    if SAMPLE_DIR.exists() and any(SAMPLE_DIR.glob("*.parquet")):
        return "sample"
    return None


def render_sample_banner() -> None:
    """Render a visible notification banner if running on sample data or missing data."""
    source = get_data_source()
    if source == "sample":
        st.info("ℹ️ **Showing bundled sample data.** Run the pipeline (`python -m pipeline demo` or `.\\run_demo.ps1`) to generate full results.")
    elif source is None:
        st.warning(
            "⚠️ **No data directory found.** Run the pipeline demo to generate results:\n\n"
            "```powershell\n.\\run_demo.ps1\n# or\npython -m pipeline demo\n```"
        )


def _resolve_table_path(table_name: str) -> Optional[Path]:
    """Find the table in SERVING_DIR first, then SAMPLE_DIR."""
    serving_path = SERVING_DIR / f"{table_name}.parquet"
    if serving_path.exists():
        return serving_path

    sample_path = SAMPLE_DIR / f"{table_name}.parquet"
    if sample_path.exists():
        return sample_path

    return None


def _get_file_mtime(p: Optional[Path]) -> float:
    if p is not None and p.exists():
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0
    return 0.0


@st.cache_data(ttl=300)
def _load_cached_table(
    table_path_str: str,
    columns_tuple: Optional[tuple],
    where_clause: Optional[str],
    mtime: float,
) -> pd.DataFrame:
    p = Path(table_path_str)
    if not p.exists():
        return pd.DataFrame()
    columns = list(columns_tuple) if columns_tuple else None
    try:
        # Use forward slashes for DuckDB read_parquet path on Windows
        posix_path = p.as_posix()
        col_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_str} FROM read_parquet('{posix_path}')"
        if where_clause:
            query += f" WHERE {where_clause}"

        con = duckdb.connect(database=":memory:")
        df = con.execute(query).df()
        con.close()
        return df
    except Exception:
        # Fallback to direct pandas read if DuckDB encounters an issue
        try:
            df = pd.read_parquet(p, columns=columns)
            return df
        except Exception:
            return pd.DataFrame()


def load_serving_table(
    table_name: str,
    columns: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a table using DuckDB over Parquet with column projection and filtering.
    Falls back gracefully to empty DataFrame if file is missing.
    Automatically invalidates cache when the underlying file on disk is modified.
    """
    p = _resolve_table_path(table_name)
    if p is None or not p.exists():
        return pd.DataFrame()

    mtime = _get_file_mtime(p)
    cols_tup = tuple(columns) if columns else None
    return _load_cached_table(str(p), cols_tup, where_clause, mtime)


def clear_data_cache() -> None:
    """Explicitly clear Streamlit data cache."""
    st.cache_data.clear()


def load_all_dashboard_data() -> Dict[str, pd.DataFrame]:
    """Load published serving tables."""
    table_names = [
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
        "cascading_predictions",
        "cascading_timeline",
    ]
    data = {}
    for name in table_names:
        data[name] = load_serving_table(name)
    return data
