"""About page: System Architecture, Mathematical Optimization Formulation, SLA Contracts, and Engineering Deep Dive."""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_loader import load_serving_table, render_sample_banner
from app.labels import GLOSSARY, SCENARIO_INFO, POLICY_LABELS, POLICY_DESCRIPTIONS
from app.state import render_sidebar_filters
from app.theme import inject_custom_theme, render_brand_header, render_system_status_bar

st.set_page_config(page_title="About & System Architecture", page_icon="ℹ️", layout="wide")

inject_custom_theme()
render_brand_header(subtitle="System Architecture, Data Contracts & Mathematical Optimization Specification")

render_sample_banner()

# Load Serving Data
hospitals_df = load_serving_table("hospitals")
state_df = load_serving_table("state_daily")
render_sidebar_filters(hospitals_df, state_df)

# ==============================================================================
# SECTION 1: EXECUTIVE QUICKSTART WORKFLOW
# ==============================================================================
st.markdown("### ⏱️ Operational Workflow & Decision Navigation")
st.markdown("""
1. **Executive Overview (Home)**:
   - Identify critical facility deficits and imminent batch expirations across the 14-day operational horizon.
   - Inspect optimization-directed transfer orders and export dispatch manifests as CSV or JSON integration payloads.
   - Benchmark quantified fill-rate gains and stockout reductions against the uncoordinated baseline.
2. **Hospital Administrator Portal**:
   - Drill into facility-level inventory health, days-of-stock runway, and scheduled incoming/outgoing transfers.
3. **Scenario & Stress-Test Benchmarks**:
   - Empirically stress-test redistribution policies across outbreak surges, supplier delays, and road infrastructure shocks.
4. **Historical Replay & Forecast Audit**:
   - Audit model predictions issued at time *t* against actual ground-truth patient demand within 95% uncertainty envelopes.
5. **Model Evaluation & Ablations**:
   - Inspect quantitative ML metrics (WAPE, MAE, RMSE), Random Forest risk classification recall, and Outpatient Clinic (OPD) lead-lag feature ablation.
""")

st.markdown("---")

# ==============================================================================
# SECTION 2: END-TO-END DATA INTELLIGENCE ARCHITECTURE
# ==============================================================================
st.markdown("### ⚙️ End-to-End System Architecture")
st.markdown("""
The platform implements a strict separation of concerns between heavy offline compute pipelines and ultra-fast, zero-compute presentation:

```
[ Multi-Facility Simulation ]  -->  Daily Time-Stepped State, OPD Clinic Visits & Shelf-Life Aging
              │
              ▼
[ Distributed Spark ETL Engine ]  -->  Feature Engineering, Windowed Lags & Data Harmonization
              │
              ▼
[ Hybrid Forecasting & Risk ]  -->  LightGBM + Moving Average Baselines + Random Forest Classifiers
              │
              ▼
[ OR-Tools MILP Optimizer ]    -->  Multi-Commodity FEFO Flow Optimization with Receiving Caps
              │
              ▼
[ Curated Serving Lakehouse ]  -->  12 Compressed Parquet Tables (Snappy, Columnar Projection)
              │
              ▼
[ DuckDB In-Memory Engine ]    -->  In-Memory SQL Execution & Predicate Pushdown (P99 < 15ms)
              │
              ▼
[ Streamlit Presentation UI ]  -->  Zero-Compute Web Dashboard (Pure Presentation, No Model Training)
```
""")

c_arch1, c_arch2 = st.columns(2)
with c_arch1:
    st.markdown("##### 🛡️ Architectural Guarantees")
    st.markdown("""
    - **Zero-Compute Web Layer**: The Streamlit interface acts purely as a presentation consumer over precomputed Parquet serving tables. It executes no model training, simulation loops, or heavy optimization solvers.
    - **Predicate Pushdown & Column Projection**: DuckDB reads only required columns and applies WHERE filters directly over Parquet metadata, minimizing memory usage and I/O.
    - **Deterministic Idempotency**: All simulated data generation and model benchmarking are bound to reproducible pseudo-random seeds.
    """)

with c_arch2:
    st.markdown("##### 📐 Mathematical Optimization Formulation")
    st.markdown("""
    - **Objective**: Minimize total network penalty:
      $$\\min \\sum_{h,m,t} \\left( w_1 \\cdot \\text{Deficit}_{h,m,t} + w_2 \\cdot \\text{Wastage}_{h,m,t} + \\sum_{j} c_{h,j} \\cdot \\text{Transfer}_{h,j,m,t} \\right)$$
    - **FEFO Priority**: Earliest-expiring batches are prioritized for local consumption and transfer dispatch.
    - **Receiving Facility Bound**: Strict caps prevent dumping surplus into facilities beyond their forecast absorption capacity.
    - **Transit Physics**: Route-specific transit lead times enforce realistic delivery delays.
    """)

st.markdown("---")

# ==============================================================================
# SECTION 3: PROJECT GLOSSARY & DOMAIN TAXONOMY
# ==============================================================================
st.markdown("### 📖 Domain Glossary & Engineering Taxonomy")
glossary_cols = st.columns(2)
items = list(GLOSSARY.items())
half = len(items) // 2

with glossary_cols[0]:
    for term, definition in items[:half]:
        with st.expander(f"🔹 **{term}**", expanded=False):
            st.write(definition)

with glossary_cols[1]:
    for term, definition in items[half:]:
        with st.expander(f"🔹 **{term}**", expanded=False):
            st.write(definition)

st.markdown("---")

# ==============================================================================
# SECTION 4: OPERATIONAL SLA & METHODOLOGICAL BOUNDARIES
# ==============================================================================
st.markdown("### ⚠️ Operational Boundaries & Data Provenance")
st.markdown("""
1. **Simulation Provenance**: Hospital facility locations, patient consumption patterns, and batch inventory ages are synthesized using daily time-stepped simulation to enable repeatable research benchmarking without exposing protected patient health information (PHI).
2. **Transportation Calibrations**: Cost-per-kilometer rates (₹2.5/km) and fixed vehicle dispatch fees (₹100) are parameterized based on regional public health logistics benchmarks.
3. **Decoupled Architecture**: In production deployments, the offline pipeline runs on a scheduled batch orchestrator (e.g., Airflow / Prefect), publishing immutable Parquet lakehouse snapshots consumed by this zero-compute presentation service.
""")

st.caption("Smart Medicine Redistribution Platform • Production Decision Support System • Department of Health & Family Welfare")
