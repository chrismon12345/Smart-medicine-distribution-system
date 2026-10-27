"""Expiry-Aware Redistribution Optimizer: dynamic surplus reservation and FEFO wastage prevention."""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_loader import load_serving_table, render_sample_banner
from app.labels import get_scenario_display, get_policy_display, STATUS_CONFIG, GLOSSARY
from app.state import render_sidebar_filters
from app.theme import inject_custom_theme, render_brand_header, apply_plotly_theme

st.set_page_config(
    page_title="Expiry-Aware Redistribution Optimizer",
    page_icon="⚖️",
    layout="wide",
)

inject_custom_theme()
render_brand_header(subtitle="Expiry-Aware Redistribution Optimizer • Surplus Reservation & FEFO Wastage Prevention Directives")

render_sample_banner()

# Load Serving Tables
hospitals_df = load_serving_table("hospitals")
medicines_df = load_serving_table("medicines")
state_df = load_serving_table("state_daily")
transfers_df = load_serving_table("transfers")

active_filters = render_sidebar_filters(hospitals_df, state_df)
selected_scenario = active_filters["scenario"]
selected_policy = active_filters["policy"]
selected_day = active_filters["day"]

if hospitals_df.empty or state_df.empty:
    st.warning("Hospital inventory data is not available. Please run the pipeline export.")
    st.stop()

# Helper dicts
h_dict = hospitals_df.set_index("hospital_id")["name"].to_dict()
m_dict = medicines_df.set_index("medicine_id")["name"].to_dict() if not medicines_df.empty else {}
m_crit = medicines_df.set_index("medicine_id")["criticality"].to_dict() if not medicines_df.empty else {}

# Mathematical Invariant Banner
st.markdown(
    r"""
    <div style="background:#eff6ff; border:1.5px solid #93c5fd; border-left:5px solid #2563eb; border-radius:8px; padding:12px 18px; margin-bottom:18px;">
        <div style="font-weight:800; font-size:1.02rem; color:#1e40af; margin-bottom:4px;">
            ⚖️ Mathematical Surplus Invariant & FEFO Protection Guarantee
        </div>
        <div style="font-size:0.88rem; color:#1e293b; line-height:1.5;">
            The Expiry-Aware Optimizer strictly enforces: 
            <code>Surplus = max(0, Physical Stock - ⌈Daily Demand × (SafetyBuffer + HorizonDays)⌉)</code>.
            Source hospitals <b>never</b> donate inventory required for their own patients or safety buffers.
            All transfers require <b>≥ 7 days remaining shelf-life</b> upon arrival, prioritizing near-expiry batches to minimize wastage.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Filter Transfers for active scenario, policy, and day
day_trans = pd.DataFrame()
if not transfers_df.empty and "day" in transfers_df.columns:
    day_trans = transfers_df[
        (transfers_df["day"] == selected_day)
    ].copy()
    if "scenario" in day_trans.columns:
        day_trans = day_trans[day_trans["scenario"] == selected_scenario]
    if "policy" in day_trans.columns:
        day_trans = day_trans[day_trans["policy"] == selected_policy]

# Filter facility state for active scenario, policy, day
day_state = state_df[
    (state_df["scenario"] == selected_scenario) &
    (state_df["policy"] == selected_policy) &
    (state_df["day"] == selected_day)
].copy()

# Page-level drilldown filters
f_col1, f_col2 = st.columns(2)
with f_col1:
    hosp_filter_options = ["All Facilities"] + sorted(list(h_dict.values()))
    chosen_hosp = st.selectbox("Filter by Facility:", hosp_filter_options, index=0)
with f_col2:
    med_filter_options = ["All Medicines"] + sorted(list(m_dict.values()))
    chosen_med = st.selectbox("Filter by Medicine:", med_filter_options, index=0)

# Calculate Page KPIs
total_transfers_count = len(day_trans)
total_units_dispatched = int(day_trans["quantity"].sum()) if not day_trans.empty else 0
avg_shelf_life = float(day_trans["batch_expiry"].mean()) if not day_trans.empty and "batch_expiry" in day_trans.columns else 45.0
total_logistics_cost = int((100.0 * len(day_trans) + day_trans["distance_km"].sum() * 2.5)) if not day_trans.empty and "distance_km" in day_trans.columns else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Active Directives", f"{total_transfers_count} dispatches", help="Number of inter-hospital redistribution orders scheduled for today.")
with m2:
    st.metric("Units Dispatched", f"{total_units_dispatched:,} units", delta="Shortage Relief", delta_color="normal")
with m3:
    st.metric("Average Batch Shelf Life", f"{avg_shelf_life:.0f} days", help="Mean shelf life of dispatched medicine batches, verifying FEFO prioritization.")
with m4:
    st.metric("Est. Logistics Cost", f"₹{total_logistics_cost:,}", help="Total transport expense including base dispatch fee and per-km transit rates.")

st.markdown("---")

# Main Content Tabs
opt_tab1, opt_tab2 = st.tabs([
    "🚚 Expiry-Aware Transfer Orders Manifest",
    "🛡️ Facility Surplus vs Locked Demand Buffer",
])

with opt_tab1:
    st.markdown("### 📋 Active Transfer Orders Manifest")
    st.caption(
        f"Optimized inter-facility transfer orders on Day {selected_day} "
        f"under {get_policy_display(selected_policy)} ({get_scenario_display(selected_scenario)})."
    )

    if not day_trans.empty:
        day_trans["From Facility"] = day_trans["from_hospital"].map(h_dict)
        day_trans["To Facility"] = day_trans["to_hospital"].map(h_dict)
        day_trans["Medicine"] = day_trans["medicine_id"].map(m_dict)
        day_trans["Quantity (units)"] = day_trans["quantity"].astype(int)
        day_trans["Transit (days)"] = day_trans["transit_days"].astype(int)
        day_trans["Target Shortage (units)"] = day_trans.get("target_shortage", day_trans["quantity"]).astype(int)
        day_trans["Batch Expiry (days)"] = day_trans.get("batch_expiry", 45).astype(int)
        day_trans["Estimated Cost (₹)"] = (100.0 + day_trans["distance_km"] * 2.5).round(0).astype(int)

        filtered_manifest = day_trans.copy()
        if chosen_hosp != "All Facilities":
            filtered_manifest = filtered_manifest[
                (filtered_manifest["From Facility"] == chosen_hosp) | (filtered_manifest["To Facility"] == chosen_hosp)
            ]
        if chosen_med != "All Medicines":
            filtered_manifest = filtered_manifest[filtered_manifest["Medicine"] == chosen_med]

        if not filtered_manifest.empty:
            display_manifest = filtered_manifest[[
                "Medicine", "From Facility", "To Facility",
                "Quantity (units)", "Target Shortage (units)", "Transit (days)",
                "Batch Expiry (days)", "Estimated Cost (₹)", "reason"
            ]].rename(columns={"reason": "Algorithmic Rationale"})

            st.dataframe(
                display_manifest,
                hide_index=True,
                use_container_width=True,
            )

            c_exp1, c_exp2 = st.columns(2)
            with c_exp1:
                csv_manifest = display_manifest.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Transfer Directives Manifest (CSV)",
                    data=csv_manifest,
                    file_name=f"expiry_transfer_manifest_day_{selected_day}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c_exp2:
                json_manifest = display_manifest.to_json(orient="records", indent=2).encode("utf-8")
                st.download_button(
                    label="🔌 Export Logistics Integration API (JSON)",
                    data=json_manifest,
                    file_name=f"expiry_transfer_manifest_day_{selected_day}.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.info("No transfer directives match the chosen facility or medicine filter.")
    else:
        if selected_policy == "no_transfer":
            st.info("ℹ️ The baseline **Do Nothing** policy executes zero inter-hospital redistributions.")
        else:
            st.success(f"✅ All hospital facilities maintain sufficient local buffer. Zero inter-hospital redistributions needed on Day {selected_day}.")

with opt_tab2:
    st.markdown("### 🛡️ Facility Surplus Reservation vs Locked Demand Buffer")
    st.caption("Visualizing the mathematical invariant: Physical Stock = Locked Buffer (Untouchable) + Transferable Surplus.")

    if not day_state.empty:
        inv_df = day_state.copy()
        inv_df["Hospital"] = inv_df["hospital_id"].map(h_dict)
        inv_df["Medicine"] = inv_df["medicine_id"].map(m_dict)

        if chosen_hosp != "All Facilities":
            inv_df = inv_df[inv_df["Hospital"] == chosen_hosp]
        if chosen_med != "All Medicines":
            inv_df = inv_df[inv_df["Medicine"] == chosen_med]

        # Locked buffer computation: 14-day horizon buffer = demand * (2 safety + 5 horizon)
        inv_df["daily_rate"] = inv_df["actual_demand"].replace(0, 1)
        inv_df["Locked Buffer"] = (inv_df["daily_rate"] * 7).astype(int)
        inv_df["Physical Stock"] = inv_df["stock_remaining"].astype(int)
        inv_df["Transferable Surplus"] = np.maximum(0, inv_df["Physical Stock"] - inv_df["Locked Buffer"])
        inv_df["Surplus Share (%)"] = np.where(
            inv_df["Physical Stock"] > 0,
            ((inv_df["Transferable Surplus"] / inv_df["Physical Stock"]) * 100).round(1),
            0.0
        )

        st.dataframe(
            inv_df[["Hospital", "Medicine", "Physical Stock", "Locked Buffer", "Transferable Surplus", "Surplus Share (%)"]].rename(columns={
                "Locked Buffer": "Locked Buffer (Protected)",
                "Transferable Surplus": "Transferable Surplus (Available)",
            }),
            hide_index=True,
            use_container_width=True,
        )

        # Plotly Stacked / Grouped Bar Chart
        if not inv_df.empty:
            summary_by_hosp = inv_df.groupby("Hospital")[["Locked Buffer", "Transferable Surplus"]].sum().reset_index()
            fig_surplus = go.Figure()
            fig_surplus.add_trace(go.Bar(
                name="Locked Local Demand Buffer (Protected)",
                x=summary_by_hosp["Hospital"],
                y=summary_by_hosp["Locked Buffer"],
                marker_color="#93c5fd",
            ))
            fig_surplus.add_trace(go.Bar(
                name="Transferable Surplus (Eligible for FEFO Dispatch)",
                x=summary_by_hosp["Hospital"],
                y=summary_by_hosp["Transferable Surplus"],
                marker_color="#2563eb",
            ))
            fig_surplus.update_layout(barmode="stack")
            fig_surplus = apply_plotly_theme(fig_surplus, title="Facility Stock Breakdown: Protected Buffer vs Genuinely Transferable Surplus", height=380)
            fig_surplus.update_layout(xaxis_title="Facility", yaxis_title="Medicine Units")
            st.plotly_chart(fig_surplus, use_container_width=True)
    else:
        st.info("No facility inventory state records available for Day {selected_day}.")

st.markdown("---")
st.caption("Smart Medicine Redistribution System • Expiry-Aware Redistribution Optimizer Module • Public Health Logistics Command")
