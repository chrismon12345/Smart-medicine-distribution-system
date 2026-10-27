"""Network-Wide Cascading Shortage Predictor & Geospatial Risk Map."""
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
    page_title="Network-Wide Cascading Shortage Predictor",
    page_icon="🗺️",
    layout="wide",
)

inject_custom_theme()
render_brand_header(subtitle="Network-Wide Cascading Shortage Predictor • Geospatial Risk Map & Multi-Day Vulnerability Propagation")

render_sample_banner()

# Load Serving Tables
hospitals_df = load_serving_table("hospitals")
medicines_df = load_serving_table("medicines")
state_df = load_serving_table("state_daily")
transfers_df = load_serving_table("transfers")
cascading_pred_df = load_serving_table("cascading_predictions")
cascading_time_df = load_serving_table("cascading_timeline")

active_filters = render_sidebar_filters(hospitals_df, state_df)
selected_scenario = active_filters["scenario"]
selected_policy = active_filters["policy"]
selected_day = active_filters["day"]

if hospitals_df.empty:
    st.warning("Hospital geographical data is not available. Please run the pipeline export.")
    st.stop()

# Helper dicts
h_dict = hospitals_df.set_index("hospital_id")["name"].to_dict()
h_lat = hospitals_df.set_index("hospital_id")["latitude"].to_dict()
h_lon = hospitals_df.set_index("hospital_id")["longitude"].to_dict()
m_dict = medicines_df.set_index("medicine_id")["name"].to_dict() if not medicines_df.empty else {}

# Scenario Shock Context Banner
st.markdown(
    f"""
    <div style="background:#f0f9ff; border:1.5px solid #bae6fd; border-left:5px solid #0284c7; border-radius:8px; padding:12px 18px; margin-bottom:18px;">
        <div style="font-weight:800; font-size:1.02rem; color:#0369a1; margin-bottom:4px;">
            🌐 Multi-Day Shortage Propagation Simulation • {get_scenario_display(selected_scenario)}
        </div>
        <div style="font-size:0.88rem; color:#1e293b; line-height:1.5;">
            Simulates daily inventory drawdowns over a 14-day horizon. Distinguishes <b>Direct Shortages</b> (primary shock epicenters) 
            from <b>Cascading Deficits</b> (secondary stockouts triggered when facilities assist neighbors or exhaust buffers). 
            Interactive map displays real-time geospatial vulnerability across all healthcare network nodes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Filter Cascading Predictions for current scenario (or fallback if scenario has no direct shock)
sc_pred = cascading_pred_df[cascading_pred_df["scenario"] == selected_scenario].copy() if not cascading_pred_df.empty else pd.DataFrame()
if sc_pred.empty and not cascading_pred_df.empty:
    # If the exact scenario key isn't in predictions, display available simulation data
    sc_pred = cascading_pred_df.copy()

sc_timeline = cascading_time_df[cascading_time_df["scenario"] == selected_scenario].copy() if not cascading_time_df.empty else pd.DataFrame()
if sc_timeline.empty and not cascading_time_df.empty:
    sc_timeline = cascading_time_df.copy()

# Compute Executive KPIs
tot_deficit = int(sc_pred["total_deficit_units"].sum()) if not sc_pred.empty else 0
direct_deficit = int(sc_pred[sc_pred["shortage_type"] == "Direct"]["total_deficit_units"].sum()) if not sc_pred.empty else 0
cascading_deficit = int(sc_pred[sc_pred["shortage_type"].str.contains("Cascading", na=False)]["total_deficit_units"].sum()) if not sc_pred.empty else 0
earliest_day = int(sc_pred["earliest_stockout_day"].min()) if not sc_pred.empty and sc_pred["earliest_stockout_day"].notna().any() else None
affected_facilities = int(sc_pred["hospital_id"].nunique()) if not sc_pred.empty else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Projected Deficit", f"{tot_deficit:,} units", help="Combined cumulative medicine shortfall across all affected network hospitals.")
with k2:
    st.metric("Direct Epicenter Deficit", f"{direct_deficit:,} units", delta=f"{affected_facilities} facilities at risk", delta_color="inverse" if direct_deficit > 0 else "normal")
with k3:
    st.metric("Cascading (Secondary) Deficit", f"{cascading_deficit:,} units", delta="Prevented by Closed-Loop MILP" if selected_policy == "optimizer" else "Unmitigated Ripple", delta_color="normal" if selected_policy == "optimizer" else "inverse")
with k4:
    earliest_txt = f"Day {earliest_day}" if earliest_day is not None else "None (Protected)"
    st.metric("Earliest Network Stockout", earliest_txt, help="The earliest operational day on which an unmitigated stockout occurs.")

st.markdown("---")

# Main Page Tabs
casc_tab1, casc_tab2, casc_tab3 = st.tabs([
    "🗺️ Interactive Geospatial Risk Map & Active Routing",
    "⚠️ Cascading Vulnerability Registry",
    "📊 14-Day Deficit Propagation Timeline",
])

with casc_tab1:
    st.markdown("### 🗺️ Geospatial Shortage Risk & Redistribution Network Map")
    st.caption("Interactive OpenStreetMap displaying hospital vulnerability states, direct shock epicenters, and active transfer vectors.")

    # Prepare map dataset by aggregating per hospital
    hosp_map_df = hospitals_df.copy()

    # Aggregate deficit and status per hospital from sc_pred
    if not sc_pred.empty:
        hosp_deficits = sc_pred.groupby("hospital_id").agg(
            total_def=("total_deficit_units", "sum"),
            earliest_so=("earliest_stockout_day", "min"),
            has_direct=("shortage_type", lambda s: (s == "Direct").any()),
            has_cascading=("shortage_type", lambda s: s.str.contains("Cascading").any()),
            root_causes=("root_cause", lambda c: "; ".join(c.dropna().unique()[:2])),
        ).reset_index()
        hosp_map_df = hosp_map_df.merge(hosp_deficits, on="hospital_id", how="left")
        hosp_map_df["total_def"] = hosp_map_df["total_def"].fillna(0).astype(int)
        hosp_map_df["earliest_so"] = hosp_map_df["earliest_so"].fillna(99).astype(int)
    else:
        hosp_map_df["total_def"] = 0
        hosp_map_df["earliest_so"] = 99
        hosp_map_df["has_direct"] = False
        hosp_map_df["has_cascading"] = False
        hosp_map_df["root_causes"] = "Stable Inventory"

    # Classify each facility
    conditions = [
        hosp_map_df["has_direct"] == True,
        hosp_map_df["has_cascading"] == True,
    ]
    choices_status = [
        "🚨 Direct Shortage Epicenter",
        "⚡ Cascading Deficit (Secondary)",
    ]
    choices_color = ["#ef4444", "#f97316"]

    hosp_map_df["Risk Category"] = np.select(conditions, choices_status, default="✅ Balanced / Safe Facility")
    hosp_map_df["Marker Color"] = np.select(conditions, choices_color, default="#10b981")
    hosp_map_df["Marker Size"] = np.where(
        hosp_map_df["total_def"] > 0,
        np.clip(14 + (hosp_map_df["total_def"] / 30.0), 14, 26),
        10
    )

    # Build Interactive Plotly Map with Scattermap
    fig_map = go.Figure()

    # Trace 1: Active Transfer Routes between hospitals
    day_trans = transfers_df[
        (transfers_df["day"] == selected_day)
    ].copy() if not transfers_df.empty and "day" in transfers_df.columns else pd.DataFrame()
    if not day_trans.empty and "scenario" in day_trans.columns:
        day_trans = day_trans[day_trans["scenario"] == selected_scenario]
    if not day_trans.empty and "policy" in day_trans.columns:
        day_trans = day_trans[day_trans["policy"] == selected_policy]

    if not day_trans.empty:
        route_lats = []
        route_lons = []
        for _, tr in day_trans.iterrows():
            f_id = tr.get("from_hospital")
            t_id = tr.get("to_hospital")
            if f_id in h_lat and t_id in h_lat:
                route_lats.extend([h_lat[f_id], h_lat[t_id], None])
                route_lons.extend([h_lon[f_id], h_lon[t_id], None])

        if route_lats:
            fig_map.add_trace(go.Scattermap(
                lat=route_lats,
                lon=route_lons,
                mode="lines",
                line=dict(width=2.5, color="#2563eb"),
                name="🚚 Active Transfer Vectors",
                hoverinfo="skip",
            ))

    # Add facility marker traces grouped by category for clean legend
    for cat, col in [
        ("🚨 Direct Shortage Epicenter", "#ef4444"),
        ("⚡ Cascading Deficit (Secondary)", "#f97316"),
        ("✅ Balanced / Safe Facility", "#10b981"),
    ]:
        sub_h = hosp_map_df[hosp_map_df["Risk Category"] == cat]
        if not sub_h.empty:
            custom_hovers = [
                f"<b>{r['name']}</b> ({r['hospital_id']})<br>"
                f"District: {r['district']} | Type: {r['hospital_type']}<br>"
                f"Status: {cat}<br>"
                f"Projected Deficit: <b>{r['total_def']:,} units</b><br>"
                f"Earliest Stockout: Day {r['earliest_so'] if r['earliest_so'] < 99 else 'None'}<br>"
                f"Root Cause: {r['root_causes']}"
                for _, r in sub_h.iterrows()
            ]
            fig_map.add_trace(go.Scattermap(
                lat=sub_h["latitude"],
                lon=sub_h["longitude"],
                mode="markers+text",
                marker=dict(
                    size=sub_h["Marker Size"],
                    color=col,
                    opacity=0.9,
                ),
                text=sub_h["name"],
                textposition="top right",
                name=cat,
                hoverinfo="text",
                hovertext=custom_hovers,
            ))

    # Center on Kerala
    kerala_lat = 10.5
    kerala_lon = 76.2
    fig_map.update_layout(
        map_style="open-street-map",
        map_center=dict(lat=kerala_lat, lon=kerala_lon),
        map_zoom=6.8,
        height=540,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Map Legend / Facility Status Breakdown
    st.markdown("#### 🏥 Facility Risk Distribution Across Health Network")
    breakdown_df = hosp_map_df[[
        "name", "hospital_type", "district", "bed_count", "Risk Category", "total_def"
    ]].rename(columns={
        "name": "Facility Name",
        "hospital_type": "Facility Type",
        "district": "District",
        "bed_count": "Beds",
        "total_def": "Projected Deficit (units)",
    })
    st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

with casc_tab2:
    st.markdown("### ⚠️ Multi-Facility Cascading Vulnerability Registry")
    st.caption("Detailed listing of all facilities and medicines facing direct or cascading stockouts over the 14-day planning horizon.")

    if not sc_pred.empty:
        reg_df = sc_pred[[
            "hospital_name", "district", "medicine_name", "shortage_type",
            "earliest_stockout_day", "days_until_stockout", "total_deficit_units",
            "root_cause", "propagation_path"
        ]].rename(columns={
            "hospital_name": "Hospital",
            "district": "District",
            "medicine_name": "Medicine",
            "shortage_type": "Shortage Classification",
            "earliest_stockout_day": "Earliest Stockout (Day)",
            "days_until_stockout": "Warning Lead Time (Days)",
            "total_deficit_units": "Cumulative Deficit (units)",
            "root_cause": "Primary Root Cause",
            "propagation_path": "Cascading Propagation Path",
        })

        st.dataframe(
            reg_df,
            hide_index=True,
            use_container_width=True,
        )

        c_c1, c_c2 = st.columns(2)
        with c_c1:
            csv_reg = reg_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Vulnerability Registry (CSV)",
                data=csv_reg,
                file_name=f"cascading_vulnerability_registry_{selected_scenario}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_c2:
            json_reg = reg_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                label="🔌 Export Vulnerability Schema (JSON)",
                data=json_reg,
                file_name=f"cascading_vulnerability_registry_{selected_scenario}.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.success("✅ Zero cascading or direct shortages projected under current operating conditions. Health network is resilient.")

with casc_tab3:
    st.markdown("### 📊 14-Day Deficit Propagation Progression")
    st.caption("Day-by-day progression comparing primary direct shortages at shock epicenters against secondary cascading deficits.")

    if not sc_timeline.empty:
        daily_agg = sc_timeline.groupby(["day", "shortage_type"])["deficit"].sum().reset_index()

        fig_prog = px.bar(
            daily_agg,
            x="day",
            y="deficit",
            color="shortage_type",
            color_discrete_map={
                "Direct": "#ef4444",
                "Cascading (Secondary)": "#f97316",
            },
            barmode="stack",
            labels={
                "day": "Simulated Operational Day",
                "deficit": "Medicine Deficit Units",
                "shortage_type": "Shortage Nature",
            },
        )
        fig_prog = apply_plotly_theme(fig_prog, title=f"Daily Network Deficit Evolution: Direct vs Cascading ({get_scenario_display(selected_scenario)})", height=380)
        st.plotly_chart(fig_prog, use_container_width=True)

        st.markdown(f"""
        > **Mitigation Analysis**: Under **{get_policy_display(selected_policy)}**, inter-hospital redistribution continuously intercepts 
        > cascading ripples by pre-positioning surplus medicine batches before donor hospital safety buffers are breached.
        """)
    else:
        st.info("No daily shortage progression data available for this scenario.")

st.markdown("---")
st.caption("Smart Medicine Redistribution System • Network-Wide Cascading Shortage Predictor • Public Health Logistics Command")
