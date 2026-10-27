"""Hospital Administrator Portal: facility inventory, stock positions, and transfer manifests."""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_loader import load_serving_table, render_sample_banner
from app.labels import STATUS_CONFIG, GLOSSARY, get_scenario_display, get_policy_display
from app.state import render_sidebar_filters
from app.theme import inject_custom_theme, render_brand_header, render_system_status_bar

st.set_page_config(
    page_title="Hospital Administrator - Inventory & Orders",
    page_icon="👨‍⚕️",
    layout="wide",
)

inject_custom_theme()
render_brand_header(subtitle="Facility Administrator Portal • Local Inventory Management & Dispatch Orders")

render_sample_banner()

# Load Published Serving Tables
hospitals_df = load_serving_table("hospitals")
medicines_df = load_serving_table("medicines")
state_df = load_serving_table("state_daily")
transfers_df = load_serving_table("transfers")

active_filters = render_sidebar_filters(hospitals_df, state_df)
selected_scenario = active_filters["scenario"]
selected_policy = active_filters["policy"]
selected_day = active_filters["day"]

if hospitals_df.empty:
    st.warning("No hospital records available. Please run the pipeline demo.")
    st.stop()

# Hospital Facility Selector
h_dict = hospitals_df.set_index("hospital_id")["name"].to_dict()
h_options = list(h_dict.keys())
h_names = [f"{h_dict[h_id]} ({h_id})" for h_id in h_options]

c_sel, c_meta = st.columns([2, 3])
with c_sel:
    selected_h_idx = st.selectbox("Select Hospital Facility:", range(len(h_options)), format_func=lambda i: h_names[i])
    selected_h_id = h_options[selected_h_idx]

hosp_info = hospitals_df[hospitals_df["hospital_id"] == selected_h_id].iloc[0]
with c_meta:
    rural_badge = '<span class="badge badge-warning">Rural Facility</span>' if hosp_info['is_rural'] else '<span class="badge badge-info">Urban Hub</span>'
    st.markdown(
        f"""
        <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:24px;">
            <span class="badge badge-neutral">ID: {selected_h_id}</span>
            <span class="badge badge-info">{hosp_info['hospital_type']}</span>
            <span class="badge badge-neutral">District: {hosp_info['district']}</span>
            <span class="badge badge-neutral">Capacity: {hosp_info['bed_count']} Beds</span>
            {rural_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Name mappings
m_name_map = medicines_df.set_index("medicine_id")["name"].to_dict() if not medicines_df.empty else {}
m_crit_map = medicines_df.set_index("medicine_id")["criticality"].to_dict() if not medicines_df.empty else {}

# Filter Facility Daily State
hosp_state = state_df[
    (state_df["scenario"] == selected_scenario) &
    (state_df["policy"] == selected_policy) &
    (state_df["day"] == selected_day) &
    (state_df["hospital_id"] == selected_h_id)
].copy()

# Summary Metrics for Facility
h_tot_stock = int(hosp_state["stock_remaining"].sum()) if not hosp_state.empty and "stock_remaining" in hosp_state.columns else 0
h_tot_short = int(hosp_state["unmet_demand"].sum()) if not hosp_state.empty and "unmet_demand" in hosp_state.columns else 0
h_tot_exp = int(hosp_state["expired_units"].sum()) if not hosp_state.empty and "expired_units" in hosp_state.columns else 0

if not hosp_state.empty:
    hosp_state["Medicine"] = hosp_state["medicine_id"].map(m_name_map)
    hosp_state["Criticality"] = hosp_state["medicine_id"].map(m_crit_map)
    hosp_state["Daily Demand"] = hosp_state["actual_demand"].replace(0, 1)
    hosp_state["Days of Stock"] = (hosp_state["stock_remaining"] / hosp_state["Daily Demand"]).round(1)
    critical_items = int((hosp_state["Days of Stock"] <= 2).sum())
else:
    critical_items = 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Facility Total Stock", f"{h_tot_stock:,} units", help="Total physical stock remaining at this facility today.")
with m2:
    st.metric("Impending Shortages", f"{h_tot_short:,} units", delta=f"{h_tot_short} deficit", delta_color="inverse" if h_tot_short > 0 else "normal")
with m3:
    st.metric("Stock at Expiry Risk", f"{h_tot_exp:,} units", delta="FEFO Monitored", delta_color="inverse" if h_tot_exp > 0 else "normal")
with m4:
    st.metric("Critical Items (≤2d Stock)", f"{critical_items} medicines", delta="Urgent Reorder" if critical_items > 0 else "Balanced", delta_color="inverse" if critical_items > 0 else "normal")

st.markdown("---")

# Section 1: Stock & Days of Supply Table
st.markdown("### 📦 Stock & Days of Supply by Medicine")
st.caption(f"Active stock levels and days of supply for **{hosp_info['name']}** on Day {selected_day}.")

if not hosp_state.empty:
    hosp_state["Status"] = np.where(
        hosp_state["unmet_demand"] > 0, STATUS_CONFIG["critical_shortage"]["badge"],
        np.where(hosp_state["Days of Stock"] <= 3, STATUS_CONFIG["low_stock"]["badge"], STATUS_CONFIG["normal"]["badge"])
    )
    st.dataframe(
        hosp_state[["Medicine", "Criticality", "stock_remaining", "Days of Stock", "unmet_demand", "Status"]].rename(columns={
            "stock_remaining": "Current Stock",
            "unmet_demand": "Unmet Deficit",
        }),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info(f"No daily inventory state records available for this facility on Day {selected_day}.")

st.markdown("---")

# Section 2: Facility Transfer Orders Manifest (Incoming & Outgoing Directives)
st.markdown("### 🚚 Facility Transfer Manifests (Incoming & Outgoing Directives)")
st.caption(
    f"Active transfer orders involving **{hosp_info['name']}** on Day {selected_day} "
    f"under **{get_policy_display(selected_policy)}** ({get_scenario_display(selected_scenario)})."
)

h_trans = transfers_df[
    (transfers_df["day"] == selected_day) &
    ((transfers_df["from_hospital"] == selected_h_id) | (transfers_df["to_hospital"] == selected_h_id))
].copy() if not transfers_df.empty and "day" in transfers_df.columns else pd.DataFrame()

if not h_trans.empty:
    if "scenario" in h_trans.columns:
        h_trans = h_trans[h_trans["scenario"] == selected_scenario]
    if "policy" in h_trans.columns:
        h_trans = h_trans[h_trans["policy"] == selected_policy]

if not h_trans.empty:
    h_trans["From Facility"] = h_trans["from_hospital"].map(h_dict)
    h_trans["To Facility"] = h_trans["to_hospital"].map(h_dict)
    h_trans["Medicine"] = h_trans["medicine_id"].map(m_name_map)
    h_trans["Direction"] = np.where(
        h_trans["from_hospital"] == selected_h_id,
        "OUTGOING (Surplus Dispatch)",
        "INCOMING (Relief Inflow)",
    )
    h_trans["Quantity (units)"] = h_trans["quantity"].astype(int)
    h_trans["Transit (days)"] = h_trans["transit_days"].astype(int)
    h_trans["Estimated Cost (₹)"] = (100.0 + h_trans["distance_km"] * 2.5).round(0).astype(int)

    if "reason" not in h_trans.columns or h_trans["reason"].isna().all():
        h_trans["Reason"] = np.where(
            h_trans["from_hospital"] == selected_h_id,
            "Dispatched from surplus buffer to relieve destination stockout without risking local safety stock.",
            "Replenishment scheduled to arrive before local deficit causes patient stockout.",
        )
    else:
        h_trans["Reason"] = h_trans["reason"]

    display_trans = h_trans[[
        "Direction", "Medicine", "From Facility", "To Facility",
        "Quantity (units)", "Transit (days)", "Estimated Cost (₹)", "Reason"
    ]].rename(columns={"Transit (days)": "Transit Days"})

    st.dataframe(
        display_trans,
        hide_index=True,
        use_container_width=True,
    )

    cdl1, cdl2 = st.columns(2)
    with cdl1:
        csv_orders = display_trans.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Facility Transfer Manifest (CSV)",
            data=csv_orders,
            file_name=f"transfer_manifest_{selected_h_id}_day_{selected_day}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download official transfer manifest for facility dispatch.",
        )
    with cdl2:
        json_orders = display_trans.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            label="🔌 Export Logistics Integration (JSON)",
            data=json_orders,
            file_name=f"transfer_manifest_{selected_h_id}_day_{selected_day}.json",
            mime="application/json",
            use_container_width=True,
            help="Export structured JSON conforming to the Logistics Dispatch Integration Schema.",
        )
else:
    if selected_policy == "no_transfer":
        st.info("ℹ️ **Do Nothing (Baseline)** executes zero inter-hospital redistributions.")
    else:
        st.success(f"✅ No inter-hospital transfers required for {hosp_info['name']} on Day {selected_day}. Local stock is self-sufficient.")

st.markdown("---")
st.caption("Smart Medicine Redistribution Platform • Hospital Administrator Decision Support • Department of Health & Family Welfare")
