"""Shared application state and unified sidebar filtering layer."""
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
from app.labels import (
    POLICY_LABELS,
    LABEL_TO_POLICY,
    SCENARIO_INFO,
    LABEL_TO_SCENARIO,
    get_scenario_display,
    get_policy_display,
    GLOSSARY,
)


def init_session_state(state_df: pd.DataFrame) -> None:
    """Initialize session state and sync with st.query_params."""
    # Compute the operational day with the highest network shortage
    default_day = 7
    if not state_df.empty and "unmet_demand" in state_df.columns:
        day_shortages = state_df.groupby("day")["unmet_demand"].sum()
        if not day_shortages.empty and day_shortages.max() > 0:
            default_day = int(day_shortages.idxmax())

    # Read from query parameters if present
    params = st.query_params
    q_scenario = params.get("scenario", "normal")
    q_policy = params.get("policy", "optimizer")
    try:
        q_day = int(params.get("day", default_day))
    except (ValueError, TypeError):
        q_day = default_day

    if "scenario" not in st.session_state:
        st.session_state["scenario"] = q_scenario if q_scenario in SCENARIO_INFO else "normal"
    if "policy" not in st.session_state:
        st.session_state["policy"] = q_policy if q_policy in POLICY_LABELS else "optimizer"
    if "day" not in st.session_state:
        st.session_state["day"] = q_day
    if "district" not in st.session_state:
        st.session_state["district"] = "All Districts"
    if "hospital_type" not in st.session_state:
        st.session_state["hospital_type"] = "All Facilities"
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = "Government Officer"


def render_guided_tour() -> None:
    """Render high-priority operational workflow presets."""
    with st.sidebar.expander("⚡ **Executive Operational Presets**", expanded=False):
        st.caption("Quickly evaluate calibrated operational states across the health network:")

        if st.button("1️⃣ Normal Operations", use_container_width=True, help="Observe baseline routine inventory and redistribution"):
            st.session_state["scenario"] = "normal"
            st.session_state["policy"] = "optimizer"
            st.session_state["day"] = 5
            st.query_params.update({"scenario": "normal", "policy": "optimizer", "day": 5})
            st.rerun()

        if st.button("2️⃣ Monsoon Surge Outbreak", use_container_width=True, help="Simulate seasonal fever surge across districts"):
            st.session_state["scenario"] = "monsoon_surge"
            st.session_state["policy"] = "optimizer"
            st.session_state["day"] = 6
            st.query_params.update({"scenario": "monsoon_surge", "policy": "optimizer", "day": 6})
            st.rerun()

        if st.button("3️⃣ Policy Comparison Benchmark", use_container_width=True, help="Benchmark Smart Plan against Do Nothing"):
            st.session_state["scenario"] = "monsoon_surge"
            st.session_state["policy"] = "optimizer"
            st.query_params.update({"scenario": "monsoon_surge", "policy": "optimizer"})
            st.switch_page("pages/2_Scenario_Comparison.py")

        if st.button("4️⃣ Forecast Replay Audit", use_container_width=True, help="Inspect ML demand forecasts vs actual consumption"):
            st.session_state["scenario"] = "normal"
            st.query_params.update({"scenario": "normal"})
            st.switch_page("pages/3_Replay.py")

        if st.button("5️⃣ Model Evaluation", use_container_width=True, help="Review forecasting metrics, baselines, and ablation tests"):
            st.switch_page("pages/4_Model_Evaluation.py")


def render_sidebar_filters(
    hospitals_df: pd.DataFrame,
    state_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Render the standard sidebar filters displayed identically across all pages."""
    init_session_state(state_df)

    st.sidebar.markdown("### 🎛️ Operational Controls")

    # 1. Operational Presets Expander
    render_guided_tour()

    # 2. Operating Scenario
    scenario_display_options = [
        info["display_name"] for info in SCENARIO_INFO.values()
    ]
    curr_sc_display = get_scenario_display(st.session_state["scenario"])
    sc_idx = scenario_display_options.index(curr_sc_display) if curr_sc_display in scenario_display_options else 0

    selected_sc_disp = st.sidebar.selectbox(
        "Operating Scenario",
        scenario_display_options,
        index=sc_idx,
        help="Simulated operational environment (outbreaks, disruptions, or standard conditions).",
    )
    selected_sc = LABEL_TO_SCENARIO.get(selected_sc_disp, "normal")
    st.session_state["scenario"] = selected_sc

    # Display Scenario Tag (Seen Before vs New Situation)
    sc_info = SCENARIO_INFO.get(selected_sc, {})
    tag = sc_info.get("tag", "Standard")
    tag_color = "#10b981" if tag == "Seen Before" else "#f59e0b"
    st.sidebar.markdown(
        f"<div style='font-size:11px; padding:2px 8px; border-radius:4px; display:inline-block; background-color:{tag_color}22; color:{tag_color}; border:1px solid {tag_color}; font-weight:600; margin-bottom:8px;'>{tag.upper()}</div>",
        unsafe_allow_html=True,
    )

    # 3. Policy Selector
    policy_display_options = list(POLICY_LABELS.values())
    curr_pol_display = get_policy_display(st.session_state["policy"])
    pol_idx = policy_display_options.index(curr_pol_display) if curr_pol_display in policy_display_options else 0

    selected_pol_disp = st.sidebar.selectbox(
        "Redistribution Policy",
        policy_display_options,
        index=pol_idx,
        help=GLOSSARY.get("Smart plan", "Inter-hospital redistribution policy"),
    )
    selected_pol = LABEL_TO_POLICY.get(selected_pol_disp, "optimizer")
    st.session_state["policy"] = selected_pol

    # 4. Operational Day Slider
    max_day = 13
    if not state_df.empty and "day" in state_df.columns:
        max_day = int(state_df["day"].max())

    curr_day = min(int(st.session_state.get("day", 7)), max_day)
    selected_day = st.sidebar.slider(
        "Simulated Operational Day",
        min_value=0,
        max_value=max_day,
        value=curr_day,
        help="The specific day within the 14-day simulation horizon to inspect.",
    )
    st.session_state["day"] = selected_day

    # 5. Advanced Filters Expander (District & Hospital Type)
    has_active_advanced = (
        st.session_state.get("district", "All Districts") != "All Districts"
        or st.session_state.get("hospital_type", "All Facilities") != "All Facilities"
    )
    with st.sidebar.expander("⚙️ Advanced Filters", expanded=has_active_advanced):
        districts = ["All Districts"]
        if not hospitals_df.empty and "district" in hospitals_df.columns:
            districts += sorted(hospitals_df["district"].dropna().unique().tolist())
        curr_dist = st.session_state.get("district", "All Districts")
        dist_idx = districts.index(curr_dist) if curr_dist in districts else 0
        selected_district = st.selectbox("District", districts, index=dist_idx)
        st.session_state["district"] = selected_district

        facility_types = ["All Facilities"]
        if not hospitals_df.empty and "hospital_type" in hospitals_df.columns:
            facility_types += sorted(hospitals_df["hospital_type"].dropna().unique().tolist())
        curr_ht = st.session_state.get("hospital_type", "All Facilities")
        ht_idx = facility_types.index(curr_ht) if curr_ht in facility_types else 0
        selected_type = st.selectbox("Facility Type", facility_types, index=ht_idx)
        st.session_state["hospital_type"] = selected_type

    # Sync query parameters for easy sharing
    st.query_params.update({
        "scenario": selected_sc,
        "policy": selected_pol,
        "day": selected_day,
    })

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reload / Refresh Cache", use_container_width=True, help="Invalidate cached data and reload fresh tables from disk"):
        from app.data_loader import clear_data_cache
        clear_data_cache()
        st.rerun()
    st.sidebar.caption("Smart Medicine Redistribution System • Health Logistics Command")

    return {
        "scenario": selected_sc,
        "policy": selected_pol,
        "day": selected_day,
        "district": selected_district,
        "hospital_type": selected_type,
    }
