"""Scenario Comparison: Empirical benchmark of policies across operational stress scenarios."""
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
from app.labels import (
    POLICY_LABELS,
    SCENARIO_INFO,
    get_scenario_display,
    get_policy_display,
    get_scenario_tag,
    GLOSSARY,
)
from app.state import render_sidebar_filters
from app.theme import inject_custom_theme, render_brand_header, render_system_status_bar, apply_plotly_theme

st.set_page_config(page_title="Scenario Comparison - Policy Benchmarks", page_icon="📊", layout="wide")

inject_custom_theme()
render_brand_header(subtitle="Policy Benchmarking & Stress Testing • Outbreaks, Supplier Shocks & Epidemic Surges")

render_sample_banner()

# Load Serving Data
hospitals_df = load_serving_table("hospitals")
state_df = load_serving_table("state_daily")
summary_df = load_serving_table("evaluation_optimizer")
catalog_df = load_serving_table("scenario_catalog")

render_sidebar_filters(hospitals_df, state_df)

if summary_df.empty:
    st.warning("No scenario summary metrics available. Please run the pipeline demo.")
    st.stop()

# Enrich Summary DataFrame with Plain Labels
df = summary_df.copy()
df["Scenario"] = df["scenario"].map(get_scenario_display)
df["Policy"] = df["policy"].map(get_policy_display)
df["Category"] = df["scenario"].map(get_scenario_tag)
df["Fill Rate (%)"] = (df["fill_rate_mean"] * 100.0).round(1)

# Tabbed Display: Seen Scenarios vs New Situations
tab_seen, tab_unseen, tab_all = st.tabs([
    "🟢 Baseline & Seasonal ('Seen Before')",
    "🟠 Stress Tests & Disruptions ('New Situations')",
    "📋 Complete Benchmark Matrix",
])

def render_scenario_charts(subset_df: pd.DataFrame, category_title: str):
    if subset_df.empty:
        st.info("No data available for this category.")
        return

    # Dynamic Takeaway Calculation
    opt_sub = subset_df[subset_df["policy"] == "optimizer"]
    base_sub = subset_df[subset_df["policy"] == "no_transfer"]
    if not opt_sub.empty and not base_sub.empty:
        diff_stockout = base_sub["stockout_days_mean"].mean() - opt_sub["stockout_days_mean"].mean()
        diff_waste = base_sub["wastage_units_mean"].mean() - opt_sub["wastage_units_mean"].mean()
        takeaway_msg = (
            f"💡 **Takeaway**: Across {category_title.lower()}, **Smart plan** reduced average stockout duration by "
            f"**{diff_stockout:.1f} facility-days** and prevented **{diff_waste:,.0f} units** of expired wastage compared with **Do nothing**."
        )
        st.markdown(takeaway_msg)

    c1, c2 = st.columns(2)

    with c1:
        fig_stockout = px.bar(
            subset_df,
            x="Scenario",
            y="stockout_days_mean",
            error_y="stockout_days_std" if "stockout_days_std" in subset_df.columns else None,
            color="Policy",
            barmode="group",
            labels={"stockout_days_mean": "Stockout Days (Mean ± Std)"},
            color_discrete_map={
                get_policy_display("optimizer"): "#10b981",
                get_policy_display("greedy"): "#38bdf8",
                get_policy_display("no_transfer"): "#ef4444",
            },
        )
        fig_stockout = apply_plotly_theme(fig_stockout, title="Stockout Duration (Facility-Days with Unmet Need)", height=330)
        st.plotly_chart(fig_stockout, use_container_width=True)

    with c2:
        fig_fill = px.bar(
            subset_df,
            x="Scenario",
            y="Fill Rate (%)",
            color="Policy",
            barmode="group",
            labels={"Fill Rate (%)": "Fill Rate (%)"},
            color_discrete_map={
                get_policy_display("optimizer"): "#10b981",
                get_policy_display("greedy"): "#38bdf8",
                get_policy_display("no_transfer"): "#ef4444",
            },
        )
        fig_fill = apply_plotly_theme(fig_fill, title="Patient Prescription Fill Rate (%)", height=330)
        fig_fill.update_layout(yaxis_range=[0, 105])
        st.plotly_chart(fig_fill, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig_waste = px.bar(
            subset_df,
            x="Scenario",
            y="wastage_units_mean",
            error_y="wastage_units_std" if "wastage_units_std" in subset_df.columns else None,
            color="Policy",
            barmode="group",
            labels={"wastage_units_mean": "Wastage Units (Mean ± Std)"},
            color_discrete_map={
                get_policy_display("optimizer"): "#10b981",
                get_policy_display("greedy"): "#38bdf8",
                get_policy_display("no_transfer"): "#ef4444",
            },
        )
        fig_waste = apply_plotly_theme(fig_waste, title="Expired Medicine Units Written Off", height=330)
        st.plotly_chart(fig_waste, use_container_width=True)

    with c4:
        fig_cost = px.bar(
            subset_df,
            x="Scenario",
            y="transport_cost_mean",
            color="Policy",
            barmode="group",
            labels={"transport_cost_mean": "Transport Cost (₹)"},
            color_discrete_map={
                get_policy_display("optimizer"): "#10b981",
                get_policy_display("greedy"): "#38bdf8",
                get_policy_display("no_transfer"): "#ef4444",
            },
        )
        fig_cost = apply_plotly_theme(fig_cost, title="Logistics & Transport Cost (₹)", height=330)
        st.plotly_chart(fig_cost, use_container_width=True)


with tab_seen:
    st.markdown("#### Baseline & Seasonal Scenarios ('Seen Before')")
    st.caption("Scenarios matching historical distribution parameters (Normal, Monsoon Surge, Low Initial Stock, Moderate Dengue).")
    render_scenario_charts(df[df["Category"] == "Seen Before"], "Seen Scenarios")

with tab_unseen:
    st.markdown("#### Out-of-Distribution Stress Tests ('New Situations')")
    st.caption("Unseen distribution shifts testing algorithm generalization (Severe 4x Dengue Epidemic, Supplier Disruption, Flooded Rural Roads).")
    render_scenario_charts(df[df["Category"] == "New Situation"], "Stress Scenarios")

with tab_all:
    st.markdown("#### Comprehensive Multi-Seed Scenario Benchmark")
    st.caption("Summary figures averaged across random seeds with standard deviation spread.")
    st.dataframe(
        df[[
            "Category", "Scenario", "Policy", "seeds_count",
            "stockout_days_mean", "stockout_days_std",
            "Fill Rate (%)",
            "wastage_units_mean", "wastage_units_std",
            "transport_cost_mean"
        ]].rename(columns={
            "seeds_count": "Seeds",
            "stockout_days_mean": "Stockout Days (Mean)",
            "stockout_days_std": "Stockout (Std)",
            "wastage_units_mean": "Wastage Units (Mean)",
            "wastage_units_std": "Wastage (Std)",
            "transport_cost_mean": "Transport (₹)",
        }),
        hide_index=True,
        use_container_width=True,
    )
