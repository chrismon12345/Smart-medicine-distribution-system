"""Light Blue UI Design System, Medical Logo, Distinct Font/Button Color Hierarchy, and Plotly Theming."""
from typing import Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Light Blue Healthcare Design System
LIGHT_BLUE_THEME_CSS = """
<style>
/* Modern Clean Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0f172a !important;
}

/* Light Blue Calming Canvas Background */
.stApp {
    background: linear-gradient(180deg, #f0f7ff 0%, #f8fafc 100%) !important;
}

/* Remove excessive empty top spacing */
.block-container,
div[data-testid="stMainBlockContainer"],
div.block-container,
section.main > div.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
}

header[data-testid="stHeader"] {
    height: 1.5rem !important;
    background: transparent !important;
}

/* Sidebar: Crisp White with Soft Light Blue Border */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 2px solid #dbeafe !important;
    box-shadow: 2px 0 12px rgba(37, 99, 235, 0.04);
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Headings: Deep Navy-Slate (Distinct from blue buttons) */
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
}

/* Body Text: High-Contrast Charcoal Slate */
p, span, label, div {
    color: #334155;
}

label[data-testid="stWidgetLabel"] p {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}

/* Action Buttons: Vibrant Royal Blue (Distinct from Charcoal Text) */
div.stButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1.5px solid #1d4ed8 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    box-shadow: 0 2px 5px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.15s ease-in-out !important;
}

div.stButton > button:hover {
    background-color: #1d4ed8 !important;
    border-color: #1e40af !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px);
}

div.stButton > button:active {
    transform: translateY(0);
}

/* Download Manifest Buttons: Distinct Sapphire Blue */
div.stDownloadButton > button {
    background-color: #0284c7 !important;
    color: #ffffff !important;
    border: 1.5px solid #0369a1 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    box-shadow: 0 2px 5px rgba(2, 132, 199, 0.2) !important;
    transition: all 0.15s ease-in-out !important;
}

div.stDownloadButton > button:hover {
    background-color: #0369a1 !important;
    border-color: #075985 !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35) !important;
    transform: translateY(-1px);
}

/* KPI Metric Cards: Crisp White with Royal Blue Top Accent */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1.5px solid #dbeafe !important;
    border-top: 4px solid #2563eb !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.05) !important;
    transition: all 0.2s ease-in-out;
}

div[data-testid="stMetric"]:hover {
    border-color: #93c5fd !important;
    border-top: 4px solid #1d4ed8 !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.12) !important;
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    letter-spacing: -0.02em;
}

/* Distinct Semantic Badges & Keys */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: 0.02em;
}

.badge-success {
    background-color: #dcfce7;
    color: #15803d;
    border: 1.5px solid #86efac;
}

.badge-warning {
    background-color: #fef3c7;
    color: #b45309;
    border: 1.5px solid #fcd34d;
}

.badge-danger {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1.5px solid #fca5a5;
}

.badge-info {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1.5px solid #93c5fd;
}

.badge-neutral {
    background-color: #f8fafc;
    color: #334155;
    border: 1.5px solid #cbd5e1;
}

/* Table Containers */
div[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border: 1.5px solid #dbeafe !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.04) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #dbeafe !important;
    border-radius: 8px !important;
}

/* Form Inputs & Selects */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #2563eb !important;
}

/* Modern High-Visibility Pill Tabs */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: #e0f2fe !important;
    padding: 6px 8px !important;
    border-radius: 12px !important;
    border: 1.5px solid #bae6fd !important;
    margin-bottom: 1.25rem !important;
}

div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 9px 18px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    color: #0369a1 !important;
    background-color: transparent !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
}

div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background-color: rgba(37, 99, 235, 0.1) !important;
    color: #1d4ed8 !important;
}

div[data-testid="stTabs"] [aria-selected="true"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.35) !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* Brand Header Block */
.brand-header-container {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: -0.5rem !important;
    margin-bottom: 12px;
    padding-bottom: 6px;
}

.brand-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.025em;
    margin: 0;
    line-height: 1.15;
}

.brand-subtitle {
    font-size: 0.88rem;
    font-weight: 600;
    color: #2563eb;
    margin: 0;
}
</style>
"""


def inject_custom_theme() -> None:
    """Inject the unified light blue theme stylesheet into Streamlit."""
    st.markdown(LIGHT_BLUE_THEME_CSS, unsafe_allow_html=True)


def render_brand_header(subtitle: str = "Multi-Facility Public Health Logistics & Inventory Redistribution DSS") -> None:
    """Render the official brand header with the medical shield logo and original system name."""
    logo_svg = """
    <svg width="46" height="46" viewBox="0 0 54 54" fill="none" xmlns="http://www.w3.org/2000/svg">
        <!-- Hospital Protective Shield Base in Royal Blue -->
        <path d="M27 3L7 11V25C7 37 15.5 48.2 27 51C38.5 48.2 47 37 47 25V11L27 3Z" fill="#2563eb" stroke="#1d4ed8" stroke-width="2"/>
        <path d="M27 6L10 13V25C10 35.5 17.5 45.3 27 48C36.5 45.3 44 35.5 44 25V13L27 6Z" fill="#3b82f6"/>
        <!-- Pure White Medical Cross -->
        <rect x="23" y="15" width="8" height="24" rx="2" fill="#ffffff"/>
        <rect x="15" y="23" width="24" height="8" rx="2" fill="#ffffff"/>
        <!-- Heartbeat Rhythm -->
        <path d="M12 27H18L21 21L25 32L28 25L30 27H42" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
    header_html = f"""
    <div class="brand-header-container">
        <div>{logo_svg}</div>
        <div>
            <h1 class="brand-title">Smart Medicine Redistribution System</h1>
            <p class="brand-subtitle">{subtitle}</p>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_system_status_bar(*args, **kwargs) -> None:
    """No-op: Telemetry bar removed as requested by user."""
    pass


def apply_plotly_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 340,
    show_legend: bool = True,
) -> go.Figure:
    """Apply light blue theme aesthetics, dark high-contrast typography, and crisp gridlines to Plotly figures."""
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#0f172a"),
        height=height,
        margin=dict(l=24, r=24, t=44 if title else 24, b=24),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#2563eb",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color="#ffffff",
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="#ffffff",
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#1e293b", weight=600),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dbeafe",
            borderwidth=1,
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=14, weight=800, color="#0f172a"),
                x=0.0,
                xanchor="left",
            )
        )

    # High-contrast gridlines
    fig.update_xaxes(
        gridcolor="#f1f5f9",
        zerolinecolor="#cbd5e1",
        tickfont=dict(size=11, color="#334155", weight=600),
        title_font=dict(size=12, color="#2563eb", weight=700),
    )
    fig.update_yaxes(
        gridcolor="#f1f5f9",
        zerolinecolor="#cbd5e1",
        tickfont=dict(size=11, color="#334155", weight=600),
        title_font=dict(size=12, color="#2563eb", weight=700),
    )
    return fig
