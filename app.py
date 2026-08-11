import os

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Construction Safety Risk Predictor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# UI HELPERS & GLOBAL STYLES
# =========================================================

PAGES = [
    "Overview",
    "Risk Predictor",
    "Incident Explorer",
    "Risk Patterns",
    "Preventive Actions",
    "Weekly Safety Brief",
]

RISK_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#3b82f6",
    "HIGH": "#f59e0b",
    "CRITICAL": "#ef4444",
}


def inject_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-primary: #0b0f14;
            --bg-secondary: #111827;
            --bg-card: #1a2332;
            --bg-card-hover: #1f2937;
            --border: #2d3a4f;
            --border-light: #374151;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #2563eb;
            --accent-soft: rgba(37, 99, 235, 0.15);
            --safe: #22c55e;
            --safe-soft: rgba(34, 197, 94, 0.12);
            --warning: #f59e0b;
            --warning-soft: rgba(245, 158, 11, 0.12);
            --danger: #ef4444;
            --danger-soft: rgba(239, 68, 68, 0.12);
            --radius: 12px;
            --radius-sm: 8px;
            --shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #0b0f14 0%, #0f172a 100%);
            color: var(--text-primary);
        }

        /* Hide default sidebar */
        section[data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Hero header */
        .app-hero {
            background: linear-gradient(135deg, #1a2332 0%, #111827 50%, #0f172a 100%);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 2rem 2.5rem;
            margin-bottom: 1.25rem;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }

        .app-hero::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 280px;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(37, 99, 235, 0.06));
            pointer-events: none;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--accent-soft);
            color: #93c5fd;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(37, 99, 235, 0.25);
            margin-bottom: 0.75rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0 0 0.5rem 0;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1.15rem;
            font-weight: 500;
            color: #cbd5e1;
            margin: 0 0 0.75rem 0;
        }

        .hero-desc {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin: 0;
            max-width: 720px;
            line-height: 1.6;
        }

        /* Top navigation */
        .top-nav-wrapper {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.35rem;
            margin-bottom: 1.75rem;
            box-shadow: var(--shadow);
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-container-marker)
            div[data-testid="stHorizontalBlock"] {
            gap: 0.25rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-container-marker)
            div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: transparent !important;
            border: none !important;
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
            padding: 0.65rem 0.5rem !important;
            border-radius: var(--radius-sm) !important;
            transition: all 0.15s ease !important;
            box-shadow: none !important;
            white-space: nowrap !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-container-marker)
            div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.05) !important;
            color: var(--text-primary) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-container-marker)
            div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background: var(--accent) !important;
            border: 1px solid rgba(37, 99, 235, 0.5) !important;
            color: #fff !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            padding: 0.65rem 0.5rem !important;
            border-radius: var(--radius-sm) !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
            white-space: nowrap !important;
        }

        /* Page sections */
        .page-header {
            margin-bottom: 1.5rem;
        }

        .page-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0 0 0.35rem 0;
        }

        .page-subtitle {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin: 0;
            line-height: 1.5;
        }

        .section-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 1rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-light);
        }

        .section-title-sm {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 0.75rem 0;
        }

        /* Metric cards */
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.15rem 1.25rem;
            height: 100%;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
        }

        .metric-card.default { border-top: 3px solid var(--accent); }
        .metric-card.critical { border-top: 3px solid var(--danger); }
        .metric-card.high { border-top: 3px solid var(--warning); }
        .metric-card.medium { border-top: 3px solid #3b82f6; }
        .metric-card.low { border-top: 3px solid var(--safe); }
        .metric-card.neutral { border-top: 3px solid var(--border-light); }

        .metric-card-icon {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }

        .metric-card-label {
            font-size: 0.78rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }

        .metric-card-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.1;
        }

        /* Risk result banner */
        .risk-banner {
            border-radius: var(--radius);
            padding: 1rem 1.25rem;
            margin: 1rem 0;
            font-weight: 500;
            font-size: 0.95rem;
            border: 1px solid;
        }

        .risk-banner.critical {
            background: var(--danger-soft);
            border-color: rgba(239, 68, 68, 0.35);
            color: #fca5a5;
        }

        .risk-banner.high {
            background: var(--warning-soft);
            border-color: rgba(245, 158, 11, 0.35);
            color: #fcd34d;
        }

        .risk-banner.medium {
            background: var(--accent-soft);
            border-color: rgba(37, 99, 235, 0.35);
            color: #93c5fd;
        }

        .risk-banner.low {
            background: var(--safe-soft);
            border-color: rgba(34, 197, 94, 0.35);
            color: #86efac;
        }

        /* Result metrics */
        .result-metric {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.25rem;
            text-align: center;
        }

        .result-metric-label {
            font-size: 0.78rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.5rem;
        }

        .result-metric-value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        /* List items */
        .list-item {
            padding: 0.6rem 0.85rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-light);
            border-radius: var(--radius-sm);
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-primary);
        }

        .list-item-factor {
            padding: 0.65rem 0.85rem;
            background: rgba(255, 255, 255, 0.02);
            border-left: 3px solid var(--accent);
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .recommendation-item {
            padding: 0.6rem 0.85rem;
            background: var(--safe-soft);
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: var(--radius-sm);
            margin-bottom: 0.45rem;
            font-size: 0.9rem;
            color: #bbf7d0;
        }

        .toolbox-item {
            padding: 0.5rem 0.85rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        /* Insight callout */
        .insight-box {
            background: var(--accent-soft);
            border: 1px solid rgba(37, 99, 235, 0.25);
            border-radius: var(--radius);
            padding: 1rem 1.25rem;
            color: #bfdbfe;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        /* Checkbox styling */
        .stCheckbox label span {
            font-size: 0.92rem !important;
            color: var(--text-primary) !important;
        }

        div[data-testid="stCheckbox"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-light);
            border-radius: var(--radius-sm);
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.4rem;
        }

        /* Dataframe */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
        }

        /* Inputs */
        div[data-testid="stSelectbox"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextArea"] label {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }

        /* Footer */
        .app-footer {
            margin-top: 2.5rem;
            padding: 1rem 1.25rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            font-size: 0.78rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        /* Responsive nav */
        @media (max-width: 900px) {
            .hero-title { font-size: 1.5rem; }
            .hero-subtitle { font-size: 1rem; }
            .app-hero { padding: 1.5rem; }
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-container-marker)
                div[data-testid="stHorizontalBlock"] button[kind="secondary"],
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-container-marker)
                div[data-testid="stHorizontalBlock"] button[kind="primary"] {
                font-size: 0.72rem !important;
                padding: 0.5rem 0.25rem !important;
            }
        }

        @media (max-width: 640px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .metric-card-value { font-size: 1.35rem; }
        }

        hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="app-hero">
            <div class="hero-badge">🦺 Construction Safety Analytics</div>
            <h1 class="hero-title">Construction Safety Risk Predictor</h1>
            <p class="hero-subtitle">Predict construction activity risk before work begins</p>
            <p class="hero-desc">
                Analyze planned activities, identify recurring safety patterns,
                and receive preventive safety recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav():
    if "current_page" not in st.session_state:
        st.session_state.current_page = PAGES[0]

    with st.container():
        st.markdown(
            '<div class="nav-container-marker"></div><div class="top-nav-wrapper"></div>',
            unsafe_allow_html=True,
        )
        nav_cols = st.columns(len(PAGES))
        for idx, page_name in enumerate(PAGES):
            with nav_cols[idx]:
                is_active = st.session_state.current_page == page_name
                if st.button(
                    page_name,
                    key=f"nav_btn_{page_name}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_page = page_name
                    st.rerun()

    return st.session_state.current_page


def render_page_header(title, subtitle=""):
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-header">
            <h2 class="page-title">{title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, icon="", variant="default"):
    st.markdown(
        f"""
        <div class="metric-card {variant}">
            <div class="metric-card-icon">{icon}</div>
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_risk_banner(risk_level, message):
    level = risk_level.lower()
    st.markdown(
        f'<div class="risk-banner {level}">{message}</div>',
        unsafe_allow_html=True,
    )


def render_result_metric(label, value, color="#f1f5f9"):
    st.markdown(
        f"""
        <div class="result-metric">
            <div class="result-metric-label">{label}</div>
            <div class="result-metric-value" style="color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_fig(fig, title=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter, sans-serif"),
        title=dict(
            text=title or fig.layout.title.text,
            font=dict(color="#f1f5f9", size=14),
        ) if (title or fig.layout.title.text) else None,
        xaxis=dict(
            gridcolor="rgba(45, 58, 79, 0.5)",
            linecolor="#2d3a4f",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor="rgba(45, 58, 79, 0.5)",
            linecolor="#2d3a4f",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(font=dict(color="#94a3b8")),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            This application is an educational MVP for analyzing historical
            safety patterns. It is not a certified safety assessment system
            and should not replace qualified safety professionals, site
            procedures, inspections, risk assessments, or applicable regulations.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DATA PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "incidents.csv")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_incidents():
    df = pd.read_csv(DATA_PATH)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


try:
    incidents = load_incidents()
    data_loaded = True
except Exception as e:
    incidents = pd.DataFrame()
    data_loaded = False
    data_error = str(e)


# =========================================================
# APP SHELL — HEADER & NAVIGATION
# =========================================================

inject_global_css()
render_hero()
page = render_top_nav()


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    render_page_header(
        "Safety Overview",
        "Dashboard summary of historical construction safety incidents and risk distribution.",
    )

    if not data_loaded:

        st.error("Could not load the incident dataset.")

        st.code(data_error)

    else:

        # -------------------------------------------------
        # BASIC COUNTS
        # -------------------------------------------------

        total_incidents = len(incidents)

        high_count = int(
            (incidents["risk_level"] == "HIGH").sum()
        )

        medium_count = int(
            (incidents["risk_level"] == "MEDIUM").sum()
        )

        low_count = int(
            (incidents["risk_level"] == "LOW").sum()
        )

        critical_count = int(
            (incidents["risk_level"] == "CRITICAL").sum()
        )

        # -------------------------------------------------
        # KPI CARDS
        # -------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            render_metric_card("Total Incidents", total_incidents, "📊", "default")

        with col2:
            render_metric_card("Critical Risk", critical_count, "🔴", "critical")

        with col3:
            render_metric_card("High Risk", high_count, "🟠", "high")

        with col4:
            render_metric_card("Medium Risk", medium_count, "🔵", "medium")

        with col5:
            render_metric_card("Low Risk", low_count, "🟢", "low")

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # RISK DISTRIBUTION
        # -------------------------------------------------

        render_section_title("Risk Level Distribution")

        risk_counts = (
            incidents["risk_level"]
            .value_counts()
            .reindex(
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = ["Risk Level", "Incidents"]

        risk_color_map = {
            "LOW": RISK_COLORS["LOW"],
            "MEDIUM": RISK_COLORS["MEDIUM"],
            "HIGH": RISK_COLORS["HIGH"],
            "CRITICAL": RISK_COLORS["CRITICAL"],
        }

        fig_risk = px.bar(
            risk_counts,
            x="Risk Level",
            y="Incidents",
            title="Incidents by Risk Level",
            text="Incidents",
            color="Risk Level",
            color_discrete_map=risk_color_map,
        )

        fig_risk.update_traces(
            textposition="outside"
        )

        fig_risk.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Number of Incidents",
            showlegend=False,
        )

        style_plotly_fig(fig_risk)
        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

        # -------------------------------------------------
        # TWO COLUMN CHART SECTION
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        # -------------------------------------------------
        # ACTIVITY DISTRIBUTION
        # -------------------------------------------------

        with col1:

            render_section_title("Incidents by Activity")

            activity_counts = (
                incidents["activity_type"]
                .value_counts()
                .reset_index()
            )

            activity_counts.columns = [
                "Activity",
                "Incidents"
            ]

            fig_activity = px.bar(
                activity_counts,
                x="Incidents",
                y="Activity",
                orientation="h",
                title="Incidents by Construction Activity",
                color="Incidents",
                color_continuous_scale=["#1e3a5f", "#2563eb"],
            )

            fig_activity.update_layout(showlegend=False, coloraxis_showscale=False)
            style_plotly_fig(fig_activity)
            st.plotly_chart(
                fig_activity,
                use_container_width=True
            )

        # -------------------------------------------------
        # SEVERITY DISTRIBUTION
        # -------------------------------------------------

        with col2:

            render_section_title("Severity Distribution")

            severity_counts = (
                incidents["severity"]
                .value_counts()
                .reset_index()
            )

            severity_counts.columns = [
                "Severity",
                "Incidents"
            ]

            fig_severity = px.pie(
                severity_counts,
                names="Severity",
                values="Incidents",
                title="Incident Severity",
                color_discrete_sequence=["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#64748b"],
            )

            style_plotly_fig(fig_severity)
            st.plotly_chart(
                fig_severity,
                use_container_width=True
            )

        # -------------------------------------------------
        # INCIDENT TREND
        # -------------------------------------------------

        render_section_title("Incident Trend Over Time")

        daily_incidents = (
            incidents
            .dropna(subset=["date"])
            .groupby("date")
            .size()
            .reset_index(name="Incidents")
            .sort_values("date")
        )

        fig_trend = px.line(
            daily_incidents,
            x="date",
            y="Incidents",
            markers=True,
            title="Daily Incident Count",
        )

        fig_trend.update_traces(line_color="#2563eb", marker_color="#3b82f6")
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Incidents"
        )

        style_plotly_fig(fig_trend)
        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )

        # -------------------------------------------------
        # HIGH-RISK ACTIVITIES
        # -------------------------------------------------

        render_section_title("High-Risk Activities")

        high_risk_data = incidents[
            incidents["risk_level"].isin(
                ["HIGH", "CRITICAL"]
            )
        ]

        high_risk_activity = (
            high_risk_data["activity_type"]
            .value_counts()
            .reset_index()
        )

        high_risk_activity.columns = [
            "Activity",
            "High/Critical Incidents"
        ]

        fig_high_risk = px.bar(
            high_risk_activity,
            x="Activity",
            y="High/Critical Incidents",
            title="Activities with High or Critical Risk",
            color="High/Critical Incidents",
            color_continuous_scale=["#92400e", "#ef4444"],
        )

        fig_high_risk.update_layout(showlegend=False, coloraxis_showscale=False)
        fig_high_risk.update_layout(
            xaxis_title="Activity",
            yaxis_title="High/Critical Incidents"
        )

        style_plotly_fig(fig_high_risk)
        st.plotly_chart(
            fig_high_risk,
            use_container_width=True
        )

        # -------------------------------------------------
        # MOST COMMON HIGH-RISK ACTIVITY
        # -------------------------------------------------

        if not high_risk_activity.empty:

            top_activity = high_risk_activity.iloc[0]["Activity"]
            top_activity_count = int(
                high_risk_activity.iloc[0]["High/Critical Incidents"]
            )

            st.markdown(
                f"""
                <div class="insight-box">
                    Most common high/critical-risk activity in this dataset:
                    <strong>{top_activity}</strong> ({top_activity_count} incidents).
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# RISK PREDICTOR PLACEHOLDER
# =========================================================

# =========================================================
# RISK PREDICTOR
# =========================================================

elif page == "Risk Predictor":

    render_page_header(
        "🔎 Construction Risk Predictor",
        "Enter planned construction activity details. The trained machine-learning model "
        "will estimate risk level, score, confidence, and contributing factors.",
    )

    # -----------------------------------------------------
    # Import prediction and recommendation modules
    # -----------------------------------------------------

    from src.predictor import predict
    from src.recommendations import get_recommendations
    from src.toolbox_talk import get_toolbox_topics

    # -----------------------------------------------------
    # Load available values from historical data
    # -----------------------------------------------------

    if data_loaded:

        activity_options = sorted(
            incidents["activity_type"]
            .dropna()
            .unique()
            .tolist()
        )

        location_options = sorted(
            incidents["location_type"]
            .dropna()
            .unique()
            .tolist()
        )

        weather_options = sorted(
            incidents["weather"]
            .dropna()
            .unique()
            .tolist()
        )

        shift_options = sorted(
            incidents["shift"]
            .dropna()
            .unique()
            .tolist()
        )

    else:

        activity_options = [
            "Working at Height",
            "Lifting",
            "Scaffolding",
            "Excavation",
            "Electrical Work",
            "Material Handling",
            "Welding",
            "Confined Space",
            "Vehicle Movement",
            "Housekeeping"
        ]

        location_options = [
            "Roof",
            "Electrical Room",
            "Excavation Area",
            "Warehouse",
            "Loading Area"
        ]

        weather_options = [
            "Clear",
            "Rain",
            "Adverse"
        ]

        shift_options = [
            "Day",
            "Night"
        ]

    # -----------------------------------------------------
    # INPUT FORM
    # -----------------------------------------------------

    render_section_title("Activity Details")

    col1, col2 = st.columns(2)

    with col1:

        activity_type = st.selectbox(
            "Activity Type",
            activity_options
        )

        location_type = st.selectbox(
            "Location Type",
            location_options
        )

        weather = st.selectbox(
            "Weather",
            weather_options
        )

        shift = st.selectbox(
            "Shift",
            shift_options
        )

    with col2:

        ppe_compliance_pct = st.slider(
            "PPE Compliance (%)",
            min_value=0,
            max_value=100,
            value=85,
            step=1
        )

        previous_incidents_30d = st.number_input(
            "Previous Incidents in Last 30 Days",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

        crew_size = st.number_input(
            "Crew Size",
            min_value=1,
            max_value=100,
            value=6,
            step=1
        )

    description = st.text_area(
        "Activity / Hazard Description",
        value="",
        placeholder="Example: Unprotected edge observed during elevated work."
    )

    st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button(
        "🚨 Predict Risk",
        type="primary",
        use_container_width=True
    ):

        activity_data = {
            "activity_type": activity_type,
            "location_type": location_type,
            "weather": weather,
            "shift": shift,
            "ppe_compliance_pct": ppe_compliance_pct,
            "previous_incidents_30d": previous_incidents_30d,
            "crew_size": crew_size,
            "description": description
        }

        try:

            result = predict(
                activity_data,
                strict=True
            )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            render_section_title("Prediction Result")

            risk_level = result["risk_level"]
            risk_score = result["risk_score"]
            confidence = result["confidence"]

            # -------------------------------------------------
            # RESULT METRICS
            # -------------------------------------------------

            risk_color = RISK_COLORS.get(risk_level, "#f1f5f9")

            col1, col2, col3 = st.columns(3)

            with col1:
                render_result_metric("Risk Level", risk_level, risk_color)

            with col2:
                render_result_metric("Risk Score", f"{risk_score:.1f} / 100", risk_color)

            with col3:

                confidence_display = confidence

                if confidence_display <= 1:
                    confidence_display = confidence_display * 100

                render_result_metric(
                    "Model Confidence",
                    f"{confidence_display:.1f}%",
                    "#94a3b8",
                )

            # -------------------------------------------------
            # RISK MESSAGE
            # -------------------------------------------------

            if risk_level == "CRITICAL":

                render_risk_banner(
                    risk_level,
                    "⚠️ CRITICAL RISK — Immediate attention and "
                    "appropriate safety controls are required before work.",
                )

            elif risk_level == "HIGH":

                render_risk_banner(
                    risk_level,
                    "⚠️ HIGH RISK — Review hazards and verify "
                    "preventive controls before starting work.",
                )

            elif risk_level == "MEDIUM":

                render_risk_banner(
                    risk_level,
                    "MEDIUM RISK — Review the activity hazards "
                    "and verify appropriate controls.",
                )

            else:

                render_risk_banner(
                    risk_level,
                    "LOW RISK — Continue with normal safety controls "
                    "and monitoring.",
                )

            # -------------------------------------------------
            # PROBABILITIES
            # -------------------------------------------------

            render_section_title("Risk Probabilities")

            probabilities = result.get(
                "probabilities",
                {}
            )

            if probabilities:

                probability_df = pd.DataFrame(
                    {
                        "Risk Level": list(probabilities.keys()),
                        "Probability (%)": [
                            value * 100
                            for value in probabilities.values()
                        ]
                    }
                )

                fig_probability = px.bar(
                    probability_df,
                    x="Risk Level",
                    y="Probability (%)",
                    text="Probability (%)",
                    title="Model Probability by Risk Level",
                    color="Risk Level",
                    color_discrete_map=RISK_COLORS,
                )

                fig_probability.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                fig_probability.update_layout(showlegend=False)
                style_plotly_fig(fig_probability)
                st.plotly_chart(
                    fig_probability,
                    use_container_width=True
                )

            # -------------------------------------------------
            # CONTRIBUTING FACTORS
            # -------------------------------------------------

            render_section_title("🔍 Top Contributing Factors")

            top_factors = result.get(
                "top_factors",
                []
            )

            if top_factors:

                for factor in top_factors:

                    if isinstance(factor, dict):

                        name = factor.get(
                            "label",
                            "Factor"
                        )

                        contribution = factor.get(
                            "contribution"
                        )

                        direction = factor.get(
                            "direction",
                            ""
                        )

                        if contribution is not None:

                            st.markdown(
                                f'<div class="list-item-factor">'
                                f'<strong>{name}</strong> {direction} '
                                f'(contribution: {contribution:.4f})'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        else:

                            st.markdown(
                                f'<div class="list-item-factor">'
                                f'<strong>{name}</strong> {direction}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    else:

                        st.markdown(
                            f'<div class="list-item-factor">• {factor}</div>',
                            unsafe_allow_html=True,
                        )

            else:

                st.markdown(
                    '<div class="list-item">No major contributing factors were returned.</div>',
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------
            # RECOMMENDATIONS
            # -------------------------------------------------

            render_section_title("🛡️ Recommended Preventive Actions")

            recommendations = get_recommendations(
                activity_type
            )

            for recommendation in recommendations:

                st.markdown(
                    f'<div class="recommendation-item">✅ {recommendation}</div>',
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------
            # TOOLBOX TALK
            # -------------------------------------------------

            render_section_title("🗣️ Toolbox Talk Topics")

            toolbox_topics = get_toolbox_topics(
                activity_type
            )

            for topic in toolbox_topics:

                st.markdown(
                    f'<div class="toolbox-item">• {topic}</div>',
                    unsafe_allow_html=True,
                )

        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(e)


# =========================================================
# INCIDENT EXPLORER PLACEHOLDER
# =========================================================

# =========================================================
# INCIDENT EXPLORER
# =========================================================

elif page == "Incident Explorer":

    render_page_header(
        "📋 Incident Explorer",
        "Explore historical construction safety incidents and filter by activity, severity, and risk level.",
    )

    if not data_loaded:
        st.error("Incident data could not be loaded.")
    else:

        # -------------------------------------------------
        # FILTERS
        # -------------------------------------------------

        render_section_title("Filter Incidents")

        col1, col2, col3 = st.columns(3)

        with col1:

            risk_options = ["All"] + sorted(
                incidents["risk_level"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_risk = st.selectbox(
                "Risk Level",
                risk_options
            )

        with col2:

            activity_options = ["All"] + sorted(
                incidents["activity_type"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_activity = st.selectbox(
                "Activity Type",
                activity_options
            )

        with col3:

            severity_options = ["All"] + sorted(
                incidents["severity"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_severity = st.selectbox(
                "Severity",
                severity_options
            )

        search_text = st.text_input(
            "Search incident description",
            placeholder="Example: cable, fall, scaffold, excavation..."
        )

        # -------------------------------------------------
        # APPLY FILTERS
        # -------------------------------------------------

        filtered = incidents.copy()

        if selected_risk != "All":
            filtered = filtered[
                filtered["risk_level"] == selected_risk
            ]

        if selected_activity != "All":
            filtered = filtered[
                filtered["activity_type"] == selected_activity
            ]

        if selected_severity != "All":
            filtered = filtered[
                filtered["severity"] == selected_severity
            ]

        if search_text.strip():

            filtered = filtered[
                filtered["description"]
                .fillna("")
                .str.contains(
                    search_text.strip(),
                    case=False,
                    na=False
                )
            ]

        # -------------------------------------------------
        # RESULTS SUMMARY
        # -------------------------------------------------

        st.markdown("<div style='margin:1.25rem 0;'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            render_metric_card("Matching Incidents", len(filtered), "📋", "default")

        with col2:

            high_critical_count = len(
                filtered[
                    filtered["risk_level"].isin(
                        ["HIGH", "CRITICAL"]
                    )
                ]
            )

            render_metric_card("High / Critical", high_critical_count, "⚠️", "high")

        with col3:

            if len(filtered) > 0:

                high_critical_pct = (
                    high_critical_count /
                    len(filtered)
                ) * 100

            else:
                high_critical_pct = 0

            render_metric_card(
                "High / Critical %",
                f"{high_critical_pct:.1f}%",
                "📈",
                "medium",
            )

        st.markdown("<div style='margin:1.25rem 0;'></div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # INCIDENT TABLE
        # -------------------------------------------------

        render_section_title("Incident Records")

        display_columns = [
            "incident_id",
            "date",
            "time",
            "activity_type",
            "location_type",
            "description",
            "severity",
            "risk_level",
            "weather",
            "shift",
            "ppe_compliance_pct",
            "previous_incidents_30d",
            "crew_size"
        ]

        available_columns = [
            column
            for column in display_columns
            if column in filtered.columns
        ]

        st.dataframe(
            filtered[available_columns].sort_values(
                by="date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # RISK DISTRIBUTION OF FILTERED DATA
        # -------------------------------------------------

        if len(filtered) > 0:

            render_section_title("Filtered Risk Distribution")

            risk_counts = (
                filtered["risk_level"]
                .value_counts()
                .reindex(
                    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    fill_value=0
                )
                .reset_index()
            )

            risk_counts.columns = [
                "Risk Level",
                "Incidents"
            ]

            fig = px.bar(
                risk_counts,
                x="Risk Level",
                y="Incidents",
                text="Incidents",
                title="Incidents by Risk Level",
                color="Risk Level",
                color_discrete_map=RISK_COLORS,
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(showlegend=False)
            style_plotly_fig(fig)
            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# RISK PATTERNS PLACEHOLDER
# =========================================================

elif page == "Risk Patterns":

    render_page_header(
        "📊 Risk Patterns",
        "Identify recurring patterns in historical construction safety incidents "
        "to understand which activities and conditions are associated with higher risk.",
    )

    # Load incident data
    incidents_path = os.path.join(BASE_DIR, "data", "incidents.csv")
    incidents_df = pd.read_csv(incidents_path)

    # ---------------------------------------------------------
    # HIGH / CRITICAL INCIDENTS
    # ---------------------------------------------------------

    high_risk_df = incidents_df[
        incidents_df["risk_level"].isin(["HIGH", "CRITICAL"])
    ].copy()

    col1, col2 = st.columns(2)

    with col1:

        render_section_title("High-Risk Activity Patterns")

        activity_counts = (
            high_risk_df["activity_type"]
            .value_counts()
            .sort_values(ascending=True)
        )

        activity_chart_df = activity_counts.reset_index()
        activity_chart_df.columns = ["Activity", "Incidents"]

        fig_activity = px.bar(
            activity_chart_df,
            x="Incidents",
            y="Activity",
            orientation="h",
            title="HIGH / CRITICAL by Activity",
            color="Incidents",
            color_continuous_scale=["#92400e", "#ef4444"],
        )
        fig_activity.update_layout(showlegend=False, coloraxis_showscale=False)
        style_plotly_fig(fig_activity)
        st.plotly_chart(fig_activity, use_container_width=True)

        st.caption(
            "Activities with more HIGH or CRITICAL incidents may require "
            "additional preventive controls."
        )

    with col2:

        render_section_title("High-Risk Locations")

        location_counts = (
            high_risk_df["location_type"]
            .value_counts()
            .sort_values(ascending=True)
        )

        location_chart_df = location_counts.reset_index()
        location_chart_df.columns = ["Location", "Incidents"]

        fig_location = px.bar(
            location_chart_df,
            x="Incidents",
            y="Location",
            orientation="h",
            title="HIGH / CRITICAL by Location",
            color="Incidents",
            color_continuous_scale=["#1e3a5f", "#f59e0b"],
        )
        fig_location.update_layout(showlegend=False, coloraxis_showscale=False)
        style_plotly_fig(fig_location)
        st.plotly_chart(fig_location, use_container_width=True)

    # ---------------------------------------------------------
    # WEATHER PATTERNS
    # ---------------------------------------------------------

    render_section_title("Risk by Weather Condition")

    weather_table = pd.crosstab(
        incidents_df["weather"],
        incidents_df["risk_level"]
    )

    st.dataframe(
        weather_table,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # PPE COMPLIANCE
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        render_section_title("PPE Compliance and Risk")

        ppe_summary = (
            incidents_df
            .groupby("risk_level")["ppe_compliance_pct"]
            .mean()
            .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        )

        ppe_chart_df = ppe_summary.reset_index()
        ppe_chart_df.columns = ["Risk Level", "Avg PPE %"]

        fig_ppe = px.bar(
            ppe_chart_df,
            x="Risk Level",
            y="Avg PPE %",
            title="Average PPE Compliance by Risk Level",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
        )
        fig_ppe.update_layout(showlegend=False)
        style_plotly_fig(fig_ppe)
        st.plotly_chart(fig_ppe, use_container_width=True)

        st.caption(
            "Average PPE compliance percentage observed "
            "for each historical risk level."
        )

    with col2:

        render_section_title("🔎 Identified Safety Patterns")

        top_activity = (
            high_risk_df["activity_type"].value_counts().idxmax()
            if not high_risk_df.empty
            else "N/A"
        )

        top_location = (
            high_risk_df["location_type"].value_counts().idxmax()
            if not high_risk_df.empty
            else "N/A"
        )

        adverse_high_risk = (
            high_risk_df["weather"].eq("Adverse").sum()
            if not high_risk_df.empty
            else 0
        )

        st.markdown(
            f"""
            <div class="insight-box">
                • <strong>{top_activity}</strong> is the activity with the most HIGH/CRITICAL incidents.<br><br>
                • <strong>{top_location}</strong> is the location most frequently associated with HIGH/CRITICAL incidents.<br><br>
                • <strong>{adverse_high_risk}</strong> HIGH/CRITICAL incidents occurred under adverse weather conditions.<br><br>
                These patterns can be used to prioritize preventive safety measures.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PREVENTIVE ACTIONS PLACEHOLDER
# =========================================================

elif page == "Preventive Actions":

    render_page_header(
        "🛡️ Preventive Actions",
        "Review recommended preventive safety controls based on the "
        "construction activity and predicted risk level.",
    )

    # ---------------------------------------------------------
    # INPUTS
    # ---------------------------------------------------------

    render_section_title("Activity & Risk Selection")

    col1, col2 = st.columns(2)

    with col1:
        selected_activity = st.selectbox(
            "Construction Activity",
            [
                "Working at Height",
                "Electrical Work",
                "Excavation",
                "Scaffolding",
                "Welding",
                "Lifting",
                "Material Handling",
                "Vehicle Movement",
                "Confined Space",
                "Housekeeping"
            ]
        )

    with col2:
        selected_risk = st.selectbox(
            "Risk Level",
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            index=2
        )

    st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # PREVENTIVE ACTION DATABASE
    # ---------------------------------------------------------

    preventive_actions = {

        "Working at Height": [
            "Verify fall protection controls before starting work.",
            "Check edge protection and guardrails.",
            "Inspect ladders, scaffolds, and access equipment.",
            "Ensure workers use appropriate fall-arrest systems.",
            "Conduct a pre-task safety briefing."
        ],

        "Electrical Work": [
            "Inspect electrical cables and connections before work.",
            "Verify isolation and lockout/tagout procedures.",
            "Ensure electrical panels are properly secured.",
            "Use appropriate electrical PPE.",
            "Keep unauthorized personnel away from electrical work areas."
        ],

        "Excavation": [
            "Inspect excavation edges before starting work.",
            "Provide suitable shoring or protective systems where required.",
            "Keep materials and equipment away from excavation edges.",
            "Check for underground utilities before excavation.",
            "Conduct regular excavation inspections."
        ],

        "Scaffolding": [
            "Inspect scaffolding before use.",
            "Verify guardrails and toe boards are installed.",
            "Ensure the scaffold is properly supported and stable.",
            "Prevent unauthorized modifications.",
            "Use safe access and egress routes."
        ],

        "Welding": [
            "Inspect welding equipment before use.",
            "Remove combustible materials from the work area.",
            "Use appropriate welding PPE.",
            "Provide adequate ventilation.",
            "Maintain suitable fire prevention controls."
        ],

        "Lifting": [
            "Inspect lifting equipment before operation.",
            "Verify load capacity and lifting plan.",
            "Keep personnel away from suspended loads.",
            "Use appropriate rigging and lifting accessories.",
            "Conduct a pre-lift safety briefing."
        ],

        "Material Handling": [
            "Inspect materials and handling equipment.",
            "Use correct manual handling techniques.",
            "Avoid overloading workers or equipment.",
            "Keep pathways clear.",
            "Use mechanical assistance for heavy loads where appropriate."
        ],

        "Vehicle Movement": [
            "Separate pedestrians and moving vehicles.",
            "Check vehicle condition before operation.",
            "Use designated traffic routes.",
            "Ensure adequate visibility and lighting.",
            "Use trained and authorized vehicle operators."
        ],

        "Confined Space": [
            "Verify confined-space entry requirements.",
            "Test the atmosphere before entry.",
            "Provide appropriate ventilation.",
            "Maintain communication with workers inside.",
            "Prepare an emergency rescue plan."
        ],

        "Housekeeping": [
            "Keep work areas clean and organized.",
            "Remove trip and slip hazards.",
            "Store materials safely.",
            "Maintain clear emergency access routes.",
            "Conduct regular housekeeping inspections."
        ]
    }

    # ---------------------------------------------------------
    # RISK-BASED ADDITIONAL ACTIONS
    # ---------------------------------------------------------

    risk_actions = {

        "LOW": [
            "Follow standard site safety procedures.",
            "Perform routine pre-task checks."
        ],

        "MEDIUM": [
            "Review the task-specific risk assessment.",
            "Conduct a pre-task safety briefing.",
            "Increase supervision during the activity."
        ],

        "HIGH": [
            "Review and approve the task-specific risk assessment.",
            "Increase safety supervision.",
            "Verify all critical safety controls before work.",
            "Conduct a documented pre-task briefing."
        ],

        "CRITICAL": [
            "Do not begin work until critical safety controls are verified.",
            "Require supervisor or safety-officer review.",
            "Conduct a detailed task-specific risk assessment.",
            "Confirm emergency and rescue arrangements.",
            "Document control verification before starting work."
        ]
    }

    toolbox_topics = {

        "Working at Height": [
            "Fall protection",
            "Edge protection",
            "Safe scaffold and ladder access",
            "Pre-use inspection of access equipment",
            "Emergency response for elevated work"
        ],

        "Electrical Work": [
            "Electrical isolation",
            "Lockout/tagout",
            "Electrical PPE",
            "Cable and panel inspection",
            "Emergency response to electrical incidents"
        ],

        "Excavation": [
            "Excavation safety",
            "Underground utility hazards",
            "Edge protection",
            "Safe access and egress",
            "Emergency response"
        ],

        "Scaffolding": [
            "Scaffold inspection",
            "Guardrails and toe boards",
            "Safe access",
            "Load limits",
            "Fall prevention"
        ],

        "Welding": [
            "Fire prevention",
            "Welding PPE",
            "Ventilation",
            "Hot-work controls",
            "Emergency response"
        ],

        "Lifting": [
            "Safe lifting practices",
            "Rigging inspection",
            "Suspended-load hazards",
            "Communication during lifting",
            "Emergency procedures"
        ],

        "Material Handling": [
            "Manual handling",
            "Safe lifting techniques",
            "Material storage",
            "Housekeeping",
            "Use of mechanical assistance"
        ],

        "Vehicle Movement": [
            "Pedestrian-vehicle separation",
            "Traffic routes",
            "Vehicle inspection",
            "Blind spots",
            "Safe vehicle operation"
        ],

        "Confined Space": [
            "Atmospheric testing",
            "Ventilation",
            "Entry procedures",
            "Communication",
            "Emergency rescue"
        ],

        "Housekeeping": [
            "Slip and trip prevention",
            "Material storage",
            "Clear access routes",
            "Waste management",
            "Routine inspections"
        ]
    }

    # ---------------------------------------------------------
    # DISPLAY ACTIVITY ACTIONS
    # ---------------------------------------------------------

    render_section_title("Recommended Preventive Controls")

    actions = preventive_actions.get(
        selected_activity,
        []
    )

    for action in actions:
        st.checkbox(action, value=False)

    # ---------------------------------------------------------
    # RISK-SPECIFIC ACTIONS
    # ---------------------------------------------------------

    render_section_title(
        f"Additional Actions for {selected_risk} Risk"
    )

    for action in risk_actions[selected_risk]:
        st.checkbox(action, value=False)

    # ---------------------------------------------------------
    # TOOLBOX TALK
    # ---------------------------------------------------------

    render_section_title("🗣️ Toolbox Talk Topics")

    for topic in toolbox_topics[selected_activity]:
        st.markdown(
            f'<div class="toolbox-item">• {topic}</div>',
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # RISK WARNING
    # ---------------------------------------------------------

    st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

    if selected_risk == "CRITICAL":
        render_risk_banner(
            selected_risk,
            "⚠️ CRITICAL RISK: Work should not begin until the required "
            "safety controls have been verified by qualified personnel.",
        )

    elif selected_risk == "HIGH":
        render_risk_banner(
            selected_risk,
            "⚠️ HIGH RISK: Verify task-specific controls and increase "
            "safety supervision before starting work.",
        )

    elif selected_risk == "MEDIUM":
        render_risk_banner(
            selected_risk,
            "ℹ️ MEDIUM RISK: Review the task risk assessment and "
            "complete the recommended controls.",
        )

    else:
        render_risk_banner(
            selected_risk,
            "✅ LOW RISK: Continue following standard site safety procedures.",
        )


# =========================================================
# WEEKLY SAFETY BRIEF PLACEHOLDER
# =========================================================

elif page == "Weekly Safety Brief":

    render_page_header(
        "🗓️ Weekly Safety Brief",
        "A quick safety briefing based on recent historical incident patterns. "
        "Use this information to focus attention on recurring hazards.",
    )

    # ---------------------------------------------------------
    # LOAD INCIDENT DATA
    # ---------------------------------------------------------

    incidents_path = os.path.join(BASE_DIR, "data", "incidents.csv")
    incidents_df = pd.read_csv(incidents_path)

    incidents_df["date"] = pd.to_datetime(incidents_df["date"])

    # Use the latest 7 days available in the dataset
    latest_date = incidents_df["date"].max()
    start_date = latest_date - pd.Timedelta(days=6)

    weekly_df = incidents_df[
        (incidents_df["date"] >= start_date)
        & (incidents_df["date"] <= latest_date)
    ].copy()

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    render_section_title("Weekly Safety Summary")

    total_incidents = len(weekly_df)

    high_critical = weekly_df[
        weekly_df["risk_level"].isin(["HIGH", "CRITICAL"])
    ]

    high_critical_count = len(high_critical)

    critical_count = len(
        weekly_df[weekly_df["risk_level"] == "CRITICAL"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_metric_card("Incidents", total_incidents, "📋", "default")

    with col2:
        render_metric_card("High / Critical", high_critical_count, "⚠️", "high")

    with col3:
        render_metric_card("Critical", critical_count, "🔴", "critical")

    st.caption(
        f"Period: {start_date.strftime('%d %b %Y')} "
        f"to {latest_date.strftime('%d %b %Y')}"
    )

    st.markdown("<div style='margin:1.25rem 0;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # RISK DISTRIBUTION
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        render_section_title("Risk Level Distribution")

        risk_counts = (
            weekly_df["risk_level"]
            .value_counts()
            .reindex(
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                fill_value=0
            )
        )

        risk_chart_df = risk_counts.reset_index()
        risk_chart_df.columns = ["Risk Level", "Incidents"]

        fig_weekly_risk = px.bar(
            risk_chart_df,
            x="Risk Level",
            y="Incidents",
            title="Weekly Risk Distribution",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
        )
        fig_weekly_risk.update_layout(showlegend=False)
        style_plotly_fig(fig_weekly_risk)
        st.plotly_chart(fig_weekly_risk, use_container_width=True)

    with col2:

        render_section_title("Weather Conditions")

        if not weekly_df.empty:

            weather_counts = weekly_df["weather"].value_counts().reset_index()
            weather_counts.columns = ["Weather", "Incidents"]

            fig_weather = px.bar(
                weather_counts,
                x="Weather",
                y="Incidents",
                title="Incidents by Weather",
                color="Incidents",
                color_continuous_scale=["#1e3a5f", "#3b82f6"],
            )
            fig_weather.update_layout(showlegend=False, coloraxis_showscale=False)
            style_plotly_fig(fig_weather)
            st.plotly_chart(fig_weather, use_container_width=True)

    # ---------------------------------------------------------
    # TOP ACTIVITIES
    # ---------------------------------------------------------

    render_section_title("Activities Requiring Attention")

    if not high_critical.empty:

        top_activities = (
            high_critical["activity_type"]
            .value_counts()
            .head(5)
        )

        st.dataframe(
            top_activities.rename("HIGH/CRITICAL incidents"),
            use_container_width=True
        )

    else:

        render_risk_banner(
            "LOW",
            "No HIGH or CRITICAL incidents were found in this period.",
        )

    # ---------------------------------------------------------
    # COMMON HAZARDS
    # ---------------------------------------------------------

    render_section_title("Common Incident Descriptions")

    if not high_critical.empty:

        descriptions = (
            high_critical["description"]
            .value_counts()
            .head(5)
        )

        for description, count in descriptions.items():

            st.markdown(
                f'<div class="list-item"><strong>{description}</strong> — {count} incident(s)</div>',
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # PPE
    # ---------------------------------------------------------

    render_section_title("Average PPE Compliance")

    if not weekly_df.empty:

        avg_ppe = weekly_df["ppe_compliance_pct"].mean()

        render_metric_card(
            "Average PPE Compliance",
            f"{avg_ppe:.1f}%",
            "🦺",
            "neutral",
        )

    # ---------------------------------------------------------
    # SAFETY BRIEF
    # ---------------------------------------------------------

    render_section_title("📢 Safety Brief")

    if high_critical_count > 0:

        top_activity = (
            high_critical["activity_type"]
            .value_counts()
            .idxmax()
        )

        render_risk_banner(
            "HIGH",
            f"During this period, {high_critical_count} HIGH/CRITICAL "
            f"incident(s) were recorded. "
            f"The activity requiring the most attention was "
            f"{top_activity}.",
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title-sm">Before starting work:</div>
                <div class="list-item">Review the task-specific risk assessment.</div>
                <div class="list-item">Verify required PPE and safety controls.</div>
                <div class="list-item">Conduct a pre-task toolbox talk.</div>
                <div class="list-item">Check work-area conditions.</div>
                <div class="list-item">Ensure workers understand emergency procedures.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        render_risk_banner(
            "LOW",
            "No HIGH or CRITICAL incidents were recorded during "
            "the selected weekly period.",
        )

    st.markdown(
        """
        <div class="insight-box">
            This weekly brief is based on historical incident data and
            is intended to support safety discussions. It does not
            replace professional site risk assessments or safety procedures.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SAFETY DISCLAIMER
# =========================================================

render_footer()