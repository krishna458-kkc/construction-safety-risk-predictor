"""Reusable UI components implementing the Mercury design language for Construction Safety Intelligence."""

from typing import Any, List, Optional
import streamlit as st
from src.ui.theme import (
    COLOR_ACCENT_COBALT,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    SAFETY_COLORS,
)

PAGES = [
    "Overview",
    "Risk Predictor",
    "Analytics",
    "Records",
    "Reports",
    "Safety",
    "AI Safety",
    "Projects",
]

NAV_GLYPHS = {
    "Overview": "⌂",
    "Risk Predictor": "◈",
    "Analytics": "▥",
    "Records": "▤",
    "Reports": "▧",
    "Safety": "🛡",
    "AI Safety": "✦",
    "Projects": "▣",
}


def render_global_header(current_project: Optional[str] = None) -> None:
    """Render the top application header with technical branding and active project/status chips."""
    project_label = current_project if current_project else "Site: All Projects"
    st.markdown(
        f"""
        <div class="cs-header">
            <div class="cs-brand-group">
                <div class="cs-brand-icon">🦺</div>
                <div>
                    <h1 class="cs-brand-title">Construction Safety Intelligence</h1>
                    <p class="cs-brand-subtitle">Predictive Risk Engineering & Operations</p>
                </div>
            </div>
            <div class="cs-header-meta">
                <div class="cs-chip active">
                    <span>🏢</span>
                    <span>{project_label}</span>
                </div>
                <div class="cs-chip">
                    <span class="cs-status-dot"></span>
                    <span>System Active</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_navigation() -> str:
    """Render the 8-item horizontal navigation pill bar."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = PAGES[0]

    with st.container():
        st.markdown(
            '<div class="nav-marker"></div><div class="cs-nav-wrapper"></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(PAGES))
        for idx, page_name in enumerate(PAGES):
            with cols[idx]:
                is_active = st.session_state.current_page == page_name
                glyph = NAV_GLYPHS.get(page_name, "•")
                if st.button(
                    f"{glyph} {page_name}",
                    key=f"cs_nav_btn_{page_name}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_page = page_name
                    st.rerun()

    return st.session_state.current_page


def render_page_hero(
    title: str,
    subtitle: str,
    tagline: str = "CONSTRUCTION SAFETY INTELLIGENCE",
) -> None:
    """Render architectural page header with spacious typography."""
    st.markdown(
        f"""
        <div class="cs-page-hero">
            <div class="cs-page-tagline">{tagline}</div>
            <h2 class="cs-page-title">{title}</h2>
            <p class="cs-page-desc">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, subtitle: Optional[str] = None) -> None:
    """Render a clean graphite section header with optional subtitle."""
    subtitle_html = (
        f'<span class="cs-section-subtitle">{subtitle}</span>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="cs-section-header">
            <h3 class="cs-section-title">{title}</h3>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: Any,
    subtitle: Optional[str] = None,
    variant: str = "default",
) -> None:
    """Render a flat graphite metric card with optional semantic top border."""
    border_class = f"border-{variant.lower()}" if variant != "default" else ""
    subtitle_html = (
        f'<div class="cs-metric-subtitle">{subtitle}</div>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="cs-metric-card {border_class}">
            <div>
                <div class="cs-metric-label">{label}</div>
                <div class="cs-metric-value">{value}</div>
            </div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str,
    message: str,
    icon: str = "◌",
) -> None:
    """Render a reusable, pristine empty state for non-populated data sources."""
    st.markdown(
        f"""
        <div class="cs-empty-state">
            <div class="cs-empty-icon">{icon}</div>
            <div class="cs-empty-title">{title}</div>
            <div class="cs-empty-desc">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_level: str) -> str:
    """Return HTML string for semantic risk badge."""
    level = risk_level.upper()
    variant = level.lower()
    return f'<span class="cs-badge cs-badge-{variant}">● {level} RISK</span>'


def render_footer() -> None:
    """Render professional engineering safety disclaimer footer."""
    st.markdown(
        """
        <div class="cs-footer">
            <p class="cs-footer-text">
                <strong>Construction Safety Intelligence Platform</strong> — Predictive risk analytics
                designed to enhance jobsite hazard awareness. This system operates as a decision-support
                tool and does not replace site safety managers, formal risk assessments (JHA/JSA),
                OSHA/regulatory compliance, or certified engineered safety plans.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
