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
    "Projects",
]

NAV_GLYPHS = {
    "Overview": "▦",
    "Risk Predictor": "◈",
    "Analytics": "▥",
    "Records": "▤",
    "Reports": "▧",
    "Safety": "🛡",
    "Projects": "▣",
}


def render_global_header(current_project: Optional[str] = None) -> None:
    """Render the top application header with technical branding and active project/status chips."""
    project_label = current_project if current_project else "Site: All Projects"
    html = (
        f'<div class="cs-header">'
        f'<div class="cs-brand-group">'
        f'<div class="cs-brand-icon">🦺</div>'
        f'<div>'
        f'<h1 class="cs-brand-title">Construction Safety Intelligence</h1>'
        f'<p class="cs-brand-subtitle">Predictive Risk Engineering & Operations</p>'
        f'</div>'
        f'</div>'
        f'<div class="cs-header-meta">'
        f'<div class="cs-chip active">'
        f'<span>🏢</span>'
        f'<span>{project_label}</span>'
        f'</div>'
        f'<div class="cs-chip">'
        f'<span class="cs-status-dot"></span>'
        f'<span>System Active</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


NAV_WEIGHTS = [1.0, 1.35, 1.05, 0.95, 0.95, 0.9, 0.98]


def render_top_navigation() -> str:
    """Render the floating Command Center navigation bar."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = PAGES[0]

    st.markdown('<div class="cs-command-nav-anchor"></div>', unsafe_allow_html=True)
    with st.container():
        cols = st.columns(NAV_WEIGHTS)
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
    html = (
        f'<div class="cs-page-hero">'
        f'<div class="cs-page-tagline">{tagline}</div>'
        f'<h2 class="cs-page-title">{title}</h2>'
        f'<p class="cs-page-desc">{subtitle}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_section_heading(title: str, subtitle: Optional[str] = None) -> None:
    """Render a clean graphite section header with optional subtitle."""
    subtitle_html = (
        f'<span class="cs-section-subtitle">{subtitle}</span>' if subtitle else ""
    )
    html = (
        f'<div class="cs-section-header">'
        f'<h3 class="cs-section-title">{title}</h3>'
        f'{subtitle_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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
    html = (
        f'<div class="cs-metric-card {border_class}">'
        f'<div>'
        f'<div class="cs-metric-label">{label}</div>'
        f'<div class="cs-metric-value">{value}</div>'
        f'</div>'
        f'{subtitle_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_project_card(
    project_name: str,
    location: str,
    start_date: str,
    status: str,
    assessment_count: int = 0,
    is_active: bool = False,
) -> None:
    """Render a clean Mercury project portfolio card with status badge and recorded assessment metrics."""
    badge_style = "border-color: #5266eb; background: rgba(82,102,235,0.08);" if is_active else ""
    active_badge_html = '<span class="cs-badge cs-badge-low" style="margin-left: 0.6rem;">CURRENT ACTIVE SITE</span>' if is_active else ''
    
    html = (
        f'<div class="cs-card" style="{badge_style}">'
        f'<div class="cs-card-header">'
        f'<div>'
        f'<strong style="color: #ededf3; font-size: 1.05rem;">{project_name}</strong>'
        f'{active_badge_html}'
        f'</div>'
        f'<span class="cs-badge cs-badge-medium">{status}</span>'
        f'</div>'
        f'<div style="display: flex; gap: 2rem; color: #c3c3cc; font-size: 0.85rem; margin-top: 0.5rem;">'
        f'<div>📍 Location: <strong style="color: #ededf3;">{location}</strong></div>'
        f'<div>🗓 Started: <strong style="color: #ededf3;">{start_date}</strong></div>'
        f'<div>◈ Recorded Assessments: <strong style="color: #ededf3;">{assessment_count}</strong></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state(
    title: str,
    message: str,
    icon: str = "◌",
) -> None:
    """Render a reusable, pristine empty state for non-populated data sources."""
    html = (
        f'<div class="cs-empty-state">'
        f'<div class="cs-empty-icon">{icon}</div>'
        f'<div class="cs-empty-title">{title}</div>'
        f'<div class="cs-empty-desc">{message}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_risk_badge(risk_level: str) -> str:
    """Return HTML string for semantic risk badge."""
    level = risk_level.upper()
    variant = level.lower()
    return f'<span class="cs-badge cs-badge-{variant}">● {level} RISK</span>'


def render_footer() -> None:
    """Render professional engineering safety disclaimer footer."""
    html = (
        '<div class="cs-footer">'
        '<p class="cs-footer-text">'
        '<strong>Construction Safety Intelligence Platform</strong> — Predictive risk analytics '
        'designed to enhance jobsite hazard awareness. This system operates as a decision-support '
        'tool and does not replace site safety managers, formal risk assessments (JHA/JSA), '
        'OSHA/regulatory compliance, or certified engineered safety plans.'
        '</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
