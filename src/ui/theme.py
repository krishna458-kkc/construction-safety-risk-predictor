"""Mercury Design System tokens and CSS injector for Construction Safety Intelligence."""

# ==============================================================================
# MERCURY DESIGN SYSTEM CONSTANTS
# ==============================================================================

COLOR_CANVAS = "#171721"               # Near-black canvas
COLOR_SURFACE = "#1e1e2a"              # Graphite card panel
COLOR_SURFACE_INTERACTIVE = "#272735"  # Secondary interactive surface / hover
COLOR_SURFACE_HOVER = "#2e2e3f"        # Elevated surface on hover
COLOR_TEXT_PRIMARY = "#ededf3"         # High-contrast ivory text
COLOR_TEXT_SECONDARY = "#c3c3cc"       # Technical secondary text
COLOR_TEXT_MUTED = "#8e8e9c"           # Muted text / captions
COLOR_BORDER_STRUCTURAL = "#70707d"    # Structural border
COLOR_BORDER_LIGHT = "rgba(112, 112, 125, 0.28)" # Subtle outline
COLOR_BORDER_SUBTLE = "rgba(226, 227, 237, 0.08)" # Very subtle divider
COLOR_ACCENT_COBALT = "#5266eb"        # Primary interface accent
COLOR_ACCENT_HOVER = "#4353cc"         # Cobalt hover
COLOR_ACCENT_SOFT = "rgba(82, 102, 235, 0.14)" # Cobalt tint
COLOR_WHITE = "#ffffff"

# Semantic Safety Colors (Reserved strictly for risk/safety status)
SAFETY_COLORS = {
    "LOW": "#22c55e",       # Green
    "MEDIUM": "#3b82f6",    # Blue
    "HIGH": "#f59e0b",      # Orange
    "CRITICAL": "#ef4444",  # Red
}

SAFETY_COLORS_SOFT = {
    "LOW": "rgba(34, 197, 94, 0.14)",
    "MEDIUM": "rgba(59, 130, 246, 0.14)",
    "HIGH": "rgba(245, 158, 11, 0.14)",
    "CRITICAL": "rgba(239, 68, 68, 0.16)",
}

SAFETY_COLORS_BORDER = {
    "LOW": "rgba(34, 197, 94, 0.4)",
    "MEDIUM": "rgba(59, 130, 246, 0.4)",
    "HIGH": "rgba(245, 158, 11, 0.4)",
    "CRITICAL": "rgba(239, 68, 68, 0.45)",
}


def get_mercury_css() -> str:
    """Return the complete scoped CSS for the Mercury-adapted safety platform."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {{
        --canvas: {COLOR_CANVAS};
        --surface: {COLOR_SURFACE};
        --surface-interactive: {COLOR_SURFACE_INTERACTIVE};
        --surface-hover: {COLOR_SURFACE_HOVER};
        --text-primary: {COLOR_TEXT_PRIMARY};
        --text-secondary: {COLOR_TEXT_SECONDARY};
        --text-muted: {COLOR_TEXT_MUTED};
        --border-structural: {COLOR_BORDER_STRUCTURAL};
        --border-light: {COLOR_BORDER_LIGHT};
        --border-subtle: {COLOR_BORDER_SUBTLE};
        --accent: {COLOR_ACCENT_COBALT};
        --accent-hover: {COLOR_ACCENT_HOVER};
        --accent-soft: {COLOR_ACCENT_SOFT};
        --safety-low: {SAFETY_COLORS["LOW"]};
        --safety-medium: {SAFETY_COLORS["MEDIUM"]};
        --safety-high: {SAFETY_COLORS["HIGH"]};
        --safety-critical: {SAFETY_COLORS["CRITICAL"]};
    }}

    /* Global reset and font styling */
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: var(--canvas) !important;
        color: var(--text-primary) !important;
        -webkit-font-smoothing: antialiased;
    }}

    /* Hide standard Streamlit header chrome and sidebar */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    /* Layout constraints */
    .block-container {{
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding-top: 1.25rem !important;
        padding-bottom: 4rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }}

    /* Architectural Header */
    .cs-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0 1.25rem 0;
        border-bottom: 1px solid var(--border-light);
        margin-bottom: 1.5rem;
    }}

    .cs-brand-group {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }}

    .cs-brand-icon {{
        width: 32px;
        height: 32px;
        background: var(--surface-interactive);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        color: var(--accent);
    }}

    .cs-brand-title {{
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-primary);
        margin: 0;
    }}

    .cs-brand-subtitle {{
        font-size: 0.75rem;
        color: var(--text-muted);
        letter-spacing: 0.02em;
        margin: 0;
    }}

    .cs-header-meta {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}

    .cs-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        color: var(--text-secondary);
    }}

    .cs-chip.active {{
        border-color: rgba(82, 102, 235, 0.4);
        background: var(--accent-soft);
        color: #b4c2fb;
    }}

    .cs-status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--safety-low);
        display: inline-block;
    }}

    /* Top Navigation Pill Bar */
    .cs-nav-wrapper {{
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 36px;
        padding: 0.35rem 0.45rem;
        margin-bottom: 2rem;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-marker)
        div[data-testid="stHorizontalBlock"] {{
        gap: 0.35rem !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-marker)
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {{
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 0.65rem !important;
        border-radius: 24px !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
        white-space: nowrap !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-marker)
        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {{
        background: var(--surface-interactive) !important;
        color: var(--text-primary) !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-marker)
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {{
        background: var(--accent) !important;
        border: 1px solid var(--accent) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 0.65rem !important;
        border-radius: 24px !important;
        box-shadow: none !important;
        white-space: nowrap !important;
    }}

    /* Page Hero / Section Titles */
    .cs-page-hero {{
        margin-bottom: 2rem;
    }}

    .cs-page-tagline {{
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.4rem;
    }}

    .cs-page-title {{
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }}

    .cs-page-desc {{
        font-size: 0.95rem;
        color: var(--text-secondary);
        max-width: 760px;
        line-height: 1.6;
        margin: 0;
    }}

    .cs-section-header {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin: 1.75rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-light);
    }}

    .cs-section-title {{
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.01em;
        margin: 0;
    }}

    .cs-section-subtitle {{
        font-size: 0.82rem;
        color: var(--text-muted);
        margin: 0;
    }}

    /* Flat Graphite Cards */
    .cs-card {{
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: none;
    }}

    .cs-card-flat {{
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }}

    .cs-card-header {{
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    /* Metric Panels */
    .cs-metric-card {{
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .cs-metric-card.border-low {{ border-top: 2px solid var(--safety-low); }}
    .cs-metric-card.border-medium {{ border-top: 2px solid var(--safety-medium); }}
    .cs-metric-card.border-high {{ border-top: 2px solid var(--safety-high); }}
    .cs-metric-card.border-critical {{ border-top: 2px solid var(--safety-critical); }}
    .cs-metric-card.border-accent {{ border-top: 2px solid var(--accent); }}

    .cs-metric-label {{
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }}

    .cs-metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }}

    .cs-metric-subtitle {{
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 0.4rem;
    }}

    /* Semantic Risk Badges */
    .cs-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}

    .cs-badge-low {{
        background: {SAFETY_COLORS_SOFT["LOW"]};
        color: {SAFETY_COLORS["LOW"]};
        border: 1px solid {SAFETY_COLORS_BORDER["LOW"]};
    }}

    .cs-badge-medium {{
        background: {SAFETY_COLORS_SOFT["MEDIUM"]};
        color: #60a5fa;
        border: 1px solid {SAFETY_COLORS_BORDER["MEDIUM"]};
    }}

    .cs-badge-high {{
        background: {SAFETY_COLORS_SOFT["HIGH"]};
        color: {SAFETY_COLORS["HIGH"]};
        border: 1px solid {SAFETY_COLORS_BORDER["HIGH"]};
    }}

    .cs-badge-critical {{
        background: {SAFETY_COLORS_SOFT["CRITICAL"]};
        color: #f87171;
        border: 1px solid {SAFETY_COLORS_BORDER["CRITICAL"]};
    }}

    /* Risk Assessment Result Panel */
    .cs-risk-assessment {{
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.5rem 0;
    }}

    .cs-risk-assessment.critical {{ border-left: 4px solid var(--safety-critical); }}
    .cs-risk-assessment.high {{ border-left: 4px solid var(--safety-high); }}
    .cs-risk-assessment.medium {{ border-left: 4px solid var(--safety-medium); }}
    .cs-risk-assessment.low {{ border-left: 4px solid var(--safety-low); }}

    .cs-risk-banner {{
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        font-weight: 500;
        line-height: 1.5;
        border: 1px solid;
    }}

    .cs-risk-banner.critical {{
        background: {SAFETY_COLORS_SOFT["CRITICAL"]};
        border-color: {SAFETY_COLORS_BORDER["CRITICAL"]};
        color: #fca5a5;
    }}

    .cs-risk-banner.high {{
        background: {SAFETY_COLORS_SOFT["HIGH"]};
        border-color: {SAFETY_COLORS_BORDER["HIGH"]};
        color: #fcd34d;
    }}

    .cs-risk-banner.medium {{
        background: {SAFETY_COLORS_SOFT["MEDIUM"]};
        border-color: {SAFETY_COLORS_BORDER["MEDIUM"]};
        color: #93c5fd;
    }}

    .cs-risk-banner.low {{
        background: {SAFETY_COLORS_SOFT["LOW"]};
        border-color: {SAFETY_COLORS_BORDER["LOW"]};
        color: #86efac;
    }}

    /* Contributing Factor Items */
    .cs-factor-item {{
        background: var(--surface-interactive);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.88rem;
    }}

    .cs-factor-name {{
        color: var(--text-primary);
        font-weight: 500;
    }}

    .cs-factor-meta {{
        font-size: 0.78rem;
        color: var(--text-muted);
        font-family: monospace;
    }}

    /* Recommendation & Checklist Items */
    .cs-action-item {{
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        font-size: 0.9rem;
        color: var(--text-primary);
        line-height: 1.45;
    }}

    .cs-action-icon {{
        color: var(--safety-low);
        font-size: 0.95rem;
        margin-top: 0.1rem;
    }}

    /* Empty States */
    .cs-empty-state {{
        background: var(--surface);
        border: 1px dashed var(--border-structural);
        border-radius: 12px;
        padding: 3.5rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }}

    .cs-empty-icon {{
        font-size: 2.2rem;
        color: var(--text-muted);
        margin-bottom: 0.85rem;
    }}

    .cs-empty-title {{
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }}

    .cs-empty-desc {{
        font-size: 0.9rem;
        color: var(--text-secondary);
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.6;
    }}

    /* Form Controls & Inputs Overrides */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stTextArea"] label {{
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.35rem !important;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stTextArea"] textarea {{
        background-color: var(--surface) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-size: 0.9rem !important;
    }}

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    div[data-testid="stTextArea"] textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }}

    /* Global Buttons: Pill Controls */
    .stButton > button[kind="primary"] {{
        background-color: var(--accent) !important;
        color: #ffffff !important;
        border-radius: 32px !important;
        border: 1px solid var(--accent) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.6rem !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        background-color: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }}

    .stButton > button[kind="secondary"] {{
        background-color: var(--surface-interactive) !important;
        color: var(--text-primary) !important;
        border-radius: 32px !important;
        border: 1px solid var(--border-light) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.6rem !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }}

    .stButton > button[kind="secondary"]:hover {{
        background-color: var(--surface-hover) !important;
        border-color: var(--border-structural) !important;
    }}

    /* Streamlit Segmented Control / Radio Buttons */
    div[data-testid="stRadio"] > div {{
        background-color: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 28px;
        padding: 0.25rem;
        gap: 0.35rem;
    }}

    div[data-testid="stRadio"] label {{
        border-radius: 20px !important;
        padding: 0.4rem 0.9rem !important;
        margin: 0 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        transition: all 0.15s ease;
    }}

    /* Dataframe Styling */
    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border-light) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        background: var(--surface) !important;
    }}

    /* Checkbox list */
    div[data-testid="stCheckbox"] {{
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.4rem;
    }}

    div[data-testid="stCheckbox"] label span {{
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
    }}

    /* Footer */
    .cs-footer {{
        margin-top: 3.5rem;
        padding: 1.25rem 0;
        border-top: 1px solid var(--border-light);
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.78rem;
        color: var(--text-muted);
    }}

    .cs-footer-text {{
        max-width: 800px;
        line-height: 1.5;
        margin: 0;
    }}

    /* ==========================================================================
       COMMAND CENTER FLOATING NAVIGATION SYSTEM
       ========================================================================== */

    /* Container wrapper for Command Center navigation */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-command-nav-anchor),
    div[data-testid="stVerticalBlock"]:has(> div > div > .cs-command-nav-anchor) > div:nth-child(2),
    div[data-testid="stHorizontalBlock"]:has(button[key*="cs_nav_btn_"]) {{
        background: rgba(22, 22, 33, 0.94) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(112, 112, 125, 0.25) !important;
        border-radius: 14px !important;
        padding: 0.28rem 0.35rem !important;
        margin-top: -0.6rem !important;
        margin-bottom: 1.75rem !important;
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.52), 0 2px 8px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.07) !important;
        position: sticky !important;
        top: 0.65rem !important;
        z-index: 999 !important;
    }}

    div[data-testid="stHorizontalBlock"]:has(button[key*="cs_nav_btn_"]) {{
        gap: 0.2rem !important;
    }}

    /* Command Center Buttons: Unified Common Properties */
    button[key*="cs_nav_btn_"] {{
        white-space: nowrap !important;
        font-size: 0.81rem !important;
        letter-spacing: 0.01em !important;
        padding: 0.46rem 0.22rem !important;
        text-overflow: ellipsis !important;
    }}

    /* Command Center Buttons: Inactive Item */
    button[key*="cs_nav_btn_"][kind="secondary"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 9px !important;
        color: #9d9dae !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    button[key*="cs_nav_btn_"][kind="secondary"]:hover {{
        background: rgba(255, 255, 255, 0.06) !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        color: #ededf3 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3) !important;
    }}

    /* Command Center Buttons: Active Item */
    button[key*="cs_nav_btn_"][kind="primary"] {{
        background: rgba(82, 102, 235, 0.22) !important;
        background-color: rgba(82, 102, 235, 0.22) !important;
        border: 1px solid rgba(82, 102, 235, 0.75) !important;
        border-radius: 9px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 0 14px rgba(82, 102, 235, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.18), inset 0 -2px 0 rgba(82, 102, 235, 0.85) !important;
        transform: none !important;
    }}

    button[key*="cs_nav_btn_"][kind="primary"]:hover {{
        background: rgba(82, 102, 235, 0.30) !important;
        background-color: rgba(82, 102, 235, 0.30) !important;
        border-color: rgba(82, 102, 235, 0.95) !important;
        color: #ffffff !important;
        box-shadow: 0 0 18px rgba(82, 102, 235, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.22), inset 0 -2px 0 rgba(82, 102, 235, 0.95) !important;
    }}

    /* Responsive adjustments */
    @media (max-width: 1100px) {{
        button[key*="cs_nav_btn_"] {{
            font-size: 0.74rem !important;
            padding: 0.40rem 0.15rem !important;
        }}
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
        .cs-page-title {{ font-size: 1.5rem; }}
        .cs-header {{ flex-direction: column; align-items: flex-start; gap: 0.75rem; }}
        .cs-metric-value {{ font-size: 1.45rem; }}
        div[data-testid="stHorizontalBlock"]:has(button[key*="cs_nav_btn_"]) {{
            position: relative !important;
            top: 0 !important;
            flex-wrap: wrap !important;
        }}
        div[data-testid="stHorizontalBlock"]:has(button[key*="cs_nav_btn_"]) > div {{
            flex: 1 1 calc(25% - 0.25rem) !important;
            min-width: 75px !important;
        }}
    }}
    </style>
    """


def inject_mercury_css() -> None:
    """Inject the Mercury design system CSS into the Streamlit app."""
    import streamlit as st
    st.markdown(get_mercury_css(), unsafe_allow_html=True)
