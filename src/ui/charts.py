"""Mercury chart styling helper for Plotly visualizations in Construction Safety Intelligence."""

from typing import Any, Optional
import plotly.graph_objects as go
from src.ui.theme import (
    COLOR_ACCENT_COBALT,
    COLOR_BORDER_LIGHT,
    COLOR_BORDER_STRUCTURAL,
    COLOR_SURFACE,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    SAFETY_COLORS,
)

# Standard Plotly Discrete Palettes
RISK_COLOR_DISCRETE_MAP = {
    "LOW": SAFETY_COLORS["LOW"],
    "MEDIUM": SAFETY_COLORS["MEDIUM"],
    "HIGH": SAFETY_COLORS["HIGH"],
    "CRITICAL": SAFETY_COLORS["CRITICAL"],
}

MONO_ACCENT_SCALE = ["#272735", "#3a3a4f", "#5266eb", "#6f82ff"]
SAFETY_ALERT_SCALE = ["#3a2e22", "#92400e", "#f59e0b", "#ef4444"]


def style_mercury_chart(
    fig: Any,
    title: Optional[str] = None,
    height: int = 340,
    show_legend: bool = False,
) -> Any:
    """Apply Mercury aesthetic to any Plotly figure: flat graphite surface, subtle grids, Inter fonts."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=44 if title else 20, b=20),
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color=COLOR_TEXT_SECONDARY,
            size=12,
        ),
        title=dict(
            text=title or (fig.layout.title.text if fig.layout.title else None),
            font=dict(
                family="Inter, sans-serif",
                color=COLOR_TEXT_PRIMARY,
                size=14,
                weight=600,
            ),
            x=0.0,
            xanchor="left",
            y=0.98,
        ) if (title or (fig.layout.title and fig.layout.title.text)) else None,
        xaxis=dict(
            gridcolor="rgba(112, 112, 125, 0.16)",
            linecolor="rgba(112, 112, 125, 0.28)",
            tickfont=dict(color=COLOR_TEXT_SECONDARY, size=11),
            zerolinecolor="rgba(112, 112, 125, 0.25)",
        ),
        yaxis=dict(
            gridcolor="rgba(112, 112, 125, 0.16)",
            linecolor="rgba(112, 112, 125, 0.28)",
            tickfont=dict(color=COLOR_TEXT_SECONDARY, size=11),
            zerolinecolor="rgba(112, 112, 125, 0.25)",
        ),
        legend=dict(
            font=dict(color=COLOR_TEXT_SECONDARY, size=11),
            bgcolor="rgba(30, 30, 42, 0.8)",
            bordercolor="rgba(112, 112, 125, 0.2)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ) if show_legend else dict(visible=False),
        hoverlabel=dict(
            bgcolor=COLOR_SURFACE,
            bordercolor=COLOR_ACCENT_COBALT,
            font=dict(family="Inter, sans-serif", color=COLOR_TEXT_PRIMARY, size=12),
        ),
    )
    return fig
