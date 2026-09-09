"""Reusable UI components implementing the Mercury design language for Construction Safety Intelligence."""

from typing import Any, List, Optional
import streamlit as st
import streamlit.components.v1 as st_components
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
    """Render the React Bits Gooey Nav Command Center navigation bar."""
    # Synchronize with query parameters on first load if present
    if "current_page" not in st.session_state:
        query_page = st.query_params.get("page")
        if query_page:
            query_page_clean = query_page.replace("+", " ").strip()
            matched = False
            for page_name in PAGES:
                if page_name.lower() == query_page_clean.lower():
                    st.session_state.current_page = page_name
                    matched = True
                    break
            if not matched:
                st.session_state.current_page = PAGES[0]
        else:
            st.session_state.current_page = PAGES[0]

    st.markdown('<div class="cs-command-nav-anchor cs-gooey-nav-anchor"></div>', unsafe_allow_html=True)
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
                    try:
                        st.query_params["page"] = page_name
                    except Exception:
                        pass
                    st.rerun()

    # Liquid Gooey particles and transition enhancer (0px height invisible iframe)
    render_gooey_nav_enhancer()

    return st.session_state.current_page


def render_gooey_nav_enhancer() -> None:
    """Inject React Bits Gooey Nav liquid particle transition enhancer into the Streamlit app."""
    gooey_script = """<!DOCTYPE html>
<html>
<head><style>body { margin: 0; padding: 0; overflow: hidden; background: transparent; }</style></head>
<body>
<script>
(function() {
    try {
        const parentDoc = window.parent ? window.parent.document : document;

        // Ensure SVG filter exists
        if (!parentDoc.getElementById('cs-gooey-filter-svg')) {
            const svg = parentDoc.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.id = 'cs-gooey-filter-svg';
            svg.style.position = 'absolute';
            svg.style.width = '0';
            svg.style.height = '0';
            svg.style.pointerEvents = 'none';
            svg.setAttribute('aria-hidden', 'true');
            svg.innerHTML = '<defs><filter id="cs-gooey-filter" x="-20%" y="-20%" width="140%" height="140%">' +
                '<feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />' +
                '<feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 16 -6" result="goo" />' +
                '<feComposite in="SourceGraphic" in2="goo" operator="atop" />' +
                '</filter></defs>';
            parentDoc.body.appendChild(svg);
        }

        function initGooeyParticles() {
            const navBar = parentDoc.querySelector('div[data-testid="stHorizontalBlock"]:has(button[key*="cs_nav_btn_"])');
            if (!navBar) return;

            let particlesLayer = navBar.querySelector('.cs-gooey-liquid-layer');
            if (!particlesLayer) {
                particlesLayer = parentDoc.createElement('div');
                particlesLayer.className = 'cs-gooey-liquid-layer';
                navBar.appendChild(particlesLayer);
            }

            const activeBtn = navBar.querySelector('button[key*="cs_nav_btn_"][kind="primary"]');
            if (!activeBtn) return;

            const curPage = activeBtn.innerText.trim();
            const prevPage = sessionStorage.getItem('cs_gooey_active_page');

            if (prevPage && prevPage !== curPage) {
                const allBtns = Array.from(navBar.querySelectorAll('button[key*="cs_nav_btn_"]'));
                let prevBtn = null;
                for (let i = 0; i < allBtns.length; i++) {
                    if (allBtns[i].innerText.trim() === prevPage) {
                        prevBtn = allBtns[i];
                        break;
                    }
                }

                if (prevBtn) {
                    spawnGooeyParticles(particlesLayer, prevBtn, activeBtn, navBar);
                }
            }

            sessionStorage.setItem('cs_gooey_active_page', curPage);

            function spawnGooeyParticles(container, fromEl, toEl, parentEl) {
                const parentRect = parentEl.getBoundingClientRect();
                const fromRect = fromEl.getBoundingClientRect();
                const toRect = toEl.getBoundingClientRect();

                const fromCenterX = fromRect.left - parentRect.left + fromRect.width / 2;
                const fromCenterY = fromRect.top - parentRect.top + fromRect.height / 2;
                const toCenterX = toRect.left - parentRect.left + toRect.width / 2;
                const toCenterY = toRect.top - parentRect.top + toRect.height / 2;

                const dx = toCenterX - fromCenterX;
                const dy = toCenterY - fromCenterY;
                const particleCount = 10;
                const duration = 440;

                for (let i = 0; i < particleCount; i++) {
                    const bubble = parentDoc.createElement('div');
                    bubble.className = 'cs-gooey-bubble';

                    const size = Math.floor(Math.random() * 5 + 4);
                    bubble.style.width = (size * 2) + 'px';
                    bubble.style.height = (size * 2) + 'px';

                    const startX = fromCenterX + (Math.random() - 0.5) * (fromRect.width * 0.4) - size;
                    const startY = fromCenterY + (Math.random() - 0.5) * 8 - size;

                    const progress = 0.3 + Math.random() * 0.7;
                    const spreadY = (Math.random() - 0.5) * 18;
                    const endX = fromCenterX + dx * progress + (Math.random() - 0.5) * 10 - size;
                    const endY = fromCenterY + dy * progress + spreadY - size;

                    bubble.style.transform = 'translate3d(' + startX + 'px, ' + startY + 'px, 0) scale(1)';
                    bubble.style.opacity = '0.92';
                    container.appendChild(bubble);

                    const startTime = performance.now();
                    const delay = Math.random() * 40;

                    function animateBubble(now) {
                        const elapsed = now - startTime - delay;
                        if (elapsed < 0) {
                            requestAnimationFrame(animateBubble);
                            return;
                        }
                        const t = Math.min(1, elapsed / duration);
                        const ease = 1 - Math.pow(1 - t, 3);

                        const curX = startX + (endX - startX) * ease;
                        const curY = startY + (endY - startY) * ease;
                        const scale = 1 - t * 0.85;
                        const alpha = 1 - t * 0.95;

                        bubble.style.transform = 'translate3d(' + curX + 'px, ' + curY + 'px, 0) scale(' + scale + ')';
                        bubble.style.opacity = alpha;

                        if (t < 1) {
                            requestAnimationFrame(animateBubble);
                        } else {
                            bubble.remove();
                        }
                    }
                    requestAnimationFrame(animateBubble);
                }
            }
        }

        setTimeout(initGooeyParticles, 30);
    } catch(e) {}
})();
</script>
</body>
</html>
"""
    st_components.html(gooey_script, height=0)



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


def render_pixel_snow_background(active: bool = True) -> None:
    """Render React Bits-inspired interactive Pixel Snow background effect on the Overview page."""
    if not active:
        cleanup_script = """
        <!DOCTYPE html>
        <html>
        <head><style>body { margin: 0; padding: 0; display: none; }</style></head>
        <body>
        <script>
        (function() {
            try {
                const parentDoc = window.parent ? window.parent.document : document;
                const canvas = parentDoc.getElementById('rf-pixel-snow-canvas');
                if (canvas) {
                    canvas.style.display = 'none';
                }
            } catch(e) {}
        })();
        </script>
        </body>
        </html>
        """
        st_components.html(cleanup_script, height=0)
        return

    pixel_snow_html = """
    <!DOCTYPE html>
    <html>
    <head><style>body { margin: 0; padding: 0; overflow: hidden; background: transparent; }</style></head>
    <body>
    <script>
    (function() {
        try {
            const parentDoc = window.parent ? window.parent.document : document;
            const parentWin = window.parent || window;

            let canvas = parentDoc.getElementById('rf-pixel-snow-canvas');
            if (!canvas) {
                canvas = parentDoc.createElement('canvas');
                canvas.id = 'rf-pixel-snow-canvas';
                canvas.style.position = 'fixed';
                canvas.style.top = '0';
                canvas.style.left = '0';
                canvas.style.width = '100vw';
                canvas.style.height = '100vh';
                canvas.style.pointerEvents = 'none';
                canvas.style.zIndex = '0';
                canvas.style.opacity = '0.72';
                parentDoc.body.appendChild(canvas);
            }
            canvas.style.display = 'block';

            // Ensure magnetic experiment canvas is hidden
            const magCanvas = parentDoc.getElementById('rf-magnetic-hero-canvas');
            if (magCanvas) magCanvas.style.display = 'none';

            const ctx = canvas.getContext('2d');
            let width = (canvas.width = parentWin.innerWidth || 1200);
            let height = (canvas.height = parentWin.innerHeight || 800);

            function resize() {
                width = canvas.width = parentWin.innerWidth || 1200;
                height = canvas.height = parentWin.innerHeight || 800;
            }
            parentWin.addEventListener('resize', resize);

            // Create pixel snow particle collection (slightly increased particle density)
            const PARTICLE_COUNT = Math.min(225, Math.max(110, Math.floor(width / 8.2)));
            const particles = [];

            // Construction tech color palette: soft white, technical slate, electric cyan, soft cobalt
            const COLORS = [
                'rgba(248, 250, 252, ', // Crisp ivory / soft white
                'rgba(148, 163, 184, ', // Technical slate
                'rgba(56, 189, 248, ',  // Electric cyan
                'rgba(59, 130, 246, '   // Cobalt blue
            ];

            for (let i = 0; i < PARTICLE_COUNT; i++) {
                const depth = Math.random() * 0.8 + 0.2; // 0.2 (far) to 1.0 (near)
                particles.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    depth: depth,
                    size: Math.floor(depth * 3.5 + 1.5), // 2px to 5px square pixels
                    baseSpeedY: depth * 0.75 + 0.35,     // Downward drift speed
                    vx: 0,
                    vy: 0,
                    swayFreq: Math.random() * 0.02 + 0.01,
                    swayAmp: Math.random() * 0.6 + 0.2,
                    phase: Math.random() * Math.PI * 2,
                    colorPrefix: COLORS[Math.floor(Math.random() * COLORS.length)],
                    baseAlpha: depth * 0.45 + 0.15
                });
            }

            let mouseX = -9999;
            let mouseY = -9999;
            let targetMouseX = -9999;
            let targetMouseY = -9999;

            parentWin.addEventListener('mousemove', function(e) {
                targetMouseX = e.clientX;
                targetMouseY = e.clientY;
            });

            parentWin.addEventListener('mouseleave', function() {
                targetMouseX = -9999;
                targetMouseY = -9999;
            });

            let time = 0;

            function animate() {
                if (canvas.style.display === 'none') return;

                ctx.clearRect(0, 0, width, height);
                time += 0.02;

                // Smooth cursor interpolation
                if (targetMouseX > -5000) {
                    if (mouseX < -5000) {
                        mouseX = targetMouseX;
                        mouseY = targetMouseY;
                    } else {
                        mouseX += (targetMouseX - mouseX) * 0.16;
                        mouseY += (targetMouseY - mouseY) * 0.16;
                    }
                } else {
                    mouseX = -9999;
                    mouseY = -9999;
                }

                // Slightly enhanced interactive deflection radius and smooth responsive force
                const interactionRadius = 195;
                const interactionRadiusSq = interactionRadius * interactionRadius;

                // Query image elements on page to guarantee snow NEVER renders over images
                const imgEls = parentDoc.querySelectorAll('div[data-testid="stImage"], .stImage img, .stImage');
                const imgBoxes = [];
                for (let k = 0; k < imgEls.length; k++) {
                    const rect = imgEls[k].getBoundingClientRect();
                    if (rect.width > 20 && rect.height > 20) {
                        imgBoxes.push({
                            left: rect.left - 2,
                            top: rect.top - 2,
                            right: rect.right + 2,
                            bottom: rect.bottom + 2
                        });
                    }
                }

                for (let i = 0; i < particles.length; i++) {
                    const p = particles[i];

                    // Base natural downward pixel motion with gentle sway
                    const sway = Math.sin(time * p.swayFreq + p.phase) * p.swayAmp;
                    p.y += p.baseSpeedY + p.vy;
                    p.x += sway + p.vx;

                    // Enhanced interactive mouse deflection with natural fluid lift
                    if (mouseX > -5000) {
                        const dx = p.x - mouseX;
                        const dy = p.y - mouseY;
                        const distSq = dx * dx + dy * dy;

                        if (distSq < interactionRadiusSq && distSq > 4) {
                            const dist = Math.sqrt(distSq);
                            const normDist = dist / interactionRadius;
                            const force = (1 - normDist) * (p.depth * 4.6);
                            const angle = Math.atan2(dy, dx);

                            // Smooth deflection push + subtle vortex lift
                            p.vx += Math.cos(angle) * force * 0.90;
                            p.vy += (Math.sin(angle) * force * 0.90) - force * 0.32;
                        }
                    }

                    // Natural velocity damping
                    p.vx *= 0.92;
                    p.vy *= 0.92;

                    // Wrap-around screen bounds
                    if (p.y > height + 10) {
                        p.y = -10;
                        p.x = Math.random() * width;
                    } else if (p.y < -15) {
                        p.y = height + 5;
                    }
                    if (p.x > width + 10) {
                        p.x = -10;
                    } else if (p.x < -10) {
                        p.x = width + 10;
                    }

                    // Strict exclusion: skip rendering if particle is within any image area
                    let isOverImage = false;
                    for (let k = 0; k < imgBoxes.length; k++) {
                        const box = imgBoxes[k];
                        if (p.x >= box.left && p.x <= box.right && p.y >= box.top && p.y <= box.bottom) {
                            isOverImage = true;
                            break;
                        }
                    }
                    if (isOverImage) continue;

                    // Render crisp square pixel particle
                    const currentAlpha = Math.min(0.88, Math.max(0.08, p.baseAlpha + Math.abs(p.vx + p.vy) * 0.12));
                    ctx.fillStyle = p.colorPrefix + currentAlpha + ')';
                    ctx.fillRect(Math.round(p.x), Math.round(p.y), p.size, p.size);
                }

                parentWin._rfPixelSnowFrame = requestAnimationFrame(animate);
            }

            if (parentWin._rfPixelSnowFrame) {
                cancelAnimationFrame(parentWin._rfPixelSnowFrame);
            }
            animate();
        } catch(e) {}
    })();
    </script>
    </body>
    </html>
    """
    st_components.html(pixel_snow_html, height=0)


def render_magnetic_hero_background(active: bool = True) -> None:
    """Render Shaders.com Magnetic Hero-inspired interactive magnetic particle field on Overview page."""
    if not active:
        cleanup_script = """
        <!DOCTYPE html>
        <html>
        <head><style>body { margin: 0; padding: 0; display: none; }</style></head>
        <body>
        <script>
        (function() {
            try {
                const parentDoc = window.parent ? window.parent.document : document;
                const canvas = parentDoc.getElementById('rf-magnetic-hero-canvas');
                if (canvas) {
                    canvas.style.display = 'none';
                }
            } catch(e) {}
        })();
        </script>
        </body>
        </html>
        """
        st_components.html(cleanup_script, height=0)
        return

    magnetic_hero_html = """
    <!DOCTYPE html>
    <html>
    <head><style>body { margin: 0; padding: 0; overflow: hidden; background: transparent; }</style></head>
    <body>
    <script>
    (function() {
        try {
            const parentDoc = window.parent ? window.parent.document : document;
            const parentWin = window.parent || window;

            let canvas = parentDoc.getElementById('rf-magnetic-hero-canvas');
            if (!canvas) {
                canvas = parentDoc.createElement('canvas');
                canvas.id = 'rf-magnetic-hero-canvas';
                canvas.style.position = 'fixed';
                canvas.style.top = '0';
                canvas.style.left = '0';
                canvas.style.width = '100vw';
                canvas.style.height = '100vh';
                canvas.style.pointerEvents = 'none';
                canvas.style.zIndex = '0';
                canvas.style.opacity = '0.75';
                parentDoc.body.appendChild(canvas);
            }
            canvas.style.display = 'block';

            const ctx = canvas.getContext('2d');
            let width = (canvas.width = parentWin.innerWidth || 1200);
            let height = (canvas.height = parentWin.innerHeight || 800);

            function resize() {
                width = canvas.width = parentWin.innerWidth || 1200;
                height = canvas.height = parentWin.innerHeight || 800;
                initGrid();
            }
            parentWin.addEventListener('resize', resize);

            let particles = [];

            function initGrid() {
                particles = [];
                // Calculate grid spacing based on viewport
                const spacing = Math.max(28, Math.min(42, Math.floor(Math.sqrt((width * height) / 600))));
                const cols = Math.floor(width / spacing) + 2;
                const rows = Math.floor(height / spacing) + 2;

                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                        const jitterX = (Math.random() - 0.5) * (spacing * 0.45);
                        const jitterY = (Math.random() - 0.5) * (spacing * 0.45);
                        const ox = c * spacing + jitterX;
                        const oy = r * spacing + jitterY;

                        particles.push({
                            x: ox,
                            y: oy,
                            originX: ox,
                            originY: oy,
                            vx: 0,
                            vy: 0,
                            baseRadius: Math.random() * 0.8 + 1.2,
                            phase: Math.random() * Math.PI * 2,
                            idleFreq: Math.random() * 0.015 + 0.008,
                            idleAmp: Math.random() * 0.4 + 0.1
                        });
                    }
                }
            }

            initGrid();

            let mouseX = -9999;
            let mouseY = -9999;
            let targetMouseX = -9999;
            let targetMouseY = -9999;

            parentWin.addEventListener('mousemove', function(e) {
                targetMouseX = e.clientX;
                targetMouseY = e.clientY;
            });

            parentWin.addEventListener('mouseleave', function() {
                targetMouseX = -9999;
                targetMouseY = -9999;
            });

            let time = 0;

            function animate() {
                if (canvas.style.display === 'none') return;

                ctx.clearRect(0, 0, width, height);
                time += 0.02;

                // Smooth cursor interpolation
                if (targetMouseX > -5000) {
                    if (mouseX < -5000) {
                        mouseX = targetMouseX;
                        mouseY = targetMouseY;
                    } else {
                        mouseX += (targetMouseX - mouseX) * 0.18;
                        mouseY += (targetMouseY - mouseY) * 0.18;
                    }
                } else {
                    mouseX = -9999;
                    mouseY = -9999;
                }

                const magneticRadius = 230;
                const magneticRadiusSq = magneticRadius * magneticRadius;
                const springK = 0.045; // Spring return stiffness
                const damping = 0.87;  // Viscous damping

                for (let i = 0; i < particles.length; i++) {
                    const p = particles[i];

                    // Calm idle breathing movement
                    const idleX = Math.cos(time * p.idleFreq + p.phase) * p.idleAmp;
                    const idleY = Math.sin(time * p.idleFreq + p.phase) * p.idleAmp;

                    // Physical magnetic attraction + vortex flow force
                    if (mouseX > -5000) {
                        const dx = mouseX - p.x;
                        const dy = mouseY - p.y;
                        const distSq = dx * dx + dy * dy;

                        if (distSq < magneticRadiusSq && distSq > 2) {
                            const dist = Math.sqrt(distSq);
                            const normDist = dist / magneticRadius;
                            // Smooth non-linear magnetic pull curve
                            const force = Math.pow(1 - normDist, 1.3) * 8.2;
                            const angle = Math.atan2(dy, dx);

                            // Strong attraction toward cursor with fluid rotational swirl
                            p.vx += Math.cos(angle) * force * 0.96 - Math.sin(angle) * force * 0.28;
                            p.vy += Math.sin(angle) * force * 0.96 + Math.cos(angle) * force * 0.28;
                        }
                    }

                    // Hooke's Law spring force back to origin equilibrium
                    const homeDx = (p.originX + idleX) - p.x;
                    const homeDy = (p.originY + idleY) - p.y;
                    p.vx += homeDx * springK;
                    p.vy += homeDy * springK;

                    // Apply damping
                    p.vx *= damping;
                    p.vy *= damping;

                    // Update position
                    p.x += p.vx;
                    p.y += p.vy;

                    // Render particle with dynamic excitation shimmer
                    const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
                    let color;
                    let radius = p.baseRadius;

                    if (speed > 2.0) {
                        // High magnetic excitation: construction amber / gold spark
                        color = 'rgba(245, 158, 11, ' + Math.min(0.95, 0.60 + speed * 0.1) + ')';
                        radius += 1.6;
                    } else if (speed > 0.7) {
                        // Medium excitation: electric cyan glow
                        color = 'rgba(56, 189, 248, ' + Math.min(0.88, 0.45 + speed * 0.16) + ')';
                        radius += 1.0;
                    } else if (speed > 0.25) {
                        // Subtle excitation: soft cobalt
                        color = 'rgba(96, 165, 250, ' + Math.min(0.70, 0.28 + speed * 0.22) + ')';
                        radius += 0.4;
                    } else {
                        // Resting state: calm technical slate / ivory
                        color = 'rgba(148, 163, 184, 0.35)';
                    }

                    ctx.beginPath();
                    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();
                }

                parentWin._rfMagneticHeroFrame = requestAnimationFrame(animate);
            }

            if (parentWin._rfMagneticHeroFrame) {
                cancelAnimationFrame(parentWin._rfMagneticHeroFrame);
            }
            animate();
        } catch(e) {}
    })();
    </script>
    </body>
    </html>
    """
    st_components.html(magnetic_hero_html, height=0)


