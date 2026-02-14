"""
MRARFAI v5 Theme — Command Center Edition
═══════════════════════════════════════════
Sprocomm 禾苗品牌 · 指挥中心风格
  🟢 Neon Green #00FF88 → 主色调 (活跃/成功/CTA)
  🔵 蓝叶 #00A0C8 → 信息色 (数据/分析/智能)
  🔴 红叶 #D94040 → 警报色 (风险/预警/危险)
  ⚡ Accent Orange #FF8800 → 警告色

Typography: Space Grotesk (headlines) + JetBrains Mono (everything else)
Corners: 0px (sharp industrial style)

Usage:
    from ui_theme import inject_theme
    inject_theme()
"""

import streamlit as st

# ── Sprocomm Brand Colors ──────────────────────────────────────
SPROCOMM_GREEN = "#8CBF3F"   # 绿叶 — 品牌原色
SPROCOMM_BLUE  = "#00A0C8"   # 蓝叶 — 信息
SPROCOMM_RED   = "#D94040"   # 红叶 — 警报

# ── Command Center Palette ─────────────────────────────────────
NEON_GREEN     = "#00FF88"   # 主色 — 指挥中心风格
WARN_ORANGE    = "#FF8800"   # 警告色

COLORS = {
    # Backgrounds — pure dark command center
    "bg_deep":       "#0C0C0C",
    "bg_base":       "#080808",
    "bg_elevated":   "#111111",
    "bg_overlay":    "#1a1a1a",
    "bg_glass":      "rgba(12,12,12,0.85)",

    # Borders — sharp, subtle
    "border_subtle": "#2f2f2f",
    "border_default":"#2f2f2f",
    "border_hover":  "rgba(0,255,136,0.30)",

    # Text
    "text_primary":  "#FFFFFF",
    "text_secondary":"#8a8a8a",
    "text_muted":    "#6a6a6a",

    # Brand — Command Center
    "accent":        NEON_GREEN,
    "accent_dim":    "rgba(0,255,136,0.10)",
    "accent_glow":   "rgba(0,255,136,0.20)",
    "info":          SPROCOMM_BLUE,
    "info_dim":      "rgba(0,160,200,0.10)",
    "info_border":   "rgba(0,160,200,0.25)",
    "danger":        SPROCOMM_RED,
    "danger_dim":    "rgba(217,64,64,0.10)",
    "danger_border": "rgba(217,64,64,0.25)",

    # Semantic (derived)
    "success":       NEON_GREEN,
    "warning":       WARN_ORANGE,
}


def inject_theme():
    """Inject Sprocomm-branded theme CSS."""
    st.markdown(_build_css(), unsafe_allow_html=True)


def _build_css() -> str:
    c = COLORS
    return f"""<style>
/* ================================================================
   MRARFAI v5 — Command Center Theme
   ⚡ Neon #00FF88  🔵 Blue #00A0C8  🔴 Red #D94040
   Typography: Space Grotesk + JetBrains Mono
   Corners: 0px sharp industrial
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
    --bg-deep: {c['bg_deep']};
    --bg-base: {c['bg_base']};
    --bg-elevated: {c['bg_elevated']};
    --bg-overlay: {c['bg_overlay']};
    --bg-glass: {c['bg_glass']};
    --border-subtle: {c['border_subtle']};
    --border-default: {c['border_default']};
    --border-hover: {c['border_hover']};
    --text-1: {c['text_primary']};
    --text-2: {c['text_secondary']};
    --text-3: {c['text_muted']};
    --accent: {c['accent']};
    --accent-dim: {c['accent_dim']};
    --accent-glow: {c['accent_glow']};
    --neon: {NEON_GREEN};
    --sp-green: {NEON_GREEN};
    --sp-blue: {SPROCOMM_BLUE};
    --sp-red: {SPROCOMM_RED};
    --warn: {WARN_ORANGE};
    --info: {c['info']};
    --info-dim: {c['info_dim']};
    --danger: {c['danger']};
    --danger-dim: {c['danger_dim']};
    --warning: {c['warning']};
    --radius-sm: 0px;
    --radius-md: 0px;
    --radius-lg: 0px;
    --font-sans: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
}}

/* ── Hide Streamlit branding ──────────────────────────────── */
#MainMenu, footer, header,
.stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{
    display: none !important;
}}

/* ── Global ───────────────────────────────────────────────── */
.stApp {{
    background: var(--bg-deep) !important;
}}

.block-container {{
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1600px;
}}

/* ── Sidebar — Command Center ────────────────────────────── */
[data-testid="stSidebar"] {{
    background: var(--bg-base) !important;
    border-right: 1px solid var(--border-subtle) !important;
}}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {{
    font-family: var(--font-mono) !important;
    color: var(--text-2) !important;
    font-size: 0.78rem !important;
}}

.sidebar-label {{
    font-family: var(--font-mono) !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-3) !important;
    padding: 1.2rem 0 0.4rem !important;
    border-top: 1px solid var(--border-subtle);
    margin-top: 0.8rem;
}}

/* ── Typography — Command Center ─────────────────────────── */
.stMarkdown p {{
    font-family: var(--font-mono) !important;
    color: var(--text-1) !important;
    line-height: 1.65 !important;
    font-size: 0.85rem !important;
}}

h1, h2, h3 {{
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}}

code, .stCode {{
    font-family: var(--font-mono) !important;
}}

/* ── Tabs — Command Center ────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--bg-base) !important;
    gap: 0 !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 0 !important;
    overflow-x: auto;
    scrollbar-width: none;
}}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}

.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-3) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.65rem 1rem !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    white-space: nowrap !important;
    transition: all 0.15s ease !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: var(--text-2) !important;
    background: rgba(255,255,255,0.03) !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: var(--neon) !important;
    border-bottom-color: var(--neon) !important;
    background: var(--bg-deep) !important;
}}

/* ── Buttons — Command Center ────────────────────────────── */
.stButton > button {{
    background: var(--bg-elevated) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.12s ease !important;
}}

.stButton > button:hover {{
    border-color: var(--neon) !important;
    background: rgba(0,255,136,0.06) !important;
    color: var(--neon) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.10) !important;
}}

/* ── Inputs — Command Center ─────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div,
.stNumberInput input {{
    background: var(--bg-elevated) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    caret-color: var(--neon) !important;
}}

.stTextInput input:focus,
.stTextArea textarea:focus {{
    border-color: var(--neon) !important;
    box-shadow: 0 0 0 2px var(--accent-dim), 0 0 20px var(--accent-glow) !important;
}}

/* ── Chat Input — neon focus ─────────────────────────────── */
[data-testid="stChatInput"] {{
    background: var(--bg-base) !important;
    border-top: 1px solid var(--border-subtle) !important;
}}

[data-testid="stChatInput"] textarea {{
    background: var(--bg-elevated) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}}

[data-testid="stChatInput"] textarea:focus {{
    border-color: var(--neon) !important;
    box-shadow: 0 0 0 2px var(--accent-dim), 0 0 30px var(--accent-glow) !important;
}}

/* ── Chat Messages — terminal style ──────────────────────── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0.8rem 0 !important;
}}

/* ── Metrics — Command Center KPI ────────────────────────── */
[data-testid="stMetric"] {{
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 0 !important;
    padding: 1.2rem !important;
    transition: all 0.15s ease !important;
}}

[data-testid="stMetric"]:hover {{
    border-color: var(--neon) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.05) !important;
}}

[data-testid="stMetric"] label {{
    font-family: var(--font-mono) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-2) !important;
}}

[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
}}

/* ── Expanders — Command Center ───────────────────────────── */
.streamlit-expanderHeader {{
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-2) !important;
    letter-spacing: 0.02em !important;
}}

.streamlit-expanderContent {{
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
}}

/* ── File Uploader — Command Center ──────────────────────── */
[data-testid="stFileUploader"] {{
    background: var(--bg-elevated) !important;
    border: 1px dashed var(--border-default) !important;
    border-radius: 0 !important;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: var(--neon) !important;
    background: rgba(0,255,136,0.04) !important;
}}

/* ── Scrollbars — thin, dark ─────────────────────────────── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.06); border-radius: 0; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(0,255,136,0.15); }}

/* ── DataFrames — sharp style ────────────────────────────── */
.stDataFrame {{
    border-radius: 0 !important;
}}

.stDataFrame [data-testid="stDataFrameContainer"] {{
    border: 1px solid var(--border-subtle) !important;
    border-radius: 0 !important;
}}

/* ════════════════════════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
   ════════════════════════════════════════════════════════════ */

/* ── Agent Card — Command Center ──────────────────────────── */
.agent-card {{
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 0;
    padding: 0.75rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: all 0.15s ease;
    margin-bottom: 0.5rem;
}}

.agent-card:hover {{ border-color: var(--neon); box-shadow: 0 0 15px rgba(0,255,136,0.05); }}

.agent-card .agent-avatar {{
    width: 32px; height: 32px;
    border-radius: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; flex-shrink: 0;
}}

.agent-card .agent-name {{
    font-family: var(--font-mono);
    font-size: 0.78rem; font-weight: 700; color: var(--text-1);
    letter-spacing: 0.03em;
}}

.agent-card .agent-role {{
    font-family: var(--font-mono);
    font-size: 0.58rem; color: var(--text-3);
    text-transform: uppercase; letter-spacing: 0.1em;
}}

.agent-card .agent-status {{
    font-family: var(--font-mono);
    font-size: 0.58rem; padding: 0.15rem 0.5rem;
    border-radius: 0; font-weight: 700; flex-shrink: 0;
    margin-left: auto; letter-spacing: 0.05em;
}}

.status-running {{
    background: {c['info_dim']};
    color: {SPROCOMM_BLUE};
    border: 1px solid {c['info_border']};
}}

.status-complete {{
    background: var(--accent-dim);
    color: {NEON_GREEN};
    border: 1px solid rgba(0,255,136,0.25);
}}

.status-waiting {{
    background: rgba(255,255,255,0.03);
    color: var(--text-3);
    border: 1px solid var(--border-subtle);
}}

.status-error {{
    background: {c['danger_dim']};
    color: {SPROCOMM_RED};
    border: 1px solid {c['danger_border']};
}}

/* ── Thinking Timeline — Terminal ─────────────────────────── */
.thinking-timeline {{
    background: var(--bg-base);
    border: 1px solid var(--border-subtle);
    border-left: 2px solid rgba(0,255,136,0.30);
    border-radius: 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}}

.thinking-step {{
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.3rem 0;
    margin-left: 0.55rem;
    border-left: 1px solid var(--border-subtle);
    padding-left: 1rem;
}}

.thinking-step .step-dot {{
    width: 6px; height: 6px;
    border-radius: 0;
    margin-top: 0.35rem; flex-shrink: 0;
    margin-left: -1.35rem;
}}

.thinking-step .step-text {{
    font-family: var(--font-mono);
    font-size: 0.72rem; color: var(--text-2); line-height: 1.5;
}}

.thinking-step .step-meta {{
    font-family: var(--font-mono);
    font-size: 0.58rem; color: var(--neon);
    margin-left: auto; flex-shrink: 0; white-space: nowrap;
    font-weight: 600;
}}

/* ── Quality Badge — Command Center ───────────────────────── */
.quality-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.7rem;
    border-radius: 0;
    font-family: var(--font-mono);
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0.25rem 0.25rem 0.25rem 0;
}}

.quality-pass {{
    background: rgba(0,255,136,0.08);
    color: {NEON_GREEN};
    border: 1px solid rgba(0,255,136,0.25);
}}

.quality-fail {{
    background: rgba(255,136,0,0.08);
    color: {WARN_ORANGE};
    border: 1px solid rgba(255,136,0,0.25);
}}

/* ── HITL Card — Command Center ───────────────────────────── */
.hitl-card {{
    background: var(--bg-base);
    border: 1px solid var(--border-subtle);
    border-radius: 0;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}}

.hitl-gauge {{
    width: 42px; height: 42px;
    border-radius: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono);
    font-size: 0.72rem; font-weight: 700; flex-shrink: 0;
}}

.hitl-high {{
    background: rgba(0,255,136,0.10);
    color: {NEON_GREEN};
    border: 2px solid rgba(0,255,136,0.40);
}}

.hitl-medium {{
    background: rgba(255,136,0,0.10);
    color: {WARN_ORANGE};
    border: 2px solid rgba(255,136,0,0.35);
}}

.hitl-low {{
    background: {c['danger_dim']};
    color: {SPROCOMM_RED};
    border: 2px solid {c['danger_border']};
}}

.hitl-info {{ flex: 1; }}

.hitl-info .hitl-level {{
    font-family: var(--font-mono);
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
}}

.hitl-info .hitl-action {{
    font-family: var(--font-mono);
    font-size: 0.72rem; color: var(--text-2); margin-top: 0.1rem;
}}

.hitl-triggers {{
    font-family: var(--font-mono);
    font-size: 0.55rem; color: var(--text-3);
    text-align: right; flex-shrink: 0;
}}

/* ── Trace Bar — Command Center ───────────────────────────── */
.trace-bar {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.35rem 0.8rem;
    background: var(--bg-base);
    border: 1px solid var(--border-subtle);
    border-radius: 0;
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--text-3);
    margin: 0.4rem 0;
    letter-spacing: 0.03em;
}}

.trace-bar .trace-value {{
    color: var(--neon); font-weight: 700;
}}

/* ── Hero Section — Command Center ───────────────────────── */
.hero-section {{
    text-align: center;
    padding: 3.5rem 2rem 2.5rem;
}}

.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--neon);
    background: rgba(0,255,136,0.06);
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 0;
    padding: 0.4rem 1rem;
    margin-bottom: 1.2rem;
}}

.hero-title {{
    font-family: var(--font-sans);
    font-size: 2.8rem; font-weight: 700;
    color: var(--text-1); line-height: 1.15;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}}

.hero-sub {{
    font-family: var(--font-mono);
    font-size: 0.85rem; color: var(--text-2);
    max-width: 500px; margin: 0 auto; line-height: 1.6;
}}

/* ── Sprocomm Logo — Command Center ───────────────────────── */
.sprocomm-logo {{
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
}}

.sprocomm-mark {{
    width: 32px; height: 32px;
    background: {NEON_GREEN};
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-sans);
    font-weight: 700; font-size: 0.9rem;
    color: #0C0C0C;
}}

.sprocomm-text {{
    font-family: var(--font-sans);
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-1);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}

.sprocomm-sub {{
    font-family: var(--font-mono);
    font-size: 0.5rem;
    color: var(--text-3);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}

/* ── Spinner — neon ───────────────────────────────────────── */
@keyframes v5-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes neon-pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.3;}} }}

.v5-spinner {{
    width: 16px; height: 16px;
    border: 2px solid var(--border-subtle);
    border-top-color: var(--neon);
    border-radius: 0;
    animation: v5-spin 0.6s linear infinite;
    display: inline-block;
    vertical-align: middle;
    margin-right: 0.4rem;
}}

/* ── Bento Metrics Grid — Command Center ─────────────────── */
.bento-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 0.8rem 0;
}}

.bento-cell {{
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 0;
    padding: 1rem 1.2rem;
    transition: all 0.15s ease;
}}

.bento-cell:hover {{
    border-color: var(--neon);
    box-shadow: 0 0 15px rgba(0,255,136,0.05);
}}

.bento-label {{
    font-family: var(--font-mono);
    font-size: 0.55rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-2); margin-bottom: 0.4rem;
}}

.bento-value {{
    font-family: var(--font-sans);
    font-size: 1.6rem; font-weight: 700;
    color: var(--text-1); line-height: 1.2;
    letter-spacing: -0.5px;
}}

.bento-delta {{
    font-family: var(--font-mono);
    font-size: 0.65rem; font-weight: 700; margin-top: 0.2rem;
}}

.bento-delta.positive {{ color: var(--neon); }}
.bento-delta.negative {{ color: var(--sp-red); }}

/* ── Section Headers — Command Center ────────────────────── */
.section-header {{
    font-family: var(--font-mono);
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-3);
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border-subtle);
    margin: 1.5rem 0 0.5rem;
}}

/* ── Status Bar — Command Center ─────────────────────────── */
.status-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    background: rgba(0,255,136,0.04);
    border: 1px solid rgba(0,255,136,0.15);
    border-radius: 0;
    margin-bottom: 14px;
}}

.status-bar .status-dot {{
    width: 6px; height: 6px;
    background: var(--neon);
    border-radius: 50%;
    animation: neon-pulse 2s ease-in-out infinite;
}}

.status-bar .status-text {{
    font-family: var(--font-mono);
    font-size: 0.72rem; font-weight: 700;
    color: var(--neon);
    letter-spacing: 0.08em;
}}

.status-bar .status-meta {{
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-3);
    margin-left: auto;
    letter-spacing: 0.04em;
}}

</style>"""
