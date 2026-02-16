"""
MRARFAI v10 — Design System
============================
Single Source of Truth for all visual styling.

References: Linear, Vercel, Cursor, Raycast, Arc
Typography: JetBrains Mono + Inter
Palette: True black + 4 grays + white
"""

import streamlit as st

# ── Design Tokens ──────────────────────────────────────────────

COLORS = {
    # Backgrounds
    "bg_void":     "#000000",
    "bg_deep":     "#050505",
    "bg_base":     "#0A0A0A",
    "bg_elevated": "#111111",
    "bg_surface":  "#171717",
    "bg_overlay":  "#1C1C1C",
    "bg_glass":    "rgba(10,10,10,0.85)",

    # Borders
    "border_ghost":   "rgba(255,255,255,0.04)",
    "border_subtle":  "rgba(255,255,255,0.07)",
    "border_default": "rgba(255,255,255,0.10)",
    "border_hover":   "rgba(255,255,255,0.18)",
    "border_active":  "rgba(255,255,255,0.30)",

    # Text
    "text_primary":   "#EDEDED",
    "text_secondary": "#999999",
    "text_tertiary":  "#666666",
    "text_muted":     "#444444",
    "text_ghost":     "#2A2A2A",

    # Semantic
    "accent":         "#EDEDED",
    "accent_dim":     "rgba(255,255,255,0.06)",
    "accent_glow":    "rgba(255,255,255,0.12)",
    "status_active":  "#EDEDED",
    "status_ready":   "#666666",
    "status_warning": "#999999",
    "status_error":   "#555555",
}


# ── Public API ─────────────────────────────────────────────────

def inject_theme():
    """Inject main dark theme. Call once in app.py after login."""
    st.markdown(_build_main_css(), unsafe_allow_html=True)


def inject_login_theme():
    """Inject login page theme. Call in auth.py."""
    st.markdown(_build_login_css(), unsafe_allow_html=True)


# ── HTML Helpers ───────────────────────────────────────────────

def agent_card_html(icon: str, name: str, role: str, status: str = "online",
                    skills: int = 0, color: str = "#EDEDED") -> str:
    status_cls = {
        "online": "st-on", "idle": "st-idle",
        "running": "st-run", "error": "st-err",
    }.get(status, "st-idle")
    status_label = {
        "online": "ONLINE", "idle": "STANDBY",
        "running": "ACTIVE", "error": "ERROR",
    }.get(status, status.upper())
    sk = f'<span class="ag-sk">{skills} skills</span>' if skills else ''
    return f'''<div class="ag-card">
  <div class="ag-av {status_cls}">{icon}</div>
  <div class="ag-body">
    <div class="ag-name">{name}</div>
    <div class="ag-role">{role}</div>
  </div>
  {sk}
  <span class="ag-st {status_cls}">{status_label}</span>
</div>'''


def status_bar_html(text: str, meta: str = "") -> str:
    m = f'<span class="sbar-meta">{meta}</span>' if meta else ''
    return f'''<div class="sbar">
  <span class="sbar-dot"></span>
  <span class="sbar-text">{text}</span>
  {m}
</div>'''


def section_header_html(text: str) -> str:
    return f'<div class="sec-hdr">{text}</div>'


def kpi_card_html(label: str, value: str, delta: str = "",
                  positive: bool = True) -> str:
    d_cls = "kpi-up" if positive else "kpi-down"
    d_html = f'<div class="kpi-delta {d_cls}">{delta}</div>' if delta else ''
    return f'''<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {d_html}
</div>'''


def badge_html(label: str, variant: str = "default") -> str:
    return f'<span class="badge badge-{variant}">{label}</span>'


# ── Main CSS ───────────────────────────────────────────────────

def _build_main_css() -> str:
    c = COLORS
    return f"""<style>
/* ================================================================
   MRARFAI v10 Design System
   Ref: Linear · Vercel · Cursor · Raycast
   Type: JetBrains Mono + Inter
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Variables ──────────────────────────────────────────────── */
:root {{
  --bg-0: {c['bg_void']};
  --bg-1: {c['bg_deep']};
  --bg-2: {c['bg_base']};
  --bg-3: {c['bg_elevated']};
  --bg-4: {c['bg_surface']};
  --bg-5: {c['bg_overlay']};

  --b0: {c['border_ghost']};
  --b1: {c['border_subtle']};
  --b2: {c['border_default']};
  --b3: {c['border_hover']};
  --b4: {c['border_active']};

  --t1: {c['text_primary']};
  --t2: {c['text_secondary']};
  --t3: {c['text_tertiary']};
  --t4: {c['text_muted']};
  --t5: {c['text_ghost']};

  --sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  --ease: cubic-bezier(0.25, 0.1, 0.25, 1);
  --dur: 150ms;
}}

/* ── Reset Streamlit ───────────────────────────────────────── */
#MainMenu, footer, header,
.stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] {{
  display: none !important;
}}

[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"] {{
  display: none !important;
}}

/* ── Base ──────────────────────────────────────────────────── */
.stApp {{
  background: var(--bg-1) !important;
  color: var(--t2) !important;
}}

.block-container {{
  padding: 1rem 2.5rem 4rem !important;
  max-width: 1400px !important;
}}

/* ── Typography ────────────────────────────────────────────── */
.stMarkdown p {{
  font-family: var(--sans) !important;
  color: var(--t2) !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
  letter-spacing: -0.01em !important;
}}

h1 {{
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  color: var(--t1) !important;
  letter-spacing: -0.03em !important;
  font-size: 24px !important;
  line-height: 1.3 !important;
}}

h2 {{
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  color: var(--t1) !important;
  letter-spacing: -0.02em !important;
  font-size: 18px !important;
}}

h3, h4 {{
  font-family: var(--mono) !important;
  font-weight: 500 !important;
  color: var(--t2) !important;
  font-size: 11px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}}

code, .stCode {{
  font-family: var(--mono) !important;
}}

.stCaption, small {{
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--t3) !important;
}}

/* ── Tabs ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  background: transparent !important;
  gap: 0 !important;
  border-bottom: 1px solid var(--b1) !important;
  padding: 0 !important;
  overflow-x: auto;
  scrollbar-width: none;
}}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}

.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--t3) !important;
  border: none !important;
  border-bottom: 1px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 16px !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  white-space: nowrap !important;
  transition: color var(--dur) var(--ease) !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
  color: var(--t2) !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {{
  color: var(--t1) !important;
  border-bottom: 1px solid var(--t1) !important;
  background: transparent !important;
}}

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {{
  background: var(--bg-3) !important;
  color: var(--t1) !important;
  border: 1px solid var(--b2) !important;
  border-radius: 6px !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 16px !important;
  transition: all var(--dur) var(--ease) !important;
  cursor: pointer !important;
}}

.stButton > button:hover {{
  background: var(--bg-4) !important;
  border-color: var(--b3) !important;
}}

.stButton > button:active {{
  transform: scale(0.98) !important;
}}

/* ── Inputs ────────────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div,
.stNumberInput input {{
  background: var(--bg-2) !important;
  color: var(--t1) !important;
  border: 1px solid var(--b2) !important;
  border-radius: 6px !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
  caret-color: var(--t1) !important;
  transition: border-color var(--dur) var(--ease) !important;
}}

.stTextInput input:focus,
.stTextArea textarea:focus {{
  border-color: var(--b4) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.04) !important;
}}

/* ── Chat ──────────────────────────────────────────────────── */
[data-testid="stChatInput"] {{
  background: var(--bg-2) !important;
  border-top: 1px solid var(--b1) !important;
}}

[data-testid="stChatInput"] textarea {{
  background: var(--bg-3) !important;
  color: var(--t1) !important;
  border: 1px solid var(--b2) !important;
  border-radius: 6px !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
}}

[data-testid="stChatInput"] textarea:focus {{
  border-color: var(--b4) !important;
}}

[data-testid="stChatMessage"] {{
  background: transparent !important;
  border: none !important;
  padding: 16px 0 !important;
  border-bottom: 1px solid var(--b0) !important;
}}

[data-testid="stChatMessage"] p {{
  font-family: var(--sans) !important;
  font-size: 14px !important;
  line-height: 1.7 !important;
  color: var(--t2) !important;
}}

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {{
  font-family: var(--sans) !important;
  color: var(--t1) !important;
}}

[data-testid="stChatMessage"] li {{
  font-family: var(--sans) !important;
  font-size: 13px !important;
  line-height: 1.7 !important;
  color: var(--t2) !important;
}}

[data-testid="stChatMessage"] strong {{
  color: var(--t1) !important;
}}

/* ── Metrics ───────────────────────────────────────────────── */
[data-testid="stMetric"] {{
  background: var(--bg-3) !important;
  border: 1px solid var(--b1) !important;
  border-radius: 8px !important;
  padding: 16px 20px !important;
  transition: border-color var(--dur) var(--ease) !important;
}}

[data-testid="stMetric"]:hover {{
  border-color: var(--b2) !important;
}}

[data-testid="stMetricLabel"],
[data-testid="stMetric"] label {{
  font-family: var(--sans) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  color: var(--t3) !important;
}}

[data-testid="stMetricValue"] {{
  font-family: var(--mono) !important;
  font-weight: 600 !important;
  color: var(--t1) !important;
  letter-spacing: -0.02em !important;
}}

[data-testid="stMetricDelta"] {{
  font-family: var(--mono) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}}

/* ── Expanders ─────────────────────────────────────────────── */
.streamlit-expanderHeader {{
  font-family: var(--sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  background: var(--bg-3) !important;
  color: var(--t2) !important;
}}

[data-testid="stExpander"] {{
  background: var(--bg-3) !important;
  border: 1px solid var(--b1) !important;
  border-radius: 8px !important;
  transition: border-color var(--dur) var(--ease) !important;
}}

[data-testid="stExpander"]:hover {{
  border-color: var(--b2) !important;
}}

/* ── File Uploader ─────────────────────────────────────────── */
[data-testid="stFileUploader"],
.stFileUploader {{
  border: 1px dashed var(--b2) !important;
  border-radius: 8px !important;
  transition: border-color var(--dur) var(--ease) !important;
}}

[data-testid="stFileUploader"]:hover,
.stFileUploader:hover {{
  border-color: var(--b3) !important;
}}

/* ── DataFrames ────────────────────────────────────────────── */
.stDataFrame {{
  border-radius: 8px !important;
}}

.stDataFrame [data-testid="stDataFrameContainer"] {{
  border: 1px solid var(--b1) !important;
  border-radius: 8px !important;
}}

/* ── Status Container ──────────────────────────────────────── */
[data-testid="stStatus"] {{
  margin: 8px 0 !important;
}}

[data-testid="stStatus"] p {{
  font-size: 13px !important;
  line-height: 1.6 !important;
}}

/* ── Scrollbars ────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--b1); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--b3); }}

/* ================================================================
   COMPONENTS
   ================================================================ */

/* ── Top Nav ───────────────────────────────────────────────── */
.top-nav {{
  display: flex;
  align-items: center;
  gap: 12px;
  height: 48px;
  padding: 0 20px;
  background: var(--bg-2);
  border-bottom: 1px solid var(--b1);
  margin: -1rem -2.5rem 24px;
}}

.top-nav-brand {{
  font-family: var(--mono);
  font-weight: 700;
  font-size: 13px;
  color: var(--t1);
  letter-spacing: 0.04em;
}}

.top-nav-version {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t4);
  padding-left: 12px;
  border-left: 1px solid var(--b1);
}}

.top-nav-user {{
  font-family: var(--sans);
  font-size: 12px;
  color: var(--t2);
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}}

.top-nav-role {{
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  color: var(--t3);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 2px 8px;
  background: var(--bg-4);
  border-radius: 3px;
}}

/* ── Command Center ────────────────────────────────────────── */
.cmd-hero {{
  padding: 48px 0 32px;
  max-width: 640px;
}}

.cmd-title {{
  font-family: var(--sans) !important;
  font-size: 32px !important;
  font-weight: 600 !important;
  color: var(--t1) !important;
  letter-spacing: -0.03em !important;
  line-height: 1.2 !important;
  margin: 0 0 8px !important;
}}

.cmd-sub {{
  font-family: var(--sans);
  font-size: 14px;
  color: var(--t3);
  line-height: 1.5;
}}

.cmd-badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: var(--bg-3);
  border: 1px solid var(--b2);
  border-radius: 4px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--t2);
  margin-top: 16px;
}}

/* ── Section Header ────────────────────────────────────────── */
.sec-hdr {{
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--t3);
  padding-bottom: 8px;
  margin: 24px 0 12px;
  border-bottom: 1px solid var(--b1);
}}

/* ── KPI Card ──────────────────────────────────────────────── */
.kpi-card {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 16px 20px;
  transition: border-color var(--dur) var(--ease);
}}

.kpi-card:hover {{
  border-color: var(--b2);
}}

.kpi-label {{
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 8px;
}}

.kpi-value {{
  font-family: var(--mono);
  font-size: 24px;
  font-weight: 600;
  color: var(--t1);
  line-height: 1;
  letter-spacing: -0.02em;
}}

.kpi-icon {{
  font-size: 14px;
  margin-bottom: 8px;
  opacity: 0.5;
}}

.kpi-sub {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  margin-top: 6px;
}}

.kpi-delta {{
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  margin-top: 4px;
}}
.kpi-up {{ color: var(--t1); }}
.kpi-down {{ color: var(--t3); }}

/* ── Agent Cards ───────────────────────────────────────────── */
.ag-card {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  transition: border-color var(--dur) var(--ease);
  cursor: pointer;
  margin-bottom: 6px;
}}

.ag-card:hover {{
  border-color: var(--b3);
}}

.ag-av {{
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
  background: var(--bg-4);
  border-radius: 8px;
}}

.ag-body {{
  flex: 1;
  min-width: 0;
}}

.ag-name {{
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 600;
  color: var(--t1);
}}

.ag-role {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  margin-top: 2px;
}}

.ag-desc {{
  font-family: var(--sans);
  font-size: 12px;
  color: var(--t3);
  line-height: 1.4;
  margin-top: 4px;
}}

.ag-meta {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t4);
}}

.ag-sk {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t4);
  flex-shrink: 0;
}}

.ag-st {{
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 3px;
  flex-shrink: 0;
  margin-left: auto;
}}

.ag-st.st-on {{
  background: rgba(255,255,255,0.06);
  color: var(--t1);
  border: 1px solid var(--b2);
}}

.ag-st.st-run {{
  background: rgba(150,150,150,0.06);
  color: var(--t2);
  border: 1px solid rgba(150,150,150,0.15);
}}

.ag-st.st-idle {{
  background: var(--bg-4);
  color: var(--t3);
  border: 1px solid var(--b1);
}}

.ag-st.st-err {{
  background: rgba(100,100,100,0.06);
  color: var(--t4);
  border: 1px solid rgba(100,100,100,0.15);
}}

/* ── Status Bar ────────────────────────────────────────────── */
.sbar {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 6px;
  margin-bottom: 16px;
}}

.sbar-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--t1);
  flex-shrink: 0;
  animation: pulse 2.5s ease-in-out infinite;
}}

.sbar-text {{
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--t1);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}

.sbar-meta {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  margin-left: auto;
}}

/* ── Pulse Dot ─────────────────────────────────────────────── */
.pulse-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--t1);
  animation: pulse 2.5s ease-in-out infinite;
  flex-shrink: 0;
}}

/* ── Badges ────────────────────────────────────────────────── */
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  border-radius: 3px;
}}

.badge-default {{
  background: var(--bg-4);
  color: var(--t2);
  border: 1px solid var(--b1);
}}

.badge-active {{
  background: rgba(255,255,255,0.06);
  color: var(--t1);
  border: 1px solid var(--b2);
}}

.badge-dim {{
  background: var(--bg-3);
  color: var(--t3);
  border: 1px solid var(--b0);
}}

/* Legacy badge aliases */
.v-badge {{ display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; font-family: var(--mono); font-size: 11px; font-weight: 500; border-radius: 3px; }}
.v-badge-white {{ background: rgba(255,255,255,0.06); color: var(--t1); border: 1px solid var(--b2); }}
.v-badge-gray {{ background: var(--bg-4); color: var(--t2); border: 1px solid var(--b1); }}
.v-badge-dim {{ background: var(--bg-3); color: var(--t3); border: 1px solid var(--b0); }}
.v-badge-active {{ background: rgba(255,255,255,0.06); color: var(--t1); border: 1px solid var(--b2); }}

/* ── Workspace Header ──────────────────────────────────────── */
.ws-header {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 0;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--b1);
}}

.ws-icon, .ws-header-icon {{
  font-size: 18px;
}}

.ws-title, .ws-header-title {{
  font-family: var(--sans);
  font-size: 16px;
  font-weight: 600;
  color: var(--t1);
}}

.ws-header-badge {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--t3);
  padding: 2px 8px;
  background: var(--bg-4);
  border-radius: 3px;
  margin-left: auto;
}}

/* ── AI Response Container ─────────────────────────────────── */
.ai-response-container {{
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 4px 0 16px;
  animation: fade-in 0.3s var(--ease);
}}

.ai-summary {{
  background: var(--bg-3);
  border: 1px solid var(--b2);
  border-left: 2px solid var(--t1);
  border-radius: 8px;
  padding: 16px 20px;
}}

.ai-summary-label {{
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--t2);
  margin-bottom: 6px;
}}

.ai-summary-text {{
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 500;
  color: var(--t1);
  line-height: 1.7;
}}

.ai-section {{
  background: var(--bg-2);
  border: 1px solid var(--b1);
  border-left: 2px solid var(--t4);
  border-radius: 8px;
  padding: 12px 16px;
  transition: border-color var(--dur) var(--ease);
}}

.ai-section:hover {{
  border-color: var(--b2);
}}

.ai-section-header {{
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 6px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--b0);
}}

.ai-section-icon {{
  color: var(--t2);
  margin-right: 4px;
}}

.ai-section-body {{
  font-family: var(--sans);
  font-size: 13px;
  color: var(--t2);
  line-height: 1.7;
}}

/* Section variants */
.ai-section-growth {{ border-left-color: var(--t1); }}
.ai-section-growth .ai-section-header {{ color: var(--t1); }}
.ai-section-risk {{ border-left-color: var(--t3); }}
.ai-section-risk .ai-section-header {{ color: var(--t3); }}
.ai-section-action {{ border-left-color: var(--t2); }}
.ai-section-action .ai-section-header {{ color: var(--t2); }}
.ai-section-analysis {{ border-left-color: var(--t4); }}

/* Metric chips */
.ai-metric-chip {{
  display: inline;
  padding: 1px 5px;
  font-family: var(--mono);
  font-weight: 600;
  font-size: inherit;
  border-radius: 3px;
}}

.ai-metric-positive {{
  color: var(--t1);
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--b2);
}}

.ai-metric-negative {{
  color: var(--t3);
  background: rgba(100,100,100,0.06);
  border: 1px solid rgba(100,100,100,0.15);
}}

.ai-metric-neutral {{
  color: var(--t2);
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--b1);
}}

/* Action items */
.ai-action-item {{
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 8px 12px;
  margin: 3px 0;
  background: var(--bg-3);
  border-radius: 6px;
}}

.ai-action-num {{
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--t3);
  background: var(--bg-4);
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  border-radius: 4px;
}}

.ai-action-text {{
  font-family: var(--sans);
  font-size: 13px;
  color: var(--t2);
  line-height: 1.6;
}}

/* Expert cards */
.ai-expert-card {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 12px 16px;
  margin: 6px 0;
}}

.ai-expert-header {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}}

.ai-expert-icon {{ font-size: 14px; }}

.ai-expert-name {{
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 600;
  color: var(--t1);
}}

.ai-expert-role {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--t3);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-left: auto;
}}

.ai-expert-body {{
  font-family: var(--sans);
  font-size: 13px;
  color: var(--t2);
  line-height: 1.6;
  max-height: 140px;
  overflow-y: auto;
}}

/* Inline meta */
.ai-inline-meta {{
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin: 6px 0;
}}

/* ── Thinking Timeline ─────────────────────────────────────── */
.tl-container {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 12px 16px;
  margin: 8px 0;
}}

.tl-step {{
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 4px 0;
  margin-left: 8px;
  border-left: 1px solid var(--b1);
  padding-left: 16px;
}}

.tl-dot {{
  width: 6px;
  height: 6px;
  flex-shrink: 0;
  margin-top: 6px;
  margin-left: -19px;
  border-radius: 50%;
  background: var(--t3);
}}

.tl-text {{
  font-family: var(--sans);
  font-size: 13px;
  color: var(--t2);
  line-height: 1.5;
}}

.tl-meta {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  margin-left: auto;
  flex-shrink: 0;
  font-weight: 500;
}}

/* ── Quality / HITL ────────────────────────────────────────── */
.qbadge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  border-radius: 4px;
  margin: 3px 3px 3px 0;
}}

.qbadge-pass {{
  background: rgba(255,255,255,0.06);
  color: var(--t1);
  border: 1px solid var(--b2);
}}

.qbadge-fail {{
  background: rgba(100,100,100,0.06);
  color: var(--t3);
  border: 1px solid rgba(100,100,100,0.15);
}}

.hitl-card {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 12px 16px;
  margin: 6px 0;
  display: flex;
  align-items: center;
  gap: 12px;
}}

.hitl-gauge {{
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 700;
  flex-shrink: 0;
  border-radius: 8px;
}}

.hitl-high {{ background: rgba(255,255,255,0.06); color: var(--t1); border: 1px solid var(--b3); }}
.hitl-medium {{ background: rgba(150,150,150,0.06); color: var(--t2); border: 1px solid rgba(150,150,150,0.20); }}
.hitl-low {{ background: rgba(100,100,100,0.06); color: var(--t3); border: 1px solid rgba(100,100,100,0.15); }}

.hitl-info {{ flex: 1; }}
.hitl-info .hitl-level {{ font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--t2); }}
.hitl-info .hitl-action {{ font-family: var(--sans); font-size: 13px; color: var(--t2); margin-top: 2px; }}
.hitl-triggers {{ font-family: var(--mono); font-size: 11px; color: var(--t3); text-align: right; flex-shrink: 0; }}

/* ── Trace Bar ─────────────────────────────────────────────── */
.trace-bar {{
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 14px;
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 6px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  margin: 8px 0;
}}

.trace-bar .trace-value {{
  color: var(--t1);
  font-weight: 600;
}}

/* ── Welcome / Ready State ─────────────────────────────────── */
.welcome-badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  background: var(--bg-3);
  border: 1px solid var(--b2);
  border-radius: 4px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--t2);
}}

.welcome-title {{
  font-family: var(--sans);
  font-size: 20px;
  font-weight: 600;
  color: var(--t1);
  margin-bottom: 4px;
}}

.welcome-sub {{
  font-family: var(--sans);
  font-size: 13px;
  color: var(--t3);
}}

/* ── Upload Zone ───────────────────────────────────────────── */
.upload-zone {{
  max-width: 560px;
  margin: 0 auto;
  padding: 24px;
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 8px;
}}

/* ── Glass Card ────────────────────────────────────────────── */
.glass-card {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 8px;
  padding: 16px;
  transition: border-color var(--dur) var(--ease);
}}

.glass-card:hover {{
  border-color: var(--b2);
}}

/* ── Logo ──────────────────────────────────────────────────── */
.logo-mark {{
  width: 24px;
  height: 24px;
  background: var(--t1);
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-weight: 700;
  font-size: 12px;
  color: var(--bg-1);
}}

/* ── Grid Layout ───────────────────────────────────────────── */
.bento-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 8px;
  margin: 8px 0;
}}

/* ════════════════════════════════════════════════════════════
   KEYFRAMES
   ════════════════════════════════════════════════════════════ */

@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.4; }}
}}

@keyframes fade-in {{
  from {{ opacity: 0; transform: translateY(4px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

/* ════════════════════════════════════════════════════════════
   RESPONSIVE
   ════════════════════════════════════════════════════════════ */

@media (max-width: 768px) {{
  .block-container {{ padding: 0.5rem 1rem 2rem !important; }}
  .top-nav {{ margin: -0.5rem -1rem 16px; padding: 0 12px; }}
  .cmd-title {{ font-size: 24px !important; }}
  .cmd-hero {{ padding: 24px 0 16px; }}
  .kpi-value {{ font-size: 20px; }}
}}

@media (max-width: 480px) {{
  .block-container {{ padding: 0.5rem 0.75rem 1.5rem !important; }}
  .cmd-title {{ font-size: 20px !important; }}
}}

</style>"""


# ── Login CSS ──────────────────────────────────────────────────

def _build_login_css() -> str:
    return """<style>
/* ================================================================
   MRARFAI v10 — Login
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Background ── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.stApp {
  background: #050505 !important;
}

[data-testid="stMain"],
[data-testid="stAppViewContainer"] {
  background: transparent !important;
}

/* ── Hide chrome ── */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"],
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stBottom"],
#MainMenu, footer, .stDeployButton {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  overflow: hidden !important;
}

[data-testid="stAppViewContainer"] > section > div {
  padding-top: 0 !important;
}

/* ── Card ── */
[data-testid="stMainBlockContainer"] {
  max-width: 380px !important;
  margin: 12vh auto 0 auto !important;
  background: #FFFFFF !important;
  border-radius: 12px !important;
  padding: 40px 32px 32px !important;
  box-shadow: 0 24px 80px rgba(0,0,0,0.6);
  animation: card-in 0.4s ease-out;
  position: relative;
  z-index: 1;
  overflow: hidden !important;
}

@keyframes card-in {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── Inner wrappers ── */
[data-testid="stMainBlockContainer"] .block-container,
[data-testid="stMainBlockContainer"] .stMainBlockContainer {
  max-width: 100% !important; width: 100% !important;
  padding: 0 !important; margin: 0 !important;
  background: transparent !important;
}

[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
  background: transparent !important;
  padding: 0 !important; margin: 0 !important;
}

[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
  background: transparent !important;
  padding: 0 !important; margin: 0 !important;
  gap: 2px !important;
}

[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"] {
  background: transparent !important;
  margin: 0 !important; width: 100% !important;
}

/* ── Logo area ── */
.login-logo-area {
  text-align: center;
  margin-bottom: 8px;
}

.login-logo-area img.horse-logo {
  width: 48px;
  height: auto;
  margin-bottom: 12px;
  filter: brightness(0);
}

.login-logo-area .horse-icon-fallback {
  font-size: 36px;
  margin-bottom: 8px;
  line-height: 1;
}

.login-logo-area .brand-name {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 18px;
  letter-spacing: 0.1em;
  color: #0A0A0A;
}

.login-logo-area .brand-sub {
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  font-size: 11px;
  letter-spacing: 0.06em;
  color: #999;
  margin-top: 4px;
}

/* ── Divider ── */
.login-divider {
  height: 1px;
  background: #EBEBEB;
  margin: 16px 0 20px;
}

/* ── Labels ── */
.login-label {
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  font-size: 12px;
  color: #555;
  margin: 0 0 6px 0;
  padding: 12px 0 0 0;
  line-height: 1;
  display: block;
}

/* ── Error ── */
.login-error {
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  color: #E53E3E;
  padding: 10px 14px;
  margin-top: 8px;
  border: 1px solid rgba(229,62,62,0.15);
  background: rgba(229,62,62,0.04);
  border-radius: 6px;
  text-align: center;
}

/* ── Hide Streamlit labels ── */
[data-testid="stMainBlockContainer"] .stTextInput label,
[data-testid="stMainBlockContainer"] .stTextInput > label,
[data-testid="stMainBlockContainer"] .stTextInput [data-testid="stWidgetLabel"],
[data-testid="stMainBlockContainer"] [data-testid="stWidgetLabel"] {
  display: none !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
  max-height: 0 !important;
}

[data-testid="stMainBlockContainer"] [data-testid="stTextInput"],
[data-testid="stMainBlockContainer"] .stTextInput {
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

[data-testid="stMainBlockContainer"] .stTextInput div {
  background-color: #FFFFFF !important;
  background: #FFFFFF !important;
}

[data-testid="stMainBlockContainer"] .stTextInput [data-testid="stTextInputRootElement"] {
  background: #FFFFFF !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
}

[data-testid="stMainBlockContainer"] .stTextInput [data-baseweb="input"] {
  background: #FFFFFF !important;
  border: 1px solid #E0E0E0 !important;
  border-radius: 8px !important;
  padding: 0 !important;
  box-shadow: none !important;
  transition: border-color 0.15s;
}

[data-testid="stMainBlockContainer"] .stTextInput [data-baseweb="input"]:focus-within {
  border-color: #0A0A0A !important;
  box-shadow: 0 0 0 3px rgba(0,0,0,0.04) !important;
}

[data-testid="stMainBlockContainer"] .stTextInput input {
  background: transparent !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  color: #1A1A1A !important;
  caret-color: #1A1A1A !important;
  -webkit-text-fill-color: #1A1A1A !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  padding: 10px 14px !important;
  height: auto !important;
  width: 100% !important;
}

[data-testid="stMainBlockContainer"] .stTextInput input::placeholder {
  color: #BBB !important;
  -webkit-text-fill-color: #BBB !important;
}

[data-testid="stMainBlockContainer"] .stTextInput input:focus {
  border: none !important; outline: none !important; box-shadow: none !important;
}

/* Password toggle */
[data-testid="stMainBlockContainer"] .stTextInput button {
  background: #FFFFFF !important;
  border: none !important;
  box-shadow: none !important;
  color: #AAA !important;
  padding: 4px 8px !important;
}

[data-testid="stMainBlockContainer"] .stTextInput button svg {
  fill: #AAA !important; stroke: #AAA !important;
  width: 16px !important; height: 16px !important;
}

/* Autofill */
[data-testid="stMainBlockContainer"] .stTextInput input:-webkit-autofill,
[data-testid="stMainBlockContainer"] .stTextInput input:-webkit-autofill:focus {
  -webkit-box-shadow: 0 0 0 1000px #FFFFFF inset !important;
  -webkit-text-fill-color: #1A1A1A !important;
}

/* ── Sign In button ── */
[data-testid="stMainBlockContainer"] .stButton button,
[data-testid="stMainBlockContainer"] .stButton > button,
[data-testid="stMainBlockContainer"] button[data-testid="stBaseButton-secondary"],
[data-testid="stMainBlockContainer"] button[kind="secondary"],
[data-testid="stMainBlockContainer"] .stButton button[kind] {
  width: 100% !important;
  background: #0A0A0A !important;
  color: #FFFFFF !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.02em !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0 !important;
  margin-top: 16px !important;
  cursor: pointer !important;
  height: 42px !important;
  transition: background 0.15s, transform 0.1s !important;
  box-shadow: none !important;
}

[data-testid="stMainBlockContainer"] .stButton button:hover {
  background: #222 !important;
}

[data-testid="stMainBlockContainer"] .stButton button:active {
  transform: scale(0.98) !important;
}

[data-testid="stMainBlockContainer"] .stButton,
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(.stButton) {
  margin-top: 4px !important;
}

/* ── Footer ── */
.login-links-outer {
  text-align: center;
  margin-top: 20px;
  font-family: 'Inter', sans-serif;
  font-size: 12px;
}

.login-links-outer a {
  color: rgba(255,255,255,0.3);
  text-decoration: none;
  transition: color 0.15s;
}

.login-links-outer a:hover { color: rgba(255,255,255,0.6); }
.login-links-outer .sep { color: rgba(255,255,255,0.1); margin: 0 10px; }

.login-page-footer-outer {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  color: rgba(255,255,255,0.2);
  text-align: center;
  margin-top: 12px;
}

.login-status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-top: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: rgba(255,255,255,0.25);
  letter-spacing: 0.06em;
}

.login-status .pulse-dot-sm {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: rgba(255,255,255,0.3);
  animation: login-pulse 2.5s ease-in-out infinite;
}

@keyframes login-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* ── Responsive ── */
@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"] {
    max-width: 90vw !important;
    padding: 32px 24px 28px !important;
    margin-top: 8vh !important;
  }
}

@media (max-width: 480px) {
  [data-testid="stMainBlockContainer"] {
    max-width: 96vw !important;
    padding: 28px 20px 24px !important;
    margin-top: 4vh !important;
    border-radius: 8px !important;
  }
}
</style>"""
