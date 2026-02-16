"""
MRARFAI v10 — Design System
============================
Single Source of Truth for all visual styling.

Generated via ui-ux-pro-max skill design system.
References: Linear, Vercel, Cursor, Raycast
Typography: JetBrains Mono (headings/data) + IBM Plex Sans (body)
Palette: Pure OLED black #000000 + grays + #FAFAFA
Style: Dark Mode (OLED) — WCAG AAA
"""

import streamlit as st

# ── Design Tokens ──────────────────────────────────────────────

COLORS = {
    # Backgrounds — OLED pure black base (skill: Dark Mode OLED)
    "bg_void":     "#000000",
    "bg_deep":     "#000000",     # skill: --bg-black: #000000
    "bg_base":     "#0A0A0A",
    "bg_elevated": "#121212",     # skill: --bg-dark-grey: #121212
    "bg_surface":  "#18181B",     # skill: Primary: #18181B
    "bg_overlay":  "#1E1E1E",
    "bg_glass":    "rgba(0,0,0,0.85)",

    # Borders
    "border_ghost":   "rgba(255,255,255,0.04)",
    "border_subtle":  "rgba(255,255,255,0.07)",
    "border_default": "rgba(255,255,255,0.10)",
    "border_hover":   "rgba(255,255,255,0.18)",
    "border_active":  "rgba(255,255,255,0.30)",

    # Text — WCAG AAA: 7:1+ contrast on #000000
    "text_primary":   "#FAFAFA",  # skill: Text: #FAFAFA
    "text_secondary": "#A1A1AA",  # 7.4:1 on #000
    "text_tertiary":  "#71717A",  # 4.6:1 on #000 (AA compliant)
    "text_muted":     "#52525B",  # decorative only
    "text_ghost":     "#3F3F46",  # decorative only

    # Semantic
    "accent":         "#FAFAFA",  # skill: CTA: #F8FAFC
    "accent_dim":     "rgba(255,255,255,0.06)",
    "accent_glow":    "rgba(255,255,255,0.12)",
    "status_active":  "#FAFAFA",
    "status_ready":   "#71717A",
    "status_warning": "#A1A1AA",
    "status_error":   "#52525B",
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
   MRARFAI v10 Design System — Major Visual Overhaul
   Ref: Linear · Vercel · Cursor · Raycast · Arc
   Type: JetBrains Mono + IBM Plex Sans
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

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

  --sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-out: cubic-bezier(0.33, 1, 0.68, 1);
  --dur: 200ms;
  --dur-slow: 300ms;
  color-scheme: dark;
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

/* ── Base — Noise texture + dot grid ──────────────────────── */
.stApp {{
  background: var(--bg-1) !important;
  color: var(--t2) !important;
  position: relative;
}}

.stApp::before {{
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    radial-gradient(rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size: 24px 24px;
  pointer-events: none;
  z-index: 0;
}}

.block-container {{
  padding: 1rem 2.5rem 4rem !important;
  max-width: 1400px !important;
  position: relative;
  z-index: 1;
}}

/* ── Typography ────────────────────────────────────────────── */
.stMarkdown p {{
  font-family: var(--sans) !important;
  color: var(--t2) !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  letter-spacing: -0.01em !important;
}}

h1 {{
  font-family: var(--sans) !important;
  font-weight: 700 !important;
  color: var(--t1) !important;
  letter-spacing: -0.04em !important;
  font-size: 28px !important;
  line-height: 1.2 !important;
}}

h2 {{
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  color: var(--t1) !important;
  letter-spacing: -0.03em !important;
  font-size: 20px !important;
}}

h3, h4 {{
  font-family: var(--mono) !important;
  font-weight: 500 !important;
  color: var(--t3) !important;
  font-size: 11px !important;
  letter-spacing: 0.1em !important;
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

/* ── Dividers ─────────────────────────────────────────────── */
hr, [data-testid="stHorizontalBlock"] + hr {{
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--b2), transparent) !important;
  margin: 24px 0 !important;
}}

/* ── Tabs — Floating pill style ───────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  background: var(--bg-2) !important;
  gap: 2px !important;
  border-bottom: none !important;
  border: 1px solid var(--b1) !important;
  border-radius: 10px !important;
  padding: 3px !important;
  overflow-x: auto;
  scrollbar-width: none;
  width: fit-content;
}}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}

.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--t3) !important;
  border: none !important;
  border-bottom: none !important;
  border-radius: 8px !important;
  padding: 8px 16px !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  white-space: nowrap !important;
  transition: all var(--dur) var(--ease) !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
  color: var(--t1) !important;
  background: var(--bg-3) !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {{
  color: var(--t1) !important;
  background: var(--bg-4) !important;
  border-bottom: none !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}}

/* Hide tab highlight bar */
.stTabs [data-baseweb="tab-highlight"] {{
  display: none !important;
}}

/* ── Buttons — Glow on hover ──────────────────────────────── */
.stButton > button {{
  background: var(--bg-3) !important;
  color: var(--t1) !important;
  border: 1px solid var(--b2) !important;
  border-radius: 8px !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 18px !important;
  transition: all var(--dur) var(--ease) !important;
  cursor: pointer !important;
  position: relative;
}}

.stButton > button:hover {{
  background: var(--bg-4) !important;
  border-color: var(--b3) !important;
  box-shadow: 0 0 20px rgba(255,255,255,0.04), 0 4px 12px rgba(0,0,0,0.3) !important;
  transform: translateY(-1px) !important;
}}

.stButton > button:active {{
  transform: translateY(0) scale(0.98) !important;
  box-shadow: none !important;
}}

/* ── Inputs — Glowing focus ───────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div,
.stNumberInput input {{
  background: var(--bg-2) !important;
  color: var(--t1) !important;
  border: 1px solid var(--b1) !important;
  border-radius: 8px !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
  caret-color: var(--t1) !important;
  transition: all var(--dur) var(--ease) !important;
}}

.stTextInput input:focus,
.stTextArea textarea:focus {{
  border-color: var(--b4) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.06), 0 0 20px rgba(255,255,255,0.03) !important;
}}

/* ── Chat — Refined message cards ─────────────────────────── */
[data-testid="stChatInput"] {{
  background: var(--bg-2) !important;
  border-top: 1px solid var(--b1) !important;
}}

[data-testid="stChatInput"] textarea {{
  background: var(--bg-3) !important;
  color: var(--t1) !important;
  border: 1px solid var(--b1) !important;
  border-radius: 10px !important;
  font-family: var(--sans) !important;
  font-size: 14px !important;
  transition: all var(--dur) var(--ease) !important;
}}

[data-testid="stChatInput"] textarea:focus {{
  border-color: var(--b3) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.04) !important;
}}

[data-testid="stChatMessage"] {{
  background: transparent !important;
  border: none !important;
  padding: 20px 0 !important;
  border-bottom: 1px solid var(--b0) !important;
  animation: msg-in 0.3s var(--ease-out);
}}

@keyframes msg-in {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

[data-testid="stChatMessage"] p {{
  font-family: var(--sans) !important;
  font-size: 14px !important;
  line-height: 1.75 !important;
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

/* ── Metrics — Elevated cards ─────────────────────────────── */
[data-testid="stMetric"] {{
  background: var(--bg-3) !important;
  border: 1px solid var(--b1) !important;
  border-radius: 12px !important;
  padding: 20px 24px !important;
  transition: all var(--dur) var(--ease) !important;
  position: relative;
  overflow: hidden;
}}

[data-testid="stMetric"]::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}}

[data-testid="stMetric"]:hover {{
  border-color: var(--b3) !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
  transform: translateY(-2px);
}}

[data-testid="stMetricLabel"],
[data-testid="stMetric"] label {{
  font-family: var(--sans) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--t3) !important;
}}

[data-testid="stMetricValue"] {{
  font-family: var(--mono) !important;
  font-weight: 700 !important;
  color: var(--t1) !important;
  letter-spacing: -0.02em !important;
  font-size: 28px !important;
}}

[data-testid="stMetricDelta"] {{
  font-family: var(--mono) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}}

/* ── Expanders — Sleek ────────────────────────────────────── */
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
  border-radius: 10px !important;
  transition: all var(--dur) var(--ease) !important;
  overflow: hidden;
}}

[data-testid="stExpander"]:hover {{
  border-color: var(--b2) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
}}

/* ── File Uploader ─────────────────────────────────────────── */
[data-testid="stFileUploader"],
.stFileUploader {{
  border: 2px dashed var(--b2) !important;
  border-radius: 12px !important;
  transition: all var(--dur) var(--ease) !important;
  background: var(--bg-2) !important;
}}

[data-testid="stFileUploader"]:hover,
.stFileUploader:hover {{
  border-color: var(--b3) !important;
  background: var(--bg-3) !important;
  box-shadow: 0 0 30px rgba(255,255,255,0.02) !important;
}}

/* ── DataFrames ────────────────────────────────────────────── */
.stDataFrame {{
  border-radius: 10px !important;
}}

.stDataFrame [data-testid="stDataFrameContainer"] {{
  border: 1px solid var(--b1) !important;
  border-radius: 10px !important;
}}

/* ── Status Container ──────────────────────────────────────── */
[data-testid="stStatus"] {{
  margin: 8px 0 !important;
}}

[data-testid="stStatus"] p {{
  font-size: 13px !important;
  line-height: 1.6 !important;
}}

/* ── Scrollbars — Thin and elegant ────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--b2); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--b3); }}

/* ================================================================
   COMPONENTS
   ================================================================ */

/* ── Top Nav — Frosted glass ──────────────────────────────── */
.top-nav {{
  display: flex;
  align-items: center;
  gap: 16px;
  height: 52px;
  padding: 0 24px;
  background: rgba(0,0,0,0.7);
  backdrop-filter: blur(16px) saturate(1.5);
  -webkit-backdrop-filter: blur(16px) saturate(1.5);
  border-bottom: 1px solid var(--b1);
  margin: -1rem -2.5rem 28px;
  position: relative;
}}

.top-nav::after {{
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.08) 50%, transparent 100%);
}}

.top-nav-brand {{
  font-family: var(--mono);
  font-weight: 700;
  font-size: 14px;
  color: var(--t1);
  letter-spacing: 0.06em;
}}

.top-nav-version {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t4);
  padding: 2px 8px;
  background: var(--bg-4);
  border-radius: 4px;
  border: 1px solid var(--b1);
}}

.top-nav-user {{
  font-family: var(--sans);
  font-size: 12px;
  color: var(--t2);
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 10px;
}}

.top-nav-role {{
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  color: var(--t2);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 10px;
  background: var(--bg-4);
  border: 1px solid var(--b2);
  border-radius: 20px;
}}

/* ── Command Center — Hero with gradient text ─────────────── */
.cmd-hero {{
  padding: 56px 0 40px;
  max-width: 680px;
  position: relative;
}}

.cmd-title {{
  font-family: var(--sans) !important;
  font-size: 36px !important;
  font-weight: 700 !important;
  color: transparent !important;
  background: linear-gradient(135deg, #FFFFFF 0%, #888888 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  letter-spacing: -0.04em !important;
  line-height: 1.15 !important;
  margin: 0 0 12px !important;
}}

.cmd-sub {{
  font-family: var(--sans);
  font-size: 15px;
  color: var(--t3);
  line-height: 1.6;
}}

.cmd-badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 14px;
  background: var(--bg-3);
  border: 1px solid var(--b2);
  border-radius: 20px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--t2);
  margin-top: 20px;
  transition: all var(--dur) var(--ease);
}}

.cmd-badge:hover {{
  border-color: var(--b3);
  background: var(--bg-4);
}}

/* ── Section Header — With gradient line ──────────────────── */
.sec-hdr {{
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--t3);
  padding-bottom: 10px;
  margin: 32px 0 16px;
  border-bottom: 1px solid var(--b1);
  position: relative;
}}

.sec-hdr::after {{
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 60px;
  height: 1px;
  background: linear-gradient(90deg, var(--t3), transparent);
}}

/* ── KPI Card — Elevated with gradient top edge ───────────── */
.kpi-card {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 12px;
  padding: 20px 24px;
  transition: all var(--dur) var(--ease);
  position: relative;
  overflow: hidden;
}}

.kpi-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}}

.kpi-card:hover {{
  border-color: var(--b3);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  transform: translateY(-2px);
}}

.kpi-label {{
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 10px;
}}

.kpi-value {{
  font-family: var(--mono);
  font-size: 28px;
  font-weight: 700;
  color: var(--t1);
  line-height: 1;
  letter-spacing: -0.03em;
}}

.kpi-icon {{
  font-size: 16px;
  margin-bottom: 10px;
  opacity: 0.4;
}}

.kpi-sub {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  margin-top: 8px;
}}

.kpi-delta {{
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  margin-top: 6px;
}}
.kpi-up {{ color: var(--t1); }}
.kpi-down {{ color: var(--t3); }}

/* ── Agent Cards — Glassmorphism with glow ────────────────── */
.ag-card {{
  background: rgba(18,18,18,0.6);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--b1);
  border-radius: 12px;
  padding: 18px 20px;
  display: flex;
  align-items: center;
  gap: 14px;
  transition: all var(--dur) var(--ease);
  cursor: pointer;
  margin-bottom: 8px;
  position: relative;
  overflow: hidden;
}}

.ag-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  opacity: 0;
  transition: opacity var(--dur) var(--ease);
}}

.ag-card:hover {{
  border-color: var(--b3);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05);
  transform: translateY(-2px);
}}

.ag-card:hover::before {{
  opacity: 1;
}}

.ag-av {{
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
  background: var(--bg-4);
  border: 1px solid var(--b1);
  border-radius: 10px;
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
  letter-spacing: -0.01em;
}}

.ag-role {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t4);
  margin-top: 3px;
}}

.ag-desc {{
  font-family: var(--sans);
  font-size: 12px;
  color: var(--t3);
  line-height: 1.5;
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
  padding: 4px 10px;
  border-radius: 20px;
  flex-shrink: 0;
  margin-left: auto;
  letter-spacing: 0.04em;
}}

.ag-st.st-on {{
  background: rgba(255,255,255,0.08);
  color: var(--t1);
  border: 1px solid var(--b2);
  box-shadow: 0 0 12px rgba(255,255,255,0.04);
}}

.ag-st.st-run {{
  background: rgba(150,150,150,0.08);
  color: var(--t2);
  border: 1px solid rgba(150,150,150,0.15);
}}

.ag-st.st-idle {{
  background: var(--bg-4);
  color: var(--t3);
  border: 1px solid var(--b1);
}}

.ag-st.st-err {{
  background: rgba(100,100,100,0.08);
  color: var(--t4);
  border: 1px solid rgba(100,100,100,0.15);
}}

/* ── Status Bar — Gradient border ─────────────────────────── */
.sbar {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 10px;
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
}}

.sbar::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}}

.sbar-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--t1);
  flex-shrink: 0;
  animation: pulse 2s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(255,255,255,0.2);
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

/* ── Pulse Dot — With glow ────────────────────────────────── */
.pulse-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--t1);
  animation: pulse 2s ease-in-out infinite;
  box-shadow: 0 0 8px rgba(255,255,255,0.2);
  flex-shrink: 0;
}}

/* ── Badges — Pill style ──────────────────────────────────── */
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  border-radius: 20px;
}}

.badge-default {{
  background: var(--bg-4);
  color: var(--t2);
  border: 1px solid var(--b1);
}}

.badge-active {{
  background: rgba(255,255,255,0.08);
  color: var(--t1);
  border: 1px solid var(--b2);
}}

.badge-dim {{
  background: var(--bg-3);
  color: var(--t3);
  border: 1px solid var(--b0);
}}

/* Legacy badge aliases */
.v-badge {{ display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; font-family: var(--mono); font-size: 11px; font-weight: 500; border-radius: 20px; }}
.v-badge-white {{ background: rgba(255,255,255,0.08); color: var(--t1); border: 1px solid var(--b2); }}
.v-badge-gray {{ background: var(--bg-4); color: var(--t2); border: 1px solid var(--b1); }}
.v-badge-dim {{ background: var(--bg-3); color: var(--t3); border: 1px solid var(--b0); }}
.v-badge-active {{ background: rgba(255,255,255,0.08); color: var(--t1); border: 1px solid var(--b2); }}

/* ── Workspace Header ──────────────────────────────────────── */
.ws-header {{
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 0;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--b1);
}}

.ws-icon, .ws-header-icon {{
  font-size: 20px;
}}

.ws-title, .ws-header-title {{
  font-family: var(--sans);
  font-size: 17px;
  font-weight: 600;
  color: var(--t1);
  letter-spacing: -0.02em;
}}

.ws-header-badge {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--t3);
  padding: 3px 10px;
  background: var(--bg-4);
  border-radius: 20px;
  border: 1px solid var(--b1);
  margin-left: auto;
}}

/* ── AI Response Container ─────────────────────────────────── */
.ai-response-container {{
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin: 8px 0 20px;
  animation: fade-in 0.3s var(--ease);
}}

.ai-summary {{
  background: var(--bg-3);
  border: 1px solid var(--b2);
  border-left: 3px solid var(--t1);
  border-radius: 12px;
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
}}

.ai-summary::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, rgba(255,255,255,0.15), transparent);
}}

.ai-summary-label {{
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--t2);
  margin-bottom: 8px;
}}

.ai-summary-text {{
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 500;
  color: var(--t1);
  line-height: 1.75;
}}

.ai-section {{
  background: var(--bg-2);
  border: 1px solid var(--b1);
  border-left: 3px solid var(--t4);
  border-radius: 10px;
  padding: 14px 18px;
  transition: all var(--dur) var(--ease);
}}

.ai-section:hover {{
  border-color: var(--b2);
  transform: translateX(2px);
}}

.ai-section-header {{
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 8px;
  padding-bottom: 6px;
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
  line-height: 1.75;
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
  padding: 2px 7px;
  font-family: var(--mono);
  font-weight: 600;
  font-size: inherit;
  border-radius: 4px;
}}

.ai-metric-positive {{
  color: var(--t1);
  background: rgba(255,255,255,0.08);
  border: 1px solid var(--b2);
}}

.ai-metric-negative {{
  color: var(--t3);
  background: rgba(100,100,100,0.08);
  border: 1px solid rgba(100,100,100,0.15);
}}

.ai-metric-neutral {{
  color: var(--t2);
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--b1);
}}

/* Action items */
.ai-action-item {{
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 10px 14px;
  margin: 4px 0;
  background: var(--bg-3);
  border-radius: 8px;
  transition: background var(--dur) var(--ease);
}}

.ai-action-item:hover {{
  background: var(--bg-4);
}}

.ai-action-num {{
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 700;
  color: var(--t3);
  background: var(--bg-4);
  width: 22px;
  height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  border-radius: 6px;
  border: 1px solid var(--b1);
}}

.ai-action-text {{
  font-family: var(--sans);
  font-size: 13px;
  color: var(--t2);
  line-height: 1.65;
}}

/* Expert cards */
.ai-expert-card {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 12px;
  padding: 16px 20px;
  margin: 8px 0;
  transition: all var(--dur) var(--ease);
}}

.ai-expert-card:hover {{
  border-color: var(--b2);
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}}

.ai-expert-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}}

.ai-expert-icon {{ font-size: 16px; }}

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
  padding: 2px 8px;
  background: var(--bg-4);
  border-radius: 20px;
  border: 1px solid var(--b1);
}}

.ai-expert-body {{
  font-family: var(--sans);
  font-size: 13px;
  color: var(--t2);
  line-height: 1.65;
  max-height: 140px;
  overflow-y: auto;
}}

/* Inline meta */
.ai-inline-meta {{
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin: 8px 0;
}}

/* ── Thinking Timeline — Refined ──────────────────────────── */
.tl-container {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 12px;
  padding: 16px 20px;
  margin: 10px 0;
}}

.tl-step {{
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 6px 0;
  margin-left: 8px;
  border-left: 1px solid var(--b1);
  padding-left: 18px;
  transition: border-color var(--dur) var(--ease);
}}

.tl-step:hover {{
  border-left-color: var(--b3);
}}

.tl-dot {{
  width: 7px;
  height: 7px;
  flex-shrink: 0;
  margin-top: 6px;
  margin-left: -22px;
  border-radius: 50%;
  background: var(--t3);
  border: 1.5px solid var(--bg-3);
  box-shadow: 0 0 0 1px var(--b2);
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
  color: var(--t4);
  margin-left: auto;
  flex-shrink: 0;
  font-weight: 500;
}}

/* ── Quality / HITL ────────────────────────────────────────── */
.qbadge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  border-radius: 20px;
  margin: 3px 3px 3px 0;
}}

.qbadge-pass {{
  background: rgba(255,255,255,0.08);
  color: var(--t1);
  border: 1px solid var(--b2);
}}

.qbadge-fail {{
  background: rgba(100,100,100,0.08);
  color: var(--t3);
  border: 1px solid rgba(100,100,100,0.15);
}}

.hitl-card {{
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 12px;
  padding: 14px 18px;
  margin: 8px 0;
  display: flex;
  align-items: center;
  gap: 14px;
  transition: all var(--dur) var(--ease);
}}

.hitl-card:hover {{
  border-color: var(--b2);
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}}

.hitl-gauge {{
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 700;
  flex-shrink: 0;
  border-radius: 10px;
}}

.hitl-high {{ background: rgba(255,255,255,0.08); color: var(--t1); border: 1px solid var(--b3); }}
.hitl-medium {{ background: rgba(150,150,150,0.08); color: var(--t2); border: 1px solid rgba(150,150,150,0.20); }}
.hitl-low {{ background: rgba(100,100,100,0.08); color: var(--t3); border: 1px solid rgba(100,100,100,0.15); }}

.hitl-info {{ flex: 1; }}
.hitl-info .hitl-level {{ font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--t2); }}
.hitl-info .hitl-action {{ font-family: var(--sans); font-size: 13px; color: var(--t2); margin-top: 3px; }}
.hitl-triggers {{ font-family: var(--mono); font-size: 11px; color: var(--t3); text-align: right; flex-shrink: 0; }}

/* ── Trace Bar ─────────────────────────────────────────────── */
.trace-bar {{
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 16px;
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 10px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--t3);
  margin: 10px 0;
}}

.trace-bar .trace-value {{
  color: var(--t1);
  font-weight: 600;
}}

/* ── Welcome / Ready State ─────────────────────────────────── */
.welcome-badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 14px;
  background: var(--bg-3);
  border: 1px solid var(--b2);
  border-radius: 20px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--t2);
}}

.welcome-title {{
  font-family: var(--sans);
  font-size: 22px;
  font-weight: 700;
  color: var(--t1);
  margin-bottom: 6px;
  letter-spacing: -0.03em;
}}

.welcome-sub {{
  font-family: var(--sans);
  font-size: 14px;
  color: var(--t3);
  line-height: 1.6;
}}

/* ── Upload Zone — With border gradient ───────────────────── */
.upload-zone {{
  max-width: 560px;
  margin: 0 auto;
  padding: 28px;
  background: var(--bg-3);
  border: 1px solid var(--b1);
  border-radius: 12px;
  position: relative;
  overflow: hidden;
}}

.upload-zone::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}}

/* ── Glass Card — Real glassmorphism ──────────────────────── */
.glass-card {{
  background: rgba(18,18,18,0.6);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--b1);
  border-radius: 12px;
  padding: 18px;
  transition: all var(--dur) var(--ease);
  position: relative;
  overflow: hidden;
}}

.glass-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
}}

.glass-card:hover {{
  border-color: var(--b2);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}}

/* ── Logo ──────────────────────────────────────────────────── */
.logo-mark {{
  width: 28px;
  height: 28px;
  background: var(--t1);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-weight: 700;
  font-size: 13px;
  color: var(--bg-1);
}}

/* ── Grid Layout ───────────────────────────────────────────── */
.bento-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  margin: 10px 0;
}}

/* ════════════════════════════════════════════════════════════
   KEYFRAMES
   ════════════════════════════════════════════════════════════ */

@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.3; }}
}}

@keyframes fade-in {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes glow {{
  0%, 100% {{ box-shadow: 0 0 8px rgba(255,255,255,0.05); }}
  50% {{ box-shadow: 0 0 20px rgba(255,255,255,0.08); }}
}}

@keyframes gradient-shift {{
  0% {{ background-position: 0% 50%; }}
  50% {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}

/* ════════════════════════════════════════════════════════════
   SKELETON / SHIMMER LOADING
   ════════════════════════════════════════════════════════════ */

@keyframes shimmer {{
  0% {{ background-position: -200px 0; }}
  100% {{ background-position: calc(200px + 100%) 0; }}
}}

.skeleton {{
  background: linear-gradient(90deg, var(--bg-3) 25%, var(--bg-4) 50%, var(--bg-3) 75%);
  background-size: 200px 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 8px;
}}

.skeleton-text {{
  height: 14px;
  margin-bottom: 8px;
  border-radius: 4px;
}}

.skeleton-card {{
  height: 120px;
  border-radius: 12px;
}}

/* ════════════════════════════════════════════════════════════
   FOCUS VISIBLE — ACCESSIBILITY
   ════════════════════════════════════════════════════════════ */

*:focus-visible {{
  outline: 2px solid rgba(255,255,255,0.30) !important;
  outline-offset: 2px !important;
  border-radius: 4px;
}}

button:focus-visible,
[role="button"]:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible {{
  outline: 2px solid rgba(255,255,255,0.30) !important;
  outline-offset: 2px !important;
}}

/* ════════════════════════════════════════════════════════════
   REDUCED MOTION — ACCESSIBILITY
   ════════════════════════════════════════════════════════════ */

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }}
  .pulse-dot {{ animation: none !important; opacity: 1 !important; }}
  .skeleton {{ animation: none !important; background: var(--bg-4) !important; }}
  .stApp::before {{ display: none; }}
}}

/* ════════════════════════════════════════════════════════════
   CURSOR & INTERACTIVITY
   ════════════════════════════════════════════════════════════ */

button,
[role="button"],
.stButton > button,
.stDownloadButton > button,
[data-baseweb="tab"],
.ag-card,
.kpi-card,
.glass-card,
.hitl-card,
a {{
  cursor: pointer !important;
}}

/* ════════════════════════════════════════════════════════════
   RESPONSIVE
   ════════════════════════════════════════════════════════════ */

@media (max-width: 768px) {{
  .block-container {{ padding: 0.5rem 1rem 2rem !important; }}
  .top-nav {{ margin: -0.5rem -1rem 16px; padding: 0 12px; height: 48px; }}
  .cmd-title {{ font-size: 26px !important; }}
  .cmd-hero {{ padding: 32px 0 20px; }}
  .kpi-value {{ font-size: 22px; }}
  .stApp::before {{ background-size: 20px 20px; }}
}}

@media (max-width: 480px) {{
  .block-container {{ padding: 0.5rem 0.75rem 1.5rem !important; }}
  .cmd-title {{ font-size: 22px !important; }}
  .stApp::before {{ display: none; }}
}}

</style>"""


# ── Login CSS ──────────────────────────────────────────────────

def _build_login_css() -> str:
    return """<style>
/* ================================================================
   MRARFAI v10 — Login
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Background ── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.stApp {
  background: #000000 !important;
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
  animation: card-in 0.3s ease-out;
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
  font-family: 'IBM Plex Sans', sans-serif;
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
  font-family: 'IBM Plex Sans', sans-serif;
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
  font-family: 'IBM Plex Sans', sans-serif;
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
  font-family: 'IBM Plex Sans', sans-serif !important;
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
  font-family: 'IBM Plex Sans', sans-serif !important;
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
  font-family: 'IBM Plex Sans', sans-serif;
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
  font-family: 'IBM Plex Sans', sans-serif;
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
  font-size: 11px;
  color: rgba(255,255,255,0.25);
  letter-spacing: 0.06em;
}

.login-status .pulse-dot-sm {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: rgba(255,255,255,0.3);
  animation: login-pulse 2s ease-in-out infinite;
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
