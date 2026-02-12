#!/usr/bin/env python3
"""
MRARFAI v9.0 — Sprocomm 禾苗 Sales Intelligence
================================================
V9.0 核心升级:
  - RLM (Recursive Language Models) 数据上下文 5K→500K+
  - LangGraph StateGraph + HITL + Reflection
  - 全新 Command Center UI (内联主题)
  - real_pipeline.py 数据管线

品牌配色: 🟢 Neon Green #00FF88  🔵 蓝叶 #00A0C8  🔴 红叶 #D94040
字体: Space Grotesk (标题) + JetBrains Mono (数据)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, tempfile
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── 数据管线: 优先 real_pipeline, 回退 analyze_clients_v2 ──
try:
    from real_pipeline import SprocommDataLoaderV2, DeepAnalyzer, ReportGeneratorV2
except ImportError:
    from analyze_clients_v2 import SprocommDataLoaderV2, DeepAnalyzer, ReportGeneratorV2

from industry_benchmark import IndustryBenchmark, generate_benchmark_section
from forecast_engine import ForecastEngine, generate_forecast_section
from ai_narrator import AINarrator, generate_narrative_section
from chat_tab import render_chat_tab
from pdf_report import render_report_section
from health_score import render_health_dashboard
from anomaly_detector import render_anomaly_dashboard
from brand_config import render_brand_settings, get_brand

# ── 微信通知 (可选) ──
try:
    from wechat_notify import render_notification_settings
    HAS_WECHAT = True
except ImportError:
    HAS_WECHAT = False

MONTHS = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']

# ============================================================
# Sprocomm 禾苗配色系统 — Command Center
# ============================================================
SP_GREEN = "#00FF88"   # ⚡ Neon Green — 主色/活跃/CTA
SP_BLUE  = "#00A0C8"   # 🔵 蓝叶 — 信息/分析/数据
SP_RED   = "#D94040"   # 🔴 红叶 — 风险/预警/危险
BRAND_GREEN = "#8CBF3F" # 原始品牌绿
ACCENT = SP_GREEN
CYAN   = SP_BLUE
GREEN  = SP_GREEN
RED    = SP_RED
ORANGE = "#FF8800"
PURPLE = "#8b5cf6"
TEXT1  = "#FFFFFF"
TEXT2  = "#8a8a8a"
CHART_COLORS = [SP_GREEN, SP_BLUE, "#3b82f6", ORANGE, SP_RED, "#ec4899", PURPLE, "#06b6d4"]
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.04)"

def plotly_layout(title="", height=400, showlegend=True):
    return dict(
        title=dict(text=title, font=dict(size=11, color=TEXT2, family="JetBrains Mono"), x=0),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT2, size=11, family="JetBrains Mono"),
        height=height, showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, family="JetBrains Mono")),
        margin=dict(l=50, r=20, t=36, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(size=10, family="JetBrains Mono"), zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(size=10, family="JetBrains Mono"), zeroline=False),
    )

def fmt(v, unit="万"):
    if v is None: return "-"
    try:
        v = float(v)
        if abs(v) >= 100: return f"{v:,.0f}{unit}"
        elif abs(v) >= 1: return f"{v:,.1f}{unit}"
        else: return f"{v:.2f}{unit}"
    except: return str(v)


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(page_title="Sprocomm AI · MRARFAI v9.0", page_icon="🌿", layout="wide")


# ============================================================
# 内联主题 — Command Center (替代 ui_theme.py)
# ============================================================
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-deep: #0C0C0C; --bg-base: #080808; --bg-elevated: #111111;
    --bg-overlay: #1a1a1a; --bg-glass: rgba(12,12,12,0.85);
    --border-subtle: #2f2f2f; --border-default: #2f2f2f;
    --border-hover: rgba(0,255,136,0.30);
    --text-1: #FFFFFF; --text-2: #8a8a8a; --text-3: #6a6a6a;
    --neon: #00FF88; --sp-green: #00FF88; --sp-blue: #00A0C8; --sp-red: #D94040;
    --warn: #FF8800; --radius-sm: 0px; --radius-md: 0px; --radius-lg: 0px;
    --font-sans: 'Space Grotesk', -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
}

#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.stApp { background: var(--bg-deep) !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1600px; }

[data-testid="stSidebar"] { background: var(--bg-base) !important; border-right: 1px solid var(--border-subtle) !important; }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMarkdown span {
    font-family: var(--font-mono) !important; color: var(--text-2) !important; font-size: 0.78rem !important;
}
.sidebar-label {
    font-family: var(--font-mono) !important; font-size: 0.6rem !important; font-weight: 700 !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important; color: var(--text-3) !important;
    padding: 1.2rem 0 0.4rem !important; border-top: 1px solid var(--border-subtle); margin-top: 0.8rem;
}

.stMarkdown p { font-family: var(--font-mono) !important; color: var(--text-1) !important; line-height: 1.65 !important; font-size: 0.85rem !important; }
h1, h2, h3 { font-family: var(--font-sans) !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--bg-base) !important; gap: 0 !important; border-bottom: 1px solid var(--border-subtle) !important; padding: 0 !important; overflow-x: auto; scrollbar-width: none; }
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-3) !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important; padding: 0.65rem 1rem !important;
    font-family: var(--font-mono) !important; font-size: 0.7rem !important; font-weight: 500 !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important; white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-2) !important; background: rgba(255,255,255,0.03) !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--neon) !important; border-bottom-color: var(--neon) !important; background: var(--bg-deep) !important; }

.stButton > button {
    background: var(--bg-elevated) !important; color: var(--text-1) !important; border: 1px solid var(--border-default) !important;
    border-radius: 0 !important; font-family: var(--font-mono) !important; font-size: 0.75rem !important;
    font-weight: 600 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important;
}
.stButton > button:hover { border-color: var(--neon) !important; background: rgba(0,255,136,0.06) !important; color: var(--neon) !important; }

.stTextInput input, .stTextArea textarea, .stSelectbox > div > div, .stNumberInput input {
    background: var(--bg-elevated) !important; color: var(--text-1) !important;
    border: 1px solid var(--border-default) !important; border-radius: 0 !important;
    font-family: var(--font-mono) !important; font-size: 0.82rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--neon) !important; box-shadow: 0 0 0 2px rgba(0,255,136,0.10) !important; }

[data-testid="stMetric"] { background: var(--bg-elevated) !important; border: 1px solid var(--border-subtle) !important; padding: 0.8rem 1rem !important; }
[data-testid="stMetric"] label { font-family: var(--font-mono) !important; font-size: 0.6rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--text-3) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: var(--font-sans) !important; font-weight: 700 !important; color: var(--text-1) !important; }

[data-testid="stExpander"] { background: var(--bg-elevated) !important; border: 1px solid var(--border-subtle) !important; border-radius: 0 !important; }
[data-testid="stExpander"] summary { font-family: var(--font-mono) !important; font-size: 0.78rem !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border-subtle) !important; border-radius: 0 !important; }
.stDataFrame th { background: var(--bg-elevated) !important; }

[data-testid="stChatInput"] { background: var(--bg-base) !important; border-top: 1px solid var(--border-subtle) !important; }
[data-testid="stChatInput"] textarea { background: var(--bg-elevated) !important; color: var(--text-1) !important; border: 1px solid var(--border-default) !important; border-radius: 0 !important; font-family: var(--font-mono) !important; }
[data-testid="stChatInput"] textarea:focus { border-color: var(--neon) !important; box-shadow: 0 0 0 2px rgba(0,255,136,0.10), 0 0 30px rgba(0,255,136,0.08) !important; }
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0.8rem 0 !important; }

.section-header { font-family: var(--font-mono); font-size: 0.58rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3); padding-bottom: 0.4rem; border-bottom: 1px solid var(--border-subtle); margin: 1.5rem 0 0.5rem; }
.status-bar { display: flex; align-items: center; gap: 10px; padding: 10px 18px; background: rgba(0,255,136,0.04); border: 1px solid rgba(0,255,136,0.15); margin-bottom: 14px; }
.status-bar .status-dot { width: 6px; height: 6px; background: var(--neon); border-radius: 50%; animation: neon-pulse 2s ease-in-out infinite; }
.status-bar .status-text { font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; color: var(--neon); letter-spacing: 0.08em; }
.status-bar .status-meta { font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-3); margin-left: auto; }

.agent-card { background: var(--bg-elevated); border: 1px solid var(--border-subtle); padding: 0.8rem 1rem; margin: 0.3rem 0; transition: border-color 0.15s; }
.agent-card:hover { border-color: rgba(0,255,136,0.25); }

@keyframes neon-pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
</style>""", unsafe_allow_html=True)


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    # Command Center Logo
    st.markdown(f"""
    <div style="padding:6px 0 14px 0;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:32px; height:32px; background:{SP_GREEN}; display:flex;
                 align-items:center; justify-content:center; flex-shrink:0;">
                <span style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                      font-size:0.85rem; color:#0C0C0C;">S</span>
            </div>
            <div>
                <div style="font-size:0.88rem; font-weight:700; color:#FFFFFF;
                     letter-spacing:0.1em; font-family:'Space Grotesk',sans-serif;
                     text-transform:uppercase;">SPROCOMM</div>
                <div style="font-size:0.5rem; color:#6a6a6a; font-family:'JetBrains Mono',monospace;
                     letter-spacing:0.1em; text-transform:uppercase;">MRARFAI v9.0 · RLM Engine</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Data section
    st.markdown('<div class="sidebar-label">DATA</div>', unsafe_allow_html=True)
    rev_file = st.file_uploader("金额报表 (.xlsx)", type=['xlsx'], key='rev', label_visibility="collapsed")
    if rev_file: st.caption(f"✓ {rev_file.name}")
    else: st.caption("拖入金额报表 .xlsx")

    qty_file = st.file_uploader("数量报表 (.xlsx)", type=['xlsx'], key='qty', label_visibility="collapsed")
    if qty_file: st.caption(f"✓ {qty_file.name}")
    else: st.caption("拖入数量报表 .xlsx")

    st.divider()

    # AI Engine section
    st.markdown('<div class="sidebar-label">AI ENGINE</div>', unsafe_allow_html=True)
    ai_enabled = st.toggle("启用 AI 叙事", value=False)
    if ai_enabled:
        ai_provider = st.selectbox("模型", ['DeepSeek', 'Claude'], label_visibility="collapsed")
        api_key = st.text_input("API Key", type="password", label_visibility="collapsed", placeholder="sk-...")
    else:
        ai_provider, api_key = 'DeepSeek', None

    st.session_state["ai_provider"] = ai_provider
    st.session_state["api_key"] = api_key or ""

    st.divider()

    # Multi-Agent section
    st.markdown('<div class="sidebar-label">MULTI-AGENT</div>', unsafe_allow_html=True)
    use_multi = st.toggle("启用 Multi-Agent", value=False, key="use_multi_agent")
    if use_multi:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:6px; padding:6px 10px;
             background:rgba(0,255,136,0.06); border:1px solid rgba(0,255,136,0.15);
             margin-top:4px;">
            <div style="width:5px; height:5px; border-radius:50%; background:{SP_GREEN};
                 animation:neon-pulse 2s ease-in-out infinite;"></div>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.58rem;
                  color:#6a6a6a; letter-spacing:0.05em;">V9 AGENTS [ACTIVE] · RLM · HITL</span>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div style="text-align:center; opacity:0.3; font-size:0.5rem; color:#6a6a6a;
         margin-top:40px; font-family:'JetBrains Mono',monospace;
         letter-spacing:0.1em; text-transform:uppercase;">
        SPROCOMM · 01401.HK<br>MRARFAI v9.0 · 36K+ lines
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 欢迎页 (未上传数据时)
# ============================================================
if not rev_file or not qty_file:
    st.markdown(f"""
    <div style="text-align:center; padding:50px 0 28px 0;">
        <div style="margin-bottom:20px;">
            <span style="display:inline-flex; align-items:center; gap:8px;
                padding:6px 16px;
                background:rgba(0,255,136,0.06); border:1px solid rgba(0,255,136,0.25);
                font-size:0.62rem; color:{SP_GREEN}; font-weight:700;
                letter-spacing:0.1em; font-family:'JetBrains Mono',monospace;
                text-transform:uppercase;">
                <span style="width:6px;height:6px;border-radius:50%;background:{SP_GREEN};"></span>
                V9.0 · RLM MULTI-AGENT INTELLIGENCE
            </span>
        </div>
        <h1 style="font-size:3rem; font-weight:700; color:{SP_GREEN}; letter-spacing:-2px;
            margin:0; line-height:1.1; font-family:'Space Grotesk',sans-serif;">
            SPROCOMM
        </h1>
        <h1 style="font-size:3rem; font-weight:700; color:#FFFFFF; letter-spacing:-2px;
            margin:0; line-height:1.1; font-family:'Space Grotesk',sans-serif;">
            SALES INTELLIGENCE
        </h1>
        <p style="color:#8a8a8a; font-size:0.82rem; margin-top:16px; max-width:500px;
           margin-left:auto; margin-right:auto; line-height:1.6;
           font-family:'JetBrains Mono',monospace;">
            // 多智能体协作 · RLM递归语言模型 · 500K+上下文 · 实时预警系统
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="agent-card" style="border-left:2px solid {SP_GREEN};
            display:block; text-align:left; padding:1.2rem;">
            <div style="width:36px;height:36px;background:rgba(0,255,136,0.08);
                 display:flex;align-items:center;justify-content:center;margin-bottom:12px;">
                <span style="color:{SP_GREEN};font-size:1.1rem;">◈</span>
            </div>
            <h4 style="color:#FFFFFF;font-family:'Space Grotesk',sans-serif;font-size:0.9rem;
                letter-spacing:0.03em;margin:0 0 8px 0;">RLM MULTI-AGENT</h4>
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8a8a8a;
               line-height:1.5;margin:0;">Route → Experts → Synthesize → Reflect → HITL</p>
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:rgba(0,255,136,0.5);
               margin:8px 0 0 0;letter-spacing:0.03em;">// 36,000+ LINES · 26 MODULES</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="agent-card" style="border-left:2px solid {SP_BLUE};
            display:block; text-align:left; padding:1.2rem;">
            <div style="width:36px;height:36px;background:rgba(0,160,200,0.08);
                 display:flex;align-items:center;justify-content:center;margin-bottom:12px;">
                <span style="color:{SP_BLUE};font-size:1.1rem;">◇</span>
            </div>
            <h4 style="color:#FFFFFF;font-family:'Space Grotesk',sans-serif;font-size:0.9rem;
                letter-spacing:0.03em;margin:0 0 8px 0;">12-DIMENSION ANALYTICS</h4>
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8a8a8a;
               line-height:1.5;margin:0;">客户·价量·预警·增长·产品·区域</p>
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:rgba(0,160,200,0.5);
               margin:8px 0 0 0;letter-spacing:0.03em;">// CONTEXT WINDOW 500K+ CHARS</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="agent-card" style="border-left:2px solid {SP_RED};
            display:block; text-align:left; padding:1.2rem;">
            <div style="width:36px;height:36px;background:rgba(217,64,64,0.08);
                 display:flex;align-items:center;justify-content:center;margin-bottom:12px;">
                <span style="color:{SP_RED};font-size:1.1rem;">◆</span>
            </div>
            <h4 style="color:#FFFFFF;font-family:'Space Grotesk',sans-serif;font-size:0.9rem;
                letter-spacing:0.03em;margin:0 0 8px 0;">5-LAYER GUARDRAILS</h4>
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8a8a8a;
               line-height:1.5;margin:0;">输入过滤·Prompt注入·幻觉检测</p>
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:rgba(217,64,64,0.5);
               margin:8px 0 0 0;letter-spacing:0.03em;">// 99.5% SECURITY PASS</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center; margin-top:28px;">
        <p style="color:#6a6a6a; font-size:0.75rem; font-family:'JetBrains Mono',monospace;">
            ← UPLOAD <strong style="color:{SP_GREEN};">金额报表</strong> &
            <strong style="color:{SP_BLUE};">数量报表</strong> TO BEGIN
        </p>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ============================================================
# 数据加载
# ============================================================
@st.cache_data(show_spinner=False)
def run_full_analysis(rev_bytes, qty_bytes):
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f1:
        f1.write(rev_bytes); rp = f1.name
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f2:
        f2.write(qty_bytes); qp = f2.name
    loader = SprocommDataLoaderV2(rp, qp)
    data = loader.load_all()
    analyzer = DeepAnalyzer(data)
    results = analyzer.run_all()
    bench = IndustryBenchmark(data, results).run()
    forecast = ForecastEngine(data, results).run()
    os.unlink(rp); os.unlink(qp)
    return data, results, bench, forecast

with st.spinner("🌿 数据加载 + 深度分析中..."):
    data, results, benchmark, forecast = run_full_analysis(rev_file.read(), qty_file.read())

active = sum(1 for c in data['客户金额'] if c['年度金额'] > 0)
st.markdown(f"""
<div class="status-bar">
    <div class="status-dot"></div>
    <span class="status-text">DATA LOADED</span>
    <span class="status-meta">{active} clients · 12 dimensions · V9.0 RLM · {datetime.now().strftime('%H:%M:%S')}</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Tabs — V9.0 布局
# ============================================================
tabs = st.tabs([
    "🧠 Agent", "📊 总览", "👥 客户分析", "💰 价量分解", "🚨 预警中心",
    "📈 增长机会", "🏭 产品结构", "🌍 区域分析",
    "🌐 行业对标", "🔮 预测", "✏️ CEO备忘录",
    "❤️ 健康评分", "🔬 异常检测", "🔔 通知推送", "🎨 品牌设置", "📥 导出",
])


# ---- Tab 0: Agent Chat ----
with tabs[0]:
    render_chat_tab(data, results, benchmark, forecast, ai_provider, api_key)


# ---- Tab 1: 总览 ----
with tabs[1]:
    yoy = data['总YoY']
    qs = data['数量汇总']
    high_risk = [a for a in results['流失预警'] if '高' in a['风险']]
    hr_amt = sum(a['年度金额'] for a in high_risk)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("全年营收", f"{data['总营收']:,.0f}万", f"+{yoy['增长率']*100:.1f}% YoY")
    c2.metric("出货量", f"{qs['全年实际']/10000:,.0f}万台", f"完成率 {qs['全年实际']/qs['全年计划']*100:.0f}%")
    c3.metric("活跃客户", f"{active}家")
    c4.metric("高风险", f"{len(high_risk)}家", f"涉及 {hr_amt:,.0f}万")
    c5.metric("增长机会", f"{len(results['增长机会'])}个")

    st.markdown("")
    findings = results['核心发现']
    st.markdown('<div class="section-header">KEY FINDINGS</div>', unsafe_allow_html=True)
    fcols = st.columns(min(len(findings), 3))
    for i, f in enumerate(findings):
        with fcols[i % len(fcols)]:
            st.markdown(f'<div class="agent-card"><p>{f}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    m_data = data['月度总营收']
    col1, col2 = st.columns(2)
    with col1:
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=MONTHS, y=m_data,
                marker=dict(color=[SP_GREEN if v == max(m_data) else "rgba(140,191,63,0.30)" for v in m_data]),
                text=[f"{v:,.0f}" for v in m_data], textposition="outside", textfont=dict(size=10, color=TEXT2),
            ))
            fig.update_layout(**plotly_layout("月度营收趋势（万元）", 380, False))
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if HAS_PLOTLY:
            cat_data = results['类别趋势']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[c['类别'] for c in cat_data], y=[c['2025金额'] for c in cat_data],
                name="2025", marker_color=SP_GREEN, text=[f"{c['2025金额']:,.0f}" for c in cat_data],
                textposition="outside", textfont=dict(size=10)))
            fig.add_trace(go.Bar(x=[c['类别'] for c in cat_data], y=[c['2024金额'] for c in cat_data],
                name="2024", marker_color="rgba(100,116,139,0.3)"))
            fig.update_layout(**plotly_layout("业务类别 YoY", 380), barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    q = [sum(m_data[i:i+3]) for i in range(0, 12, 3)]
    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.metric("Q1", f"{q[0]:,.0f}")
    qc2.metric("Q2", f"{q[1]:,.0f}")
    qc3.metric("Q3", f"{q[2]:,.0f}", "峰值季度")
    qc4.metric("Q4", f"{q[3]:,.0f}", f"{(q[3]/q[2]-1)*100:+.1f}%")

    with st.expander("📋 业务类别同比明细"):
        cat_df = pd.DataFrame(results['类别趋势'])
        for col in ['2025金额', '2024金额', '增长额']:
            if col in cat_df.columns:
                cat_df[col] = cat_df[col].apply(lambda x: round(float(x)) if pd.notna(x) else 0)
        st.dataframe(cat_df, use_container_width=True, hide_index=True,
            column_config={'2025金额': st.column_config.NumberColumn(format="%,d"),
                '2024金额': st.column_config.NumberColumn(format="%,d"),
                '增长额': st.column_config.NumberColumn(format="%,d")})


# ---- Tab 2: 客户分析 ----
with tabs[2]:
    tiers = results['客户分级']
    tier_counts = {t: sum(1 for x in tiers if x['等级']==t) for t in ['A','B','C']}
    tier_rev = {t: sum(x['年度金额'] for x in tiers if x['等级']==t) for t in ['A','B','C']}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"A级 · {tier_counts['A']}家", f"{tier_rev['A']:,.0f}万", f"占比 {tier_rev['A']/data['总营收']*100:.1f}%")
    c2.metric(f"B级 · {tier_counts['B']}家", f"{tier_rev['B']:,.0f}万")
    c3.metric(f"C级 · {tier_counts['C']}家", f"{tier_rev['C']:,.0f}万")
    c4.metric("Top4 集中度", f"{tiers[3]['累计占比']}%", "⚠️ 偏高" if tiers[3]['累计占比']>50 else "✅ 健康")

    filter_tier = st.multiselect("筛选等级", ['A','B','C'], default=['A','B','C'])
    filtered = [t for t in tiers if t['等级'] in filter_tier]
    tier_df = pd.DataFrame(filtered)
    for col in ['年度金额', 'H1', 'H2']:
        if col in tier_df.columns:
            tier_df[col] = tier_df[col].apply(lambda x: round(float(x)) if pd.notna(x) else 0)
    st.dataframe(tier_df, use_container_width=True, hide_index=True,
        column_config={'年度金额': st.column_config.NumberColumn(format="%,d"),
            'H1': st.column_config.NumberColumn(format="%,d"),
            'H2': st.column_config.NumberColumn(format="%,d")})

    if HAS_PLOTLY:
        st.markdown('<div class="section-header">CONCENTRATION CURVE</div>', unsafe_allow_html=True)
        cum = [t['累计占比'] for t in tiers]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cum)+1)), y=cum, mode='lines+markers',
            line=dict(color=SP_GREEN, width=2), marker=dict(size=5, color=SP_GREEN),
            fill='tozeroy', fillcolor='rgba(0,255,136,0.06)'))
        fig.add_hline(y=80, line_dash="dash", line_color=ORANGE,
            annotation_text="80% 帕累托线", annotation_font=dict(size=10, color=ORANGE))
        fig.update_layout(**plotly_layout("客户集中度曲线", 350, False))
        fig.update_xaxes(title_text="客户排名")
        fig.update_yaxes(title_text="累计占比 (%)")
        st.plotly_chart(fig, use_container_width=True)


# ---- Tab 3: 价量分解 ----
with tabs[3]:
    pv = results['价量分解']
    if pv:
        # 统计
        quality_counts = {}
        for p in pv:
            q = p.get('质量评估', '未知')
            quality_counts[q] = quality_counts.get(q, 0) + 1

        st.markdown('<div class="section-header">PRICE-VOLUME DECOMPOSITION</div>', unsafe_allow_html=True)
        pv_df = pd.DataFrame(pv)
        st.dataframe(pv_df, use_container_width=True, hide_index=True)

        if HAS_PLOTLY:
            fig = go.Figure()
            for i, p in enumerate(pv[:10]):
                color = SP_GREEN if '优质' in p.get('质量评估', '') else (SP_RED if '齐跌' in p.get('质量评估', '') else SP_BLUE)
                fig.add_trace(go.Bar(
                    name=p['客户'], x=[p['客户']], y=[p.get('年度金额', 0)],
                    marker_color=color, showlegend=False,
                    text=[f"{p.get('年度金额', 0):,.0f}"], textposition="outside"))
            fig.update_layout(**plotly_layout("客户价量分布", 380, False))
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab 4: 预警中心 ----
with tabs[4]:
    alerts = results['流失预警']
    high_risk = [a for a in alerts if '高' in a['风险']]
    med_risk = [a for a in alerts if '中' in a['风险']]

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 高风险", f"{len(high_risk)}家", f"涉及 {sum(a['年度金额'] for a in high_risk):,.0f}万")
    c2.metric("🟡 中风险", f"{len(med_risk)}家")
    c3.metric("总预警", f"{len(alerts)}家")

    st.markdown('<div class="section-header">HIGH RISK CLIENTS</div>', unsafe_allow_html=True)
    for a in high_risk:
        st.error(f"🔴 **{a['客户']}** — ¥{a['年度金额']:,.0f}万 — {a.get('原因', a.get('风险', ''))}")
    for a in med_risk:
        st.warning(f"🟡 **{a['客户']}** — ¥{a['年度金额']:,.0f}万 — {a.get('原因', a.get('风险', ''))}")


# ---- Tab 5: 增长机会 ----
with tabs[5]:
    growth = results['增长机会']
    st.markdown(f'<div class="section-header">GROWTH OPPORTUNITIES · {len(growth)} FOUND</div>', unsafe_allow_html=True)
    for g in growth:
        with st.expander(f"📈 **{g.get('客户', '未知')}** — {g.get('类型', '')} — {g.get('说明', '')}", expanded=False):
            for k, v in g.items():
                if k not in ('客户',):
                    st.markdown(f"**{k}**: {v}")


# ---- Tab 6: 产品结构 ----
with tabs[6]:
    pm = data.get('产品结构', data.get('类别YoY', []))
    if pm:
        st.markdown('<div class="section-header">PRODUCT MIX</div>', unsafe_allow_html=True)
        pm_df = pd.DataFrame(pm)
        st.dataframe(pm_df, use_container_width=True, hide_index=True)

        if HAS_PLOTLY:
            fig = go.Figure()
            for i, p in enumerate(pm):
                fig.add_trace(go.Bar(
                    x=[p['类别']], y=[p['2025金额']],
                    marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                    name=p['类别'],
                    text=[f"{p['2025金额']:,.0f}"], textposition="outside"))
            fig.update_layout(**plotly_layout("2025 产品结构（万元）", 380))
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab 7: 区域分析 ----
with tabs[7]:
    reg = results['区域洞察']
    c1, c2, c3 = st.columns(3)
    c1.metric("覆盖区域", f"{len(reg['详细'])}个")
    c2.metric("Top3 集中度", f"{reg['Top3集中度']}%")
    c3.metric("HHI", f"{reg['赫芬达尔指数']}", "⚠️ 高度集中" if reg['赫芬达尔指数']>2500 else "✅")
    if reg['赫芬达尔指数'] > 2500:
        st.warning(f"⚠️ HHI={reg['赫芬达尔指数']}（>2500），区域依赖风险")
    st.dataframe(pd.DataFrame(reg['详细']), use_container_width=True, hide_index=True)
    if HAS_PLOTLY:
        regions = reg['详细']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[r['区域'] for r in regions], y=[r['金额'] for r in regions],
            marker_color=[SP_GREEN if i == 0 else "rgba(140,191,63,0.25)" for i in range(len(regions))],
            text=[f"{r['金额']:,.0f}" for r in regions], textposition="outside", textfont=dict(size=10)))
        fig.update_layout(**plotly_layout("区域出货分布（万元）", 380, False))
        st.plotly_chart(fig, use_container_width=True)


# ---- Tab 8: 行业对标 ----
with tabs[8]:
    st.caption("数据来源：IDC / Counterpoint / 公司年报")
    mp = benchmark['市场定位']
    for k, v in mp.items():
        st.markdown(f'<div class="agent-card"><h4>{k}</h4><p>{v}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    cb = benchmark['竞争对标']
    comp_data = []
    for name in ['华勤', '闻泰', '龙旗', '禾苗']:
        comp_data.append({'公司': f"{'→ ' if name=='禾苗' else ''}{name}",
            '营收(亿)': cb['营收'].get(name, '-'), '增速': cb['增速'].get(name, '-'),
            '毛利率': cb['毛利率'].get(name, '-')})
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    st.info(f"📊 客户集中度：{cb['客户集中度']}")

    st.markdown("")
    for v in benchmark['客户外部视角']:
        with st.expander(f"**{v['客户']}**", expanded=v['客户']=='HMD'):
            if '外部' in v: st.markdown(f"🌐 **外部趋势**：{v['外部']}")
            if '禾苗' in v: st.markdown(f"📊 **禾苗表现**：{v['禾苗']}")
            st.markdown(f"🎯 **判断**：{v['判断']}")
            if '根因' in v: st.error(f"🔍 **根因分析**：{v['根因']}")

    col1, col2 = st.columns(2)
    with col1:
        for r in benchmark['结构性风险']:
            with st.expander(f"🔴 {r['风险']}"):
                st.markdown(f"**行业**：{r['行业']}\n\n**禾苗**：{r['禾苗']}")
                st.success(f"→ {r['建议']}")
    with col2:
        for o in benchmark['战略机会']:
            with st.expander(f"🚀 {o['机会']}（{o['数据']}）"):
                st.markdown(f"**行业**：{o['行业']}")
                st.success(f"→ {o['行动']}")


# ---- Tab 9: 预测 ----
with tabs[9]:
    t = forecast['总营收预测']
    c1, c2, c3 = st.columns(3)
    c1.metric("Q1 乐观", f"{t['置信区间']['乐观(+15%)']:,.0f}万")
    c2.metric("Q1 基准", f"{t['置信区间']['基准']:,.0f}万", "⬅️ 核心预测")
    c3.metric("Q1 悲观", f"{t['置信区间']['悲观(-15%)']:,.0f}万")
    st.caption(f"参考：Q1 2025 {t['参考']['Q1_2025实际']:,.0f}万 | Q4 2025 {t['参考']['Q4_2025实际']:,.0f}万")

    with st.expander("🔍 预测方法"):
        for k, v in t['方法说明'].items():
            st.markdown(f"- **{k}**：{v}")

    st.markdown("")
    cp_df = pd.DataFrame(forecast['客户预测'])
    for col in ['Q4实际', 'Q1预测']:
        if col in cp_df.columns:
            cp_df[col] = pd.to_numeric(cp_df[col], errors='coerce').round(0)
    st.dataframe(cp_df, use_container_width=True, hide_index=True,
        column_config={'Q4实际': st.column_config.NumberColumn(format="%,.0f"),
            'Q1预测': st.column_config.NumberColumn(format="%,.0f")})

    with st.expander("📋 品类预测 2026E"):
        st.dataframe(pd.DataFrame(forecast['品类预测']), use_container_width=True, hide_index=True)

    st.markdown("")
    scenarios = forecast['风险场景']
    if HAS_PLOTLY:
        names = list(scenarios.keys())
        values = [scenarios[n]['全年预测'] for n in names]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[n.split('(')[0] for n in names], y=values,
            marker_color=[SP_GREEN, SP_BLUE, ORANGE, SP_RED],
            text=[f"{v/10000:.1f}亿" for v in values],
            textposition="outside", textfont=dict(size=13, color=TEXT2)))
        fig.update_layout(**plotly_layout("2026 情景预测", 400, False))
        st.plotly_chart(fig, use_container_width=True)
    cols = st.columns(4)
    for i, (name, sc) in enumerate(scenarios.items()):
        with cols[i]:
            st.metric(name.split('(')[0], f"{sc['全年预测']/10000:.1f}亿")
            st.caption(sc['假设'])


# ---- Tab 10: CEO备忘录 ----
with tabs[10]:
    if ai_enabled and api_key:
        if st.button("🧠 用AI生成深度叙事", type="primary", use_container_width=True):
            narrator = AINarrator(data, results, benchmark, forecast)
            with st.spinner("AI 分析中..."):
                ai_text = narrator.generate(api_key, ai_provider.lower())
            st.markdown(ai_text)
            st.download_button("📥 下载", ai_text, "ai_memo.md", "text/markdown")
    narrator = AINarrator(data, results, benchmark, forecast)
    memo = narrator._template_narrative()
    with st.expander("📄 内置战略备忘录", expanded=not ai_enabled):
        st.markdown(memo)


# ---- Tab 11: 健康评分 ----
with tabs[11]:
    health_scores = render_health_dashboard(data, results)


# ---- Tab 12: 异常检测 ----
with tabs[12]:
    st.caption("基于统计模型 (Z-Score · IQR · 趋势断裂 · 波动率 · 系统性风险)")
    render_anomaly_dashboard(data, results)


# ---- Tab 13: 通知推送 ----
with tabs[13]:
    if HAS_WECHAT:
        _hs = health_scores if 'health_scores' in dir() and health_scores else None
        render_notification_settings(results, _hs)
    else:
        st.info("微信通知模块未加载 — 请确保 wechat_notify.py 在项目目录中")


# ---- Tab 14: 品牌设置 ----
with tabs[14]:
    render_brand_settings()


# ---- Tab 15: 导出 ----
with tabs[15]:
    render_report_section(data, results, benchmark, forecast)

    st.markdown("")
    st.markdown("#### 其他格式")
    gen = ReportGeneratorV2(data, results)
    base_report = gen.generate()
    bench_section = generate_benchmark_section(benchmark)
    forecast_section = generate_forecast_section(forecast)
    narrator = AINarrator(data, results, benchmark, forecast)
    memo = narrator._template_narrative()
    footer = "\n---\n> MRARFAI 销售分析"
    if footer in base_report:
        parts = base_report.split(footer)
        full = parts[0] + bench_section + forecast_section + memo + footer + parts[1]
    else:
        full = base_report + bench_section + forecast_section + memo
    full = full.replace("Agent v2.0", "Agent v9.0").replace("智能分析系统 v2.0", "智能分析系统 v9.0")
    full = full.replace("Agent v4.0", "Agent v9.0").replace("Agent v8.0", "Agent v9.0")
    now = datetime.now().strftime('%Y%m%d')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📄 完整报告", full, f"禾苗销售分析_{now}.md", "text/markdown", use_container_width=True)
    with c2:
        json_all = json.dumps({'分析': results, '行业': benchmark, '预测': forecast},
            ensure_ascii=False, indent=2, default=str)
        st.download_button("📊 JSON数据", json_all, f"analysis_{now}.json", "application/json", use_container_width=True)
    with c3:
        st.download_button("🤖 AI Prompt", gen.generate_ai_prompt(), "ai_prompt.txt", "text/plain", use_container_width=True)
    with st.expander("📖 报告预览"):
        st.markdown(full)
