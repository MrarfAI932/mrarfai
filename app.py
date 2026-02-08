#!/usr/bin/env python3
"""
MRARFAI v4.0 — Sales Intelligence Agent
========================================
全面升级UI：参考 ChatGPT / Perplexity / Linear 设计语言
Agent-first · 自包含 · 无额外依赖
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

from analyze_clients_v2 import SprocommDataLoaderV2, DeepAnalyzer, ReportGeneratorV2
from industry_benchmark import IndustryBenchmark, generate_benchmark_section
from forecast_engine import ForecastEngine, generate_forecast_section
from ai_narrator import AINarrator, generate_narrative_section
from chat_tab import render_chat_tab
from pdf_report import render_report_section
from health_score import render_health_dashboard
from wechat_notify import render_notification_settings

MONTHS = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']

# ============================================================
# 配色（全部内置，不需要 theme.py）
# ============================================================
ACCENT = "#6366f1"
CYAN = "#22d3ee"
GREEN = "#10b981"
RED = "#ef4444"
ORANGE = "#f59e0b"
PURPLE = "#a855f7"
TEXT1 = "#f1f5f9"
TEXT2 = "#94a3b8"
COLORS = [ACCENT, CYAN, GREEN, ORANGE, PURPLE, RED, "#ec4899", "#14b8a6"]
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(99,102,241,0.08)"

def plotly_layout(title="", height=400, showlegend=True):
    return dict(
        title=dict(text=title, font=dict(size=14, color=TEXT2), x=0),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT2, size=12),
        height=height, showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(size=11), zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(size=11), zeroline=False),
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
st.set_page_config(page_title="MRARFAI · Sales Agent", page_icon="🧠", layout="wide")

# ============================================================
# 全局样式
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    .block-container { padding-top: 1rem; max-width: 1440px; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.04));
        padding: 20px 24px; border-radius: 16px;
        border: 1px solid rgba(99,102,241,0.1);
        transition: all 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(99,102,241,0.25);
        box-shadow: 0 12px 40px rgba(99,102,241,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important; font-size: 0.8rem; font-weight: 500;
        letter-spacing: 0.5px; text-transform: uppercase;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important; font-weight: 700; font-size: 1.7rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.82rem; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0c18, #111127);
        border-right: 1px solid rgba(99,102,241,0.08);
    }
    section[data-testid="stSidebar"] * { color: #94a3b8; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; background: rgba(99,102,241,0.04);
        border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 8px 18px;
        font-size: 0.82rem; font-weight: 500; color: #64748b;
    }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(99,102,241,0.08); color: #a5b4fc; }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.15) !important;
        color: #a5b4fc !important; font-weight: 600;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .streamlit-expanderHeader { font-weight: 600; font-size: 0.9rem; border-radius: 10px; }
    hr { border-color: rgba(99,102,241,0.08) !important; }

    .agent-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.03));
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 16px; padding: 20px 24px; margin: 8px 0;
        transition: all 0.3s;
    }
    .agent-card:hover { border-color: rgba(99,102,241,0.25); transform: translateY(-2px); }
    .agent-card h4 { color: #a5b4fc; margin: 0 0 8px 0; font-size: 0.92rem; font-weight: 600; }
    .agent-card p { color: #94a3b8; margin: 0; font-size: 0.85rem; line-height: 1.7; }

    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 14px; border-radius: 20px;
        background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.15);
        font-size: 0.75rem; color: #a5b4fc; font-weight: 500;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
        margin: 24px 0 16px 0; display: flex; align-items: center; gap: 10px;
    }
    .section-header .icon {
        width: 32px; height: 32px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; background: rgba(99,102,241,0.1);
    }
    .stButton button { border-radius: 10px; font-weight: 500; border: 1px solid rgba(99,102,241,0.15); }
    .stButton button:hover { border-color: rgba(99,102,241,0.4); }
    .stDownloadButton button {
        background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.15); border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px 0;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:36px; height:36px; border-radius:10px;
                 background:linear-gradient(135deg, #6366f1, #8b5cf6);
                 display:flex; align-items:center; justify-content:center; font-size:1.2rem;">🧠</div>
            <div>
                <div style="font-size:1.05rem; font-weight:700; color:#f1f5f9;">MRARFAI</div>
                <div style="font-size:0.7rem; color:#64748b;">Sales Intelligence Agent</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<p style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:1px;">📁 数据上传</p>', unsafe_allow_html=True)
    rev_file = st.file_uploader("金额报表 (.xlsx)", type=['xlsx'], key='rev', label_visibility="collapsed")
    if rev_file: st.caption(f"✓ {rev_file.name}")
    else: st.caption("拖入金额报表 .xlsx")

    qty_file = st.file_uploader("数量报表 (.xlsx)", type=['xlsx'], key='qty', label_visibility="collapsed")
    if qty_file: st.caption(f"✓ {qty_file.name}")
    else: st.caption("拖入数量报表 .xlsx")

    st.divider()
    st.markdown('<p style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:1px;">🤖 AI 引擎</p>', unsafe_allow_html=True)
    ai_enabled = st.toggle("启用 AI 叙事", value=False)
    if ai_enabled:
        ai_provider = st.selectbox("模型", ['DeepSeek', 'Claude'], label_visibility="collapsed")
        api_key = st.text_input("API Key", type="password", label_visibility="collapsed", placeholder="sk-...")
    else:
        ai_provider, api_key = 'DeepSeek', None

    st.markdown("""
    <div style="text-align:center; opacity:0.3; font-size:0.7rem; color:#64748b; margin-top:40px;">
        Sprocomm 禾苗通讯 · 01401.HK<br>Powered by MRARFAI v4.0
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 欢迎页
# ============================================================
if not rev_file or not qty_file:
    st.markdown("""
    <div style="text-align:center; padding:80px 0 40px 0;">
        <div style="margin-bottom:20px;">
            <span class="hero-badge">✨ v4.0 · Agent-Powered Analytics</span>
        </div>
        <h1 style="font-size:2.8rem; font-weight:800; color:#f1f5f9; letter-spacing:-1px; margin:0; line-height:1.2;">
            Sales Intelligence<br>
            <span style="background:linear-gradient(135deg, #6366f1, #a855f7, #22d3ee);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Agent</span>
        </h1>
        <p style="color:#64748b; font-size:1.05rem; margin-top:16px; max-width:500px; margin-left:auto; margin-right:auto; line-height:1.6;">
            上传禾苗通讯销售数据，用自然语言对话<br>获取深度洞察与战略建议
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="agent-card"><h4>🧠 对话式分析</h4><p>用中文提问，Agent 自动选择分析工具，理解数据含义，给出专业建议</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="agent-card"><h4>📊 12维深度分析</h4><p>客户分级 · 流失预警 · 价量分解 · 行业对标 · 预测引擎 · CEO备忘录</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="agent-card"><h4>🔮 智能预测</h4><p>Q1 2026 营收预测 · 情景分析 · 客户级别预测 · AI 战略叙事</p></div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; margin-top:40px;"><p style="color:#475569; font-size:0.88rem;">👈 在左侧上传<strong style="color:#a5b4fc;">金额报表</strong>和<strong style="color:#a5b4fc;">数量报表</strong>开始分析</p></div>', unsafe_allow_html=True)
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

with st.spinner("⚡ 数据加载 + 深度分析中..."):
    data, results, benchmark, forecast = run_full_analysis(rev_file.read(), qty_file.read())

active = sum(1 for c in data['客户金额'] if c['年度金额'] > 0)
st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; padding:12px 20px;
     background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.12);
     border-radius:12px; margin-bottom:16px;">
    <span style="font-size:1.1rem;">✅</span>
    <span style="color:#10b981; font-weight:600; font-size:0.88rem;">v4.0 全套分析完成</span>
    <span style="color:#475569; font-size:0.8rem; margin-left:auto;">{active}家活跃客户 · 12维分析</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "🧠 Agent", "📊 总览", "👥 客户分析", "💰 价量分解", "🚨 预警中心",
    "📈 增长机会", "🏭 产品结构", "🌍 区域分析",
    "🌐 行业对标", "🔮 预测", "✍️ CEO备忘录",
    "❤️ 健康评分", "🔔 通知推送", "📥 导出",
])

# ---- Tab 0: Agent ----
with tabs[0]:
    render_chat_tab(data, results, benchmark, forecast)

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
    st.markdown('<div class="section-header"><div class="icon">💡</div> 核心发现</div>', unsafe_allow_html=True)
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
                marker=dict(color=[ACCENT if v == max(m_data) else "rgba(99,102,241,0.35)" for v in m_data]),
                text=[f"{v:,.0f}" for v in m_data], textposition="outside", textfont=dict(size=10, color=TEXT2),
            ))
            fig.update_layout(**plotly_layout("月度营收趋势（万元）", 380, False))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame({'月份': MONTHS, '金额': m_data}).set_index('月份'))
    with col2:
        if HAS_PLOTLY:
            cat_data = results['类别趋势']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[c['类别'] for c in cat_data], y=[c['2025金额'] for c in cat_data],
                name="2025", marker_color=ACCENT, text=[f"{c['2025金额']:,.0f}" for c in cat_data],
                textposition="outside", textfont=dict(size=10)))
            fig.add_trace(go.Bar(x=[c['类别'] for c in cat_data], y=[c['2024金额'] for c in cat_data],
                name="2024", marker_color="rgba(100,116,139,0.3)"))
            fig.update_layout(**plotly_layout("业务类别 YoY", 380), barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame(results['类别趋势'])[['类别','2025金额','2024金额']].set_index('类别'))

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

    st.markdown("")
    st.markdown('<div class="section-header"><div class="icon">📈</div> 集中度曲线</div>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        top15 = tiers[:15]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[t['客户'] for t in top15], y=[t['累计占比'] for t in top15],
            mode='lines+markers+text', text=[f"{t['累计占比']}%" for t in top15],
            textposition="top center", textfont=dict(size=9, color=TEXT2),
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=8, color=ACCENT, line=dict(width=2, color='white')),
            fill='tozeroy', fillcolor='rgba(99,102,241,0.05)'))
        fig.add_hline(y=80, line_dash="dash", line_color=ORANGE, annotation_text="80%线")
        fig.update_layout(**plotly_layout("Top15 累计营收占比", 360, False))
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    st.markdown('<div class="section-header"><div class="icon">🔍</div> 单客户趋势</div>', unsafe_allow_html=True)
    selected = st.selectbox("选择客户", [t['客户'] for t in tiers[:20]])
    sel_data = next((c for c in data['客户金额'] if c['客户'] == selected), None)
    if sel_data and HAS_PLOTLY:
        vals = sel_data['月度金额']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=MONTHS, y=vals,
            marker=dict(color=[ACCENT if v == max(vals) else "rgba(99,102,241,0.35)" for v in vals]),
            text=[f"{v:,.0f}" for v in vals], textposition="outside", textfont=dict(size=10)))
        fig.update_layout(**plotly_layout(f"{selected} · 月度营收（万元）", 350, False))
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: 价量分解 ----
with tabs[3]:
    st.markdown('<div class="section-header"><div class="icon">💰</div> 价量分解</div>', unsafe_allow_html=True)
    st.caption("单价 = 出货金额 ÷ 出货数量 → 判断增长质量")
    pv = results['价量分解']
    if not pv:
        st.warning("无法计算（需要金额+数量匹配）")
    else:
        quality_map = {}
        for p in pv:
            q = p['质量评估']
            if '优质' in q: k = '✅ 优质增长'
            elif '以价补量' in q: k = '⚠️ 以价补量'
            elif '量换价' in q: k = '⚠️ 以量换价'
            elif '齐跌' in q: k = '❌ 量价齐跌'
            else: k = '→ 价格稳定'
            quality_map[k] = quality_map.get(k, 0) + 1
        cols = st.columns(len(quality_map))
        for i, (k, v) in enumerate(quality_map.items()):
            cols[i].metric(k, f"{v}家")

        st.markdown("")
        pv_df = pd.DataFrame(pv)
        for col in ['年度金额', '年度数量', '均价(元)', 'H1均价', 'H2均价']:
            if col in pv_df.columns:
                pv_df[col] = pd.to_numeric(pv_df[col], errors='coerce').round(1)
        display_cols = [c for c in ['客户','年度金额','年度数量','均价(元)','H1均价','H2均价','价格变动','质量评估'] if c in pv_df.columns]
        st.dataframe(pv_df[display_cols], use_container_width=True, hide_index=True,
            column_config={'年度金额': st.column_config.NumberColumn(format="%,.0f"),
                '年度数量': st.column_config.NumberColumn(format="%,.0f"),
                '均价(元)': st.column_config.NumberColumn(format="%,.1f"),
                'H1均价': st.column_config.NumberColumn(format="%,.1f"),
                'H2均价': st.column_config.NumberColumn(format="%,.1f")})

        st.markdown("")
        st.markdown('<div class="section-header"><div class="icon">📉</div> Top5 单价趋势</div>', unsafe_allow_html=True)
        if HAS_PLOTLY:
            fig = go.Figure()
            for idx, p in enumerate(pv[:5]):
                prices = p.get('月度单价', [])
                if len(prices) == 12:
                    clean = [v if v and v > 0 else None for v in prices]
                    fig.add_trace(go.Scatter(x=MONTHS, y=clean, name=p['客户'],
                        mode='lines+markers', line=dict(color=COLORS[idx], width=2), marker=dict(size=5)))
            fig.update_layout(**plotly_layout("月度单价走势（元/台）", 380))
            st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: 预警中心 ----
with tabs[4]:
    alerts = results['流失预警']
    anomalies = results['MoM异常']
    if alerts:
        total_risk = sum(a['年度金额'] for a in alerts)
        high_alerts = [a for a in alerts if '高' in a['风险']]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("预警客户", f"{len(alerts)}家")
        c2.metric("🔴 高风险", f"{len(high_alerts)}家", "需立即关注")
        c3.metric("风险金额", f"{total_risk:,.0f}万")
        c4.metric("占总营收", f"{total_risk/data['总营收']*100:.1f}%")

        st.markdown("")
        st.markdown('<div class="section-header"><div class="icon">🔴</div> 流失风险排名</div>', unsafe_allow_html=True)
        alert_df = pd.DataFrame(alerts)
        display_cols = [c for c in ['客户', '风险', '得分', '年度金额', '原因'] if c in alert_df.columns]
        if '年度金额' in alert_df.columns:
            alert_df['年度金额'] = pd.to_numeric(alert_df['年度金额'], errors='coerce').round(0)
        st.dataframe(alert_df[display_cols], use_container_width=True, hide_index=True,
            column_config={'年度金额': st.column_config.NumberColumn(format="%,.0f"),
                '得分': st.column_config.ProgressColumn(min_value=0, max_value=120, format="%d")})

        st.markdown("")
        sel_alert = st.selectbox("预警客户走势", [a['客户'] for a in alerts], key='alert_sel')
        a_data = next((a for a in alerts if a['客户'] == sel_alert), None)
        if a_data and '月度趋势' in a_data and HAS_PLOTLY:
            vals = a_data['月度趋势']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=MONTHS, y=vals,
                marker_color=[RED if v > 0 else "rgba(239,68,68,0.2)" for v in vals],
                text=[f"{v:,.0f}" if v > 0 else "" for v in vals],
                textposition="outside", textfont=dict(size=10)))
            fig.update_layout(**plotly_layout(f"{sel_alert} · 月度走势", 350, False))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 无高风险预警")

    with st.expander("⚡ 月度异常检测"):
        if anomalies:
            anom_df = pd.DataFrame(anomalies[:20])
            for col in ['当月', '上月', '月均']:
                if col in anom_df.columns:
                    anom_df[col] = pd.to_numeric(anom_df[col], errors='coerce').round(0)
            st.dataframe(anom_df, use_container_width=True, hide_index=True)
        else:
            st.info("无显著异常")

# ---- Tab 5: 增长机会 ----
with tabs[5]:
    growth = results['增长机会']
    if growth:
        types = sorted(set(g['类型'] for g in growth))
        cols = st.columns(len(types))
        for i, t in enumerate(types):
            cols[i].metric(t, f"{sum(1 for g in growth if g['类型'] == t)}个")
        st.markdown("")
        g_df = pd.DataFrame(growth)
        if '金额' in g_df.columns:
            g_df['金额'] = pd.to_numeric(g_df['金额'], errors='coerce').round(0)
        st.dataframe(g_df, use_container_width=True, hide_index=True,
            column_config={'金额': st.column_config.NumberColumn(format="%,.0f")})
    else:
        st.info("暂无显著增长信号")

# ---- Tab 6: 产品结构 ----
with tabs[6]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header"><div class="icon">📱</div> 产品类型</div>', unsafe_allow_html=True)
        prod = results['产品结构']
        if prod:
            st.dataframe(pd.DataFrame(prod), use_container_width=True, hide_index=True)
            if HAS_PLOTLY:
                fig = go.Figure(data=[go.Pie(labels=[p['类型'] for p in prod], values=[p['全年实际'] for p in prod],
                    hole=0.5, marker_colors=[ACCENT, PURPLE, CYAN], textinfo='label+percent')])
                fig.update_layout(**plotly_layout("", 320, False))
                st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header"><div class="icon">📦</div> 订单模式</div>', unsafe_allow_html=True)
        order = results['订单模式']
        if order:
            st.dataframe(pd.DataFrame(order), use_container_width=True, hide_index=True)
            if HAS_PLOTLY:
                fig = go.Figure(data=[go.Pie(labels=[o['模式'] for o in order], values=[o['全年数量'] for o in order],
                    hole=0.5, marker_colors=[ORANGE, ACCENT, PURPLE], textinfo='label+percent')])
                fig.update_layout(**plotly_layout("", 320, False))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    st.markdown('<div class="section-header"><div class="icon">📊</div> 计划 vs 实际</div>', unsafe_allow_html=True)
    qs = data['数量汇总']
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MONTHS, y=qs['月度计划'], name="计划",
            line=dict(color="rgba(100,116,139,0.4)", width=2, dash='dash'), mode='lines'))
        fig.add_trace(go.Scatter(x=MONTHS, y=qs['月度实际'], name="实际",
            line=dict(color=GREEN, width=2.5), mode='lines+markers', marker=dict(size=6),
            fill='tonexty', fillcolor='rgba(16,185,129,0.06)'))
        fig.update_layout(**plotly_layout("月度出货：计划 vs 实际", 380))
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
            marker_color=[ACCENT if i == 0 else "rgba(99,102,241,0.3)" for i in range(len(regions))],
            text=[f"{r['金额']:,.0f}" for r in regions], textposition="outside", textfont=dict(size=10)))
        fig.update_layout(**plotly_layout("区域出货分布（万元）", 380, False))
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 8: 行业对标 ----
with tabs[8]:
    st.markdown('<div class="section-header"><div class="icon">🌐</div> 行业基准对标</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header"><div class="icon">🔮</div> 2026年前瞻预测</div>', unsafe_allow_html=True)
    t = forecast['总营收预测']
    c1, c2, c3 = st.columns(3)
    c1.metric("Q1 乐观", f"{t['置信区间']['乐观(+15%)']:,.0f}万")
    c2.metric("Q1 基准", f"{t['置信区间']['基准']:,.0f}万", "⬅️ 核心预测")
    c3.metric("Q1 悲观", f"{t['置信区间']['悲观(-15%)']:,.0f}万")
    st.caption(f"参考：Q1 2025 {t['参考']['Q1_2025实际']:,.0f}万 | Q4 2025 {t['参考']['Q4_2025实际']:,.0f}万")

    with st.expander("📐 预测方法"):
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
            marker_color=[GREEN, ACCENT, ORANGE, RED],
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
    st.markdown('<div class="section-header"><div class="icon">✍️</div> 管理层战略备忘录</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header"><div class="icon">❤️</div> 客户健康评分</div>', unsafe_allow_html=True)
    health_scores = render_health_dashboard(data, results)

# ---- Tab 12: 通知推送 ----
with tabs[12]:
    st.markdown('<div class="section-header"><div class="icon">🔔</div> 通知推送</div>', unsafe_allow_html=True)
    _hs = health_scores if 'health_scores' in dir() and health_scores else None
    render_notification_settings(results, _hs)

# ---- Tab 13: 导出 ----
with tabs[13]:
    st.markdown('<div class="section-header"><div class="icon">📥</div> 报告导出</div>', unsafe_allow_html=True)

    # PDF报告 + 邮件推送
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
    full = full.replace("Agent v2.0", "Agent v4.0").replace("智能分析系统 v2.0", "智能分析系统 v4.0")
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
