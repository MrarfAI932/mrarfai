#!/usr/bin/env python3
"""
MRARFAI v10.0 — Sprocomm 禾苗 Enterprise Agent Platform
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

# ── 加载 .env 配置 (内置 API Key) ──
def _load_env():
    """从 .env 文件加载环境变量"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if v and k not in os.environ:
                        os.environ[k] = v
_load_env()

# 内置 AI 配置
_DEFAULT_PROVIDER = os.environ.get("AI_PROVIDER", "Claude")
_DEFAULT_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")

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

# ── V10 Platform Gateway (可选) ──
try:
    from platform_gateway import get_gateway, PlatformGateway
    HAS_V10_GATEWAY = True
except ImportError:
    HAS_V10_GATEWAY = False

# ── 数据库连接器 (可选) ──
try:
    from db_connector import get_db_status, DatabaseConfig, get_connector
    HAS_DB_CONNECTOR = True
except ImportError:
    HAS_DB_CONNECTOR = False

# ── 定时报告引擎 (可选) ──
try:
    from scheduled_report import ReportScheduler, ReportConfig
    HAS_REPORT = True
except ImportError:
    HAS_REPORT = False

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
st.set_page_config(page_title="Sprocomm AI · MRARFAI v10.0", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# 登录门禁
# ============================================================
from auth import (require_login, get_current_user, logout, is_admin,
                   get_allowed_agents, can_access_agent, can_use_collab,
                   can_upload, can_export, get_role_permissions, ROLE_PERMISSIONS)
require_login()  # 未登录 → 显示登录页 → st.stop()
_current_user = get_current_user()
_user_role = _current_user.get("role", "viewer") if _current_user else "viewer"
_allowed_agents = get_allowed_agents(_user_role)
_role_perms = get_role_permissions(_user_role)

# ============================================================
# 多语言 i18n (中/英)
# ============================================================
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

_I18N = {
    "command_center": {"zh": "COMMAND CENTER", "en": "COMMAND CENTER"},
    "agents_available": {"zh": "AGENTS AVAILABLE", "en": "AGENTS AVAILABLE"},
    "select_agent": {"zh": "选择一个 Agent 进入专属工作台", "en": "Select an Agent to enter its workspace"},
    "current_role": {"zh": "当前角色", "en": "Current role"},
    "enter": {"zh": "进入", "en": "Enter"},
    "back": {"zh": "← 返回", "en": "← Back"},
    "agent_workspace": {"zh": "AGENT WORKSPACE", "en": "AGENT WORKSPACE"},
    "cross_agent_collab": {"zh": "跨Agent协作", "en": "Cross-Agent Collaboration"},
    "collab_desc": {"zh": "出货异常追踪 · 智能报价 · 月度复盘 · 供应商延迟", "en": "Shipment Tracking · Smart Pricing · Monthly Review · Supplier Delay"},
    "collab_hint": {"zh": "选择协作场景，触发多Agent链式分析", "en": "Select a scenario to trigger multi-Agent chain analysis"},
    "upload_data": {"zh": "上传自定义数据（Excel） — 替换内置样本数据", "en": "Upload Custom Data (Excel) — Replace Sample Data"},
    "upload_excel": {"zh": "上传报表", "en": "Upload Reports"},
    "upload_hint_v9": {"zh": "上传 Sprocomm 金额报表 + 数量报表 (Excel) 解锁全部分析功能", "en": "Upload revenue + quantity reports (Excel) to unlock full analysis"},
    "upload_please": {"zh": "请上传 2 个 Excel 文件（金额报表 + 数量报表）", "en": "Please upload 2 Excel files (revenue + quantity reports)"},
    "ai_enabled": {"zh": "AI 智能回答已启用", "en": "AI Smart Reply Enabled"},
    "key_builtin": {"zh": "Key 已内置", "en": "Key Built-in"},
    "ask_agent": {"zh": "提问...", "en": "Ask..."},
    "no_access": {"zh": "无权访问", "en": "Access Denied"},
    "no_upload": {"zh": "无上传权限，请联系管理员", "en": "No upload permission, please contact admin"},
    "no_collab": {"zh": "无权使用跨Agent协作", "en": "No access to cross-Agent collaboration"},
    "logout": {"zh": "登出", "en": "Logout"},
    "data_loaded": {"zh": "数据已加载", "en": "Data Loaded"},
    "scenarios": {"zh": "scenarios", "en": "scenarios"},
    "skills": {"zh": "skills", "en": "skills"},
    "online": {"zh": "ONLINE", "en": "ONLINE"},
    "need_upload": {"zh": "需上传Excel", "en": "Upload Required"},
}

def _t(key: str) -> str:
    """获取当前语言的翻译文本"""
    entry = _I18N.get(key, {})
    return entry.get(st.session_state.lang, entry.get("zh", key))


# ============================================================
# 内联主题 — Command Center (完整版)
# ============================================================
# 内联主题 — Command Center (完整版)
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

#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] { display: none !important; }

.stApp { background: var(--bg-deep) !important; }
.block-container { padding: 1rem 2rem 3rem !important; max-width: 1600px; }
/* Solid background for chat messages */
[data-testid="stChatMessage"] { background: #0C0C0C !important; }
.stChatMessage { background: #0C0C0C !important; }

/* ── Hide sidebar completely ── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
button[kind="headerNoPadding"] { display: none !important; }

/* ── Top Config Bar ── */
.top-bar {
    display:flex; align-items:center; gap:16px; padding:10px 20px;
    background:var(--bg-base); border:1px solid var(--border-subtle);
    margin-bottom:16px; flex-wrap:wrap;
}
.top-bar-logo {
    display:flex; align-items:center; gap:10px; margin-right:auto;
}
.top-bar-section {
    display:flex; align-items:center; gap:8px;
}
.top-bar-label {
    font-family:var(--font-mono); font-size:0.55rem; font-weight:700;
    letter-spacing:0.1em; text-transform:uppercase; color:var(--text-3);
}
/* ── Upload Zone (welcome page) ── */
.upload-zone {
    max-width:600px; margin:0 auto; padding:32px;
    background:var(--bg-elevated); border:1px solid var(--border-subtle);
    position:relative;
}
.upload-zone::before {
    content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
    background:linear-gradient(180deg, #00FF88, rgba(0,255,136,0.15));
}
.upload-zone .stFileUploader {
    border:1px dashed rgba(255,255,255,0.08) !important;
    transition:border-color 0.2s;
}
.upload-zone .stFileUploader:hover {
    border-color:rgba(0,255,136,0.25) !important;
}

.stMarkdown p { font-family: var(--font-mono) !important; color: var(--text-2) !important; font-size: 0.82rem !important; line-height: 1.7 !important; }
.stMarkdown h1 { font-family: var(--font-sans) !important; color: var(--text-1) !important; font-weight: 700 !important; letter-spacing: -0.03em !important; }
.stMarkdown h2 { font-family: var(--font-sans) !important; color: var(--text-1) !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
.stMarkdown h3, .stMarkdown h4 { font-family: var(--font-mono) !important; color: var(--neon) !important; font-weight: 700 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; font-size: 0.72rem !important; }

.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border-subtle); }
.stTabs [data-baseweb="tab"] { font-family: var(--font-mono) !important; font-size: 0.62rem !important; font-weight: 600 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: var(--text-3) !important; border-radius: 0 !important; padding: 0.7rem 1rem !important; }
.stTabs [aria-selected="true"] { color: var(--neon) !important; border-bottom: 2px solid var(--neon) !important; background: transparent !important; }
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }

.agent-card { background: var(--bg-elevated); border: 1px solid var(--border-subtle); padding: 0.8rem 1rem; margin: 0.3rem 0; transition: border-color 0.15s; display: flex; align-items: center; gap: 0.75rem; }
.agent-card:hover { border-color: rgba(0,255,136,0.25); }
.agent-card .agent-avatar { width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;flex-shrink:0; }
.agent-card .agent-name { font-family:var(--font-mono);font-size:0.78rem;font-weight:700;color:#fff;letter-spacing:0.03em; }
.agent-card .agent-role { font-family:var(--font-mono);font-size:0.58rem;color:#6a6a6a;text-transform:uppercase;letter-spacing:0.1em; }
.agent-card .agent-status { font-family:var(--font-mono);font-size:0.58rem;padding:0.15rem 0.5rem;font-weight:700;flex-shrink:0;margin-left:auto;letter-spacing:0.05em; }
.status-complete { background:rgba(0,255,136,0.08);color:#00FF88;border:1px solid rgba(0,255,136,0.25); }
.status-running { background:rgba(0,160,200,0.08);color:#00A0C8;border:1px solid rgba(0,160,200,0.25); }
.status-error { background:rgba(217,64,64,0.08);color:#D94040;border:1px solid rgba(217,64,64,0.25); }

.thinking-timeline { background:#080808;border:1px solid #2f2f2f;border-left:2px solid rgba(0,255,136,0.30);padding:0.8rem 1rem;margin:0.5rem 0; }
.thinking-step { display:flex;align-items:flex-start;gap:0.6rem;padding:0.3rem 0;margin-left:0.55rem;border-left:1px solid #2f2f2f;padding-left:1rem; }
.thinking-step .step-dot { width:6px;height:6px;margin-top:0.35rem;flex-shrink:0;margin-left:-1.35rem; }
.thinking-step .step-text { font-family:var(--font-mono);font-size:0.72rem;color:#aaa;line-height:1.5; }
.thinking-step .step-meta { font-family:var(--font-mono);font-size:0.58rem;color:#00FF88;margin-left:auto;flex-shrink:0;white-space:nowrap;font-weight:600; }

.quality-badge { display:inline-flex;align-items:center;gap:0.4rem;padding:0.35rem 0.7rem;font-family:var(--font-mono);font-size:0.65rem;font-weight:700;letter-spacing:0.05em;margin:0.25rem 0.25rem 0.25rem 0; }
.quality-pass { background:rgba(0,255,136,0.08);color:#00FF88;border:1px solid rgba(0,255,136,0.25); }
.quality-fail { background:rgba(255,136,0,0.08);color:#FF8800;border:1px solid rgba(255,136,0,0.25); }

.hitl-card { background:#080808;border:1px solid #2f2f2f;padding:0.7rem 1rem;margin:0.5rem 0;display:flex;align-items:center;gap:0.8rem; }
.hitl-gauge { width:42px;height:42px;display:flex;align-items:center;justify-content:center;font-family:var(--font-mono);font-size:0.72rem;font-weight:700;flex-shrink:0; }
.hitl-high { background:rgba(0,255,136,0.10);color:#00FF88;border:2px solid rgba(0,255,136,0.40); }
.hitl-medium { background:rgba(255,136,0,0.10);color:#FF8800;border:2px solid rgba(255,136,0,0.35); }
.hitl-low { background:rgba(217,64,64,0.10);color:#D94040;border:2px solid rgba(217,64,64,0.35); }
.hitl-info { flex:1; }
.hitl-info .hitl-level { font-family:var(--font-mono);font-size:0.65rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase; }
.hitl-info .hitl-action { font-family:var(--font-mono);font-size:0.72rem;color:#aaa;margin-top:0.1rem; }
.hitl-triggers { font-family:var(--font-mono);font-size:0.55rem;color:#6a6a6a;text-align:right;flex-shrink:0; }

.trace-bar { display:flex;align-items:center;gap:1.2rem;padding:0.6rem 1rem;background:#080808;border:1px solid #2f2f2f;font-family:var(--font-mono);font-size:0.6rem;color:#6a6a6a;margin:0.8rem 0;letter-spacing:0.03em; }
.trace-bar .trace-value { color:#00FF88;font-weight:700; }

/* ======================================== */
/* V9.0 UI POLISH — Animations & Keyframes */
/* ======================================== */
@keyframes neon-pulse {
    0%, 100% { opacity:1; box-shadow:0 0 0 0 rgba(0,255,136,0.5); }
    50%      { opacity:0.7; box-shadow:0 0 8px 4px rgba(0,255,136,0.15); }
}
@keyframes fade-in-up {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes glow-border {
    0%, 100% { border-color:rgba(0,255,136,0.12); }
    50%      { border-color:rgba(0,255,136,0.30); }
}
@keyframes v9-spin { to { transform:rotate(360deg); } }
@keyframes badge-glow {
    0%, 100% { box-shadow:0 0 0 0 rgba(0,255,136,0.08); }
    50%      { box-shadow:0 0 12px rgba(0,255,136,0.12), 0 0 0 1px rgba(0,255,136,0.20); }
}

/* ── Status Bar (post-load) ── */
.status-bar {
    display:flex; align-items:center; gap:8px;
    padding:8px 14px; margin-bottom:16px;
    background:rgba(0,255,136,0.03); border:1px solid rgba(0,255,136,0.10);
    animation:fade-in-up 0.5s ease-out;
}
.status-bar .status-dot {
    width:6px; height:6px; border-radius:50%; background:#00FF88;
    animation:neon-pulse 2s ease-in-out infinite; flex-shrink:0;
}
.status-bar .status-text {
    font-family:var(--font-mono); font-size:0.62rem; font-weight:700;
    color:#00FF88; letter-spacing:0.1em; text-transform:uppercase;
}
.status-bar .status-meta {
    font-family:var(--font-mono); font-size:0.55rem; color:#6a6a6a;
    letter-spacing:0.05em; margin-left:auto;
}

/* ── Section Header ── */
.section-header {
    font-family:var(--font-mono); font-size:0.65rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase; color:var(--text-2);
    padding:8px 0 6px; margin:16px 0 10px;
    border-bottom:2px solid var(--neon);
    display:inline-block;
}

/* ── Welcome Page ── */
.welcome-badge {
    display:inline-flex; align-items:center; gap:8px;
    padding:6px 16px;
    background:rgba(0,255,136,0.06); border:1px solid rgba(0,255,136,0.25);
    font-size:0.62rem; color:#00FF88; font-weight:700;
    letter-spacing:0.1em; font-family:var(--font-mono);
    text-transform:uppercase;
    animation:badge-glow 3s ease-in-out infinite;
}
.welcome-badge .badge-dot {
    width:6px; height:6px; border-radius:50%; background:#00FF88;
    animation:neon-pulse 2s ease-in-out infinite;
}
.welcome-title-green {
    font-size:3rem; font-weight:700; color:#00FF88; letter-spacing:-2px;
    margin:0; line-height:1.1; font-family:'Space Grotesk',sans-serif;
    text-shadow: 0 0 30px rgba(0,255,136,0.20), 0 0 60px rgba(0,255,136,0.06);
}
.welcome-title-white {
    font-size:3rem; font-weight:700; color:#FFFFFF; letter-spacing:-2px;
    margin:0; line-height:1.1; font-family:'Space Grotesk',sans-serif;
}
.welcome-card {
    background:var(--bg-elevated); border:1px solid var(--border-subtle);
    border-left:2px solid var(--neon); display:block; text-align:left;
    padding:1.2rem; position:relative; overflow:hidden;
    transition:transform 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.welcome-card::after {
    content:""; position:absolute; bottom:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg, transparent, rgba(0,255,136,0.25), transparent);
    opacity:0; transition:opacity 0.3s;
}
.welcome-card:hover {
    transform:translateY(-2px); border-color:rgba(0,255,136,0.25);
    box-shadow:0 6px 24px rgba(0,0,0,0.3);
}
.welcome-card:hover::after { opacity:1; }

/* ── Logo Box (reused in top bar) ── */
.sidebar-logo-box {
    width:32px; height:32px; background:transparent;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
/* ── File uploader styling ── */
.stFileUploader {
    border:1px dashed rgba(255,255,255,0.08) !important;
    transition:border-color 0.2s;
}
.stFileUploader:hover {
    border-color:rgba(0,255,136,0.25) !important;
}
.stCaption, small {
    font-family:var(--font-mono) !important; font-size:0.55rem !important;
    color:#6a6a6a !important; letter-spacing:0.03em !important;
}
.agent-active-badge {
    display:flex; align-items:center; gap:6px; padding:4px 8px;
    background:rgba(0,255,136,0.06); border:1px solid rgba(0,255,136,0.15);
    margin-top:4px;
}
.agent-active-badge .pulse-dot {
    width:5px; height:5px; border-radius:50%; background:#00FF88;
    animation:neon-pulse 2s ease-in-out infinite;
    box-shadow:0 0 6px rgba(0,255,136,0.4);
}

/* ── Metric Card Enhancements ── */
[data-testid="stMetric"] {
    background:var(--bg-elevated) !important; border:1px solid var(--border-subtle) !important;
    padding:14px 16px !important; position:relative;
    transition:border-color 0.2s, box-shadow 0.2s;
    overflow:hidden;
}
[data-testid="stMetric"]::before {
    content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
    background:linear-gradient(180deg, #00FF88, rgba(0,255,136,0.15));
}
[data-testid="stMetric"]:hover {
    border-color:rgba(0,255,136,0.20) !important;
    box-shadow:0 0 12px rgba(0,255,136,0.06);
}
[data-testid="stMetricLabel"] {
    font-family:var(--font-mono) !important; font-size:0.55rem !important;
    font-weight:700 !important; letter-spacing:0.1em !important;
    text-transform:uppercase !important; color:var(--text-3) !important;
}
[data-testid="stMetricValue"] {
    font-family:var(--font-sans) !important; font-weight:700 !important;
    color:var(--text-1) !important;
}
[data-testid="stMetricDelta"] {
    font-family:var(--font-mono) !important; font-size:0.65rem !important;
}

/* ── Tab Enhancements ── */
.stTabs [aria-selected="true"] {
    text-shadow:0 0 8px rgba(0,255,136,0.3) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background:rgba(255,255,255,0.02) !important;
}

/* ── Expander Enhancements ── */
.streamlit-expanderHeader {
    font-family:var(--font-mono) !important; font-size:0.72rem !important;
    background:var(--bg-elevated) !important;
}
[data-testid="stExpander"] {
    background:var(--bg-elevated) !important;
    border:1px solid var(--border-subtle) !important;
    transition:border-color 0.2s;
}
[data-testid="stExpander"]:hover {
    border-color:rgba(0,255,136,0.15) !important;
}

/* ── Chat Area — Breathing Room ── */
[data-testid="stChatMessage"] {
    padding:1rem 1.2rem !important; margin-bottom:0.8rem !important;
}
[data-testid="stChatMessage"] p {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.82rem !important; line-height:1.85 !important;
    color:#d4d4d4 !important;
}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2 {
    font-family:'Space Grotesk',sans-serif !important;
    color:#FFFFFF !important; margin-top:1rem !important;
}
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    font-family:'Space Grotesk',sans-serif !important;
    color:#e2e8f0 !important; font-size:0.95rem !important;
    margin-top:0.8rem !important;
}
[data-testid="stChatMessage"] li {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.8rem !important; line-height:1.8 !important;
    color:#b4b4b4 !important; margin-bottom:0.3rem !important;
}
[data-testid="stChatMessage"] strong {
    color:#FFFFFF !important;
}
/* Chat input */
[data-testid="stChatInput"] {
    border-top:1px solid #2f2f2f !important;
    padding-top:0.8rem !important;
}
[data-testid="stChatInput"] textarea {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.82rem !important;
}
/* Status expander in chat */
[data-testid="stStatus"] {
    margin:0.6rem 0 !important;
}
[data-testid="stStatus"] p {
    font-size:0.75rem !important; line-height:1.6 !important;
}

/* ── Suggestion Chips ── */
.stButton button {
    font-family:'JetBrains Mono',monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#2f2f2f; border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:#00FF88; }

/* ════════════════════════════════════════════════════════ */
/* AI RESPONSE — Card-based Answer Layout                  */
/* ════════════════════════════════════════════════════════ */

.ai-response-container {
    display:flex; flex-direction:column; gap:10px;
    margin:4px 0 14px; animation:fade-in-up 0.4s ease-out;
}

/* Summary — Hero card */
.ai-summary {
    background:rgba(0,255,136,0.04); border:1px solid rgba(0,255,136,0.15);
    border-left:3px solid #00FF88; padding:16px 20px; position:relative;
}
.ai-summary::after {
    content:""; position:absolute; bottom:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg, #00FF88, transparent 80%); opacity:0.25;
}
.ai-summary-label {
    font-family:var(--font-mono); font-size:0.55rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase; color:#00FF88; margin-bottom:6px;
}
.ai-summary-text {
    font-family:var(--font-sans); font-size:0.92rem; font-weight:500;
    color:#FFFFFF; line-height:1.75;
}

/* Section cards */
.ai-section {
    background:#0C0C0C; border:1px solid rgba(255,255,255,0.06);
    border-left:3px solid #6a6a6a; padding:12px 16px;
    transition:border-color 0.2s;
}
.ai-section:hover { border-color:rgba(255,255,255,0.12); }
.ai-section-header {
    font-family:var(--font-mono); font-size:0.62rem; font-weight:700;
    letter-spacing:0.08em; text-transform:uppercase; color:#8a8a8a;
    margin-bottom:6px; padding-bottom:4px;
    border-bottom:1px solid rgba(255,255,255,0.04);
}
.ai-section-icon { color:#00FF88; margin-right:4px; }
.ai-section-body {
    font-family:var(--font-mono); font-size:0.8rem;
    color:#b4b4b4; line-height:1.8;
}

/* Section color variants */
.ai-section-growth { border-left-color:#00FF88; }
.ai-section-growth .ai-section-header { color:#00FF88; }
.ai-section-risk { border-left-color:#D94040; }
.ai-section-risk .ai-section-header { color:#D94040; }
.ai-section-action { border-left-color:#00A0C8; }
.ai-section-action .ai-section-header { color:#00A0C8; }
.ai-section-analysis { border-left-color:#8a8a8a; }

/* Metric chips — inline number highlights */
.ai-metric-chip {
    display:inline; padding:1px 5px;
    font-family:var(--font-mono); font-weight:700;
    font-size:inherit; letter-spacing:0.02em;
}
.ai-metric-positive {
    color:#00FF88; background:rgba(0,255,136,0.08);
    border:1px solid rgba(0,255,136,0.20);
}
.ai-metric-negative {
    color:#D94040; background:rgba(217,64,64,0.08);
    border:1px solid rgba(217,64,64,0.20);
}
.ai-metric-neutral {
    color:#FFFFFF; background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.10);
}

/* Action items */
.ai-action-item {
    display:flex; align-items:flex-start; gap:10px;
    padding:8px 12px; margin:4px 0;
    background:rgba(0,160,200,0.03); border-left:2px solid rgba(0,160,200,0.30);
}
.ai-action-num {
    font-family:var(--font-mono); font-size:0.6rem; font-weight:700;
    color:#00A0C8; background:rgba(0,160,200,0.10);
    border:1px solid rgba(0,160,200,0.25);
    width:20px; height:20px; display:flex; align-items:center;
    justify-content:center; flex-shrink:0; margin-top:1px;
}
.ai-action-text {
    font-family:var(--font-mono); font-size:0.8rem; color:#ccc; line-height:1.7;
}

/* Expert mini-cards (inside expander) */
.ai-expert-card {
    background:#080808; border:1px solid #2f2f2f;
    padding:10px 14px; margin:6px 0; border-left:3px solid #6a6a6a;
}
.ai-expert-header {
    display:flex; align-items:center; gap:8px; margin-bottom:6px;
}
.ai-expert-icon { font-size:0.9rem; }
.ai-expert-name {
    font-family:var(--font-sans); font-size:0.78rem; font-weight:700;
    letter-spacing:0.03em;
}
.ai-expert-role {
    font-family:var(--font-mono); font-size:0.52rem; color:#6a6a6a;
    text-transform:uppercase; letter-spacing:0.1em; margin-left:auto;
}
.ai-expert-body {
    font-family:var(--font-mono); font-size:0.75rem; color:#999;
    line-height:1.7; max-height:140px; overflow-y:auto;
}

/* Inline meta row */
.ai-inline-meta {
    display:flex; align-items:center; gap:8px;
    flex-wrap:wrap; margin:6px 0;
}

/* ── Welcome Background Gradient ── */
.welcome-bg {
    position:relative;
}
.welcome-bg::before {
    content:""; position:absolute; top:-60px; left:50%; transform:translateX(-50%);
    width:600px; height:300px;
    background:radial-gradient(ellipse, rgba(0,255,136,0.04) 0%, transparent 70%);
    pointer-events:none; z-index:0;
}

/* ── 移动端适配 ── */
@media (max-width: 768px) {
    .block-container { padding: 0.5rem 0.8rem 2rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.5rem !important; padding: 0.5rem 0.5rem !important; letter-spacing: 0.05em !important; }
    .stTabs [data-baseweb="tab-list"] { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .upload-zone { padding: 16px; max-width: 100%; }
    .agent-card { padding: 0.5rem 0.6rem; }
    .top-bar { padding: 6px 10px; gap: 8px; }
    [data-testid="stDataFrame"] { overflow-x: auto !important; }
    .js-plotly-plot { overflow-x: auto !important; }
}
@media (max-width: 480px) {
    .block-container { padding: 0.3rem 0.5rem 1.5rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.45rem !important; padding: 0.4rem 0.35rem !important; }
}

</style>""", unsafe_allow_html=True)

# ============================================================
# 顶部导航栏 (替代侧边栏)
# ============================================================
_user = get_current_user()

# Top bar — logo + user + logout + lang toggle
_bar1, _bar_lang, _bar2 = st.columns([5, 0.5, 0.8])
with _bar1:
    # 读取 topbar logo
    _topbar_logo_b64 = ""
    try:
        with open("logo_b64.txt", "r") as _tf:
            _topbar_logo_b64 = _tf.read().strip()
    except Exception:
        pass
    _topbar_logo_html = f'<img src="data:image/png;base64,{_topbar_logo_b64}" style="width:28px;height:auto;filter:brightness(0) invert(1);" />' if _topbar_logo_b64 else '<span style="font-weight:700;font-size:0.85rem;color:#FFF;">M</span>'

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; padding:4px 0;">
        <div style="width:32px;height:32px;display:flex;align-items:center;justify-content:center;">
            {_topbar_logo_html}
        </div>
        <div>
            <span style="font-size:0.88rem; font-weight:700; color:#FFFFFF;
                 letter-spacing:0.1em; font-family:'Space Grotesk',sans-serif;
                 text-transform:uppercase;">MRARFAI</span>
            <span style="font-size:0.5rem; color:#6a6a6a; font-family:'JetBrains Mono',monospace;
                 letter-spacing:0.08em; margin-left:12px;">V10.0 · Enterprise Agent Platform</span>
        </div>
        <div style="margin-left:auto; display:flex; align-items:center; gap:8px;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.55rem;
                 color:#6a6a6a; letter-spacing:0.05em;">
                👤 {_user['display_name']} · <span style="color:#FFFFFF;">{_role_perms.get('label', _user['role'].upper())}</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with _bar_lang:
    _lang_label = "EN" if st.session_state.lang == "zh" else "中"
    if st.button(_lang_label, key="lang_toggle", use_container_width=True):
        st.session_state.lang = "en" if st.session_state.lang == "zh" else "zh"
        st.rerun()
with _bar2:
    if st.button(_t("logout"), key="logout_btn", type="secondary", use_container_width=True):
        logout()
        st.rerun()

# ============================================================
# 智能文件检测 + 数据加载
# ============================================================

_REV_MARKERS = ['2025数据', '2024数据', '数据', '与年度目标对比', '目标对比', '目标']
_QTY_MARKERS = ['数量汇总', '汇总', '数量']


def _detect_file_type(file_bytes: bytes) -> str:
    """检测 Excel → 'revenue' / 'quantity' / 'unknown'"""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        tmp.write(file_bytes); tmp.close()
        xls = pd.ExcelFile(tmp.name); sheets = xls.sheet_names; xls.close()
        os.unlink(tmp.name)
        for m in _QTY_MARKERS:
            if any(m in s for s in sheets):
                return 'quantity'
        for m in _REV_MARKERS:
            if any(m in s for s in sheets):
                return 'revenue'
        if len(sheets) >= 3:
            return 'revenue'
        return 'unknown'
    except Exception:
        return 'unknown'


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


# ── 读取 Logo base64 ──
_logo_b64 = ""
try:
    with open("logo_b64.txt", "r") as _lf:
        _logo_b64 = _lf.read().strip()
except Exception:
    pass

# ── MRARFAI Logo + 品牌 (居中) ──
_bl, _bc, _br = st.columns([1, 2, 1])
with _bc:
    if _logo_b64:
        st.markdown(f"""
        <div style="text-align:center;padding:32px 0 8px 0;">
            <img src="data:image/png;base64,{_logo_b64}"
                 style="width:120px;height:auto;filter:brightness(0) invert(1);margin-bottom:8px;" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:32px 0 8px 0;">
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:800;font-size:2rem;
                  color:#FFFFFF;letter-spacing:0.18em;">MRARFAI</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# V10 COMMAND CENTER — 统一 Agent 平台 (不需要上传文件)
# ============================================================
ai_provider, api_key = 'DeepSeek', None
st.session_state["ai_provider"] = ai_provider
st.session_state["api_key"] = api_key or ""

if HAS_V10_GATEWAY:
    try:
        _gw = get_gateway()
        _card = _gw.get_platform_card()

        _agent_icons = {
            "sales": "📈", "procurement": "🛒", "quality": "🔍",
            "finance": "💰", "market": "📊", "risk": "🚨", "strategist": "🔮",
        }
        _agent_names_zh = {
            "sales": "销售分析", "procurement": "采购管理", "quality": "品质检测",
            "finance": "财务分析", "market": "市场情报", "risk": "风控预警", "strategist": "战略顾问",
        }
        _agent_names_en = {
            "sales": "Sales", "procurement": "Procurement", "quality": "Quality",
            "finance": "Finance", "market": "Market Intel", "risk": "Risk Alert", "strategist": "Strategist",
        }
        _agent_names_cn = _agent_names_zh if st.session_state.lang == "zh" else _agent_names_en
        _agent_desc_zh = {
            "sales": "客户分析 · 价量分解 · 增长机会 · 流失预警",
            "procurement": "供应商评估 · PO跟踪 · 比价分析 · 延迟预警",
            "quality": "良率监控 · 退货分析 · 投诉分类 · 根因追溯",
            "finance": "应收跟踪 · 毛利分析 · 现金流预测 · 发票匹配",
            "market": "竞品监控 · 行业趋势 · 舆情追踪 · 市场全景",
            "risk": "流失预警 · 异常检测 · 风险评分 · 健康诊断",
            "strategist": "行业对标 · 场景预测 · 增长策略 · CEO备忘录",
        }
        _agent_desc_en = {
            "sales": "Client Analysis · Revenue Decomposition · Growth · Churn",
            "procurement": "Supplier Eval · PO Track · Price Compare · Delay Alert",
            "quality": "Yield Monitor · Returns · Complaints · Root Cause",
            "finance": "AR Tracking · Margin Analysis · Cash Flow · Invoice Match",
            "market": "Competitor Watch · Trends · Sentiment · Market Overview",
            "risk": "Churn Alert · Anomaly Detection · Risk Score · Health",
            "strategist": "Benchmark · Forecast · Growth Strategy · CEO Memo",
        }
        _agent_desc = _agent_desc_zh if st.session_state.lang == "zh" else _agent_desc_en
        # 哪些 Agent 有独立 engine（可直接使用）
        _has_engine = {"procurement", "quality", "finance", "market"}
        # sales 需要上传文件
        _needs_upload = {"sales", "risk", "strategist"}

        # 初始化状态
        if "active_agent" not in st.session_state:
            st.session_state.active_agent = None
        if "v10_chat_history" not in st.session_state:
            st.session_state.v10_chat_history = {}

        # ── 如果没有选择 Agent，显示主面板 ──
        if st.session_state.active_agent is None:
            # Command Center 标题
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0d1117,#161b22);padding:20px 24px;
                        border:1px solid rgba(255,255,255,0.08);margin-bottom:20px;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                    <span style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
                          font-weight:700;color:#FFF;letter-spacing:0.06em;">{_t('command_center')}</span>
                    <span style="font-size:0.5rem;color:#555;font-family:'JetBrains Mono',monospace;
                          border:1px solid #333;padding:2px 6px;">V10.0 · {len(_allowed_agents)} {_t('agents_available')}</span>
                </div>
                <div style="font-size:0.6rem;color:#555;font-family:'JetBrains Mono',monospace;">
                    {_t('select_agent')}　｜　{_t('current_role')}: {_role_perms.get('label', _user_role)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── KPI 仪表盘 ──
            _dash_label = "运营快览" if st.session_state.lang == "zh" else "Operations Snapshot"
            with st.container():
                st.markdown(f"""<div style="font-size:0.55rem;color:#555;font-family:'JetBrains Mono',monospace;
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">📊 {_dash_label}</div>""",
                    unsafe_allow_html=True)

                _kpi_cols = st.columns(5)

                # 从 Gateway 获取平台统计
                try:
                    _pstats = _gw.get_platform_stats()
                except Exception:
                    _pstats = {}

                # 从 Agent 引擎获取关键指标
                _kpi_data = []
                try:
                    _engines = _gw.collaboration.engines
                    # 采购: 订单数 + 延迟数
                    if "procurement" in _engines:
                        _pe = _engines["procurement"]
                        _orders = getattr(_pe, 'orders', [])
                        _delayed = sum(1 for o in _orders if getattr(o, 'status', '') == 'delayed')
                        _kpi_data.append(("📦", "采购订单" if st.session_state.lang == "zh" else "PO Orders",
                                          f"{len(_orders)}", f"{_delayed} 延迟" if _delayed else "正常",
                                          "#D94040" if _delayed else "#4ade80"))
                    # 品质: 良率
                    if "quality" in _engines:
                        _qe = _engines["quality"]
                        _lines = getattr(_qe, 'production_lines', [])
                        if _lines:
                            _avg_yield = sum(l.yield_rate for l in _lines) / len(_lines) * 100
                            _kpi_data.append(("🔬", "平均良率" if st.session_state.lang == "zh" else "Avg Yield",
                                              f"{_avg_yield:.1f}%", "良好" if _avg_yield > 95 else "关注",
                                              "#4ade80" if _avg_yield > 95 else "#f59e0b"))
                    # 财务: 逾期应收
                    if "finance" in _engines:
                        _fe = _engines["finance"]
                        _ar = getattr(_fe, 'accounts_receivable', [])
                        _overdue = sum(1 for a in _ar if getattr(a, 'status', '') == 'overdue')
                        _total_ar = len(_ar)
                        _kpi_data.append(("💰", "应收逾期" if st.session_state.lang == "zh" else "AR Overdue",
                                          f"{_overdue}/{_total_ar}", f"{_overdue} 笔逾期",
                                          "#D94040" if _overdue > 0 else "#4ade80"))
                    # 市场: 竞品数
                    if "market" in _engines:
                        _me = _engines["market"]
                        _comps = getattr(_me, 'competitors', {})
                        _kpi_data.append(("🏭", "竞品监控" if st.session_state.lang == "zh" else "Competitors",
                                          f"{len(_comps)}", "持续追踪", "#00A0C8"))
                except Exception:
                    pass

                # 平台请求数
                _total_req = _pstats.get("total_requests", 0)
                _avg_ms = _pstats.get("avg_duration_ms", 0)
                _kpi_data.append(("⚡", "平台请求" if st.session_state.lang == "zh" else "Requests",
                                  f"{_total_req}", f"均 {_avg_ms:.0f}ms" if _avg_ms > 0 else "就绪",
                                  "#00FF88"))

                # 渲染 KPI 卡片
                for _ki, _kpi in enumerate(_kpi_data):
                    if _ki < len(_kpi_cols):
                        with _kpi_cols[_ki]:
                            _k_icon, _k_label, _k_value, _k_sub, _k_color = _kpi
                            st.markdown(f"""<div style="background:#0c0c0c;border:1px solid #222;
                                padding:12px;text-align:center;">
                                <div style="font-size:1.2rem;">{_k_icon}</div>
                                <div style="font-size:1rem;font-weight:700;color:{_k_color};
                                    font-family:'Space Grotesk',sans-serif;margin:4px 0;">{_k_value}</div>
                                <div style="font-size:0.5rem;color:#888;font-family:'JetBrains Mono',monospace;">
                                    {_k_label}</div>
                                <div style="font-size:0.4rem;color:{_k_color};font-family:'JetBrains Mono',monospace;
                                    margin-top:2px;">{_k_sub}</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("")  # spacer

            # Agent 卡片 — 按角色过滤可见 Agent
            _visible_agents = [a for a in _card["agents"] if a in _allowed_agents]
            # 如果有协作权限，预留一个 slot 给协作卡片
            _show_collab = can_use_collab(_user_role)
            _total_cards = len(_visible_agents) + (1 if _show_collab else 0)
            _row1 = st.columns(min(4, _total_cards) if _total_cards > 0 else 4)
            _row2 = st.columns(4) if _total_cards > 4 else []

            for _i, _name in enumerate(_visible_agents):
                _col = _row1[_i] if _i < len(_row1) else _row2[_i - len(_row1)] if _i - len(_row1) < len(_row2) else None
                if _col is None:
                    continue
                _icon = _agent_icons.get(_name, "🤖")
                _cn = _agent_names_cn.get(_name, _name)
                _desc = _agent_desc.get(_name, "")
                _ac = _gw.registry.get_card(_name) if _gw.registry else None
                _sk = len(_ac.skills) if _ac else 0
                _available = _name in _has_engine
                _status_color = "#4ade80" if _available else "#f59e0b"
                _status_text = _t("online") if _available else _t("need_upload")

                with _col:
                    st.markdown(f"""
                    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.08);
                         padding:16px;text-align:center;min-height:160px;">
                        <div style="font-size:2rem;margin-bottom:4px;">{_icon}</div>
                        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                             font-size:0.8rem;color:#FFF;margin-bottom:4px;">{_cn}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.45rem;
                             color:#666;margin-bottom:8px;line-height:1.4;">{_desc}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.45rem;color:#555;">
                            {_sk} {_t('skills')}</div>
                        <div style="display:flex;align-items:center;justify-content:center;gap:4px;margin-top:4px;">
                            <div style="width:5px;height:5px;border-radius:50%;background:{_status_color};"></div>
                            <span style="font-size:0.4rem;color:{_status_color};font-family:'JetBrains Mono',monospace;">
                                {_status_text}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    if st.button(f"{_t('enter')} {_cn}", key=f"enter_{_name}", use_container_width=True):
                        st.session_state.active_agent = _name
                        st.rerun()

            # 协作卡片 — 仅有协作权限的角色可见
            if _show_collab:
                _collab_idx = len(_visible_agents)
                _collab_col = _row1[_collab_idx] if _collab_idx < len(_row1) else (_row2[_collab_idx - len(_row1)] if _row2 and _collab_idx - len(_row1) < len(_row2) else None)
                if _collab_col:
                    with _collab_col:
                        st.markdown(f"""
                        <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.08);
                             padding:16px;text-align:center;min-height:160px;">
                            <div style="font-size:2rem;margin-bottom:4px;">⚡</div>
                            <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                                 font-size:0.8rem;color:#FFF;margin-bottom:4px;">{_t('cross_agent_collab')}</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.45rem;
                                 color:#666;margin-bottom:8px;line-height:1.4;">
                                 {_t('collab_desc')}</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.45rem;color:#555;">
                                4 {_t('scenarios')}</div>
                            <div style="display:flex;align-items:center;justify-content:center;gap:4px;margin-top:4px;">
                                <div style="width:5px;height:5px;border-radius:50%;background:#4ade80;"></div>
                                <span style="font-size:0.4rem;color:#4ade80;font-family:'JetBrains Mono',monospace;">
                                    {_t('online')}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"{_t('enter')} {_t('cross_agent_collab')}", key="enter_collab", use_container_width=True):
                            st.session_state.active_agent = "_collab"
                            st.rerun()

            # ── 数据库连接状态 (管理员可见) ──
            if is_admin() and HAS_DB_CONNECTOR:
                _db_label = "数据源设置" if st.session_state.lang == "zh" else "Data Source Settings"
                with st.expander(f"🔌 {_db_label}", expanded=False):
                    _db_status = get_db_status()
                    _db_type = _db_status.get("type", "none")
                    _db_msg = _db_status.get("message", "")
                    if _db_type == "none":
                        st.markdown(f"""<div style="font-size:0.6rem;color:#f59e0b;font-family:'JetBrains Mono',monospace;">
                            ⚠ 当前使用内置样本数据。配置 .env 中的 DB_* 变量可连接真实 ERP/MES 数据库。</div>""",
                            unsafe_allow_html=True)
                    elif _db_status.get("status") == "ok":
                        st.markdown(f"""<div style="font-size:0.6rem;color:#4ade80;font-family:'JetBrains Mono',monospace;">
                            ✅ 已连接: {_db_type.upper()} — {_db_msg}</div>""",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div style="font-size:0.6rem;color:#D94040;font-family:'JetBrains Mono',monospace;">
                            ❌ 连接失败: {_db_msg}</div>""",
                            unsafe_allow_html=True)

                    st.markdown(f"""<div style="font-size:0.5rem;color:#555;font-family:'JetBrains Mono',monospace;margin-top:8px;">
                        <b>支持的数据源:</b><br>
                        · SQLite — 本地 Demo 数据库<br>
                        · MySQL — ERP (用友/金蝶/SAP)<br>
                        · PostgreSQL — MES/自研系统<br>
                        · REST API — 第三方平台接口<br><br>
                        <b>配置方法:</b> 编辑 .env 文件中的 DB_TYPE, DB_HOST, DB_PORT 等变量
                    </div>""", unsafe_allow_html=True)

            # ── 定时报告管理 (管理员可见) ──
            if is_admin() and HAS_REPORT:
                _rpt_label = "定时报告管理" if st.session_state.lang == "zh" else "Scheduled Reports"
                with st.expander(f"📧 {_rpt_label}", expanded=False):
                    _rpt_config = ReportConfig()
                    _rpt_enabled = _rpt_config.enabled

                    if _rpt_enabled:
                        st.markdown(f"""<div style="font-size:0.6rem;color:#4ade80;font-family:'JetBrains Mono',monospace;">
                            ✅ 邮件报告已启用 — {_rpt_config.schedule}
                            · SMTP: {_rpt_config.smtp_host}
                            · 收件人: {', '.join(_rpt_config.recipients)}</div>""",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div style="font-size:0.6rem;color:#f59e0b;font-family:'JetBrains Mono',monospace;">
                            ⚠ 邮件报告未配置。配置 .env 中的 REPORT_SMTP_* 变量启用自动报告推送。</div>""",
                            unsafe_allow_html=True)

                    # 手动生成报告
                    _rpt_col1, _rpt_col2 = st.columns(2)
                    with _rpt_col1:
                        if st.button("📊 生成日报", key="gen_daily", use_container_width=True):
                            with st.spinner("生成日报中…"):
                                try:
                                    _scheduler = ReportScheduler(gateway=_gw if HAS_V10_GATEWAY else None)
                                    _result = _scheduler.generate_now("daily")
                                    st.success(f"✅ 日报已生成")
                                    st.download_button(
                                        "📥 下载日报",
                                        _result["report"],
                                        f"mrarfai_daily_{datetime.now().strftime('%Y%m%d')}.md",
                                        "text/markdown",
                                        use_container_width=True,
                                    )
                                except Exception as e:
                                    st.error(f"生成失败: {e}")
                    with _rpt_col2:
                        if st.button("📈 生成周报", key="gen_weekly", use_container_width=True):
                            with st.spinner("生成周报中…"):
                                try:
                                    _scheduler = ReportScheduler(gateway=_gw if HAS_V10_GATEWAY else None)
                                    _result = _scheduler.generate_now("weekly")
                                    st.success(f"✅ 周报已生成")
                                    st.download_button(
                                        "📥 下载周报",
                                        _result["report"],
                                        f"mrarfai_weekly_{datetime.now().strftime('%Y%m%d')}.md",
                                        "text/markdown",
                                        use_container_width=True,
                                    )
                                except Exception as e:
                                    st.error(f"生成失败: {e}")

                    # 发送测试邮件
                    if _rpt_enabled:
                        if st.button("📨 发送测试邮件", key="test_email", use_container_width=True):
                            with st.spinner("发送中…"):
                                try:
                                    _scheduler = ReportScheduler(gateway=_gw if HAS_V10_GATEWAY else None)
                                    _result = _scheduler.generate_now("daily")
                                    from scheduled_report import EmailSender
                                    _email_result = EmailSender(_rpt_config).send(
                                        _result["subject"], _result["report"]
                                    )
                                    if _email_result.get("status") == "ok":
                                        st.success(f"✅ 测试邮件已发送至 {', '.join(_email_result.get('recipients', []))}")
                                    else:
                                        st.error(f"❌ 发送失败: {_email_result.get('message', '')}")
                                except Exception as e:
                                    st.error(f"发送失败: {e}")

                    st.markdown(f"""<div style="font-size:0.5rem;color:#555;font-family:'JetBrains Mono',monospace;margin-top:8px;">
                        <b>报告频率:</b> daily (日报) / weekly (周报) / monthly (月报)<br>
                        <b>推送方式:</b> SMTP 邮件 + 本地文件保存<br>
                        <b>配置方法:</b> 编辑 .env 中 REPORT_SMTP_HOST, REPORT_SMTP_USER, REPORT_RECIPIENTS 等变量
                    </div>""", unsafe_allow_html=True)

            # ── 操作日志 (管理员可见) ──
            if is_admin() and HAS_V10_GATEWAY:
                _log_label = "操作日志" if st.session_state.lang == "zh" else "Audit Log"
                with st.expander(f"📋 {_log_label}", expanded=False):
                    _audit_stats = _gw.audit.get_stats()
                    _recent_logs = _gw.audit.recent(30)

                    # 统计概览
                    _log_c1, _log_c2, _log_c3 = st.columns(3)
                    with _log_c1:
                        st.markdown(f"""<div style="text-align:center;padding:8px;background:#0a0a0a;border:1px solid #222;">
                            <div style="font-size:1rem;font-weight:700;color:#00FF88;font-family:'Space Grotesk',sans-serif;">
                                {_audit_stats.get('total_requests', 0)}</div>
                            <div style="font-size:0.4rem;color:#555;font-family:'JetBrains Mono',monospace;">
                                {'总请求' if st.session_state.lang == 'zh' else 'Total Requests'}</div>
                        </div>""", unsafe_allow_html=True)
                    with _log_c2:
                        _avg_ms = _audit_stats.get('avg_duration_ms', 0)
                        st.markdown(f"""<div style="text-align:center;padding:8px;background:#0a0a0a;border:1px solid #222;">
                            <div style="font-size:1rem;font-weight:700;color:#00A0C8;font-family:'Space Grotesk',sans-serif;">
                                {_avg_ms:.0f}ms</div>
                            <div style="font-size:0.4rem;color:#555;font-family:'JetBrains Mono',monospace;">
                                {'平均响应' if st.session_state.lang == 'zh' else 'Avg Response'}</div>
                        </div>""", unsafe_allow_html=True)
                    with _log_c3:
                        _by_status = _audit_stats.get('by_status', {})
                        _fail_count = _by_status.get('failed', 0) + _by_status.get('error', 0)
                        _fail_color = "#D94040" if _fail_count > 0 else "#4ade80"
                        st.markdown(f"""<div style="text-align:center;padding:8px;background:#0a0a0a;border:1px solid #222;">
                            <div style="font-size:1rem;font-weight:700;color:{_fail_color};font-family:'Space Grotesk',sans-serif;">
                                {_fail_count}</div>
                            <div style="font-size:0.4rem;color:#555;font-family:'JetBrains Mono',monospace;">
                                {'失败请求' if st.session_state.lang == 'zh' else 'Failed'}</div>
                        </div>""", unsafe_allow_html=True)

                    # Agent 分布
                    _by_agent = _audit_stats.get('by_agent', {})
                    if _by_agent:
                        _agent_dist_label = "Agent 请求分布" if st.session_state.lang == "zh" else "Agent Distribution"
                        st.markdown(f"""<div style="font-size:0.5rem;color:#888;font-family:'JetBrains Mono',monospace;
                            margin:12px 0 4px;">{_agent_dist_label}:</div>""", unsafe_allow_html=True)
                        for _ag, _cnt in sorted(_by_agent.items(), key=lambda x: -x[1]):
                            _pct = (_cnt / max(1, _audit_stats.get('total_requests', 1))) * 100
                            _ag_icon = _agent_icons.get(_ag, "🤖")
                            st.markdown(f"""<div style="font-size:0.5rem;font-family:'JetBrains Mono',monospace;
                                color:#e0e0e0;display:flex;align-items:center;gap:6px;padding:2px 0;">
                                <span>{_ag_icon} {_ag}</span>
                                <div style="flex:1;background:#111;height:6px;border-radius:3px;">
                                    <div style="width:{min(_pct, 100):.0f}%;background:#00FF88;height:6px;border-radius:3px;"></div>
                                </div>
                                <span style="color:#888;min-width:50px;text-align:right;">{_cnt} ({_pct:.0f}%)</span>
                            </div>""", unsafe_allow_html=True)

                    # 最近日志列表
                    if _recent_logs:
                        _recent_label = "最近请求" if st.session_state.lang == "zh" else "Recent Requests"
                        st.markdown(f"""<div style="font-size:0.5rem;color:#888;font-family:'JetBrains Mono',monospace;
                            margin:12px 0 4px;">{_recent_label}:</div>""", unsafe_allow_html=True)
                        for _log in reversed(_recent_logs[-15:]):
                            _ts = _log.get("timestamp", "")[:19].replace("T", " ")
                            _lag = _log.get("agent", "?")
                            _lq = _log.get("query", "")[:40]
                            _lst = _log.get("status", "?")
                            _ldur = _log.get("duration_ms", 0)
                            _luser = _log.get("user", "")
                            _st_color = "#4ade80" if _lst == "completed" else "#D94040" if _lst in ("failed","error") else "#f59e0b"
                            st.markdown(f"""<div style="font-size:0.45rem;font-family:'JetBrains Mono',monospace;
                                color:#aaa;padding:3px 0;border-bottom:1px solid #181818;">
                                <span style="color:#555;">{_ts}</span>
                                <span style="color:{_st_color};">●</span>
                                {_agent_icons.get(_lag,'🤖')} {_lag}
                                · {_luser}
                                · <span style="color:#e0e0e0;">{_lq}</span>
                                · {_ldur:.0f}ms
                            </div>""", unsafe_allow_html=True)

                    # API 限流状态
                    _rl_label = "API 限流状态" if st.session_state.lang == "zh" else "API Rate Limit"
                    st.markdown(f"""<div style="font-size:0.5rem;color:#888;font-family:'JetBrains Mono',monospace;
                        margin:12px 0 4px;">{_rl_label}:</div>""", unsafe_allow_html=True)
                    _rl = _gw.rate_limiter.get_usage()
                    _rl_min = _rl.get("global_last_minute", 0)
                    _rl_hr = _rl.get("global_last_hour", 0)
                    _rl_lim_min = _rl.get("global_limit_minute", 300)
                    _rl_lim_hr = _rl.get("global_limit_hour", 5000)
                    st.markdown(f"""<div style="font-size:0.5rem;font-family:'JetBrains Mono',monospace;color:#e0e0e0;">
                        ⚡ 分钟: {_rl_min}/{_rl_lim_min} · 小时: {_rl_hr}/{_rl_lim_hr}
                    </div>""", unsafe_allow_html=True)

            # ── 微信预警通知 (管理员可见) ──
            if is_admin() and HAS_WECHAT:
                _wx_label = "微信预警通知" if st.session_state.lang == "zh" else "WeChat Alert"
                with st.expander(f"📱 {_wx_label}", expanded=False):
                    _wx_webhook = st.text_input(
                        "企业微信 Webhook URL",
                        value=st.session_state.get("wecom_webhook", ""),
                        placeholder="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxx",
                        key="cmd_wecom_webhook",
                    )
                    if _wx_webhook:
                        st.session_state["wecom_webhook"] = _wx_webhook

                    _wx_c1, _wx_c2 = st.columns(2)
                    with _wx_c1:
                        _wx_auto = st.toggle(
                            "🔴 高风险自动推送" if st.session_state.lang == "zh" else "🔴 Auto Push on High Risk",
                            value=st.session_state.get("wecom_auto_alert", False),
                            key="wx_auto_alert",
                        )
                        st.session_state["wecom_auto_alert"] = _wx_auto
                    with _wx_c2:
                        if _wx_webhook and st.button("📨 发送测试", key="wx_test", use_container_width=True):
                            from wechat_notify import send_wecom_bot
                            _ok, _msg = send_wecom_bot(
                                _wx_webhook,
                                "MRARFAI 测试通知",
                                f"✅ 企业微信集成测试成功！\n\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n平台: MRARFAI V10.0",
                            )
                            if _ok:
                                st.success(f"✅ {_msg}")
                            else:
                                st.error(f"❌ {_msg}")

                    if not _wx_webhook:
                        st.markdown(f"""<div style="font-size:0.5rem;color:#555;font-family:'JetBrains Mono',monospace;">
                            配置方法: 企业微信群 → 群设置 → 群机器人 → 添加 → 复制 Webhook 地址
                        </div>""", unsafe_allow_html=True)

            st.stop()

        # ── 选择了某个 Agent → 显示专属界面 ──
        _active = st.session_state.active_agent
        _icon = _agent_icons.get(_active, "⚡")
        _cn = _agent_names_cn.get(_active, _t("cross_agent_collab"))

        # 权限守卫 — 阻止未授权访问
        if _active == "_collab" and not can_use_collab(_user_role):
            st.error(f"⚠ {_t('no_collab')}")
            st.session_state.active_agent = None
            st.rerun()
        elif _active != "_collab" and not can_access_agent(_user_role, _active):
            st.error(f"⚠ {_t('no_access')} — {_cn}")
            st.session_state.active_agent = None
            st.rerun()

        # 返回按钮 + Agent 标题
        _back_col, _title_col = st.columns([1, 5])
        with _back_col:
            if st.button(_t("back"), key="back_to_main", use_container_width=True):
                st.session_state.active_agent = None
                st.rerun()
        with _title_col:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:4px 0;">
                <span style="font-size:1.5rem;">{_icon}</span>
                <span style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;
                      font-weight:700;color:#FFF;letter-spacing:0.04em;">{_cn}</span>
                <span style="font-size:0.5rem;color:#555;font-family:'JetBrains Mono',monospace;
                      border:1px solid #333;padding:2px 6px;">{_t('agent_workspace')}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── AI 模型配置 (自动从 .env 加载，用户可覆盖) ──
        if _active != "_collab" and _active not in _needs_upload:
            _v10_provider = _DEFAULT_PROVIDER
            _v10_api_key = _DEFAULT_API_KEY
            if _v10_api_key:
                # 已内置 Key，显示状态
                st.markdown(f"""<div style="font-size:0.55rem;color:#4ade80;font-family:'JetBrains Mono',monospace;
                    padding:4px 0 8px 0;">✅ {_t('ai_enabled')} · {_v10_provider} · {_t('key_builtin')}</div>""", unsafe_allow_html=True)
            else:
                # 无内置 Key，显示输入框
                _ai_cfg_col1, _ai_cfg_col2 = st.columns([1, 3])
                with _ai_cfg_col1:
                    _v10_provider = st.selectbox("AI模型", ["Claude", "DeepSeek"],
                        key="v10_provider", label_visibility="collapsed")
                with _ai_cfg_col2:
                    _v10_api_key = st.text_input("API Key", type="password",
                        placeholder="sk-ant-... (在 .env 文件中配置可永久保存)",
                        key="v10_api_key", label_visibility="collapsed")
        else:
            _v10_provider = _DEFAULT_PROVIDER
            _v10_api_key = _DEFAULT_API_KEY

        # 初始化该 Agent 对话历史
        _chat_key = f"chat_{_active}"
        if _chat_key not in st.session_state.v10_chat_history:
            st.session_state.v10_chat_history[_chat_key] = []
        _history = st.session_state.v10_chat_history[_chat_key]

        # ── 销售/风控/战略: 需要上传 Excel 的 Agent ──
        if _active in _needs_upload:
            if not can_upload(_user_role):
                st.warning(f"⚠ {_t('no_upload')}")
                st.stop()
            st.markdown(f"""<div style="font-size:0.65rem;color:#888;font-family:'JetBrains Mono',monospace;
                margin-bottom:12px;">{_t('upload_hint_v9')}</div>""",
                unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                _t("upload_excel"), type=['xlsx'],
                accept_multiple_files=True, key=f'files_{_active}',
                label_visibility="collapsed",
            )

            _fl2, _fc2, _fr2 = st.columns([1, 2, 1])
            with _fc2:
                _b21, _b22 = st.columns(2)
                with _b21:
                    ai_enabled = st.toggle("AI 叙事", value=False, key=f"ai_toggle_{_active}")
                with _b22:
                    use_multi = st.toggle("Multi-Agent", value=False, key=f"multi_{_active}")

            if ai_enabled:
                _al2, _ac2, _ar2 = st.columns([1, 3, 1])
                with _ac2:
                    _ai21, _ai22 = st.columns(2)
                    with _ai21:
                        ai_provider = st.selectbox("模型", ['DeepSeek', 'Claude'], label_visibility="collapsed", key=f"aip_{_active}")
                    with _ai22:
                        api_key = st.text_input("Key", type="password", label_visibility="collapsed", placeholder="sk-...", key=f"aik_{_active}")
            else:
                ai_provider, api_key = 'DeepSeek', None

            st.session_state["ai_provider"] = ai_provider
            st.session_state["api_key"] = api_key or ""
            st.session_state["use_multi_agent"] = use_multi

            if not uploaded_files or len(uploaded_files) < 2:
                st.info(f"📎 {_t('upload_please')}")
                st.stop()

            # ── 加载数据后跳到下面的 tabs 逻辑 ──
            # 这里 break 出 V10 Gateway 块，让下面的原始 tab 逻辑接管
            # 先做文件分类
            _detections = []
            for f in uploaded_files[:2]:
                fb = f.read(); f.seek(0)
                _detections.append((f.name, _detect_file_type(fb), fb))

            _rev_found = _qty_found = None
            _leftovers = []
            for name, ftype, fb in _detections:
                if ftype == 'revenue' and not _rev_found:
                    _rev_found = (name, fb)
                elif ftype == 'quantity' and not _qty_found:
                    _qty_found = (name, fb)
                else:
                    _leftovers.append((name, ftype, fb))
            for name, ftype, fb in _leftovers:
                if not _rev_found:
                    _rev_found = (name, fb)
                elif not _qty_found:
                    _qty_found = (name, fb)

            if not _rev_found or not _qty_found:
                _info = " / ".join([f"{n}→{t}" for n, t, _ in _detections])
                st.error(f"⚠ 无法识别文件类型（{_info}），请上传金额报表 + 数量报表")
                st.stop()

            _rev_bytes = _rev_found[1]
            _qty_bytes = _qty_found[1]
            st.caption(f"✓ 金额: {_rev_found[0]}　|　✓ 数量: {_qty_found[0]}")

            with st.spinner("🌿 数据加载 + 深度分析中..."):
                try:
                    data, results, benchmark, forecast = run_full_analysis(_rev_bytes, _qty_bytes)
                except Exception as e:
                    st.error(f"⚠️ 数据加载失败: {e}")
                    st.stop()

            active = sum(1 for c in data['客户金额'] if c['年度金额'] > 0)
            st.markdown(f"""<div class="status-bar"><div class="status-dot"></div>
                <span class="status-text">DATA LOADED</span>
                <span class="status-meta">{active} clients · V10.0 · {datetime.now().strftime('%H:%M:%S')}</span>
            </div>""", unsafe_allow_html=True)

            # 根据进入的 Agent 显示对应 tabs（完整 V9 分析功能）
            if _active == "sales":
                _sub_tabs = st.tabs(["🧠 Agent Chat", "📊 总览", "👥 客户分析", "💰 价量分解",
                    "📈 增长机会", "🏭 产品结构", "🌍 区域分析", "📥 导出"])

                # ── Tab: Agent Chat ──
                with _sub_tabs[0]:
                    render_chat_tab(data, results, benchmark, forecast, ai_provider, api_key)

                # ── Tab: 总览 (完整 V9 版) ──
                with _sub_tabs[1]:
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
                            fig.add_trace(go.Bar(x=MONTHS, y=m_data,
                                marker=dict(color=[SP_GREEN if v == max(m_data) else "rgba(140,191,63,0.30)" for v in m_data]),
                                text=[f"{v:,.0f}" for v in m_data], textposition="outside", textfont=dict(size=10, color=TEXT2)))
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

                # ── Tab: 客户分析 (完整 V9 版 — 分级+集中度曲线) ──
                with _sub_tabs[2]:
                    tiers = results['客户分级']
                    tier_counts = {t: sum(1 for x in tiers if x['等级']==t) for t in ['A','B','C']}
                    tier_rev = {t: sum(x['年度金额'] for x in tiers if x['等级']==t) for t in ['A','B','C']}
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(f"A级 · {tier_counts['A']}家", f"{tier_rev['A']:,.0f}万", f"占比 {tier_rev['A']/data['总营收']*100:.1f}%")
                    c2.metric(f"B级 · {tier_counts['B']}家", f"{tier_rev['B']:,.0f}万")
                    c3.metric(f"C级 · {tier_counts['C']}家", f"{tier_rev['C']:,.0f}万")
                    c4.metric("Top4 集中度", f"{tiers[3]['累计占比']}%", "⚠️ 偏高" if tiers[3]['累计占比']>50 else "✅ 健康")
                    filter_tier = st.multiselect("筛选等级", ['A','B','C'], default=['A','B','C'], key="tier_filter_sales")
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
                        fig.add_trace(go.Scatter(x=list(range(1, len(cum)+1)), y=cum, mode='lines+markers',
                            line=dict(color=SP_GREEN, width=2), marker=dict(size=5, color=SP_GREEN),
                            fill='tozeroy', fillcolor='rgba(0,255,136,0.06)'))
                        fig.add_hline(y=80, line_dash="dash", line_color=ORANGE,
                            annotation_text="80% 帕累托线", annotation_font=dict(size=10, color=ORANGE))
                        fig.update_layout(**plotly_layout("客户集中度曲线", 350, False))
                        fig.update_xaxes(title_text="客户排名")
                        fig.update_yaxes(title_text="累计占比 (%)")
                        st.plotly_chart(fig, use_container_width=True)

                # ── Tab: 价量分解 (完整 V9 版 — 图表+表格) ──
                with _sub_tabs[3]:
                    pv = results['价量分解']
                    if pv:
                        quality_counts = {}
                        for p in pv:
                            _q = p.get('质量评估', '未知')
                            quality_counts[_q] = quality_counts.get(_q, 0) + 1
                        st.markdown('<div class="section-header">PRICE-VOLUME DECOMPOSITION</div>', unsafe_allow_html=True)
                        pv_df = pd.DataFrame(pv)
                        st.dataframe(pv_df, use_container_width=True, hide_index=True)
                        if HAS_PLOTLY:
                            fig = go.Figure()
                            for _i2, p in enumerate(pv[:10]):
                                color = SP_GREEN if '优质' in p.get('质量评估', '') else (SP_RED if '齐跌' in p.get('质量评估', '') else SP_BLUE)
                                fig.add_trace(go.Bar(name=p['客户'], x=[p['客户']], y=[p.get('年度金额', 0)],
                                    marker_color=color, showlegend=False,
                                    text=[f"{p.get('年度金额', 0):,.0f}"], textposition="outside"))
                            fig.update_layout(**plotly_layout("客户价量分布", 380, False))
                            st.plotly_chart(fig, use_container_width=True)

                # ── Tab: 增长机会 (完整 V9 版) ──
                with _sub_tabs[4]:
                    growth = results['增长机会']
                    st.markdown(f'<div class="section-header">GROWTH OPPORTUNITIES · {len(growth)} FOUND</div>', unsafe_allow_html=True)
                    for g in growth:
                        with st.expander(f"📈 **{g.get('客户', '未知')}** — {g.get('类型', '')} — {g.get('说明', '')}", expanded=False):
                            for k, v in g.items():
                                if k not in ('客户',):
                                    st.markdown(f"**{k}**: {v}")

                # ── Tab: 产品结构 (完整 V9 版 — 带图表) ──
                with _sub_tabs[5]:
                    pm = data.get('产品结构', data.get('类别YoY', []))
                    if pm:
                        st.markdown('<div class="section-header">PRODUCT MIX</div>', unsafe_allow_html=True)
                        pm_df = pd.DataFrame(pm)
                        st.dataframe(pm_df, use_container_width=True, hide_index=True)
                        if HAS_PLOTLY:
                            fig = go.Figure()
                            for _i3, p in enumerate(pm):
                                fig.add_trace(go.Bar(x=[p['类别']], y=[p['2025金额']],
                                    marker_color=CHART_COLORS[_i3 % len(CHART_COLORS)], name=p['类别'],
                                    text=[f"{p['2025金额']:,.0f}"], textposition="outside"))
                            fig.update_layout(**plotly_layout("2025 产品结构（万元）", 380))
                            st.plotly_chart(fig, use_container_width=True)

                # ── Tab: 区域分析 (完整 V9 版 — 带HHI+图表) ──
                with _sub_tabs[6]:
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
                            marker_color=[SP_GREEN if _i4 == 0 else "rgba(140,191,63,0.25)" for _i4 in range(len(regions))],
                            text=[f"{r['金额']:,.0f}" for r in regions], textposition="outside", textfont=dict(size=10)))
                        fig.update_layout(**plotly_layout("区域出货分布（万元）", 380, False))
                        st.plotly_chart(fig, use_container_width=True)

                # ── Tab: 导出 (完整 V9 版 — PDF+JSON+Prompt) ──
                with _sub_tabs[7]:
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
                    full = full.replace("Agent v2.0", "Agent v10.0").replace("智能分析系统 v2.0", "智能分析系统 v10.0")
                    full = full.replace("Agent v4.0", "Agent v10.0").replace("Agent v8.0", "Agent v10.0").replace("Agent v9.0", "Agent v10.0")
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

            # ── Risk Agent: 完整 V9 预警+健康评分+异常检测 ──
            elif _active == "risk":
                _sub_tabs = st.tabs(["🚨 预警中心", "❤️ 健康评分", "🔬 异常检测"])

                with _sub_tabs[0]:
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

                with _sub_tabs[1]:
                    render_health_dashboard(data, results)

                with _sub_tabs[2]:
                    st.caption("基于统计模型 (Z-Score · IQR · 趋势断裂 · 波动率 · 系统性风险)")
                    render_anomaly_dashboard(data, results)

            # ── Strategist Agent: 完整 V9 对标+预测+CEO备忘录 ──
            elif _active == "strategist":
                _sub_tabs = st.tabs(["🌐 行业对标", "🔮 预测", "✏️ CEO备忘录"])

                # ── 行业对标 (完整 V9 版) ──
                with _sub_tabs[0]:
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

                # ── 预测 (完整 V9 版 — 含图表) ──
                with _sub_tabs[1]:
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
                    for _i5, (name, sc) in enumerate(scenarios.items()):
                        with cols[_i5]:
                            st.metric(name.split('(')[0], f"{sc['全年预测']/10000:.1f}亿")
                            st.caption(sc['假设'])

                # ── CEO备忘录 (完整 V9 版 — AI生成+模板) ──
                with _sub_tabs[2]:
                    if ai_enabled and api_key:
                        if st.button("🧠 用AI生成深度叙事", type="primary", use_container_width=True, key="ai_memo_strat"):
                            narrator = AINarrator(data, results, benchmark, forecast)
                            with st.spinner("AI 分析中..."):
                                ai_text = narrator.generate(api_key, ai_provider.lower())
                            st.markdown(ai_text)
                            st.download_button("📥 下载", ai_text, "ai_memo.md", "text/markdown", key="dl_ai_memo_strat")
                    narrator = AINarrator(data, results, benchmark, forecast)
                    memo = narrator._template_narrative()
                    with st.expander("📄 内置战略备忘录", expanded=not ai_enabled):
                        st.markdown(memo)

            st.stop()

        elif _active == "_collab":
            # ── 协作场景界面 ──
            _scenarios = _gw.collaboration.scenarios
            st.markdown(f"""<div style="font-size:0.7rem;color:#888;font-family:'JetBrains Mono',monospace;
                margin-bottom:12px;">{_t('collab_hint')}</div>""", unsafe_allow_html=True)

            # 预定义场景按钮
            _scenario_list = list(_scenarios.items())
            _sc_cols = st.columns(min(4, len(_scenario_list)))
            for _i, (_sid, _scfg) in enumerate(_scenario_list):
                with _sc_cols[_i % len(_sc_cols)]:
                    _chain_icons = " → ".join([_agent_icons.get(a,"🤖") for a in _scfg["chain"]])
                    _trig_kws = _scfg.get("trigger_keywords", [])
                    if st.button(f"⚡ {_scfg['name']}", key=f"collab_{_sid}", use_container_width=True):
                        _trig = _trig_kws[0] if _trig_kws else _scfg["name"]
                        _history.append({"role": "user", "content": f"[协作] {_scfg['name']}: {_trig}"})
                        _resp = _gw.ask(_trig, user=_current_user.get("username", "anonymous"), provider=_v10_provider.lower(), api_key=_v10_api_key, chat_history=_history)
                        if _resp["type"] == "collaboration":
                            _res = _resp["result"]
                            _disp = _res.get("synthesis", "")
                            for _aname, _ares in _res.get("agent_results", {}).items():
                                _disp += f"\n\n**{_agent_icons.get(_aname,'🤖')} {_agent_names_cn.get(_aname,_aname)}**:\n"
                                try:
                                    _disp += json.dumps(json.loads(_ares) if isinstance(_ares, str) else _ares,
                                                        ensure_ascii=False, indent=2)[:600]
                                except Exception:
                                    _disp += str(_ares)[:600]
                            _history.append({"role": "assistant", "content": _disp,
                                "agent": "platform", "duration": _resp.get("duration_ms",0)})
                        st.rerun()
                    st.caption(f"{_chain_icons}\n{_scfg['description']}")

            # ── 自定义协作链构建器 ──
            _custom_label = "自定义协作链" if st.session_state.lang == "zh" else "Custom Collaboration Chain"
            with st.expander(f"🔗 {_custom_label}", expanded=False):
                _avail_agents = list(_agent_names_cn.keys())
                _agent_display = {k: f"{_agent_icons.get(k,'')} {v}" for k, v in _agent_names_cn.items()}

                _chain_hint = "选择2-5个Agent组成自定义协作链" if st.session_state.lang == "zh" else "Select 2-5 Agents to build a custom chain"
                st.markdown(f"""<div style="font-size:0.55rem;color:#888;font-family:'JetBrains Mono',monospace;">
                    {_chain_hint}</div>""", unsafe_allow_html=True)

                _selected = st.multiselect(
                    "Agent Chain",
                    options=_avail_agents,
                    format_func=lambda x: _agent_display.get(x, x),
                    key="custom_chain_agents",
                    label_visibility="collapsed",
                    max_selections=5,
                )

                if _selected and len(_selected) >= 2:
                    _chain_preview = " → ".join([f"{_agent_icons.get(a,'')} {_agent_names_cn.get(a,a)}" for a in _selected])
                    st.markdown(f"""<div style="font-size:0.65rem;color:#4ade80;font-family:'JetBrains Mono',monospace;
                        padding:8px 0;">链路: {_chain_preview}</div>""", unsafe_allow_html=True)

                    _cc1, _cc2 = st.columns([2, 1])
                    with _cc1:
                        _chain_name = st.text_input(
                            "场景名称",
                            value=f"自定义: {'+'.join(_agent_names_cn.get(a, a) for a in _selected)}",
                            key="custom_chain_name",
                            label_visibility="collapsed",
                            placeholder="输入协作场景名称...")
                    with _cc2:
                        _run_label = "执行" if st.session_state.lang == "zh" else "Run"
                        if st.button(f"⚡ {_run_label}", key="run_custom_chain", use_container_width=True):
                            # 创建临时场景并执行
                            _custom_scenario = {
                                "name": _chain_name,
                                "chain": _selected,
                                "description": f"自定义: {' → '.join(_selected)}",
                                "trigger_keywords": [],
                            }
                            _history.append({"role": "user", "content": f"[自定义协作] {_chain_name}"})
                            _result = _gw.collaboration.execute_chain(_custom_scenario, _chain_name, user=_current_user.get("username", "anonymous"))
                            if _v10_api_key:
                                _result["synthesis"] = _gw._llm_synthesize_collab(
                                    _custom_scenario,
                                    {k: v for k, v in _result.get("agent_results", {}).items()},
                                    _chain_name, _v10_provider.lower(), _v10_api_key)
                            _disp = _result.get("synthesis", "")
                            for _aname, _ares in _result.get("agent_results", {}).items():
                                _disp += f"\n\n**{_agent_icons.get(_aname,'🤖')} {_agent_names_cn.get(_aname,_aname)}**:\n"
                                try:
                                    _disp += json.dumps(json.loads(_ares) if isinstance(_ares, str) else _ares,
                                                        ensure_ascii=False, indent=2)[:600]
                                except Exception:
                                    _disp += str(_ares)[:600]
                            _history.append({"role": "assistant", "content": _disp,
                                "agent": "platform", "duration": 0})
                            st.rerun()
                elif _selected and len(_selected) < 2:
                    _min_hint = "至少选择2个Agent" if st.session_state.lang == "zh" else "Select at least 2 Agents"
                    st.caption(_min_hint)

            # ── 协作历史回溯 ──
            _mem_label = "协作历史" if st.session_state.lang == "zh" else "Collaboration History"
            with st.expander(f"📚 {_mem_label}", expanded=False):
                _mem = _gw.collaboration.memory
                _mem_records = _mem.list_recent(20)

                if not _mem_records:
                    _no_mem = "暂无协作记录。执行协作场景后，结果将自动存档。" if st.session_state.lang == "zh" else "No records yet. Run a collaboration scenario to create one."
                    st.markdown(f"""<div style="font-size:0.55rem;color:#555;font-family:'JetBrains Mono',monospace;">
                        {_no_mem}</div>""", unsafe_allow_html=True)
                else:
                    _mem_stats = _mem.get_stats()
                    _total_label = "总记录" if st.session_state.lang == "zh" else "Total records"
                    st.markdown(f"""<div style="font-size:0.5rem;color:#555;font-family:'JetBrains Mono',monospace;margin-bottom:8px;">
                        📊 {_total_label}: {_mem_stats.get('total_records', 0)}</div>""", unsafe_allow_html=True)

                    for _rec in _mem_records:
                        _ts = _rec.get("timestamp", "")[:16].replace("T", " ")
                        _sc_name = _rec.get("scenario", "?")
                        _agents = _rec.get("total_agents", 0)
                        _done = _rec.get("completed", 0)
                        _status_dot = "🟢" if _done == _agents else "🟡"
                        _mid = _rec.get("memory_id", "")

                        _rec_col1, _rec_col2 = st.columns([4, 1])
                        with _rec_col1:
                            st.markdown(f"""<div style="font-size:0.55rem;font-family:'JetBrains Mono',monospace;
                                color:#e0e0e0;padding:4px 0;border-bottom:1px solid #222;">
                                {_status_dot} <span style="color:#00A0C8;">{_sc_name}</span>
                                · {_agents} agents · {_ts}
                                · <span style="color:#555;">{_rec.get('user','')}</span></div>""",
                                unsafe_allow_html=True)
                        with _rec_col2:
                            if st.button("🔍", key=f"mem_{_mid}", help="查看详情"):
                                _full = _mem.load(_mid)
                                if _full:
                                    st.session_state[f"_show_mem_{_mid}"] = True

                        # 展开详情
                        if st.session_state.get(f"_show_mem_{_mid}", False):
                            _full = _mem.load(_mid)
                            if _full:
                                st.markdown(f"""<div style="font-size:0.5rem;font-family:'JetBrains Mono',monospace;
                                    color:#888;background:#0a0a0a;padding:12px;margin:4px 0 12px;border:1px solid #222;">
                                    <b>场景:</b> {_full.get('scenario','')}<br>
                                    <b>描述:</b> {_full.get('description','')}<br>
                                    <b>时间:</b> {_full.get('timestamp','')}<br>
                                    <b>用户:</b> {_full.get('user','')}<br>
                                    <b>执行步骤:</b> {_full.get('completed',0)}/{_full.get('total_agents',0)} 完成
                                </div>""", unsafe_allow_html=True)

                                # Agent 结果
                                for _aname, _ares in _full.get("agent_results", {}).items():
                                    _a_icon = _agent_icons.get(_aname, "🤖")
                                    _a_cn = _agent_names_cn.get(_aname, _aname)
                                    st.markdown(f"""<div style="font-size:0.5rem;font-family:'JetBrains Mono',monospace;
                                        color:#00FF88;margin-top:6px;">▸ {_a_icon} {_a_cn}</div>""",
                                        unsafe_allow_html=True)
                                    try:
                                        _parsed = json.loads(_ares) if isinstance(_ares, str) else _ares
                                        st.json(_parsed)
                                    except Exception:
                                        st.code(str(_ares)[:500], language="text")

                                # 综合分析
                                _syn = _full.get("synthesis", "")
                                if _syn:
                                    _syn_label = "综合分析" if st.session_state.lang == "zh" else "Synthesis"
                                    st.markdown(f"**📝 {_syn_label}:**")
                                    st.markdown(_syn[:1000])

                                if st.button("收起", key=f"close_mem_{_mid}"):
                                    st.session_state[f"_show_mem_{_mid}"] = False
                                    st.rerun()

        else:
            # ── V10 独立 Agent 界面 ──

            # ── Excel 数据上传（替换样本数据 — 仅有上传权限时显示） ──
            _upload_hints = {
                "procurement": "Sheet1「供应商」: 名称|类别|交期天数|质量评分|价格指数|准时率|不良率|信用等级\nSheet2「订单」: PO号|供应商|物料|数量|总额(万)|状态|创建日期|期望交期",
                "finance": "Sheet1「应收」: 客户|发票号|金额(万)|货币|到期日|状态|逾期天数\nSheet2「毛利」: 产品|客户|营收(万)|成本(万)",
                "quality": "Sheet1「良率」: 产线|产品|月份|总产数|合格数|缺陷类型1|缺陷类型2|...\nSheet2「退货」: 客户|产品|数量|原因|日期|严重度",
                "market": "Sheet1「竞品」: 公司|股票代码|营收(亿)|增速%|主要客户(逗号分隔)|优势(逗号分隔)|劣势(逗号分隔)|市场份额%",
            }
            _data_key = f"{_active}_custom_data"
            if not can_upload(_user_role):
                if _data_key in st.session_state:
                    _dfs = st.session_state[_data_key]
                    st.info(f"📊 使用已上传的自定义数据 ({sum(len(df) for df in _dfs.values())} 条记录)")
            else:
                with st.expander(f"📎 {_t('upload_data')}", expanded=False):
                    st.markdown(f"""<div style="font-size:0.55rem;color:#888;font-family:'JetBrains Mono',monospace;
                        white-space:pre-line;line-height:1.5;">{_upload_hints.get(_active, '')}</div>""", unsafe_allow_html=True)
                    _uploaded_data = st.file_uploader(
                        "上传 Excel", type=["xlsx"], key=f"upload_{_active}",
                        label_visibility="collapsed")
                    if _uploaded_data:
                        try:
                            _xls = pd.ExcelFile(_uploaded_data)
                            _sheets = _xls.sheet_names
                            _dfs = {s: pd.read_excel(_xls, s) for s in _sheets}
                            st.session_state[_data_key] = _dfs

                            # 用 from_dataframes 创建新引擎
                            _sheet_list = list(_dfs.values())
                            if _active == "procurement":
                                from agent_procurement import ProcurementEngine as _PE
                                _new_engine = _PE.from_dataframes(
                                    suppliers_df=_sheet_list[0] if len(_sheet_list) > 0 else None,
                                    orders_df=_sheet_list[1] if len(_sheet_list) > 1 else None)
                                _gw.update_engine("procurement", _new_engine)
                            elif _active == "finance":
                                from agent_finance import FinanceEngine as _FE
                                _new_engine = _FE.from_dataframes(
                                    ar_df=_sheet_list[0] if len(_sheet_list) > 0 else None,
                                    margin_df=_sheet_list[1] if len(_sheet_list) > 1 else None)
                                _gw.update_engine("finance", _new_engine)
                            elif _active == "quality":
                                from agent_quality import QualityEngine as _QE
                                _new_engine = _QE.from_dataframes(
                                    yields_df=_sheet_list[0] if len(_sheet_list) > 0 else None,
                                    returns_df=_sheet_list[1] if len(_sheet_list) > 1 else None)
                                _gw.update_engine("quality", _new_engine)
                            elif _active == "market":
                                from agent_market import MarketEngine as _ME
                                _new_engine = _ME.from_dataframes(
                                    competitors_df=_sheet_list[0] if len(_sheet_list) > 0 else None)
                                _gw.update_engine("market", _new_engine)

                            st.success(f"✅ 数据已加载！{len(_sheets)} 个Sheet · {sum(len(df) for df in _dfs.values())} 条记录")
                            for _sn, _sdf in _dfs.items():
                                st.caption(f"Sheet「{_sn}」: {len(_sdf)} 行 × {len(_sdf.columns)} 列")
                        except Exception as _ue:
                            st.error(f"⚠️ Excel 解析失败: {_ue}")
                    elif _data_key in st.session_state:
                        _dfs = st.session_state[_data_key]
                        st.info(f"📊 使用已上传的自定义数据 ({sum(len(df) for df in _dfs.values())} 条记录)")

            # ── 快捷功能按钮 ──
            _quick_queries = {
                "procurement": [("供应商评估", "供应商评估"), ("PO跟踪", "PO跟踪进度"), ("比价分析", "供应商比价分析"), ("延迟预警", "采购延迟预警")],
                "quality": [("良率监控", "良率趋势如何"), ("退货分析", "退货率分析"), ("投诉分类", "客户投诉分类"), ("根因追溯", "品质根因追溯")],
                "finance": [("应收跟踪", "应收账款逾期情况"), ("毛利分析", "毛利率分析"), ("现金流预测", "未来3个月现金流预测"), ("发票匹配", "发票匹配查询")],
                "market": [("竞品监控", "竞品市场份额对比"), ("行业趋势", "2026行业趋势报告"), ("舆情追踪", "舆情追踪分析"), ("市场概览", "ODM市场全景")],
            }
            _qs = _quick_queries.get(_active, [])
            if _qs:
                _qcols = st.columns(len(_qs))
                for _i, (_label, _query) in enumerate(_qs):
                    with _qcols[_i]:
                        if st.button(f"{_label}", key=f"aq_{_active}_{_i}", use_container_width=True):
                            _history.append({"role": "user", "content": _query})
                            _resp = _gw.ask(_query, user=_current_user.get("username", "anonymous"), provider=_v10_provider.lower(), api_key=_v10_api_key, chat_history=_history)
                            _ans = _resp.get("answer", "")
                            try:
                                _disp = json.dumps(json.loads(_ans) if isinstance(_ans, str) else _ans,
                                                   ensure_ascii=False, indent=2)
                            except Exception:
                                _disp = str(_ans)
                            _history.append({"role": "assistant", "content": _disp,
                                "agent": _resp.get("agent",""), "duration": _resp.get("duration_ms",0)})
                            st.rerun()

        # ── 对话历史 ──
        for _msg in _history:
            if _msg["role"] == "user":
                st.markdown(f"""<div style="background:#1a1f2e;padding:10px 14px;margin:6px 0;
                    border-left:3px solid #FFF;font-size:0.82rem;color:#ccc;">
                    🧑 {_msg['content']}</div>""", unsafe_allow_html=True)
            else:
                _du = _msg.get("duration", 0)
                _ag = _msg.get("agent", _active)
                _badge = f"""<span style="font-size:0.5rem;font-family:'JetBrains Mono',monospace;
                    color:#888;border:1px solid #333;padding:1px 6px;">
                    {_agent_icons.get(_ag,'🤖')} {_agent_names_cn.get(_ag,_ag)} · {_du:.0f}ms</span>"""
                st.markdown(f"""<div style="background:#0d1117;padding:12px 16px;margin:6px 0;
                    border:1px solid rgba(255,255,255,0.06);font-size:0.8rem;color:#ddd;">
                    <div style="margin-bottom:6px;">{_badge}</div>
                    <div style="white-space:pre-wrap;line-height:1.5;">{_msg['content']}</div>
                </div>""", unsafe_allow_html=True)

        # ── 自由输入框 ──
        _v10q = st.chat_input(f"{_t('ask_agent')} {_cn}", key=f"chat_input_{_active}")
        if _v10q:
            _history.append({"role": "user", "content": _v10q})
            _resp = _gw.ask(_v10q, user=_current_user.get("username", "anonymous"), provider=_v10_provider.lower(), api_key=_v10_api_key, chat_history=_history)
            if _resp["type"] == "collaboration":
                _res = _resp["result"]
                _disp = _res.get("synthesis", "")
                _history.append({"role": "assistant", "content": _disp,
                    "agent": "platform", "duration": _resp.get("duration_ms",0)})
            else:
                _ans = _resp.get("answer", "")
                try:
                    _disp = json.dumps(json.loads(_ans) if isinstance(_ans, str) else _ans,
                                       ensure_ascii=False, indent=2)
                except Exception:
                    _disp = str(_ans)
                _history.append({"role": "assistant", "content": _disp,
                    "agent": _resp.get("agent",""), "duration": _resp.get("duration_ms",0)})
            st.rerun()

        st.stop()

    except Exception as _e:
        st.error(f"V10 Command Center 加载失败: {_e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ── Fallback: 无 V10 Gateway 时的旧模式 ──
uploaded_files = st.file_uploader("上传报表", type=['xlsx'], accept_multiple_files=True, key='files', label_visibility="collapsed")
ai_enabled = False
use_multi = False
ai_provider, api_key = 'DeepSeek', None
st.session_state["ai_provider"] = ai_provider
st.session_state["api_key"] = ""
st.session_state["use_multi_agent"] = use_multi
if not uploaded_files or len(uploaded_files) < 2:
    st.stop()

# ── 智能文件分类 ──
_detections = []
for f in uploaded_files[:2]:
    fb = f.read(); f.seek(0)
    _detections.append((f.name, _detect_file_type(fb), fb))

_rev_found = _qty_found = None
_leftovers = []
for name, ftype, fb in _detections:
    if ftype == 'revenue' and not _rev_found:
        _rev_found = (name, fb)
    elif ftype == 'quantity' and not _qty_found:
        _qty_found = (name, fb)
    else:
        _leftovers.append((name, ftype, fb))
for name, ftype, fb in _leftovers:
    if not _rev_found:
        _rev_found = (name, fb)
    elif not _qty_found:
        _qty_found = (name, fb)

if not _rev_found or not _qty_found:
    _info = " / ".join([f"{n}→{t}" for n, t, _ in _detections])
    st.error(f"⚠ 无法识别文件类型（{_info}），请上传 Sprocomm 金额报表 + 数量报表")
    st.stop()

_rev_bytes = _rev_found[1]
_qty_bytes = _qty_found[1]

st.caption(f"✓ 金额: {_rev_found[0]}　|　✓ 数量: {_qty_found[0]}")

with st.spinner("🌿 数据加载 + 深度分析中..."):
    try:
        data, results, benchmark, forecast = run_full_analysis(_rev_bytes, _qty_bytes)
    except ValueError as e:
        err_msg = str(e)
        if "Worksheet named" in err_msg:
            sheet_name = err_msg.split("'")[1] if "'" in err_msg else "未知"
            st.error(f"📊 Excel 工作表不匹配")
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                 color:#8a8a8a; padding:12px; border:1px solid rgba(217,64,64,0.15);
                 background:rgba(217,64,64,0.04); margin-top:8px;">
                <p>找不到工作表 "<strong style="color:#D94040;">{sheet_name}</strong>"</p>
                <p style="margin-top:8px;">可能原因:</p>
                <p>· 上传的文件不是 Sprocomm 销售报表</p>
                <p>· Excel 文件中的 Sheet 名称已被修改</p>
                <p style="margin-top:8px;">请检查文件后重新上传。</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"⚠️ 数据格式错误: {err_msg}")
        st.stop()
    except Exception as e:
        st.error("⚠️ 数据加载失败")
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
             color:#6a6a6a; padding:12px; border:1px solid rgba(138,138,138,0.15);
             background:rgba(138,138,138,0.04); margin-top:8px;">
            <p>错误类型: <strong>{type(e).__name__}</strong></p>
            <p>详情: {str(e)[:200]}</p>
            <p style="margin-top:8px; color:#8a8a8a;">请检查 Excel 文件格式是否正确，或联系管理员。</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

active = sum(1 for c in data['客户金额'] if c['年度金额'] > 0)
st.markdown(f"""
<div class="status-bar">
    <div class="status-dot"></div>
    <span class="status-text">DATA LOADED</span>
    <span class="status-meta">{active} clients · 12 dimensions · V10.0 Enterprise · {datetime.now().strftime('%H:%M:%S')}</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Tabs — V10.0 布局
# ============================================================
_tab_labels = [
    "🧠 Agent", "📊 总览", "👥 客户分析", "💰 价量分解", "🚨 预警中心",
    "📈 增长机会", "🏭 产品结构", "🌍 区域分析",
    "🌐 行业对标", "🔮 预测", "✏️ CEO备忘录",
    "❤️ 健康评分", "🔬 异常检测", "🔔 通知推送", "🎨 品牌设置", "📥 导出",
]
if HAS_V10_GATEWAY:
    _tab_labels.append("🏗️ V10 Platform")
tabs = st.tabs(_tab_labels)


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
    full = full.replace("Agent v2.0", "Agent v10.0").replace("智能分析系统 v2.0", "智能分析系统 v10.0")
    full = full.replace("Agent v4.0", "Agent v10.0").replace("Agent v8.0", "Agent v10.0")
    full = full.replace("Agent v9.0", "Agent v10.0")
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


# ---- Tab 16: V10 Platform (可选) ----
if HAS_V10_GATEWAY:
    with tabs[16]:
        try:
            gw = get_gateway()
            card = gw.get_platform_card()
            stats = gw.get_stats()

            # ── 顶部：统一智能对话入口 ──
            st.markdown("""
            <div style="background:linear-gradient(135deg,#0d1117,#161b22);padding:24px 28px;
                        border:1px solid rgba(255,255,255,0.08);margin-bottom:20px;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <span style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;
                          font-weight:700;color:#FFF;letter-spacing:0.06em;">MRARFAI COMMAND CENTER</span>
                    <span style="font-size:0.6rem;color:#555;font-family:'JetBrains Mono',monospace;
                          border:1px solid #333;padding:2px 8px;">V10.0</span>
                </div>
                <div style="font-size:0.7rem;color:#555;font-family:'JetBrains Mono',monospace;">
                    输入任何业务问题 → 自动路由到最佳 Agent → 返回分析结果
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 初始化对话历史
            if "v10_chat_history" not in st.session_state:
                st.session_state.v10_chat_history = []

            # Agent 图标映射
            _agent_icons = {
                "sales": "📈", "procurement": "🛒", "quality": "🔍",
                "finance": "💰", "market": "📊", "risk": "🚨", "strategist": "🔮",
            }
            _agent_names = {
                "sales": "销售分析师", "procurement": "采购管理", "quality": "品质检测",
                "finance": "财务分析", "market": "市场情报", "risk": "风控预警", "strategist": "战略顾问",
            }

            # 展示历史对话
            for msg in st.session_state.v10_chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""<div style="background:#1a1f2e;padding:12px 16px;margin:8px 0;
                        border-left:3px solid #FFF;font-size:0.85rem;color:#ccc;">
                        🧑 {msg['content']}</div>""", unsafe_allow_html=True)
                else:
                    agent = msg.get("agent", "platform")
                    icon = _agent_icons.get(agent, "🤖")
                    name = _agent_names.get(agent, agent)
                    conf = msg.get("confidence", 0)
                    duration = msg.get("duration", 0)
                    _type = msg.get("type", "single_agent")

                    # 路由信息
                    _route_badge = f"""<span style="display:inline-block;font-size:0.55rem;
                        font-family:'JetBrains Mono',monospace;color:#888;
                        border:1px solid #333;padding:1px 6px;margin-left:8px;">
                        {icon} {name} · 置信度 {conf:.0%} · {duration:.0f}ms</span>"""

                    if _type == "collaboration":
                        _route_badge = f"""<span style="display:inline-block;font-size:0.55rem;
                            font-family:'JetBrains Mono',monospace;color:#f0b040;
                            border:1px solid #665520;padding:1px 6px;margin-left:8px;">
                            ⚡ 跨Agent协作 · {msg.get('scenario', '')} · {duration:.0f}ms</span>"""

                    st.markdown(f"""<div style="background:#0d1117;padding:14px 18px;margin:8px 0;
                        border:1px solid rgba(255,255,255,0.06);font-size:0.82rem;color:#ddd;">
                        <div style="margin-bottom:8px;">{_route_badge}</div>
                        <div style="white-space:pre-wrap;line-height:1.6;">{msg['content']}</div>
                    </div>""", unsafe_allow_html=True)

            # 输入框
            _v10_input = st.chat_input("输入业务问题... 例如：竞品市场份额对比 / 应收账款逾期 / 良率趋势 / 供应商比价", key="v10_chat_input")

            if _v10_input:
                # 记录用户消息
                st.session_state.v10_chat_history.append({"role": "user", "content": _v10_input})

                # 调用 Gateway
                with st.spinner("🔄 智能路由中..."):
                    resp = gw.ask(_v10_input, user=st.session_state.get("auth_user", {}).get("username", "admin"),
                                  provider=st.session_state.get("ai_provider", "claude").lower(),
                                  api_key=st.session_state.get("api_key", ""))

                # 解析响应
                if resp["type"] == "collaboration":
                    _result = resp["result"]
                    _display = _result.get("synthesis", "")
                    # 加上各 Agent 的摘要
                    for _ag, _ar in _result.get("agent_results", {}).items():
                        _display += f"\n\n**{_agent_icons.get(_ag, '🤖')} {_agent_names.get(_ag, _ag)}**:\n"
                        try:
                            _parsed = json.loads(_ar) if isinstance(_ar, str) else _ar
                            _display += json.dumps(_parsed, ensure_ascii=False, indent=2)[:800]
                        except Exception:
                            _display += str(_ar)[:800]

                    st.session_state.v10_chat_history.append({
                        "role": "assistant", "content": _display,
                        "agent": resp.get("routing", {}).get("agent", "platform"),
                        "confidence": resp.get("routing", {}).get("confidence", 0),
                        "duration": resp.get("duration_ms", 0),
                        "type": "collaboration",
                        "scenario": _result.get("scenario", ""),
                    })
                elif resp["type"] == "single_agent":
                    _answer = resp.get("answer", "")
                    try:
                        _parsed = json.loads(_answer) if isinstance(_answer, str) else _answer
                        _display = json.dumps(_parsed, ensure_ascii=False, indent=2)
                    except Exception:
                        _display = str(_answer)

                    st.session_state.v10_chat_history.append({
                        "role": "assistant", "content": _display,
                        "agent": resp.get("agent", "unknown"),
                        "confidence": resp.get("confidence", 0),
                        "duration": resp.get("duration_ms", 0),
                        "type": "single_agent",
                    })
                else:
                    st.session_state.v10_chat_history.append({
                        "role": "assistant", "content": f"❌ 错误: {resp.get('error', '未知')}",
                        "agent": "platform", "confidence": 0, "duration": 0, "type": "error",
                    })

                st.rerun()

            # ── 分割线 ──
            st.markdown("---")

            # ── Agent 状态面板 ──
            st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;
                font-weight:700;color:#FFF;margin-bottom:12px;letter-spacing:0.04em;">
                AGENT STATUS MATRIX</div>""", unsafe_allow_html=True)

            _agent_cols = st.columns(len(card["agents"]))
            for i, name in enumerate(card["agents"]):
                agent_card_info = gw.registry.get_card(name) if gw.registry else None
                icon = _agent_icons.get(name, "🤖")
                display_name = _agent_names.get(name, name)
                skills_count = len(agent_card_info.skills) if agent_card_info else 0

                # 统计该 Agent 被调用的次数
                _calls = sum(1 for m in st.session_state.v10_chat_history
                             if m.get("role") == "assistant" and m.get("agent") == name)

                with _agent_cols[i]:
                    st.markdown(f"""
                    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.08);
                         padding:14px;text-align:center;">
                        <div style="font-size:1.5rem;">{icon}</div>
                        <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                             font-size:0.7rem;color:#FFF;margin:4px 0;">{display_name}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:#555;">
                            {skills_count} skills · {_calls} calls
                        </div>
                        <div style="width:6px;height:6px;border-radius:50%;background:#4ade80;
                             margin:6px auto 0;"></div>
                    </div>""", unsafe_allow_html=True)

            # ── 快捷协作场景 ──
            st.markdown("")
            st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;
                font-weight:700;color:#FFF;margin-bottom:12px;letter-spacing:0.04em;">
                CROSS-AGENT COLLABORATION</div>""", unsafe_allow_html=True)

            _sc_cols = st.columns(len(gw.collaboration.scenarios))
            for i, (sid, sconfig) in enumerate(gw.collaboration.scenarios.items()):
                with _sc_cols[i]:
                    chain_icons = " → ".join([_agent_icons.get(a, "🤖") for a in sconfig["chain"]])
                    if st.button(f"⚡ {sconfig['name']}", key=f"collab_{sid}", use_container_width=True):
                        # 触发协作场景
                        _trigger_q = sconfig["trigger_keywords"][0]
                        st.session_state.v10_chat_history.append({"role": "user", "content": f"[协作] {_trigger_q}"})
                        resp = gw.ask(_trigger_q, user=_current_user.get("username", "anonymous"),
                                      provider=st.session_state.get("ai_provider", "claude").lower(),
                                      api_key=st.session_state.get("api_key", ""))
                        if resp["type"] == "collaboration":
                            _result = resp["result"]
                            _display = _result.get("synthesis", "")
                            st.session_state.v10_chat_history.append({
                                "role": "assistant", "content": _display,
                                "agent": "platform", "confidence": 1.0,
                                "duration": resp.get("duration_ms", 0),
                                "type": "collaboration", "scenario": _result.get("scenario", ""),
                            })
                        st.rerun()
                    st.caption(f"{chain_icons}\n{sconfig['description']}")

            # ── 快捷示例问题 ──
            st.markdown("")
            st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;
                font-weight:700;color:#FFF;margin-bottom:8px;letter-spacing:0.04em;">
                QUICK START</div>""", unsafe_allow_html=True)

            _example_qs = [
                ("🛒 供应商比价分析", "供应商比价分析"),
                ("🔍 良率趋势如何？", "良率趋势如何？"),
                ("💰 应收账款逾期", "应收账款逾期情况"),
                ("📊 竞品市场份额", "竞品市场份额对比"),
                ("💰 毛利率分析", "毛利率分析"),
                ("📊 行业趋势", "2026行业趋势报告"),
            ]
            _eq_cols = st.columns(len(_example_qs))
            for i, (label, query) in enumerate(_example_qs):
                with _eq_cols[i]:
                    if st.button(label, key=f"quick_{i}", use_container_width=True):
                        st.session_state.v10_chat_history.append({"role": "user", "content": query})
                        resp = gw.ask(query, user=_current_user.get("username", "anonymous"),
                                      provider=st.session_state.get("ai_provider", "claude").lower(),
                                      api_key=st.session_state.get("api_key", ""))
                        if resp["type"] == "single_agent":
                            _answer = resp.get("answer", "")
                            try:
                                _parsed = json.loads(_answer) if isinstance(_answer, str) else _answer
                                _display = json.dumps(_parsed, ensure_ascii=False, indent=2)
                            except Exception:
                                _display = str(_answer)
                            st.session_state.v10_chat_history.append({
                                "role": "assistant", "content": _display,
                                "agent": resp.get("agent", "unknown"),
                                "confidence": resp.get("confidence", 0),
                                "duration": resp.get("duration_ms", 0), "type": "single_agent",
                            })
                        st.rerun()

            # ── 审计日志 ──
            audit_stats = stats.get("audit", {})
            if audit_stats.get("total_requests", 0) > 0:
                with st.expander("📋 Audit Log"):
                    st.json(audit_stats)

        except Exception as e:
            st.error(f"V10 Platform 加载失败: {e}")
            import traceback
            st.code(traceback.format_exc())
