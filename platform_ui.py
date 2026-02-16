#!/usr/bin/env python3
"""
MRARFAI V10.0 — Platform UI (Streamlit)
=========================================
统一平台入口: 所有Agent在一个界面

功能:
  1. 智能对话 — 自动路由到最佳Agent
  2. Agent面板 — 查看所有Agent状态
  3. 协作视图 — 跨Agent联合分析
  4. 审计日志 — 每次调用可追溯
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="MRARFAI 企业Agent平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 主题 — from ui_theme (Single Source of Truth)
# ============================================================
from ui_theme import inject_theme
inject_theme()

# ============================================================
# Gateway 初始化
# ============================================================

@st.cache_resource
def get_gateway():
    """初始化平台网关 (缓存)"""
    import sys, os
    # 确保可以import项目文件
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    from platform_gateway import get_gateway
    return get_gateway()


def run_async(coro):
    """运行异步函数"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================
# Session State 初始化
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None  # None = 自动路由


# ============================================================
# Sidebar — Agent 管理面板
# ============================================================

def render_sidebar():
    gateway = get_gateway()
    stats = gateway.get_stats()
    agent_names = stats.get("agents", [])
    registry_agents = stats.get("registry", {}).get("agents", {})

    st.sidebar.markdown("## 🚀 MRARFAI")
    st.sidebar.markdown("**企业Agent平台** V10.0")
    st.sidebar.divider()

    # Agent选择
    st.sidebar.markdown("### 🤖 选择Agent")

    agent_options = {"🧠 智能路由 (自动)": None}
    agent_icons = {
        "sales": "📊", "risk": "⚠️", "strategist": "🎯",
        "procurement": "🛒", "quality": "🔍", "finance": "💰", "market": "🌍",
    }
    for name in agent_names:
        icon = agent_icons.get(name, "🤖")
        agent_options[f"{icon} {name.upper()}"] = name

    selected = st.sidebar.radio(
        "选择目标Agent",
        list(agent_options.keys()),
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.selected_agent = agent_options[selected]

    st.sidebar.divider()

    # Agent状态
    st.sidebar.markdown("### 📋 Agent状态")
    for name in agent_names:
        icon = agent_icons.get(name, "🤖")
        info = registry_agents.get(name, {})
        skills = info.get("skills", 0)
        tasks = info.get("tasks_processed", 0)
        st.sidebar.markdown(
            f"{icon} **{name}** — "
            f"`{skills}技能` · "
            f"`{tasks}次调用`"
        )

    st.sidebar.divider()

    # 平台统计
    st.sidebar.markdown("### 📈 平台统计")
    st.sidebar.metric("总Agent数", stats.get("total_agents", 0))
    st.sidebar.metric("总技能数", stats.get("total_skills", 0))

    audit = stats.get("audit", {})
    if audit.get("total_requests", 0) > 0:
        st.sidebar.metric("总调用次数", audit["total_requests"])
        st.sidebar.metric("平均响应", f"{audit.get('avg_duration_ms', 0):.0f}ms")


# ============================================================
# Main Content
# ============================================================

def render_main():
    gateway = get_gateway()

    # Header
    st.markdown("""
    <div class="platform-header">
        <h1>🚀 MRARFAI 企业Agent平台</h1>
        <p>禾苗科技 · 多Agent协作 · MCP + A2A 标准 · 全业务覆盖</p>
    </div>
    """, unsafe_allow_html=True)

    # Tab
    tab_chat, tab_agents, tab_collab, tab_audit = st.tabs([
        "💬 智能对话", "🤖 Agent矩阵", "🔗 协作场景", "📋 审计日志",
    ])

    # ── Tab 1: 智能对话 ──
    with tab_chat:
        render_chat(gateway)

    # ── Tab 2: Agent矩阵 ──
    with tab_agents:
        render_agents(gateway)

    # ── Tab 3: 协作场景 ──
    with tab_collab:
        render_collaboration(gateway)

    # ── Tab 4: 审计日志 ──
    with tab_audit:
        render_audit(gateway)


# ============================================================
# 💬 智能对话 Tab
# ============================================================

def render_chat(gateway):
    # 快捷问题
    st.markdown("**快捷问题:**")
    cols = st.columns(4)
    quick_questions = [
        ("📊 营收概览", "今年营收情况概览"),
        ("🛒 采购延迟", "有哪些采购延迟的订单？"),
        ("🔍 良品率", "各产线良品率报告"),
        ("💰 应收账款", "应收账款和逾期情况"),
    ]
    for i, (label, q) in enumerate(quick_questions):
        if cols[i].button(label, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            result = run_async(gateway.ask(q))
            st.session_state.messages.append({
                "role": "assistant", "content": result.get("answer", ""),
                "agent": result.get("agent", ""), "confidence": result.get("confidence", 0),
                "collaboration": result.get("collaboration", False),
                "duration_ms": result.get("duration_ms", 0),
            })
            st.rerun()

    cols2 = st.columns(4)
    quick_questions2 = [
        ("🌍 竞品分析", "ODM竞品对比分析"),
        ("⚠️ 客户流失", "哪些客户有流失风险？"),
        ("📈 行业趋势", "ODM行业趋势分析"),
        ("🔗 全链路", "客户A出货异常，全链路追踪"),
    ]
    for i, (label, q) in enumerate(quick_questions2):
        if cols2[i].button(label, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            result = run_async(gateway.ask(q))
            st.session_state.messages.append({
                "role": "assistant", "content": result.get("answer", ""),
                "agent": result.get("agent", ""),
                "confidence": result.get("confidence", 0),
                "collaboration": result.get("collaboration", False),
                "duration_ms": result.get("duration_ms", 0),
                "steps": result.get("steps", []),
            })
            st.rerun()

    st.divider()

    # 聊天历史
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-msg chat-user">
                👤 {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            agent_name = msg.get("agent", "unknown")
            conf = msg.get("confidence", 0)
            collab = "🔗协作" if msg.get("collaboration") else ""
            duration = msg.get("duration_ms", 0)

            st.markdown(f"""
            <div class="chat-msg chat-agent">
                <div class="chat-agent-tag">
                    🤖 {agent_name.upper()} · 置信度{conf:.0%} · {duration:.0f}ms {collab}
                </div>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)

            # 如果有协作步骤
            if msg.get("steps"):
                with st.expander("🔗 查看协作详情"):
                    for step in msg["steps"]:
                        status_icon = {"completed": "✅", "failed": "❌", "skipped": "⚠️"}.get(step["status"], "⏳")
                        st.markdown(f"""
                        <div class="collab-step">
                            {status_icon} <strong>[{step['agent'].upper()}]</strong> {step['question']}<br/>
                            <span style="color:#8b949e">{step.get('answer', '')[:200]}</span>
                        </div>
                        """, unsafe_allow_html=True)

    # 输入
    agent_hint = f"当前Agent: {st.session_state.selected_agent or '智能路由(自动)'}"
    user_input = st.chat_input(f"输入问题... ({agent_hint})")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        result = run_async(gateway.ask(user_input))
        st.session_state.messages.append({
            "role": "assistant", "content": result.get("answer", ""),
            "agent": result.get("agent", ""),
            "confidence": result.get("confidence", 0),
            "collaboration": result.get("collaboration", False),
            "duration_ms": result.get("duration_ms", 0),
            "steps": result.get("steps", []),
        })
        st.rerun()

    # 清空按钮
    if st.session_state.messages:
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# 🤖 Agent矩阵 Tab
# ============================================================

def render_agents(gateway):
    stats = gateway.get_stats()
    agent_names = stats.get("agents", [])
    registry_agents = stats.get("registry", {}).get("agents", {})
    # 构建 UI 需要的 agent 信息列表
    agents = []
    for name in agent_names:
        info = registry_agents.get(name, {})
        card = gateway.registry.get_card(name) if gateway.registry else None
        agents.append({
            "name": name,
            "display_name": name.upper(),
            "description": card.description if card and hasattr(card, 'description') else f"{name} agent",
            "skills": info.get("skills", 0),
            "tasks_processed": info.get("tasks_processed", 0),
        })

    # 指标行
    cols = st.columns(4)
    cols[0].markdown(f"""
    <div class="metric-box">
        <div class="value">{len(agents)}</div>
        <div class="label">在线Agent</div>
    </div>""", unsafe_allow_html=True)

    total_skills = sum(a["skills"] for a in agents)
    cols[1].markdown(f"""
    <div class="metric-box">
        <div class="value">{total_skills}</div>
        <div class="label">总技能数</div>
    </div>""", unsafe_allow_html=True)

    total_tasks = sum(a["tasks_processed"] for a in agents)
    cols[2].markdown(f"""
    <div class="metric-box">
        <div class="value">{total_tasks}</div>
        <div class="label">处理任务</div>
    </div>""", unsafe_allow_html=True)

    cols[3].markdown(f"""
    <div class="metric-box">
        <div class="value" style="color:#FFFFFF">✅</div>
        <div class="label">全部在线</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Agent卡片网格
    agent_icons = {
        "sales": "📊", "risk": "⚠️", "strategist": "🎯",
        "procurement": "🛒", "quality": "🔍", "finance": "💰", "market": "🌍",
    }
    agent_colors = {
        "sales": "#FFFFFF", "risk": "#CCCCCC", "strategist": "#AAAAAA",
        "procurement": "#A1A1AA", "quality": "#BBBBBB", "finance": "#DDDDDD", "market": "#888888",
    }

    cols = st.columns(3)
    for i, a in enumerate(agents):
        with cols[i % 3]:
            icon = agent_icons.get(a["name"], "🤖")
            color = agent_colors.get(a["name"], "#CCCCCC")
            st.markdown(f"""
            <div class="agent-card" style="border-left: 3px solid {color}">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <span class="name">{icon} {a['display_name']}</span>
                    <span class="status-badge status-online">● ONLINE</span>
                </div>
                <div class="desc">{a['description'][:60]}...</div>
                <div class="stats">
                    {a['skills']} skills · {a['tasks_processed']} tasks
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Agent Card JSON
    with st.expander("🔧 平台Agent Card (/.well-known/agent-card.json)"):
        st.json(gateway.get_platform_card())


# ============================================================
# 🔗 协作场景 Tab
# ============================================================

def render_collaboration(gateway):
    st.markdown("### 🔗 预定义协作场景")
    st.markdown("当问题涉及多个领域时，平台自动触发跨Agent协作")

    scenarios = gateway.collaboration.scenarios
    for sid, config in scenarios.items():
        with st.expander(f"📋 {config['name']}"):
            st.markdown(f"**触发关键词:** {', '.join(config['trigger_keywords'])}")
            st.markdown("**协作链路:**")
            for i, step in enumerate(config["steps"]):
                arrow = "→" if i < len(config["steps"]) - 1 else "✅"
                st.markdown(f"  {i+1}. **[{step['agent'].upper()}]** {step['question_template']} {arrow}")

            # 一键测试
            if st.button(f"🚀 测试此场景", key=f"test_{sid}"):
                test_q = config["trigger_keywords"][0] + "分析"
                result = run_async(gateway.ask(test_q))
                st.markdown("---")
                st.markdown(f"**问题:** {test_q}")
                st.markdown(f"**结果:**")
                st.text(result.get("answer", ""))
                if result.get("steps"):
                    for step in result["steps"]:
                        st.markdown(f"- [{step['agent']}] {step['status']}")


# ============================================================
# 📋 审计日志 Tab
# ============================================================

def render_audit(gateway):
    stats = gateway.audit.get_stats()

    if stats["total"] == 0:
        st.info("📋 暂无审计记录。请先在「智能对话」中提问。")
        return

    # 统计
    cols = st.columns(4)
    cols[0].metric("总调用", stats["total"])
    cols[1].metric("协作率", f"{stats.get('collaboration_rate', 0)*100:.0f}%")
    cols[2].metric("平均响应", f"{stats.get('avg_duration_ms', 0):.0f}ms")
    cols[3].metric("Agent分布", len(stats.get("by_agent", {})))

    if stats.get("by_agent"):
        st.bar_chart(stats["by_agent"])

    # 日志列表
    st.divider()
    entries = gateway.audit.recent(20)
    for e in reversed(entries):
        collab_tag = "🔗" if e.get("collab") else ""
        st.markdown(
            f"`{e['ts'][:19]}` | **{e['agent']}** | "
            f"{e['question']} | "
            f"置信度{e['confidence']} | {e['duration_ms']:.0f}ms {collab_tag}"
        )


# ============================================================
# Main
# ============================================================

render_sidebar()
render_main()
