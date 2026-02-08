#!/usr/bin/env python3
"""
MRARFAI Chat Tab v5.2 — Multi-Agent + Memory + HITL
=====================================================
"""

import streamlit as st
from agent import ask_agent, SUGGESTED_QUESTIONS

# Multi-Agent
try:
    from multi_agent import (
        ask_multi_agent, ask_multi_agent_simple, AGENT_PROFILES,
        AgentMemory, get_memory, set_memory,
    )
    HAS_MULTI_AGENT = True
except ImportError:
    HAS_MULTI_AGENT = False

# CrewAI
try:
    from crewai import Agent as _TestAgent
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False


def _get_session_memory() -> 'AgentMemory':
    """从session_state获取或创建记忆"""
    if 'agent_memory' not in st.session_state:
        st.session_state['agent_memory'] = AgentMemory() if HAS_MULTI_AGENT else None
    mem = st.session_state['agent_memory']
    if mem and HAS_MULTI_AGENT:
        set_memory(mem)
    return mem


def _render_hitl_panel(triggers: list, memory):
    """渲染Human-in-the-loop确认面板"""
    if not triggers:
        return
    
    st.markdown(f"""
    <div style="padding:14px 18px; margin:12px 0;
         background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.2);
         border-radius:12px;">
        <div style="font-size:0.85rem; font-weight:700; color:#fca5a5; margin-bottom:10px;">
            ⚠️ 需要您确认 — 风控Agent检测到 {len(triggers)} 个高风险
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, t in enumerate(triggers):
        confirmed_key = f"hitl_{t['customer']}_{i}"
        already = st.session_state.get(confirmed_key)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(
                f"**{t['risk_level']}** · **{t['customer']}** · "
                f"¥{t['amount']:,.0f}万\n\n"
                f"<span style='font-size:0.8rem; color:#94a3b8;'>"
                f"{t['reason']}</span>",
                unsafe_allow_html=True
            )
        with col2:
            if already != 'confirmed':
                if st.button("✅ 确认关注", key=f"confirm_{confirmed_key}", 
                            use_container_width=True):
                    st.session_state[confirmed_key] = 'confirmed'
                    if memory:
                        memory.add_risk_confirmation(t['customer'], True)
                    st.rerun()
            else:
                st.success("已确认", icon="✅")
        with col3:
            if already != 'dismissed':
                if st.button("➖ 暂不处理", key=f"dismiss_{confirmed_key}",
                            use_container_width=True):
                    st.session_state[confirmed_key] = 'dismissed'
                    if memory:
                        memory.add_risk_confirmation(t['customer'], False)
                    st.rerun()
            else:
                st.caption("已跳过")
        
        if i < len(triggers) - 1:
            st.markdown("<hr style='border-color:rgba(239,68,68,0.1); margin:6px 0;'>", 
                       unsafe_allow_html=True)


def render_chat_tab(data: dict, results: dict,
                     benchmark: dict = None, forecast: dict = None):
    """渲染Agent对话界面"""

    # 获取记忆
    memory = _get_session_memory() if HAS_MULTI_AGENT else None

    # ---- 头部 ----
    active_clients = sum(1 for c in data['客户金额'] if c['年度金额'] > 0)
    st.markdown("""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
        <div style="width:40px; height:40px; border-radius:12px;
             background:linear-gradient(135deg, #7c3aed, #06b6d4);
             display:flex; align-items:center; justify-content:center;
             font-size:1.2rem; box-shadow:0 4px 16px rgba(124,58,237,0.15);">🧠</div>
        <div>
            <div style="font-size:1.15rem; font-weight:700; color:#fafafa;">Sales Agent</div>
            <div style="font-size:0.72rem; color:#71717a; display:flex; align-items:center; gap:6px;">
                <span style="width:6px;height:6px;border-radius:50%;background:#10b981;display:inline-block;"></span>
                已加载 · {c}家客户 · {d}维分析就绪{mem}
            </div>
        </div>
    </div>
    """.format(
        c=active_clients, d=len(results),
        mem=f" · 🧠{len(memory.conversation_history)}轮记忆" if memory and memory.conversation_history else "",
    ), unsafe_allow_html=True)

    st.markdown("")

    # ---- AI配置 ----
    with st.expander("⚙️ AI 引擎配置", expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            provider = st.selectbox("引擎", ["deepseek", "claude"], index=0,
                help="DeepSeek：国内可用 | Claude：更强，需翻墙")
        with col2:
            key_label = "DeepSeek" if provider == "deepseek" else "Claude"
            api_key = st.text_input(
                f"{key_label} API Key", type="password",
                value=st.session_state.get(f'{provider}_key', ''),
                placeholder="sk-..." if provider == "deepseek" else "sk-ant-...",
            )
            if api_key:
                st.session_state[f'{provider}_key'] = api_key

        st.markdown("")
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
        with col_m1:
            agent_mode = st.toggle(
                "🤖 Multi-Agent",
                value=st.session_state.get('multi_agent_mode', False),
                help="4个专家Agent协作")
            st.session_state['multi_agent_mode'] = agent_mode
        with col_m2:
            if agent_mode:
                if HAS_CREWAI:
                    st.caption("✅ CrewAI + 记忆 + HITL")
                elif HAS_MULTI_AGENT:
                    st.caption("⚡ 简化模式 + 记忆")
                else:
                    st.caption("⚠️ pip install crewai")
            else:
                st.caption("单Agent · 快速")
        with col_m3:
            if memory and memory.conversation_history:
                if st.button("🧹 清除记忆", use_container_width=True):
                    memory.clear()
                    st.toast("记忆已清除")

        if agent_mode and HAS_MULTI_AGENT:
            st.markdown("""
            <div style="padding:10px 14px; background:rgba(99,102,241,0.06);
                 border:1px solid rgba(99,102,241,0.1); border-radius:10px; margin-top:8px;">
                <div style="font-size:0.75rem; color:#a5b4fc; font-weight:600; margin-bottom:6px;">
                    🏛️ 专家团队 · 内置记忆 · Human-in-the-Loop
                </div>
                <div style="display:flex; gap:10px; flex-wrap:wrap; font-size:0.72rem; color:#94a3b8;">
                    <span>📊 <b style="color:#e2e8f0;">分析师</b></span>
                    <span>🛡️ <b style="color:#e2e8f0;">风控</b></span>
                    <span>💡 <b style="color:#e2e8f0;">策略师</b></span>
                    <span>🖊️ <b style="color:#e2e8f0;">报告员</b></span>
                    <span>🧠 <b style="color:#e2e8f0;">记忆</b></span>
                    <span>⚠️ <b style="color:#e2e8f0;">HITL</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- 对话历史初始化 ----
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # ---- 空状态 ----
    if not st.session_state['chat_history']:
        st.markdown("""
        <div style="text-align:center; padding:32px 0 20px;">
            <div style="font-size:1.4rem; margin-bottom:8px;">👋</div>
            <div style="color:#a1a1aa; font-size:0.92rem; font-weight:500;">
                有什么想了解的？直接问我。</div>
            <div style="color:#71717a; font-size:0.78rem; margin-top:4px;">
                我可以分析客户、预测趋势、发现风险和机会</div>
        </div>
        """, unsafe_allow_html=True)

        questions_display = SUGGESTED_QUESTIONS[:6]
        for row in [questions_display[:3], questions_display[3:6]]:
            cols = st.columns(len(row))
            for i, q in enumerate(row):
                with cols[i]:
                    if st.button(q, key=f"sq_{q[:8]}", use_container_width=True):
                        st.session_state['pending_question'] = q
                        st.rerun()

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="v5-card"><h4>📊 数据问答</h4><p>"今年总营收多少？"</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="v5-card"><h4>🚨 风险预警</h4><p>"哪些客户可能流失？"</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="v5-card"><h4>💡 战略建议</h4><p>"CEO该关注什么？"</p></div>', unsafe_allow_html=True)

    # ---- 历史显示 ----
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.chat_message("user", avatar="👤").markdown(msg['content'])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(msg['content'])
                if msg.get('tools'):
                    tool_html = " ".join(f'<span class="tool-chip">{t}</span>' for t in msg['tools'])
                    st.markdown(f'<div style="margin-top:8px;">{tool_html}</div>', unsafe_allow_html=True)
                if msg.get('expert_outputs'):
                    with st.expander("🏛️ 各专家原始意见", expanded=False):
                        for en, eo in msg['expert_outputs'].items():
                            st.markdown(f"**{en}**"); st.markdown(eo); st.markdown("---")
                if msg.get('hitl_triggers'):
                    _render_hitl_panel(msg['hitl_triggers'], memory)

    # ---- 输入处理 ----
    pending = st.session_state.pop('pending_question', None)
    user_input = st.chat_input("输入问题...")
    question = pending or user_input

    if question:
        st.chat_message("user", avatar="👤").markdown(question)
        st.session_state['chat_history'].append({'role': 'user', 'content': question})

        current_key = st.session_state.get(f'{provider}_key', '')
        use_multi = st.session_state.get('multi_agent_mode', False) and HAS_MULTI_AGENT

        with st.chat_message("assistant", avatar="🧠"):
            ph = st.empty()

            if use_multi:
                ph.markdown("""
                <div class="agent-thinking">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <circle cx="8" cy="8" r="6" stroke="#7c3aed" stroke-width="2" stroke-dasharray="4 4">
                            <animateTransform attributeName="transform" type="rotate"
                                from="0 8 8" to="360 8 8" dur="1.5s" repeatCount="indefinite"/>
                        </circle>
                    </svg>
                    🏛️ Multi-Agent 协作中...
                </div>""", unsafe_allow_html=True)

                if HAS_CREWAI:
                    result = ask_multi_agent(
                        question=question, data=data, results=results,
                        benchmark=benchmark, forecast=forecast,
                        provider=provider, api_key=current_key, memory=memory,
                    )
                else:
                    result = ask_multi_agent_simple(
                        question=question, data=data, results=results,
                        benchmark=benchmark, forecast=forecast,
                        provider=provider, api_key=current_key, memory=memory,
                    )

                ph.empty()

                # 调度日志
                if result.get('thinking'):
                    th_html = "<br>".join(
                        f'<span style="font-size:0.73rem; color:#71717a;">{t}</span>'
                        for t in result['thinking']
                    )
                    st.markdown(f"""
                    <div style="padding:8px 12px; margin-bottom:10px;
                         background:rgba(99,102,241,0.04); border-radius:8px;
                         border:1px solid rgba(99,102,241,0.08);">
                        <span style="font-size:0.68rem; color:#6366f1; font-weight:600;">🏛️ 调度日志</span><br>
                        {th_html}
                    </div>""", unsafe_allow_html=True)

                st.markdown(result['answer'])

                if result.get('agents_used'):
                    ah = " ".join(f'<span class="tool-chip">{a}</span>' for a in result['agents_used'])
                    st.markdown(f'<div style="margin-top:8px;">{ah}</div>', unsafe_allow_html=True)

                if result.get('expert_outputs'):
                    with st.expander("🏛️ 各专家原始意见", expanded=False):
                        for en, eo in result['expert_outputs'].items():
                            st.markdown(f"**{en}**"); st.markdown(eo); st.markdown("---")

                # HITL
                hitl = result.get('hitl_triggers', [])
                if hitl:
                    _render_hitl_panel(hitl, memory)

                st.session_state['chat_history'].append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'tools': result.get('agents_used', []),
                    'expert_outputs': result.get('expert_outputs', {}),
                    'hitl_triggers': hitl,
                })

            else:
                # 单Agent
                ph.markdown("""
                <div class="agent-thinking">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <circle cx="8" cy="8" r="6" stroke="#7c3aed" stroke-width="2" stroke-dasharray="4 4">
                            <animateTransform attributeName="transform" type="rotate"
                                from="0 8 8" to="360 8 8" dur="1.5s" repeatCount="indefinite"/>
                        </circle>
                    </svg>
                    分析中...
                </div>""", unsafe_allow_html=True)

                result = ask_agent(
                    question=question, data=data, results=results,
                    benchmark=benchmark, forecast=forecast,
                    provider=provider, api_key=current_key,
                )
                ph.empty()
                st.markdown(result['answer'])
                if result['tools_used']:
                    th = " ".join(f'<span class="tool-chip">🔧 {t}</span>' for t in result['tools_used'])
                    st.markdown(f'<div style="margin-top:8px;">{th}</div>', unsafe_allow_html=True)

                st.session_state['chat_history'].append({
                    'role': 'assistant', 'content': result['answer'],
                    'tools': result['tools_used'],
                })

    # ---- 底部 ----
    if st.session_state['chat_history']:
        st.markdown("")
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state['chat_history'] = []
                st.rerun()
        with c2:
            chat_text = "\n\n".join(
                f"{'Q' if m['role']=='user' else 'A'}: {m['content']}"
                for m in st.session_state['chat_history']
            )
            st.download_button("📥 导出对话", chat_text, "agent_chat.txt", use_container_width=True)
