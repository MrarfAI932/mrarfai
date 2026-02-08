#!/usr/bin/env python3
"""
MRARFAI Chat Tab v5.0 — Agent对话界面
=====================================
设计参考：ChatGPT + Perplexity + Claude UI
"""

import streamlit as st
from agent import ask_agent, SUGGESTED_QUESTIONS


def render_chat_tab(data: dict, results: dict,
                     benchmark: dict = None, forecast: dict = None):
    """渲染Agent对话界面"""

    # ---- 头部 ----
    st.markdown("""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
        <div style="width:40px; height:40px; border-radius:12px;
             background:linear-gradient(135deg, #7c3aed, #06b6d4);
             display:flex; align-items:center; justify-content:center;
             font-size:1.2rem; box-shadow:0 4px 16px rgba(124,58,237,0.15);">
            🧠
        </div>
        <div>
            <div style="font-size:1.15rem; font-weight:700; color:#fafafa; letter-spacing:-0.5px;">
                Sales Agent
            </div>
            <div style="font-size:0.72rem; color:#71717a; display:flex; align-items:center; gap:6px;">
                <span style="width:6px;height:6px;border-radius:50%;background:#10b981;display:inline-block;"></span>
                已加载 · {clients}家客户 · {dims}维分析就绪
            </div>
        </div>
    </div>
    """.format(
        clients=sum(1 for c in data['客户金额'] if c['年度金额'] > 0),
        dims=len(results),
    ), unsafe_allow_html=True)

    st.markdown("")

    # ---- AI配置（折叠） ----
    with st.expander("⚙️ AI 引擎配置", expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            provider = st.selectbox(
                "引擎", ["deepseek", "claude"], index=0,
                help="DeepSeek：国内可用，约¥0.01/次 | Claude：更强，需翻墙"
            )
        with col2:
            key_label = "DeepSeek" if provider == "deepseek" else "Claude"
            api_key = st.text_input(
                f"{key_label} API Key", type="password",
                value=st.session_state.get(f'{provider}_key', ''),
                placeholder="sk-..." if provider == "deepseek" else "sk-ant-...",
            )
            if api_key:
                st.session_state[f'{provider}_key'] = api_key

    # ---- 初始化对话历史 ----
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # ---- 空状态：引导界面 ----
    if not st.session_state['chat_history']:
        st.markdown("""
        <div style="text-align:center; padding:32px 0 20px;">
            <div style="font-size:1.4rem; margin-bottom:8px;">👋</div>
            <div style="color:#a1a1aa; font-size:0.92rem; font-weight:500;">
                有什么想了解的？直接问我。
            </div>
            <div style="color:#71717a; font-size:0.78rem; margin-top:4px;">
                我可以分析客户、预测趋势、发现风险和机会
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 推荐问题 — 2行×3列
        questions_display = SUGGESTED_QUESTIONS[:6]
        rows = [questions_display[:3], questions_display[3:6]]
        for row in rows:
            cols = st.columns(len(row))
            for i, q in enumerate(row):
                with cols[i]:
                    if st.button(q, key=f"sq_{q[:8]}", use_container_width=True):
                        st.session_state['pending_question'] = q
                        st.rerun()

        # 功能说明卡片
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="v5-card">
                <h4>📊 数据问答</h4>
                <p>"今年总营收多少？"<br>"最大客户是谁？"</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="v5-card">
                <h4>🚨 风险预警</h4>
                <p>"哪些客户可能流失？"<br>"有什么异常波动？"</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="v5-card">
                <h4>💡 战略建议</h4>
                <p>"最值得投入的方向？"<br>"CEO该关注什么？"</p>
            </div>
            """, unsafe_allow_html=True)

    # ---- 对话历史显示 ----
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.chat_message("user", avatar="👤").markdown(msg['content'])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(msg['content'])
                if msg.get('tools'):
                    tool_html = " ".join(
                        f'<span class="tool-chip">🔧 {t}</span>'
                        for t in msg['tools']
                    )
                    st.markdown(f'<div style="margin-top:8px;">{tool_html}</div>',
                                unsafe_allow_html=True)

    # ---- 输入处理 ----
    pending = st.session_state.pop('pending_question', None)
    user_input = st.chat_input("输入问题... 例如：哪些客户有流失风险？")
    question = pending or user_input

    if question:
        st.chat_message("user", avatar="👤").markdown(question)
        st.session_state['chat_history'].append({
            'role': 'user', 'content': question
        })

        current_key = st.session_state.get(f'{provider}_key', '')

        with st.chat_message("assistant", avatar="🧠"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
            <div class="agent-thinking">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <circle cx="8" cy="8" r="6" stroke="#7c3aed" stroke-width="2" stroke-dasharray="4 4">
                        <animateTransform attributeName="transform" type="rotate"
                            from="0 8 8" to="360 8 8" dur="1.5s" repeatCount="indefinite"/>
                    </circle>
                </svg>
                分析中...正在调用分析工具
            </div>
            """, unsafe_allow_html=True)

            result = ask_agent(
                question=question,
                data=data,
                results=results,
                benchmark=benchmark,
                forecast=forecast,
                provider=provider,
                api_key=current_key,
            )

            thinking_placeholder.empty()
            st.markdown(result['answer'])

            if result['tools_used']:
                tool_html = " ".join(
                    f'<span class="tool-chip">🔧 {t}</span>'
                    for t in result['tools_used']
                )
                st.markdown(f'<div style="margin-top:8px;">{tool_html}</div>',
                            unsafe_allow_html=True)

            st.session_state['chat_history'].append({
                'role': 'assistant',
                'content': result['answer'],
                'tools': result['tools_used'],
            })

    # ---- 底部操作栏 ----
    if st.session_state['chat_history']:
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state['chat_history'] = []
                st.rerun()
        with col2:
            chat_text = "\n\n".join(
                f"{'Q' if m['role']=='user' else 'A'}: {m['content']}"
                for m in st.session_state['chat_history']
            )
            st.download_button("📥 导出对话", chat_text, "agent_chat.txt",
                               use_container_width=True)
