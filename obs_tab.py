#!/usr/bin/env python3
"""
MRARFAI Observability Dashboard v1.0
======================================
Streamlit可观测性仪表盘

四个视图：
  1. 📊 概览 — KPI卡片（查询数、延迟、成本、错误率）
  2. 📈 趋势 — 每日查询量、延迟、成本走势图
  3. 🔬 追踪详情 — 单次请求的span分解瀑布图
  4. ⚙️ 系统 — 数据库统计、导出、清理

集成方式：
  在 app.py 中添加一个tab调用 render_obs_tab()
"""

import streamlit as st
from datetime import datetime, timedelta

try:
    from observability import (
        get_tracer, get_metrics, get_store,
        export_traces_json, export_traces_csv,
    )
    HAS_OBS = True
except ImportError:
    HAS_OBS = False


def render_obs_tab():
    """渲染可观测性仪表盘"""
    if not HAS_OBS:
        st.warning("⚠️ 可观测性模块未安装。请确保 observability.py 在同目录下。")
        return

    st.markdown("""
    <div style="padding:12px 0 8px 0;">
        <span style="font-size:1.3rem; font-weight:700;">📡 可观测性中心</span>
        <span style="font-size:0.8rem; color:#888; margin-left:8px;">
            OpenTelemetry标准 · SQLite持久化 · 实时分析
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 时间范围选择
    col1, col2 = st.columns([3, 1])
    with col2:
        days = st.selectbox("时间范围", [1, 7, 14, 30], index=1,
                           format_func=lambda x: f"最近{x}天")

    metrics = get_metrics()

    # ---- TAB导航 ----
    tab1, tab2, tab3, tab4 = st.tabs(["📊 概览", "📈 趋势", "🔬 追踪详情", "⚙️ 系统"])

    with tab1:
        _render_overview(metrics, days)

    with tab2:
        _render_trends(metrics, days)

    with tab3:
        _render_trace_detail(metrics, days)

    with tab4:
        _render_system(days)


def _render_overview(metrics, days):
    """概览视图 — KPI卡片"""
    overview = metrics.get_overview(days)

    # KPI 行1: 核心指标
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        _metric_card("🔢 总查询", f"{overview['total_queries']}",
                     "次", "#60a5fa")
    with c2:
        avg_s = overview['avg_latency_ms'] / 1000
        color = "#34d399" if avg_s < 5 else "#fbbf24" if avg_s < 15 else "#f87171"
        _metric_card("⏱️ 平均延迟", f"{avg_s:.1f}", "秒", color)
    with c3:
        _metric_card("🪙 总成本", f"${overview['total_cost_usd']:.4f}",
                     "USD", "#a78bfa")
    with c4:
        err_color = "#34d399" if overview['error_rate'] < 5 else "#f87171"
        _metric_card("❌ 错误率", f"{overview['error_rate']}", "%", err_color)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # KPI 行2: 细节指标
    c5, c6, c7, c8 = st.columns(4)

    with c5:
        _metric_card("📝 总Token", f"{overview['total_tokens']:,}", "", "#60a5fa")
    with c6:
        _metric_card("📊 单次Token", f"{overview['avg_tokens_per_query']:,.0f}",
                     "平均", "#60a5fa")
    with c7:
        _metric_card("🤖 LLM调用", f"{overview['total_llm_calls']}",
                     f"({overview['avg_llm_calls']:.1f}/次)", "#a78bfa")
    with c8:
        rating = overview['avg_rating']
        stars = "⭐" * int(rating) if rating > 0 else "暂无"
        _metric_card("⭐ 用户评分", f"{rating:.1f}" if rating > 0 else "-",
                     stars, "#fbbf24")

    # 延迟分位数
    st.markdown("### 延迟分布")
    pcts = metrics.get_latency_percentiles(days)
    if pcts['count'] > 0:
        cols = st.columns(6)
        labels = ["P50", "P90", "P95", "P99", "最小", "最大"]
        keys = ["p50", "p90", "p95", "p99", "min", "max"]
        for col, label, key in zip(cols, labels, keys):
            with col:
                val = pcts[key] / 1000  # ms -> s
                st.metric(label, f"{val:.1f}s")

    # 各阶段延迟
    st.markdown("### 阶段延迟分解")
    stage_latency = metrics.get_latency_by_stage(days)
    if stage_latency:
        for item in stage_latency:
            kind = item['kind']
            avg = item['avg_ms'] / 1000
            count = item['count']
            label_map = {
                'trace': '🔄 完整请求',
                'data_query': '📊 数据查询',
                'routing': '🧭 路由',
                'llm_call': '🤖 LLM调用',
                'agent': '🧑‍💼 Agent',
                'reporter': '🖊️ 报告员',
                'kg_lookup': '📚 知识图谱',
            }
            label = label_map.get(kind, kind)
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin:4px 0;">
                <span style="width:140px; font-size:0.85rem;">{label}</span>
                <div style="flex:1; background:#1e293b; border-radius:4px; height:20px; margin:0 8px;">
                    <div style="width:{min(avg/30*100, 100):.0f}%; background:linear-gradient(90deg,#3b82f6,#60a5fa);
                         height:100%; border-radius:4px; min-width:2px;"></div>
                </div>
                <span style="font-size:0.8rem; color:#94a3b8; width:80px; text-align:right;">
                    {avg:.1f}s ({count}次)
                </span>
            </div>
            """, unsafe_allow_html=True)

    # 成本分解
    st.markdown("### 成本分解")
    cost_data = metrics.get_cost_breakdown(days)
    if cost_data['by_stage']:
        for item in cost_data['by_stage']:
            stage = item['stage'] or '未知'
            cost = item['cost'] or 0
            tokens = item['tokens'] or 0
            calls = item['calls'] or 0
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                 padding:6px 12px; margin:2px 0; background:#0f172a;
                 border-radius:6px; font-size:0.85rem;">
                <span>{stage}</span>
                <span style="color:#a78bfa">${cost:.4f} · {tokens:,} tokens · {calls}次调用</span>
            </div>
            """, unsafe_allow_html=True)

    # 路由分析
    st.markdown("### 路由统计")
    routing = metrics.get_routing_stats(days)

    if routing['by_source']:
        cols = st.columns(len(routing['by_source']))
        source_icons = {"知识图谱": "📚", "LLM": "🤖", "规则": "📏"}
        for col, item in zip(cols, routing['by_source']):
            with col:
                src = item['route_source'] or '未知'
                icon = source_icons.get(src, "❓")
                st.metric(f"{icon} {src}", item['count'])


def _render_trends(metrics, days):
    """趋势视图"""
    trend = metrics.get_daily_trend(days)

    if not trend:
        st.info("📭 暂无数据。使用Agent问答后数据将自动出现。")
        return

    # 转为简单的展示
    st.markdown("### 每日查询量")
    for item in trend:
        date = item['date'] or ''
        queries = item['queries'] or 0
        avg_latency = (item['avg_latency'] or 0) / 1000
        cost = item['cost'] or 0
        errors = item['errors'] or 0

        bar_width = min(queries * 5, 100)
        error_tag = f" <span style='color:#f87171'>({errors}错误)</span>" if errors > 0 else ""

        st.markdown(f"""
        <div style="display:flex; align-items:center; margin:3px 0;">
            <span style="width:90px; font-size:0.82rem; color:#94a3b8;">{date}</span>
            <div style="flex:1; background:#1e293b; border-radius:3px; height:18px; margin:0 8px;">
                <div style="width:{bar_width}%; background:linear-gradient(90deg,#3b82f6,#60a5fa);
                     height:100%; border-radius:3px;"></div>
            </div>
            <span style="font-size:0.78rem; color:#94a3b8; width:220px; text-align:right;">
                {queries}次 · {avg_latency:.1f}s · ${cost:.4f}{error_tag}
            </span>
        </div>
        """, unsafe_allow_html=True)


def _render_trace_detail(metrics, days):
    """追踪详情视图"""
    store = get_store()
    traces = store.get_recent_traces(limit=20)

    if not traces:
        st.info("📭 暂无追踪记录。")
        return

    # 追踪列表
    st.markdown("### 最近请求")

    for i, t in enumerate(traces):
        duration_s = (t.get('total_duration_ms', 0) or 0) / 1000
        cost = t.get('total_cost_usd', 0) or 0
        tokens = t.get('total_tokens', 0) or 0
        status = t.get('status', 'ok')
        question = (t.get('question', '') or '')[:60]
        trace_id = t.get('trace_id', '')[:8]
        timestamp = (t.get('timestamp', '') or '')[:19]
        feedback = t.get('user_feedback')

        status_icon = "✅" if status == "ok" else "❌"
        feedback_str = f"⭐{feedback}" if feedback else ""

        with st.expander(f"{status_icon} {question}... — {duration_s:.1f}s / ${cost:.4f} / {tokens} tok {feedback_str}",
                        expanded=(i == 0)):
            st.markdown(f"""
            <div style="font-size:0.8rem; color:#64748b; margin-bottom:8px;">
                Trace: <code>{trace_id}</code> · {timestamp} ·
                模式: {t.get('pattern_matched', '-')} ·
                路由: {t.get('route_source', '-')}
            </div>
            """, unsafe_allow_html=True)

            # Span分解
            spans = store.get_span_breakdown(t.get('trace_id', ''))
            if spans:
                for s in spans:
                    kind = s.get('kind', '')
                    name = s.get('name', kind)
                    dur = (s.get('duration_ms', 0) or 0) / 1000
                    stok = s.get('total_tokens', 0) or 0
                    scost = s.get('cost_usd', 0) or 0
                    sstatus = s.get('status', 'ok')

                    icon = {"trace": "🔄", "data_query": "📊", "routing": "🧭",
                            "llm_call": "🤖", "agent": "🧑‍💼", "reporter": "🖊️",
                            "kg_lookup": "📚"}.get(kind, "▪️")
                    s_color = "#34d399" if sstatus == "ok" else "#f87171"

                    tok_str = f" · {stok} tok · ${scost:.4f}" if stok > 0 else ""

                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin:2px 0;
                         padding:4px 8px; background:#0f172a; border-radius:4px;
                         border-left:3px solid {s_color};">
                        <span style="font-size:0.82rem;">{icon} {name}</span>
                        <span style="flex:1"></span>
                        <span style="font-size:0.78rem; color:#94a3b8;">
                            {dur:.2f}s{tok_str}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            # 用户反馈
            col_a, col_b = st.columns([3, 1])
            with col_a:
                feedback_val = st.slider(
                    "评分", 1, 5, 3,
                    key=f"fb_{t.get('trace_id', '')}",
                    label_visibility="collapsed"
                )
            with col_b:
                if st.button("提交反馈", key=f"fb_btn_{t.get('trace_id', '')}"):
                    tracer = get_tracer()
                    tracer.save_feedback(t.get('trace_id', ''), feedback_val)
                    st.success(f"已记录 ⭐{feedback_val}")


def _render_system(days):
    """系统视图"""
    store = get_store()
    stats = store.get_db_stats()

    st.markdown("### 数据库统计")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总Traces", stats['total_traces'])
    c2.metric("总Spans", stats['total_spans'])
    c3.metric("总反馈", stats['total_feedback'])
    c4.metric("DB大小", f"{stats['db_size_mb']} MB")

    st.markdown("---")

    # 导出
    st.markdown("### 数据导出")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 导出JSON"):
            path = export_traces_json(days=days)
            st.success(f"已导出: {path}")
    with col2:
        if st.button("📊 导出CSV"):
            path = export_traces_csv(days=days)
            st.success(f"已导出: {path}")

    # 清理
    st.markdown("### 数据清理")
    cleanup_days = st.number_input("保留天数", value=90, min_value=7, max_value=365)
    if st.button("🗑️ 清理过期数据", type="secondary"):
        store.cleanup(days=cleanup_days)
        st.success(f"已清理 {cleanup_days} 天前的数据")

    # 质量指标
    st.markdown("### 质量信号")
    quality = get_metrics().get_quality_metrics(days)
    st.markdown(f"""
    - 知识图谱纠正率: **{quality['kg_correction_rate']}%**
    - 有反馈的查询: **{quality['total_with_feedback']}** / {quality['total_queries']}
    """)

    if quality['feedback_distribution']:
        st.markdown("反馈分布：")
        for item in quality['feedback_distribution']:
            stars = "⭐" * (item['rating'] or 0)
            count = item['count'] or 0
            st.markdown(f"  {stars} — {count} 次")


def _metric_card(title: str, value: str, subtitle: str, color: str):
    """渲染KPI卡片"""
    st.markdown(f"""
    <div style="padding:12px 16px; background:#0f172a; border-radius:10px;
         border-left:4px solid {color};">
        <div style="font-size:0.75rem; color:#64748b;">{title}</div>
        <div style="font-size:1.5rem; font-weight:700; color:{color}; margin:2px 0;">
            {value}
        </div>
        <div style="font-size:0.7rem; color:#475569;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
