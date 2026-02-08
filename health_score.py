#!/usr/bin/env python3
"""
MRARFAI 客户健康评分系统 v1.0
================================
多维度评分模型：
  - 营收贡献 (25%)  — 年度金额在总营收中的占比
  - 增长趋势 (25%)  — H2 vs H1 增速, 最近3个月 vs 前3个月
  - 稳定性   (20%)  — 月度营收变异系数(CV)，0月数量
  - 价格质量 (15%)  — 均价变动方向，以价补量 vs 量价齐跌
  - 订单频率 (15%)  — 活跃月数/12，连续出货月数

输出：0-100 健康分 + 等级（A/B/C/D/F）+ 雷达图
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# 评分引擎
# ============================================================

def compute_health_scores(data: dict, results: dict) -> list:
    """
    计算所有客户的健康评分
    
    返回: [{
        '客户': str,
        '总分': float,
        '等级': str,  # A/B/C/D/F
        '营收贡献': float,
        '增长趋势': float,
        '稳定性': float,
        '价格质量': float,
        '订单频率': float,
        '年度金额': float,
        '风险标签': list,
        '建议': str,
    }]
    """
    customers = data.get('客户金额', [])
    total_rev = data.get('总营收', 1)
    pv = {p['客户']: p for p in results.get('价量分解', [])}
    alerts = {a['客户']: a for a in results.get('流失预警', [])}
    
    scored = []
    
    for c in customers:
        name = c['客户']
        annual = c.get('年度金额', 0)
        monthly = c.get('月度金额', [0]*12)
        
        if annual <= 0:
            continue
        
        # --- 1. 营收贡献 (25分) ---
        share = annual / total_rev if total_rev > 0 else 0
        if share >= 0.10:
            s_rev = 25
        elif share >= 0.05:
            s_rev = 20
        elif share >= 0.02:
            s_rev = 15
        elif share >= 0.005:
            s_rev = 10
        else:
            s_rev = 5
        
        # --- 2. 增长趋势 (25分) ---
        h1 = sum(monthly[:6])
        h2 = sum(monthly[6:])
        recent3 = sum(monthly[9:12])
        prev3 = sum(monthly[6:9])
        
        # H2 vs H1 增速
        if h1 > 0:
            h_growth = (h2 - h1) / h1
        else:
            h_growth = 1.0 if h2 > 0 else 0
        
        # 最近3月 vs 前3月
        if prev3 > 0:
            r_growth = (recent3 - prev3) / prev3
        else:
            r_growth = 1.0 if recent3 > 0 else -1.0
        
        # 综合增长得分
        avg_growth = (h_growth * 0.4 + r_growth * 0.6)
        if avg_growth >= 0.3:
            s_growth = 25
        elif avg_growth >= 0.1:
            s_growth = 20
        elif avg_growth >= 0:
            s_growth = 15
        elif avg_growth >= -0.2:
            s_growth = 10
        elif avg_growth >= -0.5:
            s_growth = 5
        else:
            s_growth = 0
        
        # --- 3. 稳定性 (20分) ---
        active_months = [m for m in monthly if m > 0]
        num_active = len(active_months)
        num_zero = 12 - num_active
        
        if num_active >= 2:
            cv = np.std(active_months) / np.mean(active_months)
        else:
            cv = 2.0
        
        # CV低 = 稳定 = 高分
        if cv < 0.3 and num_zero <= 1:
            s_stable = 20
        elif cv < 0.5 and num_zero <= 2:
            s_stable = 16
        elif cv < 0.8 and num_zero <= 4:
            s_stable = 12
        elif cv < 1.2:
            s_stable = 8
        else:
            s_stable = 4
        
        # --- 4. 价格质量 (15分) ---
        pv_info = pv.get(name, {})
        quality = pv_info.get('质量评估', '')
        
        if '优质' in quality:
            s_price = 15
        elif '价格稳定' in quality:
            s_price = 12
        elif '以价补量' in quality:
            s_price = 8
        elif '以量换价' in quality or '量换价' in quality:
            s_price = 6
        elif '齐跌' in quality:
            s_price = 3
        else:
            s_price = 9  # 默认中等
        
        # --- 5. 订单频率 (15分) ---
        # 活跃月数
        freq_ratio = num_active / 12
        # 连续性：最后N个月是否连续
        last_streak = 0
        for m in reversed(monthly):
            if m > 0:
                last_streak += 1
            else:
                break
        
        s_freq = min(15, int(freq_ratio * 10 + last_streak * 0.5))
        
        # --- 总分 ---
        total = s_rev + s_growth + s_stable + s_price + s_freq
        
        # --- 等级 ---
        if total >= 85:
            grade = 'A'
        elif total >= 70:
            grade = 'B'
        elif total >= 55:
            grade = 'C'
        elif total >= 40:
            grade = 'D'
        else:
            grade = 'F'
        
        # --- 风险标签 ---
        risk_tags = []
        if name in alerts:
            risk_tags.append(f"⚠️ {alerts[name].get('风险', '预警')}")
        if avg_growth < -0.3:
            risk_tags.append("📉 急速下滑")
        if num_zero >= 6:
            risk_tags.append("💤 半年无单")
        if cv > 1.5:
            risk_tags.append("🎲 极不稳定")
        if '齐跌' in quality:
            risk_tags.append("💀 量价齐跌")
        
        # --- 建议 ---
        if grade == 'A':
            advice = "核心客户，优先保障交付和服务"
        elif grade == 'B':
            advice = "重点维护，挖掘增长空间"
        elif grade == 'C':
            if avg_growth > 0:
                advice = "成长型客户，加大投入培育"
            else:
                advice = "关注下滑原因，及时干预"
        elif grade == 'D':
            advice = "预警客户，安排专项拜访评估"
        else:
            advice = "高危客户，评估是否值得继续投入"
        
        scored.append({
            '客户': name,
            '总分': round(total, 1),
            '等级': grade,
            '营收贡献': s_rev,
            '增长趋势': s_growth,
            '稳定性': s_stable,
            '价格质量': s_price,
            '订单频率': s_freq,
            '年度金额': annual,
            '风险标签': risk_tags,
            '建议': advice,
        })
    
    # 按总分降序
    scored.sort(key=lambda x: x['总分'], reverse=True)
    return scored


# ============================================================
# 可视化
# ============================================================

def _grade_color(grade):
    return {
        'A': '#10b981', 'B': '#6366f1',
        'C': '#f59e0b', 'D': '#f97316', 'F': '#ef4444',
    }.get(grade, '#94a3b8')


def make_health_overview_chart(scores: list):
    """等级分布饼图"""
    from collections import Counter
    grades = Counter(s['等级'] for s in scores)
    labels = ['A', 'B', 'C', 'D', 'F']
    values = [grades.get(g, 0) for g in labels]
    colors = [_grade_color(g) for g in labels]
    
    fig = go.Figure(go.Pie(
        labels=[f"{g}级 ({v}家)" for g, v in zip(labels, values)],
        values=values,
        marker=dict(colors=colors),
        hole=0.45,
        textinfo='percent+label',
        textfont=dict(size=12, color='white'),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        showlegend=False,
    )
    return fig


def make_radar_chart(score: dict):
    """单客户雷达图"""
    categories = ['营收贡献', '增长趋势', '稳定性', '价格质量', '订单频率']
    max_vals = [25, 25, 20, 15, 15]
    values = [score[c] for c in categories]
    # 归一化到0-100
    normalized = [v / m * 100 for v, m in zip(values, max_vals)]
    normalized.append(normalized[0])  # 闭合
    categories.append(categories[0])
    
    fig = go.Figure(go.Scatterpolar(
        r=normalized,
        theta=categories,
        fill='toself',
        fillcolor=f"rgba(99,102,241,0.15)",
        line=dict(color='#6366f1', width=2),
        marker=dict(size=6, color='#6366f1'),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor='rgba(148,163,184,0.15)',
                tickfont=dict(size=9, color='#64748b'),
            ),
            angularaxis=dict(
                gridcolor='rgba(148,163,184,0.15)',
                tickfont=dict(size=11, color='#cbd5e1'),
            ),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=30, b=30),
        height=280,
    )
    return fig


def make_scatter_chart(scores: list):
    """健康分 vs 营收 气泡图"""
    fig = go.Figure()
    
    for grade in ['A', 'B', 'C', 'D', 'F']:
        subset = [s for s in scores if s['等级'] == grade]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[s['年度金额'] for s in subset],
            y=[s['总分'] for s in subset],
            mode='markers+text',
            name=f'{grade}级',
            text=[s['客户'][:6] for s in subset],
            textposition='top center',
            textfont=dict(size=9, color='#94a3b8'),
            marker=dict(
                size=[max(8, min(40, s['年度金额'] / max(1, scores[0]['年度金额']) * 40)) for s in subset],
                color=_grade_color(grade),
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
            ),
        ))
    
    fig.update_layout(
        xaxis=dict(title=dict(text='年度金额 (万)', font=dict(color='#94a3b8')),
                   gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#94a3b8')),
        yaxis=dict(title=dict(text='健康评分', font=dict(color='#94a3b8')),
                   gridcolor='rgba(148,163,184,0.1)', range=[0, 105],
                   tickfont=dict(color='#94a3b8')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(l=50, r=20, t=20, b=50),
        height=350,
        legend=dict(font=dict(color='#94a3b8')),
    )
    return fig


# ============================================================
# Streamlit 渲染
# ============================================================

def render_health_dashboard(data: dict, results: dict):
    """在Streamlit中渲染客户健康评分看板"""
    import streamlit as st
    
    scores = compute_health_scores(data, results)
    if not scores:
        st.warning("无客户数据")
        return scores
    
    # 概览指标
    avg_score = np.mean([s['总分'] for s in scores])
    a_count = sum(1 for s in scores if s['等级'] == 'A')
    f_count = sum(1 for s in scores if s['等级'] in ('D', 'F'))
    risk_count = sum(1 for s in scores if s['风险标签'])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平均健康分", f"{avg_score:.1f}/100")
    c2.metric("A级客户", f"{a_count}家", help="健康分≥85")
    c3.metric("D/F级客户", f"{f_count}家", help="健康分<55，需要关注")
    c4.metric("有风险标签", f"{risk_count}家")
    
    # 两列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(make_health_overview_chart(scores), use_container_width=True, key="health_pie")
    
    with col2:
        st.plotly_chart(make_scatter_chart(scores), use_container_width=True, key="health_scatter")
    
    # 客户列表
    st.markdown("#### 客户健康评分排行")
    
    # 筛选器
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        grade_filter = st.multiselect(
            "筛选等级", ['A', 'B', 'C', 'D', 'F'],
            default=['A', 'B', 'C', 'D', 'F'],
            key="health_grade_filter"
        )
    
    filtered = [s for s in scores if s['等级'] in grade_filter]
    
    for i, s in enumerate(filtered[:20]):
        color = _grade_color(s['等级'])
        risk_html = " ".join(
            f'<span style="font-size:0.7rem; background:rgba(239,68,68,0.1); '
            f'padding:2px 6px; border-radius:4px; color:#fca5a5;">{t}</span>'
            for t in s['风险标签']
        ) if s['风险标签'] else ''
        
        with st.expander(
            f"{'🟢' if s['等级'] in ('A','B') else '🟡' if s['等级']=='C' else '🔴'} "
            f"**{s['客户']}** — {s['总分']}分 ({s['等级']}级) — ¥{s['年度金额']:,.0f}万",
            expanded=(i < 3)
        ):
            ec1, ec2 = st.columns([1, 1])
            with ec1:
                st.plotly_chart(make_radar_chart(s), use_container_width=True, key=f"radar_{i}_{s['客户'][:4]}")
            with ec2:
                st.markdown(f"""
                | 维度 | 得分 | 满分 |
                |------|------|------|
                | 营收贡献 | **{s['营收贡献']}** | 25 |
                | 增长趋势 | **{s['增长趋势']}** | 25 |
                | 稳定性 | **{s['稳定性']}** | 20 |
                | 价格质量 | **{s['价格质量']}** | 15 |
                | 订单频率 | **{s['订单频率']}** | 15 |
                """)
                if risk_html:
                    st.markdown(f"**风险标签：** {risk_html}", unsafe_allow_html=True)
                st.markdown(f"**建议：** {s['建议']}")
    
    return scores
