#!/usr/bin/env python3
"""
MRARFAI 智能异常检测引擎 v1.0
=================================
统计模型替代规则引擎，自动发现：
  1. Z-Score 异常    — 偏离均值>2σ的月份
  2. IQR 箱线图异常  — 超出1.5倍四分位距
  3. 趋势断裂检测    — 连续增长/下降突然反转
  4. 波动率异常      — 某月变异幅度远超历史
  5. 客户间关联异常  — 多个客户同时下滑（系统性风险）
"""

import numpy as np
from datetime import datetime
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 核心检测算法
# ============================================================

def zscore_detect(values: list, threshold: float = 2.0) -> list:
    """
    Z-Score 异常检测
    返回: [(month_idx, value, z_score, direction)]
    """
    arr = np.array(values, dtype=float)
    nonzero = arr[arr > 0]
    if len(nonzero) < 3:
        return []
    
    mean = np.mean(nonzero)
    std = np.std(nonzero)
    if std < 1e-6:
        return []
    
    anomalies = []
    for i, v in enumerate(arr):
        if v > 0:
            z = (v - mean) / std
            if abs(z) > threshold:
                direction = "📈 异常高" if z > 0 else "📉 异常低"
                anomalies.append((i, v, round(z, 2), direction))
    
    return anomalies


def iqr_detect(values: list, k: float = 1.5) -> list:
    """
    IQR (四分位距) 异常检测 — 对偏态分布更稳健
    返回: [(month_idx, value, bound_type)]
    """
    arr = np.array(values, dtype=float)
    nonzero = arr[arr > 0]
    if len(nonzero) < 4:
        return []
    
    q1 = np.percentile(nonzero, 25)
    q3 = np.percentile(nonzero, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    
    anomalies = []
    for i, v in enumerate(arr):
        if v > 0:
            if v > upper:
                anomalies.append((i, v, f"超上界 (>{upper:,.0f})"))
            elif v < lower and lower > 0:
                anomalies.append((i, v, f"低于下界 (<{lower:,.0f})"))
    
    return anomalies


def trend_break_detect(values: list, min_streak: int = 2) -> list:
    """
    趋势断裂检测 — 连续N月增长后突然暴跌，或连续下降后反弹
    返回: [(month_idx, break_type, magnitude)]
    """
    arr = np.array(values, dtype=float)
    breaks = []
    
    for i in range(min_streak + 1, len(arr)):
        if arr[i] <= 0 or arr[i-1] <= 0:
            continue
        
        # 检查之前min_streak个月是否连续同方向
        prev_diffs = []
        valid = True
        for j in range(i - min_streak, i):
            if arr[j] <= 0 or arr[j-1] <= 0:
                valid = False
                break
            prev_diffs.append(arr[j] - arr[j-1])
        
        if not valid or not prev_diffs:
            continue
        
        current_diff = arr[i] - arr[i-1]
        
        # 连续增长后暴跌
        if all(d > 0 for d in prev_diffs) and current_diff < 0:
            magnitude = current_diff / arr[i-1] * 100
            if abs(magnitude) > 15:
                breaks.append((i, "📉 增长趋势断裂", f"{magnitude:+.1f}%"))
        
        # 连续下降后反弹
        elif all(d < 0 for d in prev_diffs) and current_diff > 0:
            magnitude = current_diff / arr[i-1] * 100
            if magnitude > 20:
                breaks.append((i, "📈 下降趋势反转", f"{magnitude:+.1f}%"))
    
    return breaks


def volatility_detect(values: list, window: int = 3, threshold: float = 2.5) -> list:
    """
    波动率异常 — 某月环比变动幅度远超历史滚动波动率
    返回: [(month_idx, mom_change, historical_vol, ratio)]
    """
    arr = np.array(values, dtype=float)
    anomalies = []
    
    for i in range(window + 1, len(arr)):
        if arr[i] <= 0 or arr[i-1] <= 0:
            continue
        
        # 计算历史窗口内的环比变动
        hist_changes = []
        for j in range(max(1, i - window * 2), i):
            if arr[j] > 0 and arr[j-1] > 0:
                hist_changes.append((arr[j] - arr[j-1]) / arr[j-1])
        
        if len(hist_changes) < 3:
            continue
        
        hist_vol = np.std(hist_changes)
        if hist_vol < 0.01:
            continue
        
        current_change = (arr[i] - arr[i-1]) / arr[i-1]
        ratio = abs(current_change) / hist_vol
        
        if ratio > threshold:
            anomalies.append((
                i,
                f"{current_change*100:+.1f}%",
                f"历史波动率{hist_vol*100:.1f}%",
                f"{ratio:.1f}倍",
            ))
    
    return anomalies


def systemic_risk_detect(all_customers: list, month_names: list = None) -> list:
    """
    系统性风险检测 — 多个客户同月份同时下滑
    all_customers: [{'客户': str, '月度金额': [float]*12}]
    返回: [(month_idx, count, customers, avg_decline)]
    """
    if not month_names:
        month_names = [f"{i+1}月" for i in range(12)]
    
    monthly_declines = {}  # month_idx -> [(customer, decline_pct)]
    
    for c in all_customers:
        monthly = c.get('月度金额', [0] * 12)
        for i in range(1, len(monthly)):
            if monthly[i-1] > 0 and monthly[i] >= 0:
                change = (monthly[i] - monthly[i-1]) / monthly[i-1]
                if change < -0.3:  # >30%下滑
                    if i not in monthly_declines:
                        monthly_declines[i] = []
                    monthly_declines[i].append((c['客户'], change))
    
    systemic = []
    for month_idx, declines in sorted(monthly_declines.items()):
        if len(declines) >= 3:  # 3家以上同时下滑 = 系统性
            avg_decline = np.mean([d[1] for d in declines])
            customers = [d[0] for d in declines[:5]]
            systemic.append((
                month_idx,
                len(declines),
                customers,
                f"{avg_decline*100:.1f}%",
            ))
    
    return systemic


# ============================================================
# 主检测入口
# ============================================================

def run_full_detection(data: dict, results: dict) -> dict:
    """
    对所有客户运行全部异常检测算法
    
    返回: {
        'summary': {总异常数, 按类型分布, 按月份分布, 严重程度},
        'customer_anomalies': [{客户, 异常列表}],
        'systemic_risks': [],
        'top_anomalies': [],  # 最严重的异常排序
    }
    """
    customers = data.get('客户金额', [])
    month_names = [f"{i+1}月" for i in range(12)]
    
    all_anomalies = []
    customer_anomalies = []
    
    for c in customers:
        name = c['客户']
        monthly = c.get('月度金额', [0] * 12)
        annual = c.get('年度金额', 0)
        
        if annual <= 0:
            continue
        
        c_anomalies = []
        
        # 1. Z-Score
        for idx, val, z, direction in zscore_detect(monthly):
            severity = "🔴" if abs(z) > 3 else "🟡"
            c_anomalies.append({
                'type': 'Z-Score',
                'month': month_names[idx],
                'month_idx': idx,
                'detail': f"{direction} (z={z})",
                'value': val,
                'severity': severity,
                'score': abs(z),
            })
        
        # 2. IQR
        for idx, val, bound in iqr_detect(monthly):
            c_anomalies.append({
                'type': 'IQR异常',
                'month': month_names[idx],
                'month_idx': idx,
                'detail': bound,
                'value': val,
                'severity': '🟡',
                'score': 2.0,
            })
        
        # 3. 趋势断裂
        for idx, break_type, magnitude in trend_break_detect(monthly):
            is_severe = float(magnitude.replace('%', '').replace('+', '')) 
            severity = "🔴" if abs(is_severe) > 40 else "🟡"
            c_anomalies.append({
                'type': '趋势断裂',
                'month': month_names[idx],
                'month_idx': idx,
                'detail': f"{break_type} {magnitude}",
                'value': monthly[idx],
                'severity': severity,
                'score': abs(is_severe) / 10,
            })
        
        # 4. 波动率
        for idx, change, hist_vol, ratio in volatility_detect(monthly):
            ratio_val = float(ratio.replace('倍', ''))
            severity = "🔴" if ratio_val > 4 else "🟡"
            c_anomalies.append({
                'type': '波动率异常',
                'month': month_names[idx],
                'month_idx': idx,
                'detail': f"环比{change}, {hist_vol}, 偏离{ratio}",
                'value': monthly[idx],
                'severity': severity,
                'score': ratio_val,
            })
        
        if c_anomalies:
            customer_anomalies.append({
                '客户': name,
                '年度金额': annual,
                '异常数': len(c_anomalies),
                '严重异常': sum(1 for a in c_anomalies if a['severity'] == '🔴'),
                '异常列表': sorted(c_anomalies, key=lambda x: x['score'], reverse=True),
            })
            all_anomalies.extend([{**a, '客户': name, '年度金额': annual} for a in c_anomalies])
    
    # 5. 系统性风险
    systemic = systemic_risk_detect(customers, month_names)
    
    # 汇总
    customer_anomalies.sort(key=lambda x: x['严重异常'], reverse=True)
    all_anomalies.sort(key=lambda x: x['score'], reverse=True)
    
    type_dist = Counter(a['type'] for a in all_anomalies)
    month_dist = Counter(a['month'] for a in all_anomalies)
    severity_dist = Counter(a['severity'] for a in all_anomalies)
    
    return {
        'summary': {
            '总异常数': len(all_anomalies),
            '涉及客户': len(customer_anomalies),
            '严重异常': severity_dist.get('🔴', 0),
            '警告异常': severity_dist.get('🟡', 0),
            '系统性风险': len(systemic),
            '类型分布': dict(type_dist),
            '月份分布': dict(month_dist),
        },
        'customer_anomalies': customer_anomalies,
        'systemic_risks': systemic,
        'top_anomalies': all_anomalies[:20],
    }


# ============================================================
# 可视化
# ============================================================

def make_anomaly_timeline(detection: dict):
    """异常时间线热力图"""
    month_names = [f"{i+1}月" for i in range(12)]
    customers = detection['customer_anomalies'][:15]
    
    if not customers:
        return None
    
    # 构建矩阵
    matrix = []
    y_labels = []
    for c in customers:
        row = [0] * 12
        for a in c['异常列表']:
            idx = a['month_idx']
            score = a['score']
            if row[idx] < score:
                row[idx] = score
        matrix.append(row)
        y_labels.append(c['客户'][:8])
    
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=month_names,
        y=y_labels,
        colorscale=[
            [0, 'rgba(17,17,17,0.8)'],
            [0.3, 'rgba(136,136,136,0.4)'],
            [0.6, 'rgba(187,187,187,0.6)'],
            [1, 'rgba(255,255,255,0.8)'],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text='异常程度', font=dict(color='#94a3b8', size=10)),
            tickfont=dict(color='#94a3b8', size=9),
        ),
        hovertemplate='%{y} · %{x}<br>异常分: %{z:.1f}<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(l=80, r=20, t=10, b=40),
        height=max(200, len(customers) * 28 + 60),
        xaxis=dict(tickfont=dict(color='#94a3b8', size=10)),
        yaxis=dict(tickfont=dict(color='#94a3b8', size=10), autorange='reversed'),
    )
    return fig


def make_anomaly_type_chart(detection: dict):
    """异常类型分布"""
    type_dist = detection['summary']['类型分布']
    if not type_dist:
        return None
    
    colors = {
        'Z-Score': '#FFFFFF',
        'IQR异常': '#CCCCCC',
        '趋势断裂': '#999999',
        '波动率异常': '#AAAAAA',
    }

    labels = list(type_dist.keys())
    values = list(type_dist.values())
    bar_colors = [colors.get(l, '#888888') for l in labels]
    
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=bar_colors, opacity=0.8,
                   line=dict(width=1, color='rgba(255,255,255,0.1)')),
        text=values, textposition='outside',
        textfont=dict(color='#e2e8f0', size=12),
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(l=40, r=20, t=10, b=40),
        height=250,
        xaxis=dict(gridcolor='rgba(148,163,184,0.05)', tickfont=dict(color='#94a3b8')),
        yaxis=dict(gridcolor='rgba(148,163,184,0.08)', tickfont=dict(color='#94a3b8')),
    )
    return fig


def make_monthly_anomaly_chart(detection: dict):
    """月度异常分布"""
    month_dist = detection['summary']['月份分布']
    months = [f"{i+1}月" for i in range(12)]
    values = [month_dist.get(m, 0) for m in months]
    
    fig = go.Figure(go.Bar(
        x=months, y=values,
        marker=dict(
            color=[f'rgba(255,255,255,{min(1, v/max(max(values),1)*0.8+0.2)})' for v in values],
            line=dict(width=1, color='rgba(255,255,255,0.05)'),
        ),
        text=values, textposition='outside',
        textfont=dict(color='#94a3b8', size=10),
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(l=40, r=20, t=10, b=40),
        height=220,
        xaxis=dict(gridcolor='rgba(148,163,184,0.05)', tickfont=dict(color='#94a3b8')),
        yaxis=dict(gridcolor='rgba(148,163,184,0.08)', tickfont=dict(color='#94a3b8')),
    )
    return fig


# ============================================================
# Streamlit 渲染
# ============================================================

def render_anomaly_dashboard(data: dict, results: dict):
    """渲染异常检测看板"""
    import streamlit as st
    
    detection = run_full_detection(data, results)
    summary = detection['summary']
    
    # 概览指标
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总异常", f"{summary['总异常数']}个")
    c2.metric("🔴 严重", f"{summary['严重异常']}个")
    c3.metric("🟡 警告", f"{summary['警告异常']}个")
    c4.metric("涉及客户", f"{summary['涉及客户']}家")
    c5.metric("系统性风险", f"{summary['系统性风险']}次")
    
    st.markdown("")
    
    # 图表
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**异常类型分布**")
        chart = make_anomaly_type_chart(detection)
        if chart:
            st.plotly_chart(chart, use_container_width=True, key="anomaly_type_dist")
    with col2:
        st.markdown("**月度异常分布**")
        chart = make_monthly_anomaly_chart(detection)
        if chart:
            st.plotly_chart(chart, use_container_width=True, key="anomaly_month_dist")
    
    # 热力图
    st.markdown("**客户异常热力图** — 颜色越深异常越严重")
    heatmap = make_anomaly_timeline(detection)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True, key="anomaly_heatmap")
    
    # 系统性风险
    if detection['systemic_risks']:
        st.markdown("#### ⚠️ 系统性风险事件")
        st.markdown("多个客户同月份同时暴跌 (>30%)，可能是市场/行业层面的问题")
        for i, (idx, count, customers, avg_dec) in enumerate(detection['systemic_risks']):
            st.warning(
                f"**{idx+1}月** — {count}家客户同时暴跌，"
                f"平均跌幅 {avg_dec}\n\n"
                f"涉及：{', '.join(customers)}"
            )
    
    # 客户异常详情
    st.markdown("#### 🔍 客户异常详情")
    
    severity_filter = st.radio(
        "筛选", ["全部", "🔴 仅严重", "🟡 仅警告"],
        horizontal=True, key="anomaly_severity_filter"
    )
    
    for i, ca in enumerate(detection['customer_anomalies'][:20]):
        if severity_filter == "🔴 仅严重" and ca['严重异常'] == 0:
            continue
        if severity_filter == "🟡 仅警告" and ca['严重异常'] > 0:
            continue
        
        icon = "🔴" if ca['严重异常'] > 0 else "🟡"
        with st.expander(
            f"{icon} **{ca['客户']}** — {ca['异常数']}个异常 "
            f"({ca['严重异常']}个严重) — ¥{ca['年度金额']:,.0f}万",
            expanded=(i < 3 and ca['严重异常'] > 0)
        ):
            for a in ca['异常列表'][:8]:
                st.markdown(
                    f"&nbsp;&nbsp; {a['severity']} **{a['month']}** · "
                    f"`{a['type']}` · {a['detail']} · "
                    f"金额 ¥{a['value']:,.0f}万"
                )
    
    return detection
