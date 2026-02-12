"""
MRARFAI v8 — Agent Chat Tab (Sprocomm 禾苗品牌版)
Agent colors mapped to Sprocomm leaf palette:
  🟢 Atlas (分析师) — Sprocomm Green
  🔵 Shield (风控) — Sprocomm Blue
  🔴 Nova (策略师) — Sprocomm Red
  📝 Quill (报告) — blend
  ⚡ V8.0 — Adaptive Gate + Context Eng + Meta-Memory + Self-Evolution
"""

import streamlit as st
import time
import re
from typing import Optional

# V8.0 Import
try:
    from v8_patch import (
        v8_pre_process, v8_post_process, v8_get_stats,
        v8_build_telos_from_results, get_v8_status,
        HAS_V8_GATE, HAS_V8_CTX, HAS_V8_MEM, HAS_V8_EVO,
    )
    HAS_V8 = True
except ImportError:
    HAS_V8 = False

# ── Command Center palette ──────────────────────────────────────
SP_GREEN = "#00FF88"       # Neon green — command center accent
SP_BLUE  = "#00A0C8"
SP_RED   = "#D94040"
BRAND_GREEN = "#8CBF3F"    # Original Sprocomm green
C_TEXT_MUTED = "#6a6a6a"
C_TEXT_SEC   = "#8a8a8a"
C_SUCCESS    = SP_GREEN
C_WARNING    = "#FF8800"

# ── Agent Registry — mapped to Sprocomm 三叶 ───────────────────
AGENTS = {
    "📊 数据分析师": {
        "name": "数据分析师",
        "role": "DATA ANALYST",
        "color": SP_GREEN,
        "bg": "rgba(0,255,136,0.08)",
        "border": "rgba(0,255,136,0.20)",
        "icon": "📊",
    },
    "🛡️ 风控专家": {
        "name": "风控专家",
        "role": "RISK CONTROL",
        "color": SP_RED,
        "bg": "rgba(217,64,64,0.08)",
        "border": "rgba(217,64,64,0.20)",
        "icon": "🛡️",
    },
    "💡 策略师": {
        "name": "策略师",
        "role": "STRATEGIST",
        "color": SP_BLUE,
        "bg": "rgba(0,160,200,0.08)",
        "border": "rgba(0,160,200,0.20)",
        "icon": "💡",
    },
    "🖊️ 报告员": {
        "name": "报告员",
        "role": "REPORTER",
        "color": "#8a8a8a",
        "bg": "rgba(138,138,138,0.08)",
        "border": "rgba(138,138,138,0.20)",
        "icon": "🖊️",
    },
    "🔍 质量审查": {
        "name": "质量审查",
        "role": "QUALITY REVIEW",
        "color": SP_GREEN,
        "bg": "rgba(0,255,136,0.06)",
        "border": "rgba(0,255,136,0.18)",
        "icon": "🔍",
    },
}

SUGGESTIONS = [
    "今年总营收和去年比怎么样？",
    "哪些客户有流失风险？",
    "CEO 本月该关注什么？",
    "各区域增长排名",
]


# ════════════════════════════════════════════════════════════════
#  v4.0 STREAMING HELPERS
# ════════════════════════════════════════════════════════════════

def _update_agent_progress(container, active: dict, completed: set):
    """Live agent progress display during streaming — command center style."""
    if not active and not completed:
        return

    html = '<div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin:0.3rem 0;">'
    for name in completed:
        a = AGENTS.get(name, {"icon": "✅", "color": SP_GREEN})
        html += (
            f'<span style="background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.25);'
            f'padding:0.15rem 0.5rem;font-size:0.68rem;color:{SP_GREEN};'
            f'font-family:\'JetBrains Mono\',monospace;font-weight:600;letter-spacing:0.03em;">'
            f'✓ {name}</span>'
        )
    for name, status in active.items():
        a = AGENTS.get(name, {"icon": "⏳", "color": C_TEXT_MUTED})
        html += (
            f'<span style="background:rgba(138,138,138,0.06);border:1px solid rgba(138,138,138,0.20);'
            f'padding:0.15rem 0.5rem;font-size:0.68rem;color:{C_TEXT_MUTED};'
            f'font-family:\'JetBrains Mono\',monospace;font-weight:600;letter-spacing:0.03em;">'
            f'⏳ {name}</span>'
        )
    html += '</div>'
    container.markdown(html, unsafe_allow_html=True)


def _render_v4_badges(result: dict):
    """v4.0 feature status badges — command center style."""
    badges = []
    if result.get("tool_use_enabled"):
        badges.append(f'<span style="background:rgba(0,255,136,0.06);color:{SP_GREEN};'
                      f'border:1px solid rgba(0,255,136,0.20);'
                      f'padding:0.1rem 0.4rem;font-size:0.58rem;font-weight:700;'
                      f'letter-spacing:0.05em;font-family:\'JetBrains Mono\',monospace;">TOOL USE</span>')
    if result.get("guardrails_enabled"):
        badges.append(f'<span style="background:rgba(0,160,200,0.06);color:{SP_BLUE};'
                      f'border:1px solid rgba(0,160,200,0.20);'
                      f'padding:0.1rem 0.4rem;font-size:0.58rem;font-weight:700;'
                      f'letter-spacing:0.05em;font-family:\'JetBrains Mono\',monospace;">GUARDRAILS</span>')
    if result.get("streaming_enabled"):
        badges.append(f'<span style="background:rgba(138,138,138,0.06);color:#8a8a8a;'
                      f'border:1px solid rgba(138,138,138,0.20);'
                      f'padding:0.1rem 0.4rem;font-size:0.58rem;font-weight:700;'
                      f'letter-spacing:0.05em;font-family:\'JetBrains Mono\',monospace;">STREAMING</span>')
    if result.get("from_cache"):
        badges.append(f'<span style="background:rgba(255,136,0,0.06);color:{C_WARNING};'
                      f'border:1px solid rgba(255,136,0,0.20);'
                      f'padding:0.1rem 0.4rem;font-size:0.58rem;font-weight:700;'
                      f'letter-spacing:0.05em;font-family:\'JetBrains Mono\',monospace;">CACHED</span>')

    budget = result.get("budget_status")
    if budget and budget.get("level") != "normal":
        level_color = {
            "caution": C_WARNING,
            "warning": SP_RED,
            "critical": SP_RED,
        }.get(budget["level"], C_TEXT_MUTED)
        badges.append(
            f'<span style="background:rgba(217,64,64,0.06);color:{level_color};'
            f'border:1px solid rgba(217,64,64,0.20);'
            f'padding:0.1rem 0.4rem;font-size:0.58rem;font-weight:700;'
            f'letter-spacing:0.05em;font-family:\'JetBrains Mono\',monospace;">'
            f'{budget["usage_pct"]:.0f}% BUDGET</span>'
        )

    if badges:
        st.markdown(
            f'<div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin:0.4rem 0 0.2rem;">{"".join(badges)}</div>',
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════════
#  V8.0 RENDERING — Gate / Review / Eval / Memory
# ════════════════════════════════════════════════════════════════

V8_ORANGE = "#FF6B35"
V8_PURPLE = "#A855F7"
V8_CYAN   = "#06B6D4"

def _badge_html(label: str, color: str) -> str:
    """Generate a V8 badge span."""
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return (
        f'<span style="background:rgba({r},{g},{b},0.06);color:{color};'
        f'border:1px solid rgba({r},{g},{b},0.20);'
        f'padding:0.1rem 0.4rem;font-size:0.58rem;font-weight:700;'
        f"letter-spacing:0.05em;font-family:'JetBrains Mono',monospace;"
        f'">{label}</span>'
    )


def _render_v8_badges(result: dict):
    """V8.0 feature badges — gate level + review + memory."""
    if not result.get("v8_enhanced"):
        return

    badges = []
    gate = result.get("v8_gate", {})
    level = gate.get("level", "full")

    # Gate level badge
    gate_colors = {"skip": SP_GREEN, "light": V8_ORANGE, "full": V8_PURPLE}
    gate_labels = {"skip": "⚡ SKIP", "light": "🔀 LIGHT", "full": "🔥 FULL"}
    badges.append(_badge_html(gate_labels.get(level, level.upper()), gate_colors.get(level, V8_PURPLE)))

    # Gate score
    score = gate.get("score", 0)
    badges.append(_badge_html(f"GATE {score:.2f}", C_TEXT_MUTED))

    # Review badge
    review = result.get("v8_review")
    if review:
        rv_score = review.get("score", 0)
        rv_passed = review.get("passed", False)
        rv_color = SP_GREEN if rv_passed else SP_RED
        rv_label = f"{'✅' if rv_passed else '❌'} REVIEW {rv_score:.1f}"
        badges.append(_badge_html(rv_label, rv_color))

    # Eval trend badge
    v8_eval = result.get("v8_eval")
    if v8_eval:
        trend = v8_eval.get("trend", "stable")
        trend_icons = {"improving": "📈", "stable": "➡️", "declining": "📉"}
        badges.append(_badge_html(f"{trend_icons.get(trend, '➡️')} {trend.upper()}", V8_CYAN))

    # Skills badge
    skills = result.get("v8_skills", [])
    if skills:
        badges.append(_badge_html(f"🎯 {len(skills)} SKILLS", V8_ORANGE))

    # V8 status modules
    status = result.get("v8_status", {})
    active_count = sum(1 for v in status.values() if v)
    badges.append(_badge_html(f"V8 {active_count}/4", V8_PURPLE))

    if badges:
        st.markdown(
            f'<div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin:0.3rem 0 0.1rem;">{"".join(badges)}</div>',
            unsafe_allow_html=True
        )


def _render_v8_gate_card(result: dict):
    """V8.0 gate routing detail card."""
    gate = result.get("v8_gate", {})
    if not gate:
        return

    level = gate.get("level", "full")
    score = gate.get("score", 0)
    agents = gate.get("agents", [])
    reason = gate.get("reason", "")

    level_colors = {"skip": "rgba(0,255,136,0.06)", "light": "rgba(255,107,53,0.06)", "full": "rgba(168,85,247,0.06)"}
    level_borders = {"skip": "rgba(0,255,136,0.15)", "light": "rgba(255,107,53,0.15)", "full": "rgba(168,85,247,0.15)"}
    level_icons = {"skip": "⚡", "light": "🔀", "full": "🔥"}

    agents_str = " → ".join(agents) if agents else "SQL直查"

    st.markdown(f"""
    <div style="background:{level_colors.get(level, 'rgba(138,138,138,0.06)')};
         border:1px solid {level_borders.get(level, 'rgba(138,138,138,0.15)')};
         padding:8px 12px; margin:4px 0;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem;
             color:#6a6a6a; letter-spacing:0.05em; margin-bottom:4px;">
            {level_icons.get(level, '🔥')} V8 GATE · {level.upper()} · score={score:.2f}
        </div>
        <div style="font-size:0.72rem; color:#ccc;">
            {agents_str}
        </div>
        <div style="font-size:0.58rem; color:#555; margin-top:2px;">
            {reason}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_v8_review_card(result: dict):
    """V8.0 structured review card."""
    review = result.get("v8_review")
    if not review:
        return

    score = review.get("score", 0)
    passed = review.get("passed", False)
    checks = review.get("checks", {})
    blockers = review.get("blockers", [])

    bar_color = SP_GREEN if passed else SP_RED
    bar_width = min(score * 10, 100)

    checks_html = ""
    if isinstance(checks, dict):
        checks_iter = checks.items()
    elif isinstance(checks, list):
        checks_iter = ((c.get("name", f"check_{i}"), c) for i, c in enumerate(checks) if isinstance(c, dict))
    else:
        checks_iter = []
    for check_name, check_data in checks_iter:
        if not isinstance(check_data, dict):
            check_data = {"score": 0, "passed": True}
        c_score = check_data.get("score", 0)
        c_passed = check_data.get("passed", True)
        icon = "✅" if c_passed else "❌"
        checks_html += f'<span style="font-size:0.55rem;color:#888;margin-right:8px;">{icon} {check_name}: {c_score:.1f}</span>'

    blockers_html = ""
    if blockers:
        for b in blockers:
            blockers_html += f'<div style="font-size:0.55rem;color:{SP_RED};margin-top:2px;">🚫 {b}</div>'

    st.markdown(f"""
    <div style="background:rgba(138,138,138,0.04); border:1px solid rgba(138,138,138,0.10);
         padding:8px 12px; margin:4px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.6rem;
                 color:#6a6a6a; letter-spacing:0.05em;">
                📝 V8 REVIEW · {"PASS" if passed else "FAIL"}
            </span>
            <span style="font-size:0.7rem; font-weight:700; color:{bar_color};">{score:.1f}/10</span>
        </div>
        <div style="background:rgba(138,138,138,0.10); height:3px; margin:4px 0; border-radius:2px;">
            <div style="background:{bar_color}; width:{bar_width}%; height:100%; border-radius:2px;"></div>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">
            {checks_html}
        </div>
        {blockers_html}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  RENDERING
# ════════════════════════════════════════════════════════════════

def _render_agent_cards(agents_used: list, agent_outputs: dict = None):
    """Cursor 2.0 style agent status cards with Sprocomm colors."""
    if not agents_used:
        return

    cards_html = ""
    for agent_key in agents_used:
        a = AGENTS.get(agent_key, {
            "name": agent_key, "role": "AGENT", "color": SP_GREEN,
            "bg": "rgba(140,191,63,0.10)", "border": "rgba(140,191,63,0.22)", "icon": "🤖",
        })

        # All agents in agents_used list have completed
        status = '<span style="font-family:monospace;font-size:0.58rem;padding:0.15rem 0.5rem;font-weight:700;flex-shrink:0;margin-left:auto;letter-spacing:0.05em;background:rgba(0,255,136,0.08);color:#00FF88;border:1px solid rgba(0,255,136,0.25);">DONE</span>'

        cards_html += f"""
        <div style="background:#111;border:1px solid #2f2f2f;border-left:2px solid {a['color']};padding:0.8rem 1rem;margin:0.3rem 0;display:flex;align-items:center;gap:0.75rem;">
            <div style="width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-size:0.9rem;flex-shrink:0;background:{a['bg']};border:1px solid {a['border']};">{a['icon']}</div>
            <div style="flex:1;min-width:0;">
                <div style="font-family:monospace;font-size:0.78rem;font-weight:700;color:#fff;letter-spacing:0.03em;">{a['name']}</div>
                <div style="font-family:monospace;font-size:0.58rem;color:#6a6a6a;text-transform:uppercase;letter-spacing:0.1em;">{a['role']}</div>
            </div>
            {status}
        </div>"""

    st.markdown(cards_html, unsafe_allow_html=True)


def _render_thinking_timeline(thinking_log: str, total_time: float = 0):
    """Collapsible thinking timeline with color-coded steps."""
    if not thinking_log:
        return

    lines = [l.strip() for l in thinking_log.strip().split("\n") if l.strip()]
    if not lines:
        return

    t_label = f"· {total_time:.1f}s" if total_time > 0 else ""

    with st.expander(f"🧠 推理过程 {t_label} · {len(lines)} 步骤", expanded=False):
        for line in lines:
            # Color mapping
            if any(k in line for k in ["✅", "完成", "通过"]):
                color = SP_GREEN
            elif any(k in line for k in ["⚠️", "警告", "未通过", "需改进"]):
                color = C_WARNING
            elif any(k in line for k in ["❌", "错误", "失败"]):
                color = SP_RED
            elif any(k in line for k in ["🔍", "审查", "质量"]):
                color = SP_GREEN
            elif any(k in line for k in ["🎯", "HITL", "置信"]):
                color = SP_BLUE
            elif any(k in line for k in ["🧠", "记忆"]):
                color = SP_BLUE
            elif any(k in line for k in ["📊", "分析", "Atlas"]):
                color = SP_GREEN
            elif any(k in line for k in ["🛡️", "风控", "Shield"]):
                color = SP_RED
            elif any(k in line for k in ["💡", "策略", "Nova"]):
                color = SP_BLUE
            else:
                color = C_TEXT_MUTED

            time_m = re.search(r'(\d+\.?\d*)\s*[sS秒]', line)
            time_str = f' `{time_m.group(1)}s`' if time_m else ""
            st.markdown(f"<span style='color:{color};font-size:8px;'>●</span> <span style='font-family:monospace;font-size:0.75rem;color:#aaa;'>{line}</span>{time_str}", unsafe_allow_html=True)


def _render_quality_badge(critique: dict):
    """Quality badge with dimension scores."""
    if not critique:
        return

    score = critique.get("overall_score", 0)
    passed = critique.get("pass_threshold", score >= 7.0)
    iters = critique.get("iterations", 1)
    imp = critique.get("improvement", 0)

    dims = critique.get("dimension_scores", {})
    abbr_map = {"completeness":"COM","accuracy":"ACC","actionability":"ACT","clarity":"CLA","consistency":"CON"}
    dim_text = " · ".join(f"{abbr_map.get(k,k[:3].upper())}:{v}" for k, v in dims.items()) if dims else ""

    bg = "rgba(0,255,136,0.08)" if passed else "rgba(255,136,0,0.08)"
    color = SP_GREEN if passed else C_WARNING
    border = "rgba(0,255,136,0.25)" if passed else "rgba(255,136,0,0.25)"
    icon = "✓" if passed else "!"
    label = "PASS" if passed else "REVIEW"

    imp_html = f'<span style="color:{SP_GREEN};margin-left:0.3rem;">↑{imp:.1f}</span>' if imp and imp > 0 else ""
    iter_html = f'<span style="color:{C_TEXT_MUTED};margin-left:0.3rem;">×{iters}</span>' if iters and iters > 1 else ""
    dim_html = f'<span style="font-size:0.58rem;color:{C_TEXT_MUTED};margin-left:0.4rem;">{dim_text}</span>' if dim_text else ""

    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.35rem 0.7rem;
         font-family:monospace;font-size:0.65rem;font-weight:700;letter-spacing:0.05em;
         margin:0.25rem 0;background:{bg};color:{color};border:1px solid {border};">
        <span style="font-size:0.8rem;">{icon}</span>
        <span>{label}</span>
        <span style="font-weight:700;">{score:.1f}</span>
        <span style="color:{C_TEXT_MUTED};">/10</span>
        {imp_html}{iter_html}{dim_html}
    </div>""", unsafe_allow_html=True)


def _render_hitl_card(hitl_decision: dict):
    """HITL confidence card — green/yellow/red gauge."""
    if not hitl_decision:
        return

    score = hitl_decision.get("confidence_score", 0)
    level = hitl_decision.get("confidence_level", "unknown")
    action = hitl_decision.get("action", "unknown")
    triggers = hitl_decision.get("triggers", [])

    pct = int(score * 100)

    level_map = {
        "high":   ("hitl-high",   SP_GREEN, "HIGH CONFIDENCE",   "自动执行"),
        "medium": ("hitl-medium", C_WARNING, "MEDIUM CONFIDENCE", "建议确认"),
        "low":    ("hitl-low",    SP_RED,   "LOW CONFIDENCE",    "需人工审查"),
    }
    gauge_cls, color, label, act_text = level_map.get(level, ("hitl-medium", C_WARNING, level.upper(), action))

    action_map = {
        "auto_execute": "自动执行", "suggest_confirmation": "建议确认",
        "require_approval": "需人工审查", "escalate": "升级处理",
    }
    act_text = action_map.get(action, act_text)

    trig_html = ""
    if triggers:
        items = []
        for t in triggers[:3]:
            if isinstance(t, dict):
                items.append(f"• {t.get('message', t.get('reason', str(t)))}")
            else:
                items.append(f"• {t}")
        trig_html = f'<div style="font-family:monospace;font-size:0.55rem;color:#6a6a6a;text-align:right;flex-shrink:0;">{"<br>".join(items)}</div>'

    gauge_bg = f"rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.10)"
    gauge_border = f"rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.35)"

    st.markdown(f"""
    <div style="background:#080808;border:1px solid #2f2f2f;padding:0.7rem 1rem;margin:0.5rem 0;display:flex;align-items:center;gap:0.8rem;">
        <div style="width:42px;height:42px;display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:0.72rem;font-weight:700;flex-shrink:0;background:{gauge_bg};color:{color};border:2px solid {gauge_border};">{pct}</div>
        <div style="flex:1;">
            <div style="font-family:monospace;font-size:0.65rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{color};">{label}</div>
            <div style="font-family:monospace;font-size:0.72rem;color:#aaa;margin-top:0.1rem;">{act_text}</div>
        </div>
        {trig_html}
    </div>""", unsafe_allow_html=True)


def _render_trace_bar(trace_id: str = "", obs_summary: dict = None):
    """Compact trace info bar — adapted to multi_agent.py obs_summary format."""
    items = []
    if trace_id:
        items.append(f'<span style="margin-right:0.5rem;">TRACE <span style="color:#00FF88;font-weight:700;">{trace_id[:8]}</span></span>')
    if obs_summary:
        # Latency
        lb = obs_summary.get("latency_breakdown", {})
        total_ms = lb.get("total_ms", 0)
        if total_ms:
            items.append(f'<span style="margin-right:0.5rem;">LATENCY <span style="color:#00FF88;font-weight:700;">{total_ms/1000:.1f}s</span></span>')
        # Tokens
        tokens = obs_summary.get("total_tokens", 0)
        if tokens:
            items.append(f'<span style="margin-right:0.5rem;">TOKENS <span style="color:#00FF88;font-weight:700;">{tokens:,}</span></span>')
        # Cost
        cost = obs_summary.get("total_cost_usd", 0)
        if cost:
            items.append(f'<span style="margin-right:0.5rem;">COST <span style="color:#00FF88;font-weight:700;">${cost:.4f}</span></span>')
        # LLM calls
        calls = obs_summary.get("total_llm_calls", 0)
        if calls:
            items.append(f'<span style="margin-right:0.5rem;">CALLS <span style="color:#00FF88;font-weight:700;">{calls}</span></span>')

    if items:
        st.markdown(f'<div style="display:flex;align-items:center;gap:1rem;padding:0.35rem 0.8rem;background:#080808;border:1px solid #2f2f2f;font-family:monospace;font-size:0.58rem;color:#6a6a6a;margin:0.4rem 0;">{" ".join(items)}</div>', unsafe_allow_html=True)


def _render_suggestion_chips():
    """Quick-start suggestion buttons."""
    cols = st.columns(len(SUGGESTIONS))
    for i, s in enumerate(SUGGESTIONS):
        with cols[i]:
            if st.button(s, key=f"suggest_{i}", use_container_width=True):
                st.session_state["pending_question"] = s
                st.rerun()


def _render_welcome():
    """Welcome state — command center terminal style."""
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-badge">
            <span style="width:6px;height:6px;border-radius:50%;background:{SP_GREEN};display:inline-block;"></span>
            AGENT TERMINAL · v5
        </div>
        <div class="hero-title">
            <span style="color:{SP_GREEN};">SPROCOMM</span><br>
            <span style="color:#FFFFFF;">SALES INTELLIGENCE</span>
        </div>
        <div class="hero-sub">
            // 多智能体协作 · 12维深度分析 · 实时预警系统
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Agent roster — command center style
    cols = st.columns(3)
    roster = [
        ("◈", "数据分析师", "DATA ANALYST", SP_GREEN, "rgba(0,255,136,0.06)", "rgba(0,255,136,0.20)"),
        ("◆", "风控专家", "RISK CONTROL", SP_RED, "rgba(217,64,64,0.06)", "rgba(217,64,64,0.20)"),
        ("◇", "策略师", "STRATEGIST", SP_BLUE, "rgba(0,160,200,0.06)", "rgba(0,160,200,0.20)"),
    ]
    for i, (icon, name, role, color, bg, border) in enumerate(roster):
        with cols[i]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};
                        padding:1.2rem;text-align:center;">
                <div style="font-size:1.4rem;margin-bottom:0.4rem;color:{color};">{icon}</div>
                <div style="font-family:var(--font-mono);font-size:0.82rem;font-weight:700;color:{color};
                     letter-spacing:0.05em;">{name}</div>
                <div style="font-family:var(--font-mono);font-size:0.52rem;color:{C_TEXT_MUTED};
                            text-transform:uppercase;letter-spacing:0.1em;margin-top:0.15rem;">{role}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def render_chat_tab(data, results: dict, benchmark: dict = None, forecast: dict = None,
                    provider: str = "deepseek", api_key: str = None):
    """Render the Agent chat tab with Sprocomm branding."""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    use_multi = st.session_state.get("use_multi_agent", False)
    has_data = data is not None and not (hasattr(data, "empty") and data.empty)

    if not has_data:
        _render_welcome()
        st.info("请先在左侧上传数据文件")
        return

    if not use_multi:
        _render_welcome()
        st.warning("请在左侧打开 Multi-Agent 开关")
        return

    # ── Chat history ────────────────────────────────────────
    # If a new question is pending, skip old history rendering
    has_pending = "pending_question" in st.session_state
    if not st.session_state.chat_history:
        _render_welcome()
        st.markdown("---")
        _render_suggestion_chips()
    elif not has_pending:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🌿"):
                    st.markdown(msg["content"])
                    if msg.get("agents_used"):
                        _render_agent_cards(msg["agents_used"], msg.get("expert_outputs"))
                    if msg.get("thinking_log"):
                        _render_thinking_timeline(msg["thinking_log"], msg.get("total_time", 0))
                    # v4.0 feature badges (from history)
                    v4f = msg.get("v4_features", {})
                    if v4f:
                        _render_v4_badges({
                            "tool_use_enabled": v4f.get("tool_use"),
                            "guardrails_enabled": v4f.get("guardrails"),
                            "streaming_enabled": v4f.get("streaming"),
                            "from_cache": v4f.get("from_cache"),
                        })
                    if msg.get("critique"):
                        _render_quality_badge(msg["critique"])
                    if msg.get("hitl_decision"):
                        _render_hitl_card(msg["hitl_decision"])
                    if msg.get("trace_id") or msg.get("obs_summary"):
                        _render_trace_bar(msg.get("trace_id", ""), msg.get("obs_summary"))
                    # V8.0 badges from history
                    if msg.get("v8_enhanced"):
                        _render_v8_badges(msg)

    # ── Input ───────────────────────────────────────────────
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("问一个关于销售数据的问题...")
    question = pending or user_input
    if not question:
        return

    # Clear previous history - only show current Q&A
    st.session_state.chat_history = []

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🌿"):
        # v4.0: Pre-create streaming containers
        status = st.status(f"分析中: {question[:40]}...", expanded=True)
        agent_progress = st.empty()  # for live agent status

        try:
            from multi_agent import ask_multi_agent

            # v4.0: Import streaming
            try:
                from streaming import StreamCallback, EventType
                has_streaming = True
            except ImportError:
                has_streaming = False

            # Fallback: read from session_state if not passed as params
            _provider = provider or st.session_state.get("ai_provider", "deepseek")
            _api_key = api_key or st.session_state.get("api_key", "")

            if not _api_key:
                st.warning("⚠️ 请先在左侧配置 API Key")
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ 请先在左侧 AI ENGINE 中配置 API Key"})
                return

            # v4.0: Create streaming callback
            stream_cb = StreamCallback() if has_streaming else None

            with status:
                # ════ V8.0 PRE-PROCESSING ════
                v8_pre = None
                if HAS_V8:
                    st.write("⚡ V8.0 门控评估...")
                    try:
                        v8_pre = v8_pre_process(
                            question, data if isinstance(data, dict) else {},
                            results or {}, _provider.lower()
                        )
                        gate_level = v8_pre.get("gate_level", "full")
                        gate_score = v8_pre.get("gate_score", 0)
                        gate_icons = {"skip": "⚡", "light": "🔀", "full": "🔥"}
                        st.write(f"{gate_icons.get(gate_level, '🔥')} 门控: {gate_level.upper()} (score={gate_score:.2f})")
                        for t in v8_pre.get("v8_thinking", []):
                            st.write(t)
                    except Exception as e:
                        st.write(f"⚠️ V8前处理异常: {e}")
                        v8_pre = None

                st.write("🌿 连接禾苗智能体网络...")

                # v4.0: Run with streaming — use threading for live updates
                import threading
                result_holder = [None]
                error_holder = [None]

                def _run_agents():
                    try:
                        result_holder[0] = ask_multi_agent(
                            question=question,
                            data=data,
                            results=results,
                            benchmark=benchmark,
                            forecast=forecast,
                            provider=_provider.lower(),
                            api_key=_api_key,
                            stream_callback=stream_cb,
                        )
                    except Exception as e:
                        error_holder[0] = e

                t0 = time.time()
                thread = threading.Thread(target=_run_agents, daemon=True)
                thread.start()

                # v4.0: Poll streaming events for live updates
                active_agents = {}
                completed_agents = set()
                stage_log = []

                while thread.is_alive():
                    if stream_cb:
                        events = stream_cb.drain()
                        for evt in events:
                            if evt.type == EventType.STAGE_START:
                                label = evt.data.get("label", "")
                                st.write(label)
                                stage_log.append(label)
                            elif evt.type == EventType.AGENT_START:
                                name = evt.data.get("name", "")
                                active_agents[name] = "⏳ 分析中..."
                                _update_agent_progress(agent_progress, active_agents, completed_agents)
                            elif evt.type == EventType.AGENT_DONE:
                                name = evt.data.get("name", "")
                                elapsed_ms = evt.data.get("elapsed_ms", 0)
                                active_agents.pop(name, None)
                                completed_agents.add(name)
                                st.write(f"✅ {name} 完成 ({elapsed_ms/1000:.1f}s)")
                                _update_agent_progress(agent_progress, active_agents, completed_agents)
                            elif evt.type == EventType.TOOL_CALL:
                                tool = evt.data.get("tool", "")
                                agent = evt.data.get("agent", "")
                                st.write(f"🔧 {agent} → {tool}")
                            elif evt.type == EventType.STAGE_END:
                                stage = evt.data.get("stage", "")
                                if stage == "complete":
                                    break

                    import time as _t
                    _t.sleep(0.05)

                thread.join(timeout=60)
                elapsed = time.time() - t0

                if error_holder[0]:
                    raise error_holder[0]

                st.write(f"✅ 完成 · {elapsed:.1f}s")

            result = result_holder[0]
            if result is None:
                st.error("Agent 返回空结果")
                return

            # ════ V8.0 POST-PROCESSING ════
            if HAS_V8 and v8_pre is not None:
                try:
                    result = v8_post_process(
                        result, question,
                        data if isinstance(data, dict) else {},
                        results or {},
                        v8_pre
                    )
                except Exception:
                    pass

            answer = result.get("answer", "抱歉，暂时无法回答。")
            agents_used = result.get("agents_used", [])
            agent_outputs = result.get("expert_outputs", {})
            thinking_list = result.get("thinking", [])
            thinking_log = "\n".join(thinking_list) if isinstance(thinking_list, list) else str(thinking_list)
            trace_id = result.get("trace_id", "")
            obs_summary = result.get("obs_summary", {})
            critique = result.get("critique", {})
            hitl_decision = result.get("hitl_decision", {})

            st.markdown(answer)
            if agents_used: _render_agent_cards(agents_used, agent_outputs)
            if thinking_log: _render_thinking_timeline(thinking_log, elapsed)

            # v4.0: Feature badges
            _render_v4_badges(result)

            # v8.0: V8 badges + gate card + review card
            if result.get("v8_enhanced"):
                _render_v8_badges(result)
                with st.expander("🔬 V8.0 Details", expanded=False):
                    _render_v8_gate_card(result)
                    _render_v8_review_card(result)
                    # Contract issues
                    issues = result.get("v8_contract_issues", {})
                    if issues:
                        for agent_name, issue_list in issues.items():
                            for iss in issue_list:
                                st.caption(f"⚠️ [{agent_name}] {iss}")

            if critique: _render_quality_badge(critique)
            if hitl_decision: _render_hitl_card(hitl_decision)
            if trace_id or obs_summary: _render_trace_bar(trace_id, obs_summary)

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({
                "role": "assistant", "content": answer,
                "agents_used": agents_used, "expert_outputs": agent_outputs,
                "thinking_log": thinking_log, "total_time": elapsed,
                "trace_id": trace_id, "obs_summary": obs_summary,
                "critique": critique, "hitl_decision": hitl_decision,
                "v4_features": {
                    "tool_use": result.get("tool_use_enabled", False),
                    "guardrails": result.get("guardrails_enabled", False),
                    "streaming": result.get("streaming_enabled", False),
                    "from_cache": result.get("from_cache", False),
                },
                "v8_gate": result.get("v8_gate"),
                "v8_review": result.get("v8_review"),
                "v8_eval": result.get("v8_eval"),
                "v8_enhanced": result.get("v8_enhanced", False),
                "v8_status": result.get("v8_status"),
            })

        except Exception as e:
            st.error(f"Agent 执行出错: {str(e)}")
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ 执行出错: {str(e)}"})
