"""
MRARFAI v10 — Agent Chat Tab
Colors imported from ui_theme design system.
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

# ── Colors from design system ──────────────────────────────────
from ui_theme import COLORS

SP_GREEN     = COLORS["text_primary"]
SP_BLUE      = COLORS["text_tertiary"]
SP_RED       = COLORS["status_error"]
BRAND_GREEN  = COLORS["text_secondary"]
C_TEXT_MUTED = COLORS["text_muted"]
C_TEXT_SEC   = COLORS["text_secondary"]
C_SUCCESS    = COLORS["text_primary"]
C_WARNING    = COLORS["status_warning"]

# ── Agent Registry — mapped to Sprocomm 三叶 ───────────────────
AGENTS = {
    "📊 数据分析师": {
        "name": "数据分析师",
        "role": "DATA ANALYST",
        "color": SP_GREEN,
        "bg": "rgba(255,255,255,0.08)",
        "border": "rgba(255,255,255,0.20)",
        "icon": "📊",
    },
    "🛡️ 风控专家": {
        "name": "风控专家",
        "role": "RISK CONTROL",
        "color": SP_RED,
        "bg": "rgba(102,102,102,0.08)",
        "border": "rgba(102,102,102,0.20)",
        "icon": "🛡️",
    },
    "💡 策略师": {
        "name": "策略师",
        "role": "STRATEGIST",
        "color": SP_BLUE,
        "bg": "rgba(136,136,136,0.08)",
        "border": "rgba(136,136,136,0.20)",
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
        "bg": "rgba(255,255,255,0.06)",
        "border": "rgba(255,255,255,0.18)",
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
#  ANSWER PARSER — 文本墙 → 卡片化 HTML
# ════════════════════════════════════════════════════════════════

# Section header detection patterns
_SEC_PATTERNS = [
    # (regex, type, color)
    (re.compile(r'(?:核心结论|总结|概要|要点|摘要|总览)'), 'summary', SP_GREEN),
    (re.compile(r'(?:增长|驱动|亮点|机会|正面|优势|突破)'), 'growth', SP_GREEN),
    (re.compile(r'(?:风险|预警|流失|下滑|问题|挑战|威胁|隐患)'), 'risk', SP_RED),
    (re.compile(r'(?:行动|建议|下一步|策略|方案|措施|计划|应对)'), 'action', SP_BLUE),
    (re.compile(r'(?:分析|详情|对比|说明|背景|补充|结构|趋势)'), 'analysis', C_TEXT_MUTED),
]

# Regex for numbers worth highlighting: 54.1%, 41.71亿, 3.42亿元, 110.4%, -42.4%, +12家 etc.
_METRIC_RE = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*([%％万亿美元元家个月次台件倍])')

# Header line pattern: lines that look like section titles
_HEADER_RE = re.compile(
    r'^\s*(?:[#*]*\s*)?'                             # optional markdown ## or **
    r'(?:[\d一二三四五六七八九十]+[\.、）)]\s*)?'     # optional numbering
    r'[【\[]?\s*'                                     # optional brackets
    r'(.{2,15}?)'                                     # title text (2-15 chars)
    r'\s*[】\]]?\s*$'                                 # optional closing bracket
)


def _classify_header(title: str):
    """Classify a header title text into a section type."""
    for pat, sec_type, color in _SEC_PATTERNS:
        if pat.search(title):
            return sec_type, color
    return 'analysis', C_TEXT_MUTED


def _highlight_metrics(text: str, section_type: str) -> str:
    """Wrap numbers+units in colored metric chips."""
    def _replace(m):
        num_str = m.group(1)
        unit = m.group(2)
        # Determine polarity
        if section_type == 'risk' or num_str.startswith('-'):
            cls = 'ai-metric-negative'
        elif section_type in ('growth', 'summary') or num_str.startswith('+'):
            cls = 'ai-metric-positive'
        else:
            cls = 'ai-metric-neutral'
        return f'<span class="ai-metric-chip {cls}">{num_str}{unit}</span>'
    return _METRIC_RE.sub(_replace, text)


def _format_action_items(body: str) -> str:
    """Convert numbered list text into styled action items."""
    lines = [l.strip() for l in body.split('\n') if l.strip()]
    # Detect numbered items: 1. xxx  or  ① xxx  or  一、xxx
    num_re = re.compile(r'^(?:(\d+)[\.、）)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*(.*)')
    items_html = []
    counter = 1
    for line in lines:
        m = num_re.match(line)
        if m:
            text = m.group(2) if m.group(2) else line
            text = _highlight_metrics(text, 'action')
            items_html.append(
                f'<div class="ai-action-item">'
                f'<div class="ai-action-num">{counter}</div>'
                f'<div class="ai-action-text">{text}</div></div>'
            )
            counter += 1
        else:
            # Non-numbered line in action section — still render with highlight
            text = _highlight_metrics(line, 'action')
            items_html.append(
                f'<div class="ai-action-item">'
                f'<div class="ai-action-num">→</div>'
                f'<div class="ai-action-text">{text}</div></div>'
            )
    return '\n'.join(items_html)


def _split_into_sections(text: str):
    """Split answer text into typed sections by detecting header lines."""
    lines = text.split('\n')
    sections = []
    current = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                current['body'] += '\n'
            continue

        # Check if this line is a header (short, matches patterns)
        # Remove markdown bold markers for detection
        clean = re.sub(r'\*\*', '', stripped)
        is_header = False

        if len(clean) <= 20:
            sec_type, color = _classify_header(clean)
            # If it matched a known keyword, treat as header
            for pat, _, _ in _SEC_PATTERNS:
                if pat.search(clean):
                    is_header = True
                    break

        if is_header:
            if current:
                sections.append(current)
            current = {'type': sec_type, 'title': clean, 'color': color, 'body': ''}
        else:
            if current is None:
                # First lines before any header → treat as summary
                current = {'type': 'summary', 'title': '核心结论', 'color': SP_GREEN, 'body': ''}
            current['body'] += stripped + '\n'

    if current:
        sections.append(current)

    return sections


def _format_answer_html(answer: str) -> str:
    """Parse plain-text answer into structured card-based HTML."""
    if not answer:
        return ''

    # Very short answers — no parsing needed
    if len(answer.strip()) < 50:
        return (f'<div class="ai-response-container">'
                f'<div class="ai-summary"><div class="ai-summary-text">'
                f'{answer}</div></div></div>')

    # Split into sections
    sections = _split_into_sections(answer)

    # Fallback: if only 1 section or none, create summary + body
    if len(sections) <= 1:
        text = answer.strip()
        # Find first sentence break
        for sep in ['。', '，', '. ', ', ']:
            idx = text.find(sep, 30)
            if idx > 0 and idx < len(text) - 20:
                summary_text = text[:idx + len(sep)]
                body_text = text[idx + len(sep):]
                sections = [
                    {'type': 'summary', 'title': '核心结论', 'color': SP_GREEN, 'body': summary_text},
                    {'type': 'analysis', 'title': '详细分析', 'color': C_TEXT_MUTED, 'body': body_text},
                ]
                break
        else:
            sections = [{'type': 'summary', 'title': '核心结论', 'color': SP_GREEN, 'body': text}]

    # Render sections
    parts = ['<div class="ai-response-container">']
    for sec in sections:
        body = sec['body'].strip()
        if not body:
            continue

        if sec['type'] == 'summary':
            body_html = _highlight_metrics(body.replace('\n', '<br>'), 'summary')
            parts.append(
                f'<div class="ai-summary">'
                f'<div class="ai-summary-label">{sec["title"]}</div>'
                f'<div class="ai-summary-text">{body_html}</div>'
                f'</div>'
            )
        elif sec['type'] == 'action':
            items_html = _format_action_items(body)
            parts.append(
                f'<div class="ai-section ai-section-action">'
                f'<div class="ai-section-header">'
                f'<span class="ai-section-icon">▸</span> {sec["title"]}</div>'
                f'<div class="ai-section-body">{items_html}</div>'
                f'</div>'
            )
        else:
            body_html = _highlight_metrics(body.replace('\n', '<br>'), sec['type'])
            parts.append(
                f'<div class="ai-section ai-section-{sec["type"]}">'
                f'<div class="ai-section-header">'
                f'<span class="ai-section-icon">▸</span> {sec["title"]}</div>'
                f'<div class="ai-section-body">{body_html}</div>'
                f'</div>'
            )

    parts.append('</div>')
    return '\n'.join(parts)


def _inline_meta_html(critique: dict = None, hitl_decision: dict = None) -> str:
    """Compact inline HTML for quality badge + HITL confidence."""
    parts = ['<div class="ai-inline-meta">']

    if critique:
        score = critique.get("overall_score", 0)
        passed = critique.get("pass_threshold", score >= 7.0)
        bg = "rgba(255,255,255,0.08)" if passed else "rgba(204,204,204,0.08)"
        color = SP_GREEN if passed else C_WARNING
        border = "rgba(255,255,255,0.25)" if passed else "rgba(204,204,204,0.25)"
        icon = "✓" if passed else "!"
        label = "PASS" if passed else "REVIEW"
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;'
            f'font-family:Fira Sans,sans-serif;font-size:12px;font-weight:700;'
            f'background:{bg};color:{color};border:1px solid {border};">'
            f'{icon} {label} {score:.1f}/10</span>'
        )

    if hitl_decision:
        score = hitl_decision.get("confidence_score", 0)
        level = hitl_decision.get("confidence_level", "medium")
        pct = int(score * 100)
        level_map = {
            "high": (SP_GREEN, "HIGH CONFIDENCE"),
            "medium": (C_WARNING, "MEDIUM"),
            "low": (SP_RED, "LOW CONFIDENCE"),
        }
        color, label = level_map.get(level, (C_WARNING, level.upper()))
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;'
            f'font-family:Fira Sans,sans-serif;font-size:12px;font-weight:700;'
            f'background:rgba({r},{g},{b},0.08);color:{color};'
            f'border:1px solid rgba({r},{g},{b},0.25);">'
            f'{pct} {label}</span>'
        )

    parts.append('</div>')
    return ''.join(parts)


def _render_expert_mini_cards(agent_outputs: dict):
    """Render each expert's individual analysis in a styled mini-card."""
    if not agent_outputs:
        return
    for expert_name, output in agent_outputs.items():
        a = AGENTS.get(expert_name, {"icon": "🤖", "color": C_TEXT_MUTED, "role": "AGENT", "name": expert_name})
        display_text = str(output)[:500] + ("..." if len(str(output)) > 500 else "")
        display_text = display_text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        st.markdown(f'''
        <div class="ai-expert-card" style="border-left-color:{a['color']}">
            <div class="ai-expert-header">
                <span class="ai-expert-icon">{a.get('icon', '🤖')}</span>
                <span class="ai-expert-name" style="color:{a['color']}">{a.get('name', expert_name)}</span>
                <span class="ai-expert-role">{a.get('role', 'AGENT')}</span>
            </div>
            <div class="ai-expert-body">{display_text}</div>
        </div>''', unsafe_allow_html=True)


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
            f'<span style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.25);'
            f'padding:0.15rem 0.5rem;font-size:13px;color:{SP_GREEN};'
            f'font-family:Fira Sans,sans-serif;font-weight:600;letter-spacing:0.03em;">'
            f'✓ {name}</span>'
        )
    for name, status in active.items():
        a = AGENTS.get(name, {"icon": "⏳", "color": C_TEXT_MUTED})
        html += (
            f'<span style="background:rgba(138,138,138,0.06);border:1px solid rgba(138,138,138,0.20);'
            f'padding:0.15rem 0.5rem;font-size:13px;color:{C_TEXT_MUTED};'
            f'font-family:Fira Sans,sans-serif;font-weight:600;letter-spacing:0.03em;">'
            f'⏳ {name}</span>'
        )
    html += '</div>'
    container.markdown(html, unsafe_allow_html=True)


def _render_v4_badges(result: dict):
    """v4.0 feature status badges — command center style."""
    badges = []
    if result.get("tool_use_enabled"):
        badges.append(f'<span style="background:rgba(255,255,255,0.06);color:{SP_GREEN};'
                      f'border:1px solid rgba(255,255,255,0.20);'
                      f'padding:0.1rem 0.4rem;font-size:11px;font-weight:700;'
                      f'letter-spacing:0.05em;font-family:Fira Sans,sans-serif;">TOOL USE</span>')
    if result.get("guardrails_enabled"):
        badges.append(f'<span style="background:rgba(136,136,136,0.06);color:{SP_BLUE};'
                      f'border:1px solid rgba(136,136,136,0.20);'
                      f'padding:0.1rem 0.4rem;font-size:11px;font-weight:700;'
                      f'letter-spacing:0.05em;font-family:Fira Sans,sans-serif;">GUARDRAILS</span>')
    if result.get("streaming_enabled"):
        badges.append(f'<span style="background:rgba(138,138,138,0.06);color:#8a8a8a;'
                      f'border:1px solid rgba(138,138,138,0.20);'
                      f'padding:0.1rem 0.4rem;font-size:11px;font-weight:700;'
                      f'letter-spacing:0.05em;font-family:Fira Sans,sans-serif;">STREAMING</span>')
    if result.get("from_cache"):
        badges.append(f'<span style="background:rgba(204,204,204,0.06);color:{C_WARNING};'
                      f'border:1px solid rgba(204,204,204,0.20);'
                      f'padding:0.1rem 0.4rem;font-size:11px;font-weight:700;'
                      f'letter-spacing:0.05em;font-family:Fira Sans,sans-serif;">CACHED</span>')

    budget = result.get("budget_status")
    if budget and budget.get("level") != "normal":
        level_color = {
            "caution": C_WARNING,
            "warning": SP_RED,
            "critical": SP_RED,
        }.get(budget["level"], C_TEXT_MUTED)
        badges.append(
            f'<span style="background:rgba(102,102,102,0.06);color:{level_color};'
            f'border:1px solid rgba(102,102,102,0.20);'
            f'padding:0.1rem 0.4rem;font-size:11px;font-weight:700;'
            f'letter-spacing:0.05em;font-family:Fira Sans,sans-serif;">'
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

V8_ORANGE = "#AAAAAA"
V8_PURPLE = "#888888"
V8_CYAN   = "#CCCCCC"

def _badge_html(label: str, color: str) -> str:
    """Generate a V8 badge span."""
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return (
        f'<span style="background:rgba({r},{g},{b},0.06);color:{color};'
        f'border:1px solid rgba({r},{g},{b},0.20);'
        f'padding:0.1rem 0.4rem;font-size:11px;font-weight:700;'
        f"letter-spacing:0.05em;font-family:Fira Sans,sans-serif;"
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

    # ── V9.0 COLORS ──
V9_TEAL = "#CCCCCC"
V9_INDIGO = "#AAAAAA"
V9_ROSE = "#A1A1AA"

def _render_v9_badges(result: dict):
    """V9.0 论文模块活动指标 — 实际被触发的模块."""
    v9_mods = result.get("v9_modules", {})
    v9_act = result.get("v9_activity", {})
    if not v9_mods:
        return

    badges = []

    # RLM
    if v9_act.get("rlm_used"):
        badges.append(_badge_html("🔄 RLM ACTIVE", V9_TEAL))
    elif v9_mods.get("rlm"):
        badges.append(_badge_html("RLM READY", C_TEXT_MUTED))

    # Reasoning Templates
    if v9_act.get("reasoning_templates_injected"):
        badges.append(_badge_html("🧠 REASONING V9", V9_INDIGO))

    # Memory 3D
    if v9_act.get("memory_3d_retrieved"):
        badges.append(_badge_html("💾 MEM3D HIT", V9_TEAL))
    elif v9_act.get("memory_3d_saved"):
        badges.append(_badge_html("💾 MEM3D SAVED", V9_INDIGO))

    # Interpretability
    trace_steps = v9_act.get("trace_steps", 0)
    if trace_steps > 0:
        badges.append(_badge_html(f"🔍 TRACE {trace_steps}步", V9_ROSE))
    elif v9_act.get("interpretability_traced"):
        badges.append(_badge_html("🔍 TRACED", C_TEXT_MUTED))

    # Module count
    active = sum(1 for v in v9_mods.values() if v)
    badges.append(_badge_html(f"V9 {active}/7", V9_TEAL))

    if badges:
        st.markdown(
            f'<div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin:0.3rem 0 0.1rem;">{"".join(badges)}</div>',
            unsafe_allow_html=True
        )


def _render_v9_details(result: dict):
    """V9.0 详细面板 — expander 内容."""
    v9_mods = result.get("v9_modules", {})
    v9_act = result.get("v9_activity", {})
    if not v9_mods:
        return

    rows = [
        ("① RLM递归引擎", "rlm", v9_act.get("rlm_used", False), "数据递归压缩处理"),
        ("② AWM合成环境", "awm", False, "训练/测试时使用"),
        ("③ EnCompass搜索", "search_engine", False, "多路径分支搜索"),
        ("④ 推理模板", "reasoning_templates", v9_act.get("reasoning_templates_injected", False), "结构化CoT注入"),
        ("⑤ 三维记忆", "memory_3d", v9_act.get("memory_3d_saved", False), f"{'检索命中' if v9_act.get('memory_3d_retrieved') else '已保存本次'}"),
        ("⑥ 可解释性", "interpretability", v9_act.get("interpretability_traced", False), f"{v9_act.get('trace_steps', 0)}步追踪"),
        ("⑦ 评估框架", "evals_v9", False, "批量评估时使用"),
    ]

    for label, mod_key, used, desc in rows:
        installed = v9_mods.get(mod_key, False)
        if used:
            icon = "🟢"
            status = "ACTIVE"
            color = V9_TEAL
        elif installed:
            icon = "🟡"
            status = "READY"
            color = C_TEXT_MUTED
        else:
            icon = "🔴"
            status = "OFF"
            color = SP_RED

        st.caption(f"{icon} {label} · {status} · {desc}")
    gate = result.get("v8_gate", {})
    if not gate:
        return

    level = gate.get("level", "full")
    score = gate.get("score", 0)
    agents = gate.get("agents", [])
    reason = gate.get("reason", "")

    level_colors = {"skip": "rgba(255,255,255,0.06)", "light": "rgba(170,170,170,0.06)", "full": "rgba(136,136,136,0.06)"}
    level_borders = {"skip": "rgba(255,255,255,0.15)", "light": "rgba(170,170,170,0.15)", "full": "rgba(136,136,136,0.15)"}
    level_icons = {"skip": "⚡", "light": "🔀", "full": "🔥"}

    agents_str = " → ".join(agents) if agents else "SQL直查"

    st.markdown(f"""
    <div style="background:{level_colors.get(level, 'rgba(138,138,138,0.06)')};
         border:1px solid {level_borders.get(level, 'rgba(138,138,138,0.15)')};
         padding:8px 12px; margin:4px 0;">
        <div style="font-family:Fira Sans,sans-serif; font-size:12px;
             color:#888; letter-spacing:0.05em; margin-bottom:4px;">
            {level_icons.get(level, '🔥')} V8 GATE · {level.upper()} · score={score:.2f}
        </div>
        <div style="font-size:13px; color:#ccc;">
            {agents_str}
        </div>
        <div style="font-size:11px; color:#555; margin-top:2px;">
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
        checks_html += f'<span style="font-size:11px;color:#888;margin-right:8px;">{icon} {check_name}: {c_score:.1f}</span>'

    blockers_html = ""
    if blockers:
        for b in blockers:
            blockers_html += f'<div style="font-size:11px;color:{SP_RED};margin-top:2px;">🚫 {b}</div>'

    st.markdown(f"""
    <div style="background:rgba(138,138,138,0.04); border:1px solid rgba(138,138,138,0.10);
         padding:8px 12px; margin:4px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-family:Fira Sans,sans-serif; font-size:12px;
                 color:#888; letter-spacing:0.05em;">
                📝 V8 REVIEW · {"PASS" if passed else "FAIL"}
            </span>
            <span style="font-size:13px; font-weight:700; color:{bar_color};">{score:.1f}/10</span>
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
        status = ('<span style="font-family:Fira Sans,sans-serif;font-size:12px;padding:0.2rem 0.6rem;'
                  'font-weight:700;flex-shrink:0;margin-left:auto;letter-spacing:0.05em;'
                  'background:rgba(255,255,255,0.08);color:#FFFFFF;border:1px solid rgba(255,255,255,0.25);">DONE</span>')

        cards_html += f"""
        <div class="agent-card" style="border-left:2px solid {a['color']};">
            <div class="agent-avatar" style="background:{a['bg']};border:1px solid {a['border']};">{a['icon']}</div>
            <div style="flex:1;min-width:0;">
                <div class="agent-name">{a['name']}</div>
                <div class="agent-role">{a['role']}</div>
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
            st.markdown(f"<span style='color:{color};font-size:8px;'>●</span> <span style='font-family:Fira Sans,sans-serif;font-size:14px;color:#aaa;line-height:1.8;'>{line}</span>{time_str}", unsafe_allow_html=True)


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

    bg = "rgba(255,255,255,0.08)" if passed else "rgba(204,204,204,0.08)"
    color = SP_GREEN if passed else C_WARNING
    border = "rgba(255,255,255,0.25)" if passed else "rgba(204,204,204,0.25)"
    icon = "✓" if passed else "!"
    label = "PASS" if passed else "REVIEW"

    imp_html = f'<span style="color:{SP_GREEN};margin-left:0.3rem;">↑{imp:.1f}</span>' if imp and imp > 0 else ""
    iter_html = f'<span style="color:{C_TEXT_MUTED};margin-left:0.3rem;">×{iters}</span>' if iters and iters > 1 else ""
    dim_html = f'<span style="font-size:11px;color:{C_TEXT_MUTED};margin-left:0.4rem;">{dim_text}</span>' if dim_text else ""

    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.35rem 0.7rem;
         font-family:Fira Sans,sans-serif;font-size:13px;font-weight:700;letter-spacing:0.05em;
         margin:0.25rem 0;background:{bg};color:{color};border:1px solid {border};">
        <span style="font-size:14px;">{icon}</span>
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
        trig_html = f'<div style="font-family:Fira Sans,sans-serif;font-size:12px;color:#888;text-align:right;flex-shrink:0;">{"<br>".join(items)}</div>'

    gauge_bg = f"rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.10)"
    gauge_border = f"rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.35)"

    st.markdown(f"""
    <div style="background:#080808;border:1px solid #2f2f2f;padding:1rem 1.2rem;margin:0.8rem 0;display:flex;align-items:center;gap:1rem;">
        <div style="width:48px;height:48px;display:flex;align-items:center;justify-content:center;font-family:Fira Sans,sans-serif;font-size:15px;font-weight:700;flex-shrink:0;background:{gauge_bg};color:{color};border:2px solid {gauge_border};">{pct}</div>
        <div style="flex:1;">
            <div style="font-family:Fira Sans,sans-serif;font-size:14px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{color};">{label}</div>
            <div style="font-family:Fira Sans,sans-serif;font-size:14px;color:#aaa;margin-top:0.2rem;">{act_text}</div>
        </div>
        {trig_html}
    </div>""", unsafe_allow_html=True)


def _render_trace_bar(trace_id: str = "", obs_summary: dict = None):
    """Compact trace info bar — adapted to multi_agent.py obs_summary format."""
    items = []
    if trace_id:
        items.append(f'<span style="margin-right:0.5rem;">TRACE <span style="color:#FFFFFF;font-weight:700;">{trace_id[:8]}</span></span>')
    if obs_summary:
        # Latency
        lb = obs_summary.get("latency_breakdown", {})
        total_ms = lb.get("total_ms", 0)
        if total_ms:
            items.append(f'<span style="margin-right:0.5rem;">LATENCY <span style="color:#FFFFFF;font-weight:700;">{total_ms/1000:.1f}s</span></span>')
        # Tokens
        tokens = obs_summary.get("total_tokens", 0)
        if tokens:
            items.append(f'<span style="margin-right:0.5rem;">TOKENS <span style="color:#FFFFFF;font-weight:700;">{tokens:,}</span></span>')
        # Cost
        cost = obs_summary.get("total_cost_usd", 0)
        if cost:
            items.append(f'<span style="margin-right:0.5rem;">COST <span style="color:#FFFFFF;font-weight:700;">${cost:.4f}</span></span>')
        # LLM calls
        calls = obs_summary.get("total_llm_calls", 0)
        if calls:
            items.append(f'<span style="margin-right:0.5rem;">CALLS <span style="color:#FFFFFF;font-weight:700;">{calls}</span></span>')

    if items:
        st.markdown('<div class="trace-bar">' + " ".join(items) + '</div>', unsafe_allow_html=True)


def _render_suggestion_chips():
    """Quick-start suggestion buttons."""
    cols = st.columns(len(SUGGESTIONS))
    for i, s in enumerate(SUGGESTIONS):
        with cols[i]:
            if st.button(s, key=f"suggest_{i}", use_container_width=True):
                st.session_state["pending_question"] = s
                st.rerun()


def _render_welcome():
    """Welcome state — compact prompt, no hero wall."""
    st.markdown(f"""
    <div style="text-align:center; padding:24px 0 16px 0;">
        <div style="display:inline-flex; align-items:center; gap:8px; padding:6px 14px;
             background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.12);
             margin-bottom:12px;">
            <div style="width:5px;height:5px;border-radius:50%;background:#FFFFFF;
                 animation:neon-pulse 2s ease-in-out infinite;"></div>
            <span style="font-family:Fira Sans,sans-serif;font-size:11px;
                  font-weight:700;color:#FFFFFF;letter-spacing:0.1em;">AGENT TERMINAL · V10.0</span>
        </div>
        <div style="font-family:Fira Sans,sans-serif;font-size:1.2rem;font-weight:700;
             color:#FFFFFF;letter-spacing:-0.01em;margin-bottom:6px;">
            多智能体销售分析
        </div>
        <div style="font-family:Fira Sans,sans-serif;font-size:12px;
             color:#888;letter-spacing:0.02em;">
            // 12维深度分析 · 实时预警 · RLM 递归语言模型
        </div>
        <div style="display:flex;justify-content:center;gap:18px;margin-top:14px;">
            <div style="display:flex;align-items:center;gap:5px;">
                <span style="color:{SP_GREEN};font-size:12px;">◈</span>
                <span style="font-family:Fira Sans,sans-serif;font-size:10px;
                      color:#4a4a4a;letter-spacing:0.08em;">ANALYST</span>
            </div>
            <div style="display:flex;align-items:center;gap:5px;">
                <span style="color:{SP_RED};font-size:12px;">◆</span>
                <span style="font-family:Fira Sans,sans-serif;font-size:10px;
                      color:#4a4a4a;letter-spacing:0.08em;">RISK</span>
            </div>
            <div style="display:flex;align-items:center;gap:5px;">
                <span style="color:{SP_BLUE};font-size:12px;">◇</span>
                <span style="font-family:Fira Sans,sans-serif;font-size:10px;
                      color:#4a4a4a;letter-spacing:0.08em;">STRATEGY</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_ready_state():
    """Compact ready state — data loaded, agents standby, no welcome wall."""
    st.markdown(f"""
    <div style="text-align:center; padding:40px 0 20px 0;">
        <div style="display:inline-flex; align-items:center; gap:8px; padding:8px 18px;
             background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.15);
             margin-bottom:20px;">
            <div style="width:6px;height:6px;border-radius:50%;background:#FFFFFF;
                 animation:neon-pulse 2s ease-in-out infinite;"></div>
            <span style="font-family:Fira Sans,sans-serif;font-size:12px;
                  font-weight:700;color:#FFFFFF;letter-spacing:0.1em;text-transform:uppercase;">
                AGENTS READY · V10.0
            </span>
        </div>
        <div style="font-family:Fira Sans,sans-serif;font-size:1.4rem;font-weight:700;
             color:#FFFFFF;letter-spacing:-0.02em;margin-bottom:6px;">
            多智能体分析就绪
        </div>
        <div style="font-family:Fira Sans,sans-serif;font-size:13px;
             color:#888;letter-spacing:0.03em;">
            // 在下方输入问题，或点击快捷提问开始分析
        </div>
    </div>
    <div style="display:flex;justify-content:center;gap:24px;margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="color:{SP_GREEN};font-size:13px;">◈</span>
            <span style="font-family:Fira Sans,sans-serif;font-size:11px;
                  color:#888;letter-spacing:0.05em;">DATA ANALYST</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="color:{SP_RED};font-size:13px;">◆</span>
            <span style="font-family:Fira Sans,sans-serif;font-size:11px;
                  color:#888;letter-spacing:0.05em;">RISK CONTROL</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="color:{SP_BLUE};font-size:13px;">◇</span>
            <span style="font-family:Fira Sans,sans-serif;font-size:11px;
                  color:#888;letter-spacing:0.05em;">STRATEGIST</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
        st.info("请先在上方上传数据文件")
        return

    if not use_multi:
        _render_welcome()
        st.warning("请在上方打开 Multi-Agent 开关")
        return

    # ── Chat history ────────────────────────────────────────
    # If a new question is pending, skip old history rendering
    has_pending = "pending_question" in st.session_state
    if not st.session_state.chat_history:
        _render_ready_state()
        _render_suggestion_chips()
    elif not has_pending:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🌿"):
                    # ── 1. Formatted answer ──
                    formatted = _format_answer_html(msg["content"])
                    st.markdown(formatted, unsafe_allow_html=True)

                    # ── 2. Inline meta ──
                    if msg.get("critique") or msg.get("hitl_decision"):
                        st.markdown(
                            _inline_meta_html(msg.get("critique"), msg.get("hitl_decision")),
                            unsafe_allow_html=True
                        )

                    # ── 3. Agent details — collapsed ──
                    with st.expander("🔬 Agent Details", expanded=False):
                        if msg.get("agents_used"):
                            _render_agent_cards(msg["agents_used"], msg.get("expert_outputs"))
                        if msg.get("expert_outputs"):
                            _render_expert_mini_cards(msg.get("expert_outputs"))
                        v4f = msg.get("v4_features", {})
                        if v4f:
                            _render_v4_badges({
                                "tool_use_enabled": v4f.get("tool_use"),
                                "guardrails_enabled": v4f.get("guardrails"),
                                "streaming_enabled": v4f.get("streaming"),
                                "from_cache": v4f.get("from_cache"),
                            })
                        if msg.get("v8_enhanced"):
                            _render_v8_badges(msg)
                        if msg.get("v9_modules"):
                            _render_v9_badges(msg)
                        if msg.get("thinking_log"):
                            _render_thinking_timeline(msg["thinking_log"], msg.get("total_time", 0))

                    # ── 4. Trace bar ──
                    if msg.get("trace_id") or msg.get("obs_summary"):
                        _render_trace_bar(msg.get("trace_id", ""), msg.get("obs_summary"))

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
                st.warning("⚠️ 请先在上方配置 API Key")
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ 请先在上方 AI ENGINE 中配置 API Key"})
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

            # ── 1. FORMATTED ANSWER — the hero ──
            formatted_html = _format_answer_html(answer)
            st.markdown(formatted_html, unsafe_allow_html=True)

            # ── 2. INLINE META — quality + confidence compact ──
            if critique or hitl_decision:
                st.markdown(_inline_meta_html(critique, hitl_decision), unsafe_allow_html=True)

            # ── 3. AGENT DETAILS — all collapsed in one expander ──
            with st.expander("🔬 Agent Details", expanded=False):
                if agents_used:
                    _render_agent_cards(agents_used, agent_outputs)
                if agent_outputs:
                    _render_expert_mini_cards(agent_outputs)
                _render_v4_badges(result)
                if result.get("v8_enhanced"):
                    _render_v8_badges(result)
                    _render_v8_gate_card(result)
                    _render_v8_review_card(result)
                    issues = result.get("v8_contract_issues", {})
                    if issues:
                        for agent_name, issue_list in issues.items():
                            for iss in issue_list:
                                st.caption(f"⚠️ [{agent_name}] {iss}")
                if result.get("v9_modules"):
                    _render_v9_badges(result)
                    _render_v9_details(result)
                if thinking_log:
                    _render_thinking_timeline(thinking_log, elapsed)

            # ── 4. TRACE BAR — subtle bottom ──
            if trace_id or obs_summary:
                _render_trace_bar(trace_id, obs_summary)

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
                "v9_modules": result.get("v9_modules"),
                "v9_activity": result.get("v9_activity"),
            })

        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__

            # 分类友好错误信息
            if "api_key" in err_str.lower() or "authentication" in err_str.lower() or "401" in err_str:
                friendly = "🔑 API Key 无效或已过期，请在上方重新输入"
            elif "rate_limit" in err_str.lower() or "429" in err_str:
                friendly = "⏳ API 请求过于频繁，请稍等30秒后重试"
            elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
                friendly = "⏱️ 请求超时，请稍后重试（可能是网络问题）"
            elif "connection" in err_str.lower() or "network" in err_str.lower():
                friendly = "🌐 网络连接失败，请检查网络后重试"
            elif "insufficient_quota" in err_str.lower() or "402" in err_str:
                friendly = "💳 API 额度不足，请充值后重试"
            elif "model" in err_str.lower() and "not found" in err_str.lower():
                friendly = "🤖 模型不可用，请切换其他模型重试"
            else:
                friendly = f"⚠️ 分析过程中出现错误，请重试"

            st.error(friendly)
            with st.expander("🔧 技术详情", expanded=False):
                st.code(f"Error: {err_type}\n{err_str[:500]}", language="text")

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": friendly})
