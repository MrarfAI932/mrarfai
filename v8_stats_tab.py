"""
MRARFAI V8.0 — Stats & Observability Tab
==========================================
Streamlit tab showing V8 engine internals:
  - Gate distribution (skip/light/full)
  - Token/time savings
  - Memory graph stats
  - Eval trends
  - Playbook effectiveness
  - Skill inventory
"""

import streamlit as st
import json
from typing import Dict

# Colors
SP_GREEN = "#00FF88"
SP_BLUE = "#00A0C8"
SP_RED = "#D94040"
V8_ORANGE = "#FF6B35"
V8_PURPLE = "#A855F7"
V8_CYAN = "#06B6D4"

try:
    from v8_patch import (
        v8_get_stats, get_v8_status,
        HAS_V8_GATE, HAS_V8_CTX, HAS_V8_MEM, HAS_V8_EVO,
    )
    HAS_V8 = True
except ImportError:
    HAS_V8 = False


def render_v8_stats_tab():
    """Render V8 engine stats & observability."""
    if not HAS_V8:
        st.warning("⚠️ V8 模块未加载")
        return

    status = get_v8_status()
    stats = v8_get_stats()

    # ── Header ──
    active = sum(1 for v in status.values() if v)
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
        <div style="width:8px; height:8px; border-radius:50%; background:{SP_GREEN};
             animation:pulse 2s ease-in-out infinite;"></div>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.9rem;
              font-weight:700; color:#fff; letter-spacing:0.05em;">
            V8.0 ENGINE · {active}/4 MODULES ACTIVE
        </span>
    </div>
    <style>@keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.3;}} }}</style>
    """, unsafe_allow_html=True)

    # ── Module status ──
    modules = [
        ("⚡ Gate", "gate", HAS_V8_GATE, V8_ORANGE),
        ("🧠 Context", "context", HAS_V8_CTX, SP_BLUE),
        ("💾 Memory", "memory", HAS_V8_MEM, V8_PURPLE),
        ("🔄 Evolution", "evolution", HAS_V8_EVO, V8_CYAN),
    ]

    cols = st.columns(4)
    for i, (label, key, active, color) in enumerate(modules):
        with cols[i]:
            icon = "🟢" if active else "🔴"
            st.markdown(f"""
            <div style="background:rgba(138,138,138,0.04); border:1px solid rgba(138,138,138,0.10);
                 padding:8px; text-align:center;">
                <div style="font-size:1.2rem;">{icon}</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem;
                     color:{color}; letter-spacing:0.05em; margin-top:4px;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Gate Stats ──
    if HAS_V8_GATE and "gate" in stats:
        gate = stats["gate"]
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
             color:#888; letter-spacing:0.05em; margin-bottom:8px;">
            ⚡ ADAPTIVE GATE DISTRIBUTION
        </div>
        """, unsafe_allow_html=True)

        total = gate.get("total_queries", 0)
        if total > 0:
            gcols = st.columns(3)
            with gcols[0]:
                skip_pct = gate.get("skip_rate", "0%")
                st.metric("⚡ Skip", gate.get("skip_count", 0), skip_pct)
            with gcols[1]:
                light_pct = gate.get("light_rate", "0%")
                st.metric("🔀 Light", gate.get("light_count", 0), light_pct)
            with gcols[2]:
                full_pct = gate.get("full_rate", "0%")
                st.metric("🔥 Full", gate.get("full_count", 0), full_pct)

            # Savings
            st.markdown(f"""
            <div style="background:rgba(0,255,136,0.04); border:1px solid rgba(0,255,136,0.10);
                 padding:8px 12px; margin:8px 0;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.58rem;
                     color:{SP_GREEN}; letter-spacing:0.05em;">
                    💰 SAVINGS: {gate.get('agent_calls_saved', 0)} calls ·
                    ~{gate.get('estimated_tokens_saved', 0):,} tokens ·
                    ~{gate.get('estimated_time_saved_ms', 0)/1000:.1f}s
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("暂无查询记录")

        st.divider()

    # ── Cache Stats ──
    if HAS_V8_CTX and "cache" in stats:
        cache = stats["cache"]
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
             color:#888; letter-spacing:0.05em; margin-bottom:8px;">
            🧠 SEMANTIC CACHE
        </div>
        """, unsafe_allow_html=True)

        ccols = st.columns(3)
        with ccols[0]:
            st.metric("缓存条目", cache.get("size", 0))
        with ccols[1]:
            st.metric("命中次数", cache.get("hits", 0))
        with ccols[2]:
            st.metric("命中率", f"{cache.get('hit_rate', 0):.0%}")

        st.divider()

    # ── Memory Stats ──
    if HAS_V8_MEM and "memory" in stats:
        mem = stats["memory"]
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
             color:#888; letter-spacing:0.05em; margin-bottom:8px;">
            💾 META-MEMORY GRAPH
        </div>
        """, unsafe_allow_html=True)

        mcols = st.columns(4)
        with mcols[0]:
            st.metric("节点", mem.get("total_nodes", 0))
        with mcols[1]:
            st.metric("边", mem.get("total_edges", 0))
        with mcols[2]:
            st.metric("TELOS画像", mem.get("telos_count", 0))
        with mcols[3]:
            st.metric("实体", mem.get("unique_entities", 0))

        # Type breakdown
        by_type = mem.get("by_type", {})
        if by_type:
            st.caption(f"Episodic: {by_type.get('episodic', 0)} · "
                      f"Semantic: {by_type.get('semantic', 0)} · "
                      f"Procedural: {by_type.get('procedural', 0)}")

        st.divider()

    # ── Eval Stats ──
    if HAS_V8_EVO and "eval" in stats:
        ev = stats["eval"]
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
             color:#888; letter-spacing:0.05em; margin-bottom:8px;">
            🔄 AUTO-EVALUATION
        </div>
        """, unsafe_allow_html=True)

        ecols = st.columns(4)
        with ecols[0]:
            st.metric("评估次数", ev.get("total_evals", 0))
        with ecols[1]:
            st.metric("通过率", f"{ev.get('pass_rate', 0):.0%}")
        with ecols[2]:
            st.metric("平均分", f"{ev.get('avg_score', 0):.1f}")
        with ecols[3]:
            trend = ev.get("trend", "stable")
            trend_icons = {"improving": "📈", "stable": "➡️", "declining": "📉"}
            st.metric("趋势", f"{trend_icons.get(trend, '➡️')} {trend}")

    # ── Skills ──
    if HAS_V8_EVO and "skills" in stats:
        sk = stats["skills"]
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
             color:#888; letter-spacing:0.05em; margin:16px 0 8px;">
            🎯 SKILL INVENTORY · {sk.get('total_skills', 0)} skills
        </div>
        """, unsafe_allow_html=True)

        skill_list = sk.get("skills", [])
        for s in skill_list[:8]:
            name = s.get("name", "")
            success = s.get("success_count", 0)
            fail = s.get("fail_count", 0)
            total = success + fail
            rate = success / total if total > 0 else 0
            st.caption(f"  🎯 {name} · 成功率 {rate:.0%} ({success}/{total})")

    # ── Playbook ──
    if HAS_V8_CTX and "playbook" in stats:
        pb = stats["playbook"]
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
             color:#888; letter-spacing:0.05em; margin:16px 0 8px;">
            📋 EVOLVING PLAYBOOK
        </div>
        """, unsafe_allow_html=True)

        for qtype, strategies in pb.items():
            st.caption(f"  📋 {qtype}: {len(strategies)} strategies")


def render_v8_sidebar():
    """Render V8 status in sidebar."""
    if not HAS_V8:
        return

    status = get_v8_status()
    active = sum(1 for v in status.values() if v)

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:6px; padding:6px 10px;
         background:rgba(168,85,247,0.06); border:1px solid rgba(168,85,247,0.15);
         margin-top:4px;">
        <div style="width:5px; height:5px; border-radius:50%; background:{V8_PURPLE};
             animation:pulse 2s ease-in-out infinite;"></div>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.58rem;
              color:#6a6a6a; letter-spacing:0.05em;">V8.0 ENGINE [{active}/4 ACTIVE]</span>
    </div>
    <style>@keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.3;}} }}</style>
    """, unsafe_allow_html=True)
