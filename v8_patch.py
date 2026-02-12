#!/usr/bin/env python3
"""
MRARFAI V8.0 — multi_agent.py 补丁
=====================================
运行方法: python v8_patch.py
效果: 将 V8 模块注入到 multi_agent.py 中

注入点:
  1. 文件顶部: 添加 V8 import
  2. ask_multi_agent(): 在关键阶段插入 V8 处理
  3. 结果字典: 添加 V8 字段
"""

# ============================================================
# 这个文件不是补丁脚本 — 它是 multi_agent.py 的 V8 增强层
# 通过 monkey-patch 方式注入，不修改原始文件
# ============================================================

import sys
import os
import json
import time
from typing import Dict, List, Optional, Any

# 确保 V8 模块在路径上
V8_DIR = os.path.dirname(os.path.abspath(__file__))
if V8_DIR not in sys.path:
    sys.path.insert(0, V8_DIR)

# V8 模块导入
try:
    from adaptive_gate import (
        get_gate, get_factory, get_validator,
        ComplexityLevel,
    )
    HAS_V8_GATE = True
except ImportError:
    HAS_V8_GATE = False

try:
    from context_engine import (
        get_context_cache, get_playbook,
        build_working_context, compress_agent_output,
        ContextCompressor,
    )
    HAS_V8_CTX = True
except ImportError:
    HAS_V8_CTX = False

try:
    from meta_memory import (
        get_memory_graph, MemoryType,
    )
    HAS_V8_MEM = True
except ImportError:
    HAS_V8_MEM = False

try:
    from self_evolution import (
        get_reviewer, get_eval_loop, get_distiller,
    )
    HAS_V8_EVO = True
except ImportError:
    HAS_V8_EVO = False


def get_v8_status() -> Dict[str, bool]:
    """V8 模块状态"""
    return {
        "gate": HAS_V8_GATE,
        "context": HAS_V8_CTX,
        "memory": HAS_V8_MEM,
        "evolution": HAS_V8_EVO,
    }


# ============================================================
# V8 Pre-processing: 在 ask_multi_agent 之前
# ============================================================

def v8_pre_process(question: str, data: dict, results: dict,
                   provider: str = "deepseek") -> Dict[str, Any]:
    """
    V8 前处理 — 在 ask_multi_agent 之前调用

    Returns:
        {
            "gate_level": "skip|light|full",
            "gate_score": float,
            "gate_agents": [...],
            "gate_reason": str,
            "v8_memory_ctx": str,
            "v8_skills": [...],
            "v8_thinking": [...],
        }
    """
    ctx = {
        "gate_level": "full",
        "gate_score": 0.0,
        "gate_agents": [],
        "gate_reason": "",
        "v8_memory_ctx": "",
        "v8_skills": [],
        "v8_thinking": [],
    }

    # Phase I: 门控评估
    if HAS_V8_GATE:
        gate = get_gate()
        assessment = gate.route(question)
        ctx["gate_level"] = assessment.level.value
        ctx["gate_score"] = assessment.score
        ctx["gate_agents"] = assessment.agents_recommended
        ctx["gate_reason"] = assessment.reason
        ctx["v8_thinking"].append(
            f"🔀 V8门控: {assessment.level.value} "
            f"(score={assessment.score:.2f}) {assessment.reason}"
        )

    # Phase III: 元记忆加载
    if HAS_V8_MEM:
        try:
            graph = get_memory_graph()
            # 简单实体提取
            entities = _extract_entities(question, data)
            mem_ctx = graph.build_memory_context(question, entities)
            if mem_ctx:
                ctx["v8_memory_ctx"] = mem_ctx
                ctx["v8_thinking"].append(f"🧠 V8记忆: 加载 {len(mem_ctx)} 字上下文")
        except Exception:
            pass

    # Phase IV: 技能匹配
    if HAS_V8_EVO:
        try:
            distiller = get_distiller()
            skills = distiller.match_skills(question)
            ctx["v8_skills"] = [
                {"name": s.name, "strategy": s.strategy}
                for s in skills
            ]
            if skills:
                ctx["v8_thinking"].append(
                    f"🎯 V8技能: {', '.join(s.name for s in skills)}"
                )
        except Exception:
            pass

    return ctx


# ============================================================
# V8 Post-processing: 在 ask_multi_agent 之后
# ============================================================

def v8_post_process(result: dict, question: str,
                    data: dict, results: dict,
                    v8_pre: dict = None) -> dict:
    """
    V8 后处理 — 在 ask_multi_agent 之后调用

    增强 result 字典，添加 V8 字段。
    """
    v8_pre = v8_pre or {}
    answer = result.get("answer", "")
    expert_outputs = result.get("expert_outputs", {})
    agents_used = result.get("agents_used", [])
    v8_thinking = list(v8_pre.get("v8_thinking", []))

    # Phase I: 合约验证
    contract_issues = {}
    if HAS_V8_GATE:
        validator = get_validator()
        for agent_name, output in expert_outputs.items():
            clean_id = _agent_name_to_id(agent_name)
            if clean_id:
                is_valid, issues = validator.validate(clean_id, output)
                if not is_valid:
                    contract_issues[agent_name] = issues
                    v8_thinking.append(f"⚠️ 合约违反 [{clean_id}]: {'; '.join(issues)}")

    # Phase IV: 结构化审查
    v8_review = None
    if HAS_V8_EVO:
        try:
            reviewer = get_reviewer()
            context_str = json.dumps(results, ensure_ascii=False, default=str)[:3000]
            review = reviewer.review(answer, question, context_str, expert_outputs)
            v8_review = review.to_dict()
            v8_thinking.append(
                f"📝 V8审查: {review.overall_score:.1f}/10 "
                f"({'✅通过' if review.passed else '❌未通过'})"
            )
            if review.blockers:
                for b in review.blockers:
                    v8_thinking.append(f"  🚫 {b}")
        except Exception:
            pass

    # Phase IV: 自动评估
    v8_eval = None
    if HAS_V8_EVO:
        try:
            loop = get_eval_loop()
            eval_result = loop.evaluate(
                answer, question,
                metadata={
                    "agents": agents_used,
                    "gate_level": v8_pre.get("gate_level", "full"),
                }
            )
            v8_eval = {
                "trend": eval_result.get("trend"),
                "avg_score": eval_result.get("avg_score"),
                "eval_count": eval_result.get("eval_count"),
            }
            if eval_result.get("alert"):
                v8_thinking.append(f"🚨 {eval_result['alert']}")
        except Exception:
            pass

    # Phase IV: 轨迹记录
    if HAS_V8_EVO:
        try:
            distiller = get_distiller()
            score = v8_review.get("score", 7.0) if v8_review else 7.0
            distiller.record_trajectory(
                question, answer, agents_used, score, expert_outputs
            )
        except Exception:
            pass

    # Phase III: 保存到记忆图
    if HAS_V8_MEM:
        try:
            graph = get_memory_graph()
            entities = _extract_entities(question, data)
            importance = min((v8_review.get("score", 5.0) if v8_review else 5.0) / 10, 1.0)
            graph.add(
                content=f"Q: {question[:100]} → A: {answer[:200]}",
                memory_type=MemoryType.EPISODIC,
                importance=importance,
                entities=entities,
                tags=agents_used,
                source="analysis",
            )
        except Exception:
            pass

    # Phase II: 语义缓存保存
    if HAS_V8_CTX:
        try:
            cache = get_context_cache()
            cache.put(question, result)
        except Exception:
            pass

    # 添加 V8 字段到结果
    result["v8_enhanced"] = True
    result["v8_gate"] = {
        "level": v8_pre.get("gate_level", "full"),
        "score": v8_pre.get("gate_score", 0),
        "agents": v8_pre.get("gate_agents", []),
        "reason": v8_pre.get("gate_reason", ""),
    }
    result["v8_review"] = v8_review
    result["v8_eval"] = v8_eval
    result["v8_skills"] = v8_pre.get("v8_skills", [])
    result["v8_contract_issues"] = contract_issues
    result["v8_status"] = get_v8_status()

    # 合并 thinking
    original_thinking = result.get("thinking", [])
    result["thinking"] = v8_thinking + original_thinking

    return result


# ============================================================
# V8 全局统计
# ============================================================

def v8_get_stats() -> Dict:
    """获取 V8 全部统计信息"""
    stats = {"status": get_v8_status()}

    if HAS_V8_GATE:
        stats["gate"] = get_gate().get_stats()
    if HAS_V8_CTX:
        stats["cache"] = get_context_cache().get_stats()
        stats["playbook"] = get_playbook().get_all_stats()
    if HAS_V8_MEM:
        stats["memory"] = get_memory_graph().get_stats()
    if HAS_V8_EVO:
        stats["eval"] = get_eval_loop().get_report()
        stats["skills"] = get_distiller().get_stats()

    return stats


def v8_build_telos_from_results(results: dict, health_scores: list = None):
    """从分析结果批量构建 TELOS 画像"""
    if not HAS_V8_MEM:
        return

    graph = get_memory_graph()
    customers = results.get('客户金额', [])
    if isinstance(customers, list):
        for c in customers[:50]:
            try:
                graph.build_telos_from_data(c, health_scores)
            except Exception:
                pass


# ============================================================
# 辅助函数
# ============================================================

def _extract_entities(question: str, data: dict) -> List[str]:
    """从问题和数据中提取实体"""
    entities = []
    customers = data.get('客户金额', [])
    if isinstance(customers, list):
        for c in customers[:50]:
            name = c.get('客户', '')
            if name and len(name) >= 2 and name in question:
                entities.append(name)
    return entities[:10]


def _agent_name_to_id(agent_name: str) -> str:
    """Agent 显示名 → Agent ID"""
    if "分析" in agent_name:
        return "analyst"
    if "风控" in agent_name:
        return "risk"
    if "策略" in agent_name:
        return "strategist"
    return ""
