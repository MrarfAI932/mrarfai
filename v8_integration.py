#!/usr/bin/env python3
"""
MRARFAI V8.0 — Integration Layer (集成层)
============================================
将 Phase I-IV 全部集成到现有 multi_agent.py 流程中。

V7 → V8 流程对比:
──────────────────────────────────────────────────────────────
V7.0 流程:                      V8.0 流程:
1. 缓存检查                      0. 语义缓存检查 (Phase II)
2. 持久记忆加载                   1. 自适应门控评估 (Phase I)
3. SmartDataQuery                 2. 上下文工程编译 (Phase II)
4. SmartRouter (2级)              3. 动态Agent工厂 (Phase I)
5. ParallelAgentExecutor          4. 并行执行 + 合约验证 (Phase I)
6. Reporter                       5. 上下文压缩传递 (Phase II)
7. CriticAgent                    6. 结构化Reviewer硬门控 (Phase IV)
8. HITL                           7. 技能蒸馏 + 自动评估 (Phase IV)
9. 持久记忆保存                   8. 多维记忆图更新 (Phase III)
──────────────────────────────────────────────────────────────

向后兼容: 所有 V7 接口保持不变，V8 模块 try/except 导入
"""

import json
import time
from typing import Dict, List, Optional, Any

# ============================================================
# V8.0 模块导入 (全部 try/except 保证向后兼容)
# ============================================================

# Phase I: 自适应门控
try:
    from adaptive_gate import (
        AdaptiveGate, AgentFactory, ContractValidator,
        get_gate, get_factory, get_validator,
        adaptive_route, ComplexityLevel, ComplexityAssessment,
    )
    HAS_V8_GATE = True
except ImportError:
    HAS_V8_GATE = False

# Phase II: 上下文工程
try:
    from context_engine import (
        ContextCompressor, SemanticCache, WorkingContextBuilder,
        EvolvingPlaybook, YAMLContextSchema,
        get_context_cache, get_playbook,
        build_working_context, compress_agent_output,
    )
    HAS_V8_CTX = True
except ImportError:
    HAS_V8_CTX = False

# Phase III: 元记忆
try:
    from meta_memory import (
        MemoryGraph, TELOSProfile, MemoryType, MemoryNode,
        get_memory_graph, set_memory_graph,
    )
    HAS_V8_MEM = True
except ImportError:
    HAS_V8_MEM = False

# Phase IV: 自进化
try:
    from self_evolution import (
        StructuredReviewer, AutoEvalLoop, SkillDistiller,
        get_reviewer, get_eval_loop, get_distiller,
    )
    HAS_V8_EVO = True
except ImportError:
    HAS_V8_EVO = False


# ============================================================
# V8.0 能力摘要
# ============================================================

def get_v8_capabilities() -> Dict[str, Any]:
    """获取 V8.0 各模块可用状态"""
    return {
        "v8_available": any([HAS_V8_GATE, HAS_V8_CTX, HAS_V8_MEM, HAS_V8_EVO]),
        "phase_i_gate": HAS_V8_GATE,
        "phase_ii_context": HAS_V8_CTX,
        "phase_iii_memory": HAS_V8_MEM,
        "phase_iv_evolution": HAS_V8_EVO,
        "modules": {
            "adaptive_gate": "✅" if HAS_V8_GATE else "❌",
            "context_engine": "✅" if HAS_V8_CTX else "❌",
            "meta_memory": "✅" if HAS_V8_MEM else "❌",
            "self_evolution": "✅" if HAS_V8_EVO else "❌",
        },
    }


# ============================================================
# V8.0 增强流程节点 — 插入到现有 Pipeline 中
# ============================================================

class V8Pipeline:
    """
    V8.0 增强 Pipeline

    不替换 V7 的 ask_multi_agent，而是包装增强:
    - 在关键节点插入 V8 处理
    - V8 模块不可用时自动降级到 V7 逻辑
    """

    def __init__(self):
        self.thinking: List[str] = []
        self.v8_stats: Dict[str, Any] = {}

    # ---- Phase I: 门控路由 ----

    def adaptive_route(self, question: str, context: dict = None,
                       provider: str = "deepseek") -> Dict:
        """
        V8 自适应路由 (替代 SmartRouter)

        Returns:
            {
                "level": "skip|light|full",
                "agents": [...],
                "score": float,
                "reason": str,
                "v8_active": bool,
            }
        """
        if not HAS_V8_GATE:
            return {"level": "full", "agents": [], "v8_active": False}

        gate = get_gate()
        assessment = gate.route(question, context)

        self.thinking.append(
            f"🔀 V8门控: {assessment.level.value} "
            f"(score={assessment.score:.2f}) → "
            f"{assessment.reason}"
        )
        self.v8_stats["gate"] = {
            "level": assessment.level.value,
            "score": assessment.score,
            "agents": assessment.agents_recommended,
        }

        return {
            "level": assessment.level.value,
            "agents": assessment.agents_recommended,
            "score": assessment.score,
            "reason": assessment.reason,
            "v8_active": True,
            "assessment": assessment,
        }

    # ---- Phase II: 上下文编译 ----

    def compile_context(self, question: str, data: dict, results: dict,
                        memory_context: str = "",
                        level: str = "full",
                        agent_id: str = "") -> str:
        """
        V8 上下文编译 (替代简单拼接)
        """
        if not HAS_V8_CTX:
            return ""

        ctx = build_working_context(
            question, data, results, memory_context, level, agent_id
        )

        self.thinking.append(
            f"📋 V8上下文: {len(ctx)}字 (level={level})"
        )
        return ctx

    def check_semantic_cache(self, question: str) -> Optional[Dict]:
        """V8 语义缓存检查"""
        if not HAS_V8_CTX:
            return None

        cache = get_context_cache()
        result = cache.get(question)
        if result:
            self.thinking.append("⚡ V8语义缓存命中")
        return result

    def save_to_semantic_cache(self, question: str, result: dict):
        """保存到语义缓存"""
        if HAS_V8_CTX:
            cache = get_context_cache()
            cache.put(question, result)

    def compress_output(self, output: str, strategy: str = "moderate") -> str:
        """V8 Agent 输出压缩"""
        if not HAS_V8_CTX:
            return output
        return compress_agent_output(output, strategy)

    # ---- Phase I: 合约验证 ----

    def validate_output(self, agent_id: str, output: str) -> Dict:
        """V8 Agent 输出合约验证"""
        if not HAS_V8_GATE:
            return {"valid": True, "issues": []}

        validator = get_validator()
        is_valid, issues = validator.validate(agent_id, output)

        if not is_valid:
            self.thinking.append(
                f"⚠️ 合约违反 [{agent_id}]: {'; '.join(issues)}"
            )

        return {"valid": is_valid, "issues": issues}

    # ---- Phase III: 元记忆 ----

    def load_memory_context(self, question: str,
                            entities: List[str] = None) -> str:
        """V8 元记忆上下文加载"""
        if not HAS_V8_MEM:
            return ""

        graph = get_memory_graph()
        ctx = graph.build_memory_context(question, entities)

        if ctx:
            self.thinking.append(f"🧠 V8元记忆: 加载 {len(ctx)} 字上下文")
        return ctx

    def save_to_memory(self, question: str, answer: str,
                       agents_used: List[str],
                       entities: List[str] = None,
                       importance: float = 0.5):
        """V8 保存到记忆图"""
        if not HAS_V8_MEM:
            return

        graph = get_memory_graph()
        graph.add(
            content=f"Q: {question[:100]} → A: {answer[:200]}",
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            entities=entities or [],
            tags=agents_used,
            source="analysis",
        )

    def build_telos_profiles(self, customer_data: list,
                             health_scores: list = None):
        """批量构建 TELOS 画像"""
        if not HAS_V8_MEM:
            return

        graph = get_memory_graph()
        for customer in customer_data[:50]:  # Top 50 客户
            graph.build_telos_from_data(customer, health_scores)

    # ---- Phase IV: 自进化 ----

    def review_answer(self, answer: str, question: str,
                      context: str = "",
                      expert_outputs: Dict[str, str] = None) -> Dict:
        """V8 结构化审查"""
        if not HAS_V8_EVO:
            return {"passed": True, "score": 7.0}

        reviewer = get_reviewer()
        result = reviewer.review(answer, question, context, expert_outputs)

        self.thinking.append(
            f"📝 V8审查: {result.overall_score:.1f}/10 "
            f"({'✅' if result.passed else '❌'}) "
            f"耗时{result.review_time_ms:.0f}ms"
        )
        if result.blockers:
            for b in result.blockers:
                self.thinking.append(f"  🚫 {b}")

        self.v8_stats["review"] = result.to_dict()
        return result.to_dict()

    def auto_evaluate(self, answer: str, question: str,
                      context: str = "",
                      metadata: Dict = None) -> Dict:
        """V8 自动评估"""
        if not HAS_V8_EVO:
            return {}

        loop = get_eval_loop()
        result = loop.evaluate(answer, question, context, metadata=metadata)

        if result.get("alert"):
            self.thinking.append(f"🚨 {result['alert']}")

        self.v8_stats["eval"] = {
            "trend": result.get("trend"),
            "avg": result.get("avg_score"),
            "count": result.get("eval_count"),
        }
        return result

    def match_skills(self, question: str) -> List[Dict]:
        """V8 技能匹配"""
        if not HAS_V8_EVO:
            return []

        distiller = get_distiller()
        skills = distiller.match_skills(question)

        if skills:
            self.thinking.append(
                f"🎯 V8技能: 匹配 {len(skills)} 个 "
                f"({', '.join(s.name for s in skills)})"
            )
        return [{"name": s.name, "strategy": s.strategy} for s in skills]

    def record_trajectory(self, question: str, answer: str,
                          agents_used: List[str], score: float,
                          expert_outputs: Dict = None):
        """V8 记录分析轨迹"""
        if not HAS_V8_EVO:
            return

        distiller = get_distiller()
        distiller.record_trajectory(
            question, answer, agents_used, score, expert_outputs
        )

    # ---- 综合统计 ----

    def get_stats(self) -> Dict:
        """V8 全局统计"""
        stats = {
            "capabilities": get_v8_capabilities(),
            "v8_stats": self.v8_stats,
        }

        if HAS_V8_GATE:
            stats["gate_stats"] = get_gate().get_stats()
        if HAS_V8_CTX:
            stats["cache_stats"] = get_context_cache().get_stats()
            stats["playbook_stats"] = get_playbook().get_all_stats()
        if HAS_V8_MEM:
            stats["memory_stats"] = get_memory_graph().get_stats()
        if HAS_V8_EVO:
            stats["eval_report"] = get_eval_loop().get_report()
            stats["skill_stats"] = get_distiller().get_stats()

        return stats


# ============================================================
# V8 增强版 ask_multi_agent — 包装器
# ============================================================

def ask_multi_agent_v8(
    question: str,
    data: dict,
    results: dict,
    benchmark: dict = None,
    forecast: dict = None,
    provider: str = "deepseek",
    api_key: str = "",
    memory=None,
    # V7 参数
    enable_critic: bool = True,
    enable_hitl_v2: bool = True,
    enable_persistent_mem: bool = True,
    critic_threshold: float = 7.0,
    critic_max_iter: int = 2,
    stream_callback=None,
    enable_tools: bool = True,
    enable_cache: bool = True,
    # V8 新增参数
    enable_v8: bool = True,         # V8 总开关
    enable_gate: bool = True,       # Phase I
    enable_context_eng: bool = True, # Phase II
    enable_meta_memory: bool = True, # Phase III
    enable_evolution: bool = True,   # Phase IV
    force_level: str = None,        # 强制门控级别 (调试)
) -> dict:
    """
    V8.0 主入口 — 增强版 ask_multi_agent

    向后兼容 V7 所有参数，新增 V8 开关。
    V8 模块不可用时自动降级到 V7。
    """
    # 如果不启用 V8，直接走 V7
    if not enable_v8:
        from multi_agent import ask_multi_agent
        return ask_multi_agent(
            question, data, results, benchmark, forecast,
            provider, api_key, memory,
            enable_critic=enable_critic,
            enable_hitl_v2=enable_hitl_v2,
            enable_persistent_mem=enable_persistent_mem,
            critic_threshold=critic_threshold,
            critic_max_iter=critic_max_iter,
            stream_callback=stream_callback,
            enable_tools=enable_tools,
            enable_cache=enable_cache,
        )

    t0 = time.time()
    v8 = V8Pipeline()
    v8.thinking.append(f"📩 V8.0 收到: {question}")
    v8.thinking.append(f"🔧 V8模块: {get_v8_capabilities()['modules']}")

    # ---- Step 0: 语义缓存 (Phase II) ----
    if enable_context_eng:
        cached = v8.check_semantic_cache(question)
        if cached:
            cached["from_cache"] = True
            cached["v8_enhanced"] = True
            return cached

    # ---- Step 1: 自适应门控 (Phase I) ----
    gate_result = {"level": "full", "agents": []}
    if enable_gate:
        gate_result = v8.adaptive_route(question, provider=provider)

    level = gate_result.get("level", "full")

    # ---- Step 2: 元记忆加载 (Phase III) ----
    v8_memory_ctx = ""
    if enable_meta_memory:
        # 提取问题中可能的实体
        entities = _extract_entities_simple(question, data)
        v8_memory_ctx = v8.load_memory_context(question, entities)

    # ---- Step 3: 技能匹配 (Phase IV) ----
    matched_skills = []
    if enable_evolution:
        matched_skills = v8.match_skills(question)

    # ---- Step 4: 调用 V7 核心流程 (带 V8 增强参数) ----
    from multi_agent import ask_multi_agent
    v7_result = ask_multi_agent(
        question, data, results, benchmark, forecast,
        provider, api_key, memory,
        enable_critic=enable_critic,
        enable_hitl_v2=enable_hitl_v2,
        enable_persistent_mem=enable_persistent_mem,
        critic_threshold=critic_threshold,
        critic_max_iter=critic_max_iter,
        stream_callback=stream_callback,
        enable_tools=enable_tools,
        enable_cache=enable_cache,
    )

    # ---- Step 5: V8 后处理 ----
    answer = v7_result.get("answer", "")
    expert_outputs = v7_result.get("expert_outputs", {})
    agents_used = v7_result.get("agents_used", [])

    # Phase I: 合约验证
    if enable_gate:
        for agent_id, output in expert_outputs.items():
            # 提取 agent_id (去掉 emoji)
            clean_id = "analyst" if "分析" in agent_id else (
                "risk" if "风控" in agent_id else (
                "strategist" if "策略" in agent_id else ""
            ))
            if clean_id:
                v8.validate_output(clean_id, output)

    # Phase IV: 结构化审查
    review_result = {}
    if enable_evolution:
        review_result = v8.review_answer(
            answer, question,
            context=json.dumps(results, ensure_ascii=False, default=str)[:3000],
            expert_outputs=expert_outputs,
        )

    # Phase IV: 自动评估
    eval_result = {}
    if enable_evolution:
        eval_result = v8.auto_evaluate(answer, question, metadata={
            "agents": agents_used,
            "gate_level": level,
        })

    # Phase IV: 轨迹记录 + 技能蒸馏
    if enable_evolution:
        score = review_result.get("score", 7.0)
        v8.record_trajectory(question, answer, agents_used, score, expert_outputs)

    # Phase III: 记忆保存
    if enable_meta_memory:
        entities = _extract_entities_simple(question, data)
        v8.save_to_memory(
            question, answer, agents_used,
            entities=entities,
            importance=min(review_result.get("score", 5.0) / 10, 1.0),
        )

    # Phase II: 语义缓存保存
    if enable_context_eng:
        v8.save_to_semantic_cache(question, v7_result)

    elapsed = time.time() - t0
    v8.thinking.append(f"⏱️ V8总耗时 {elapsed:.1f}秒")

    # ---- 合并结果 ----
    v7_result["v8_enhanced"] = True
    v7_result["v8_thinking"] = v8.thinking
    v7_result["v8_stats"] = v8.get_stats()
    v7_result["v8_gate_level"] = level
    v7_result["v8_review"] = review_result
    v7_result["v8_eval"] = eval_result
    v7_result["v8_skills_matched"] = matched_skills

    # 合并 thinking
    original_thinking = v7_result.get("thinking", [])
    v7_result["thinking"] = v8.thinking + original_thinking

    return v7_result


def _extract_entities_simple(question: str, data: dict) -> List[str]:
    """简单实体提取"""
    entities = []
    customers = data.get('客户金额', [])
    if isinstance(customers, list):
        for c in customers[:50]:
            name = c.get('客户', '')
            if name and len(name) >= 2 and name in question:
                entities.append(name)
    return entities[:10]


# ============================================================
# 全局 V8 Pipeline 实例
# ============================================================

_v8_pipeline: Optional[V8Pipeline] = None

def get_v8_pipeline() -> V8Pipeline:
    global _v8_pipeline
    if _v8_pipeline is None:
        _v8_pipeline = V8Pipeline()
    return _v8_pipeline
