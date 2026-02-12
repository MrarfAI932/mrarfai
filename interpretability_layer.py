#!/usr/bin/env python3
"""
MRARFAI V9.0 — LatentLens Interpretability Layer
====================================================
基于 "LatentLens" (arXiv:2602.07715) + MRARFAI 可观测性需求

核心思路:
  将 Agent 内部状态映射为人类可理解的自然语言解释
  不是黑盒输出 → 而是展示"Agent 为什么这样分析"

三层解释:
  1. Intent Mapping  — 问题意图识别 + 路由解释
  2. Process Trace   — 推理过程追踪 + 决策树可视化
  3. Output Explain  — 输出归因 + 置信度分解

集成点:
  - observability.py: 扩展 Langfuse trace 加解释层
  - adaptive_gate.py: 解释门控决策(为什么 skip/light/full)
  - search_engine.py: 解释搜索路径选择
  - reasoning_templates.py: 解释模板匹配逻辑
  - memory_v9.py: 解释记忆检索和技能匹配
  - ai_narrator.py: 集成到叙事输出

效果:
  - 每次分析附带"决策解释"面板
  - 用户可点击查看: 为什么选了这些 Agent / 为什么关注这些数据
  - 审计合规: 完整的决策归因链
"""

import json
import time
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("mrarfai.interpret")


# ============================================================
# 解释数据结构
# ============================================================

@dataclass
class IntentExplanation:
    """意图解释"""
    original_query: str
    detected_intent: str              # trend / compare / risk / query / forecast
    confidence: float                 # 0-1
    intent_signals: List[Dict]        # 触发意图的信号
    route_decision: str               # skip / light / full
    route_reason: str                 # 路由原因
    alternative_intents: List[Dict] = field(default_factory=list)  # 其他可能意图

@dataclass
class ProcessStep:
    """推理过程步骤"""
    step_name: str
    agent_name: str
    action: str                       # 做了什么
    input_summary: str                # 输入摘要
    output_summary: str               # 输出摘要
    decision_points: List[Dict] = field(default_factory=list)  # 关键决策点
    data_accessed: List[str] = field(default_factory=list)      # 访问的数据
    elapsed_ms: float = 0
    tokens_used: int = 0

@dataclass
class OutputAttribution:
    """输出归因"""
    claim: str                        # 一个具体断言
    confidence: float                 # 置信度
    supporting_data: List[Dict]       # 支撑数据
    source_agents: List[str]          # 来源 Agent
    memory_used: List[str] = field(default_factory=list)    # 使用的记忆
    reasoning_chain: str = ""         # 推理链概要

@dataclass
class FullExplanation:
    """完整的解释报告"""
    query: str
    intent: IntentExplanation
    process: List[ProcessStep]
    attributions: List[OutputAttribution]
    summary_zh: str = ""              # 中文总结
    total_elapsed_ms: float = 0
    total_tokens: int = 0
    explanation_cost_pct: float = 0   # 解释本身的开销占比


# ============================================================
# Intent Mapper — 意图识别与解释
# ============================================================

class IntentMapper:
    """
    意图映射器 — 解释 Agent 如何理解用户问题
    """

    INTENT_PATTERNS = {
        "trend": {
            "keywords": ["趋势", "变化", "走势", "增长", "下降", "波动", "月度", "季度"],
            "description": "趋势分析 — 识别时间序列变化模式",
        },
        "compare": {
            "keywords": ["对比", "比较", "vs", "差异", "排名", "top", "最高", "最低"],
            "description": "对比分析 — 多维度横向比较",
        },
        "risk": {
            "keywords": ["风险", "异常", "下滑", "流失", "集中度", "预警", "危险"],
            "description": "风险检测 — 识别潜在威胁和异常",
        },
        "forecast": {
            "keywords": ["预测", "预估", "估计", "明年", "下季度", "展望", "预期"],
            "description": "预测分析 — 未来趋势推断",
        },
        "query": {
            "keywords": ["多少", "是什么", "几个", "列出", "总", "查询"],
            "description": "数据查询 — 直接检索特定数据",
        },
        "strategy": {
            "keywords": ["策略", "建议", "怎么办", "机会", "行动", "优化"],
            "description": "策略建议 — 基于数据的行动方案",
        },
    }

    COMPLEXITY_REASONS = {
        "skip": "问题较简单，直接SQL查询即可，无需调用分析Agent",
        "light": "问题中等复杂，需要1-2个专业Agent进行分析",
        "full": "问题涉及多维度/跨领域分析，需要全部Agent协作",
    }

    def explain_intent(self, query: str,
                       gate_result: Dict = None) -> IntentExplanation:
        """生成意图解释"""
        # 检测意图
        intent_scores = {}
        intent_signals = []

        for intent, config in self.INTENT_PATTERNS.items():
            score = 0
            matched_kws = []
            for kw in config["keywords"]:
                if kw in query:
                    score += 1
                    matched_kws.append(kw)
            if matched_kws:
                intent_scores[intent] = score
                intent_signals.append({
                    "intent": intent,
                    "matched_keywords": matched_kws,
                    "score": score,
                })

        # 主意图
        if intent_scores:
            primary = max(intent_scores, key=intent_scores.get)
            max_score = max(intent_scores.values())
            confidence = min(1.0, max_score / 3)
        else:
            primary = "query"
            confidence = 0.3

        # 路由解释
        route = "light"
        if gate_result:
            route = gate_result.get("level", "light")
        route_reason = self.COMPLEXITY_REASONS.get(route, "默认路由")

        # 备选意图
        alternatives = [
            {"intent": k, "score": v, "desc": self.INTENT_PATTERNS[k]["description"]}
            for k, v in sorted(intent_scores.items(), key=lambda x: -x[1])
            if k != primary
        ][:3]

        return IntentExplanation(
            original_query=query,
            detected_intent=primary,
            confidence=confidence,
            intent_signals=intent_signals,
            route_decision=route,
            route_reason=route_reason,
            alternative_intents=alternatives,
        )


# ============================================================
# Process Tracer — 推理过程追踪
# ============================================================

class ProcessTracer:
    """
    推理过程追踪器 — 记录 Agent 每一步的决策

    与 observability.py 互补:
      - observability: 底层 span/trace (技术指标)
      - ProcessTracer: 高层 "为什么" (业务解释)
    """

    def __init__(self):
        self.steps: List[ProcessStep] = []
        self._start_time = time.time()

    def trace_step(self, step_name: str, agent_name: str,
                   action: str, input_summary: str = "",
                   output_summary: str = "",
                   decision_points: List[Dict] = None,
                   data_accessed: List[str] = None,
                   elapsed_ms: float = 0, tokens: int = 0):
        """记录一个推理步骤"""
        self.steps.append(ProcessStep(
            step_name=step_name,
            agent_name=agent_name,
            action=action,
            input_summary=input_summary[:200],
            output_summary=output_summary[:300],
            decision_points=decision_points or [],
            data_accessed=data_accessed or [],
            elapsed_ms=elapsed_ms,
            tokens_used=tokens,
        ))

    def trace_gate_decision(self, query: str, level: str,
                            score: float, agents: List[str]):
        """追踪门控决策"""
        self.trace_step(
            step_name="门控路由",
            agent_name="AdaptiveGate",
            action=f"复杂度评估 → {level}",
            input_summary=query[:100],
            output_summary=f"分数={score:.2f}, 级别={level}, 推荐Agent={agents}",
            decision_points=[{
                "point": "复杂度阈值",
                "threshold": "skip<0.3, light<0.7, full≥0.7",
                "actual": f"{score:.2f} → {level}",
            }],
        )

    def trace_agent_call(self, agent_name: str, question: str,
                         output_preview: str, elapsed_ms: float,
                         tokens: int, template_used: str = ""):
        """追踪 Agent 调用"""
        self.trace_step(
            step_name=f"Agent分析: {agent_name}",
            agent_name=agent_name,
            action=f"使用模板 {template_used}" if template_used else "自由分析",
            input_summary=question[:100],
            output_summary=output_preview[:200],
            elapsed_ms=elapsed_ms,
            tokens=tokens,
        )

    def trace_memory_recall(self, agent: str, query: str,
                            memories_found: int, skills_matched: int):
        """追踪记忆检索"""
        self.trace_step(
            step_name="记忆检索",
            agent_name="MemoryV9",
            action=f"为 {agent} 检索记忆",
            input_summary=query[:80],
            output_summary=f"找到 {memories_found} 条相关记忆, {skills_matched} 个匹配技能",
        )

    def trace_search(self, strategy: str, branches: int,
                     best_score: float, calls: int):
        """追踪搜索过程"""
        self.trace_step(
            step_name="EnCompass搜索",
            agent_name="SearchEngine",
            action=f"{strategy} 搜索",
            output_summary=f"探索 {branches} 条路径, 最优分 {best_score:.3f}, {calls} 次LLM调用",
        )

    def get_trace(self) -> List[ProcessStep]:
        return self.steps

    def to_timeline(self) -> List[Dict]:
        """转换为时间线格式"""
        return [
            {
                "step": s.step_name,
                "agent": s.agent_name,
                "action": s.action,
                "input": s.input_summary,
                "output": s.output_summary,
                "decisions": s.decision_points,
                "data": s.data_accessed,
                "time_ms": s.elapsed_ms,
                "tokens": s.tokens_used,
            }
            for s in self.steps
        ]


# ============================================================
# Output Attributor — 输出归因
# ============================================================

class OutputAttributor:
    """
    输出归因 — 解释每个结论的来源

    将 Agent 输出拆分为具体断言，
    每个断言追溯到: 数据来源、推理链、使用的记忆
    """

    def attribute(self, final_output: str,
                  expert_outputs: Dict[str, str] = None,
                  data_sources: List[str] = None,
                  memories_used: List[Dict] = None) -> List[OutputAttribution]:
        """生成输出归因"""
        attributions = []

        # 将输出拆分为断言 (按句号/换行)
        claims = self._split_claims(final_output)

        for claim in claims[:10]:  # 最多10个
            attr = OutputAttribution(
                claim=claim,
                confidence=self._estimate_confidence(claim),
                supporting_data=self._find_data_support(claim, data_sources),
                source_agents=self._find_agent_sources(claim, expert_outputs),
                memory_used=[m["id"] for m in (memories_used or [])
                             if self._content_overlap(claim, m.get("content", ""))],
                reasoning_chain=self._infer_reasoning_chain(claim, expert_outputs),
            )
            attributions.append(attr)

        return attributions

    def _split_claims(self, text: str) -> List[str]:
        """将文本拆分为独立断言"""
        import re
        # 按句号、换行、分号拆分
        parts = re.split(r'[。\n；;]', text)
        claims = [p.strip() for p in parts if len(p.strip()) > 10]
        return claims

    def _estimate_confidence(self, claim: str) -> float:
        """估算断言置信度"""
        score = 0.5

        # 包含具体数字 → 更高置信
        import re
        if re.search(r'\d+[%万亿]', claim):
            score += 0.2
        if re.search(r'\d+\.\d+', claim):
            score += 0.1

        # 包含限定词 → 稍低置信
        hedges = ["可能", "也许", "大约", "估计", "推测"]
        if any(h in claim for h in hedges):
            score -= 0.1

        # 包含对比/因果 → 中等置信
        if any(kw in claim for kw in ["因此", "所以", "导致", "因为"]):
            score += 0.1

        return max(0.1, min(1.0, score))

    def _find_data_support(self, claim: str,
                           data_sources: List[str] = None) -> List[Dict]:
        """找出支撑数据"""
        supports = []
        if not data_sources:
            return supports

        for ds in data_sources[:5]:
            overlap = self._content_overlap(claim, ds)
            if overlap > 0.2:
                supports.append({
                    "source": ds[:50],
                    "overlap": round(overlap, 2),
                })
        return supports

    def _find_agent_sources(self, claim: str,
                            expert_outputs: Dict[str, str] = None) -> List[str]:
        """找出贡献的 Agent"""
        if not expert_outputs:
            return []
        sources = []
        for agent, output in expert_outputs.items():
            if self._content_overlap(claim, output) > 0.15:
                sources.append(agent)
        return sources

    def _infer_reasoning_chain(self, claim: str,
                               expert_outputs: Dict[str, str] = None) -> str:
        """推断推理链"""
        if not expert_outputs:
            return "直接数据查询"

        chain_parts = []
        for agent, output in expert_outputs.items():
            if self._content_overlap(claim, output) > 0.1:
                chain_parts.append(agent)

        if chain_parts:
            return " → ".join(chain_parts) + " → 综合结论"
        return "综合推理"

    @staticmethod
    def _content_overlap(t1: str, t2: str) -> float:
        """文本重叠度"""
        s1 = set(t1.lower().split())
        s2 = set(t2.lower().split())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / max(len(s1 | s2), 1)


# ============================================================
# 统一解释引擎
# ============================================================

class InterpretabilityEngine:
    """
    MRARFAI 可解释性引擎 — 统一入口

    用法:
        engine = InterpretabilityEngine()
        
        # 开始追踪
        engine.start_trace(question)
        
        # 各阶段追踪
        engine.explain_intent(question, gate_result)
        engine.trace_gate(...)
        engine.trace_agent(...)
        engine.trace_memory(...)
        
        # 生成完整解释
        explanation = engine.finalize(final_output, expert_outputs)
    """

    def __init__(self):
        self.intent_mapper = IntentMapper()
        self.tracer = ProcessTracer()
        self.attributor = OutputAttributor()
        self._query = ""
        self._intent = None
        self._start = 0

    def start_trace(self, query: str):
        """开始新的解释追踪"""
        self._query = query
        self._start = time.time()
        self.tracer = ProcessTracer()
        self._intent = None

    def explain_intent(self, query: str,
                       gate_result: Dict = None) -> IntentExplanation:
        """解释意图识别"""
        self._intent = self.intent_mapper.explain_intent(query, gate_result)
        return self._intent

    def trace_gate(self, query: str, level: str,
                   score: float, agents: List[str]):
        self.tracer.trace_gate_decision(query, level, score, agents)

    def trace_agent(self, agent_name: str, question: str,
                    output_preview: str, elapsed_ms: float = 0,
                    tokens: int = 0, template: str = ""):
        self.tracer.trace_agent_call(
            agent_name, question, output_preview,
            elapsed_ms, tokens, template
        )

    def trace_memory(self, agent: str, query: str,
                     found: int, skills: int):
        self.tracer.trace_memory_recall(agent, query, found, skills)

    def trace_search(self, strategy: str, branches: int,
                     best_score: float, calls: int):
        self.tracer.trace_search(strategy, branches, best_score, calls)

    def finalize(self, final_output: str,
                 expert_outputs: Dict[str, str] = None,
                 data_sources: List[str] = None,
                 memories_used: List[Dict] = None) -> FullExplanation:
        """生成完整解释报告"""
        elapsed = (time.time() - self._start) * 1000

        # 归因
        attributions = self.attributor.attribute(
            final_output, expert_outputs, data_sources, memories_used
        )

        # 总token
        total_tokens = sum(s.tokens_used for s in self.tracer.steps)

        # 中文总结
        summary = self._generate_summary(
            self._intent, self.tracer.steps, attributions
        )

        return FullExplanation(
            query=self._query,
            intent=self._intent or self.intent_mapper.explain_intent(self._query),
            process=self.tracer.steps,
            attributions=attributions,
            summary_zh=summary,
            total_elapsed_ms=elapsed,
            total_tokens=total_tokens,
        )

    def _generate_summary(self, intent, steps, attributions) -> str:
        """生成中文解释总结"""
        parts = []

        if intent:
            parts.append(f"识别意图: {intent.detected_intent} (置信度{intent.confidence:.0%})")
            parts.append(f"路由决策: {intent.route_decision} — {intent.route_reason}")

        if steps:
            agents_used = list(set(s.agent_name for s in steps))
            parts.append(f"执行了 {len(steps)} 个步骤，涉及 {', '.join(agents_used)}")

        if attributions:
            high_conf = [a for a in attributions if a.confidence > 0.7]
            parts.append(f"输出包含 {len(attributions)} 个断言，其中 {len(high_conf)} 个高置信")

        return " → ".join(parts) if parts else "无解释信息"

    def to_dict(self) -> Dict:
        """导出为 JSON 兼容字典"""
        exp = self.finalize("", {})
        return {
            "query": exp.query,
            "intent": {
                "detected": exp.intent.detected_intent,
                "confidence": exp.intent.confidence,
                "route": exp.intent.route_decision,
                "reason": exp.intent.route_reason,
            },
            "process": self.tracer.to_timeline(),
            "summary": exp.summary_zh,
        }


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MRARFAI LatentLens Interpretability v9.0 Demo")
    print("=" * 60)

    engine = InterpretabilityEngine()

    # 模拟完整分析流程
    query = "分析2025年各品牌出货趋势，找出异常并给出建议"
    print(f"\n问题: {query}")

    engine.start_trace(query)

    # 1. 意图
    intent = engine.explain_intent(query, {"level": "full", "score": 0.82})
    print(f"\n--- 意图解释 ---")
    print(f"  主意图: {intent.detected_intent} ({intent.confidence:.0%})")
    print(f"  信号: {intent.intent_signals}")
    print(f"  路由: {intent.route_decision} — {intent.route_reason}")
    print(f"  备选: {[a['intent'] for a in intent.alternative_intents]}")

    # 2. 门控
    engine.trace_gate(query, "full", 0.82, ["analyst", "risk", "strategist", "reporter"])

    # 3. 记忆
    engine.trace_memory("analyst", query, found=3, skills=1)

    # 4. Agent 调用
    engine.trace_agent("analyst", query, "HMD增长35%，Transsion稳定...",
                       elapsed_ms=1200, tokens=800, template="analyst-deep")
    engine.trace_agent("risk", query, "Top2集中度55%，存在风险...",
                       elapsed_ms=900, tokens=600, template="risk-standard")
    engine.trace_agent("strategist", query, "建议拓展平板和新品牌...",
                       elapsed_ms=1100, tokens=700, template="strategist-standard")

    # 5. 搜索
    engine.trace_search("two_level_beam", branches=9, best_score=0.87, calls=12)

    # 6. 最终解释
    explanation = engine.finalize(
        final_output="2025年出货趋势呈现分化格局。HMD同比增长35%领跑，Transsion稳定在3.2亿。但Top2集中度达55%存在风险。建议拓展3-5个新品牌。",
        expert_outputs={
            "analyst": "HMD增长35%，Transsion 3.2亿营收",
            "risk": "Top2集中度55%，建议分散",
            "strategist": "拓展平板赛道，开发新品牌",
        },
    )

    print(f"\n--- 推理过程 ({len(explanation.process)} 步) ---")
    for step in explanation.process:
        print(f"  [{step.agent_name}] {step.step_name}: {step.action}")
        if step.output_summary:
            print(f"    → {step.output_summary[:80]}")

    print(f"\n--- 输出归因 ({len(explanation.attributions)} 个断言) ---")
    for attr in explanation.attributions:
        print(f"  📝 {attr.claim[:60]}")
        print(f"     置信: {attr.confidence:.0%} | 来源: {attr.source_agents} | 链: {attr.reasoning_chain}")

    print(f"\n--- 总结 ---")
    print(f"  {explanation.summary_zh}")
    print(f"  耗时: {explanation.total_elapsed_ms:.0f}ms")
    print(f"  Tokens: {explanation.total_tokens}")

    print("\n✅ LatentLens Interpretability Layer 初始化成功")
