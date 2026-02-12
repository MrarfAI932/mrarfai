#!/usr/bin/env python3
"""
MRARFAI v3.3 — CriticAgent (Generator + Critic Pattern)
=========================================================
Google Agent Pattern #5 & #6: Generator+Critic + Iterative Refinement

报告员生成报告 → 批评家审查 → 低分则迭代优化 → 输出高质量最终报告

质量维度:
  - completeness: 是否回答了用户问题的所有方面
  - accuracy: 数据引用是否准确、结论是否有数据支撑  
  - actionability: 建议是否具体可执行
  - clarity: 表达是否清晰、结构是否合理
  - conciseness: 是否简洁、无废话
"""

import json
import time
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 质量评估数据模型
# ============================================================

@dataclass
class CritiqueResult:
    """批评结果"""
    overall_score: float          # 0-10
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    issues: list = field(default_factory=list)       # 具体问题
    suggestions: list = field(default_factory=list)   # 改进建议
    pass_threshold: bool = False  # 是否通过质量门禁
    iteration: int = 0

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "dimensions": self.dimension_scores,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "passed": self.pass_threshold,
            "iteration": self.iteration,
        }


@dataclass
class RefinementTrace:
    """迭代精炼追踪"""
    original_score: float = 0.0
    final_score: float = 0.0
    iterations: int = 0
    history: list = field(default_factory=list)  # [{iteration, score, issues}]
    total_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "original_score": self.original_score,
            "final_score": self.final_score,
            "iterations": self.iterations,
            "improvement": round(self.final_score - self.original_score, 2),
            "history": self.history,
            "total_time_ms": round(self.total_time_ms, 1),
        }


# ============================================================
# CriticAgent — 报告质量审查
# ============================================================

class CriticAgent:
    """
    批评家Agent — 审查报告质量并提供改进建议
    
    支持两种模式:
    1. LLM批评 (默认): 用LLM评估报告质量
    2. 规则批评 (fallback): 基于规则的快速评估
    """

    # 质量门禁阈值
    DEFAULT_THRESHOLD = 7.0
    MAX_ITERATIONS = 2

    CRITIC_SYSTEM_PROMPT = """你是禾苗通讯的报告质量审查专家。你的职责是严格评估分析报告的质量。

评估维度（每项0-10分）:
1. completeness（完整性）: 是否全面回答了用户问题？有无遗漏关键方面？
2. accuracy（准确性）: 数据引用是否准确？结论是否有数据支撑？有无自相矛盾？
3. actionability（可行性）: 建议是否具体？能否直接执行？有无空泛的废话？
4. clarity（清晰度）: 结构是否清晰？逻辑是否通顺？是否易于CEO阅读？
5. conciseness（简洁度）: 是否精炼？有无重复或冗余内容？

你必须严格按照以下JSON格式输出，不要有任何其他内容:
{
  "overall_score": 7.5,
  "dimensions": {
    "completeness": 8,
    "accuracy": 7,
    "actionability": 7,
    "clarity": 8,
    "conciseness": 7
  },
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}"""

    REFINER_SYSTEM_PROMPT = """你是禾苗通讯高级报告撰写人。请根据质量审查反馈，改进以下报告。

改进规则:
1. 针对每个issue，直接修正
2. 遵循每条suggestion
3. 保留原报告的正确内容和数据
4. 不要添加你没有数据支持的新论点
5. 控制500字内"""

    @staticmethod
    def critique_with_llm(
        report: str,
        question: str,
        expert_outputs: Dict[str, str],
        llm_fn,  # _call_llm_raw function
        provider: str,
        api_key: str,
        iteration: int = 0,
    ) -> CritiqueResult:
        """LLM驱动的报告质量批评"""
        experts_summary = "\n".join(f"- {k}: {v[:100]}..." for k, v in expert_outputs.items())

        user_prompt = f"""用户问题: {question}

专家分析摘要:
{experts_summary}

待审查报告:
{report}

请严格按JSON格式评估。"""

        raw = llm_fn(
            CriticAgent.CRITIC_SYSTEM_PROMPT,
            user_prompt,
            provider, api_key,
            max_tokens=500, temperature=0.1,
            _trace_name=f"critic_llm_iter{iteration}",
        )

        return CriticAgent._parse_critique(raw, iteration)

    @staticmethod
    def critique_with_rules(
        report: str,
        question: str,
        expert_outputs: Dict[str, str],
        iteration: int = 0,
    ) -> CritiqueResult:
        """规则驱动的快速质量评估（不消耗LLM token）"""
        scores = {}
        issues = []
        suggestions = []

        # completeness: 报告是否涵盖了所有专家的核心内容
        expert_keywords = set()
        for name, output in expert_outputs.items():
            # 提取关键数字
            nums = re.findall(r'[\d,]+\.?\d*[%万亿]?', output)
            expert_keywords.update(nums[:3])
        
        covered = sum(1 for kw in expert_keywords if kw in report) if expert_keywords else 0
        coverage_ratio = covered / max(len(expert_keywords), 1)
        scores["completeness"] = min(10, 5 + coverage_ratio * 5)
        if coverage_ratio < 0.5:
            issues.append("报告遗漏了部分专家的关键数据")
            suggestions.append("确保每位专家的核心数据点都被报告引用")

        # accuracy: 数据引用密度
        nums_in_report = len(re.findall(r'[\d,]+\.?\d*[%万亿]?', report))
        scores["accuracy"] = min(10, 4 + nums_in_report * 0.5)
        if nums_in_report < 3:
            issues.append("数据引用不足，缺乏量化支撑")
            suggestions.append("增加具体数字引用，每个论点至少一个数据支撑")

        # actionability: 是否有具体建议
        action_words = ["建议", "应该", "可以", "需要", "优先", "立即", "行动"]
        action_count = sum(1 for w in action_words if w in report)
        scores["actionability"] = min(10, 4 + action_count * 1.5)
        if action_count < 2:
            issues.append("缺乏具体可执行建议")
            suggestions.append("增加具体行动建议，包含时间框架和优先级")

        # clarity: 结构性
        has_sections = bool(re.search(r'[#\*\d][\.\)、]', report))
        length = len(report)
        scores["clarity"] = 7 if has_sections else 5
        if 200 <= length <= 600:
            scores["clarity"] = min(10, scores["clarity"] + 2)
        if not has_sections:
            issues.append("缺乏清晰的段落结构")
            suggestions.append("添加小标题或编号来组织内容")

        # conciseness
        if length > 800:
            scores["conciseness"] = 5
            issues.append("报告过长，超出500字限制")
            suggestions.append("精简冗余内容，控制在500字以内")
        elif length < 100:
            scores["conciseness"] = 4
            issues.append("报告过短，内容不充分")
            suggestions.append("补充更多分析和建议")
        else:
            scores["conciseness"] = 8

        overall = sum(scores.values()) / len(scores)

        return CritiqueResult(
            overall_score=round(overall, 1),
            dimension_scores=scores,
            issues=issues,
            suggestions=suggestions,
            pass_threshold=overall >= CriticAgent.DEFAULT_THRESHOLD,
            iteration=iteration,
        )

    @staticmethod
    def _parse_critique(raw: str, iteration: int) -> CritiqueResult:
        """解析LLM返回的JSON格式批评"""
        try:
            # 提取JSON
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")

            return CritiqueResult(
                overall_score=float(data.get("overall_score", 5)),
                dimension_scores=data.get("dimensions", {}),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                pass_threshold=float(data.get("overall_score", 5)) >= CriticAgent.DEFAULT_THRESHOLD,
                iteration=iteration,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # LLM返回非标格式 → fallback
            return CritiqueResult(
                overall_score=6.0,
                dimension_scores={},
                issues=["无法解析LLM批评结果"],
                suggestions=["需要人工审查"],
                pass_threshold=False,
                iteration=iteration,
            )


# ============================================================
# IterativeRefiner — 迭代精炼引擎
# ============================================================

class IterativeRefiner:
    """
    迭代精炼引擎 — Generator+Critic循环
    
    流程:
    1. Reporter 生成初稿
    2. CriticAgent 评估质量
    3. 如果低于阈值 → Refiner 重写 → 回到步骤2
    4. 最多 MAX_ITERATIONS 次迭代
    5. 返回最终版本 + 迭代追踪
    """

    @staticmethod
    def refine(
        initial_report: str,
        question: str,
        expert_outputs: Dict[str, str],
        llm_fn,
        provider: str,
        api_key: str,
        threshold: float = None,
        max_iterations: int = None,
        use_llm_critic: bool = True,
    ) -> Tuple[str, CritiqueResult, RefinementTrace]:
        """
        迭代精炼报告直到通过质量门禁
        
        Returns:
            (final_report, final_critique, refinement_trace)
        """
        threshold = threshold or CriticAgent.DEFAULT_THRESHOLD
        max_iter = max_iterations or CriticAgent.MAX_ITERATIONS
        trace = RefinementTrace()
        t0 = time.time()

        current_report = initial_report

        for i in range(max_iter + 1):  # 0=initial critique, 1..N=refinements
            # 批评
            if use_llm_critic and api_key:
                critique = CriticAgent.critique_with_llm(
                    current_report, question, expert_outputs,
                    llm_fn, provider, api_key, iteration=i,
                )
            else:
                critique = CriticAgent.critique_with_rules(
                    current_report, question, expert_outputs, iteration=i,
                )

            trace.history.append({
                "iteration": i,
                "score": critique.overall_score,
                "issues": critique.issues[:3],
                "passed": critique.pass_threshold,
            })

            if i == 0:
                trace.original_score = critique.overall_score

            # 通过门禁 → 结束
            if critique.pass_threshold:
                trace.final_score = critique.overall_score
                trace.iterations = i
                trace.total_time_ms = (time.time() - t0) * 1000
                return current_report, critique, trace

            # 最后一轮 → 即使没通过也结束
            if i == max_iter:
                trace.final_score = critique.overall_score
                trace.iterations = i
                trace.total_time_ms = (time.time() - t0) * 1000
                return current_report, critique, trace

            # 精炼
            current_report = IterativeRefiner._refine_report(
                current_report, critique, question, expert_outputs,
                llm_fn, provider, api_key, iteration=i + 1,
            )

        # 不应到达这里
        trace.final_score = critique.overall_score
        trace.iterations = max_iter
        trace.total_time_ms = (time.time() - t0) * 1000
        return current_report, critique, trace

    @staticmethod
    def _refine_report(
        report: str,
        critique: CritiqueResult,
        question: str,
        expert_outputs: Dict[str, str],
        llm_fn,
        provider: str,
        api_key: str,
        iteration: int,
    ) -> str:
        """基于批评反馈精炼报告"""
        issues_text = "\n".join(f"- {i}" for i in critique.issues)
        suggestions_text = "\n".join(f"- {s}" for s in critique.suggestions)
        experts_summary = "\n".join(f"- {k}: {v[:150]}..." for k, v in expert_outputs.items())

        prompt = f"""原始问题: {question}

专家分析:
{experts_summary}

当前报告 (得分 {critique.overall_score}/10):
{report}

质量审查发现的问题:
{issues_text}

改进建议:
{suggestions_text}

请改进报告，解决以上问题。500字内。"""

        refined = llm_fn(
            CriticAgent.REFINER_SYSTEM_PROMPT,
            prompt,
            provider, api_key,
            max_tokens=800, temperature=0.3,
            _trace_name=f"refiner_llm_iter{iteration}",
        )

        return refined


# ============================================================
# 快捷API — 供multi_agent.py调用
# ============================================================

def critique_and_refine(
    report: str,
    question: str,
    expert_outputs: Dict[str, str],
    llm_fn,
    provider: str = "deepseek",
    api_key: str = "",
    threshold: float = 7.0,
    max_iterations: int = 2,
    use_llm_critic: bool = True,
    enabled: bool = True,
) -> Tuple[str, Optional[dict], Optional[dict]]:
    """
    快捷入口 — critique并可选refine
    
    Returns:
        (final_report, critique_dict_or_None, trace_dict_or_None)
    """
    if not enabled:
        return report, None, None

    final, critique, trace = IterativeRefiner.refine(
        report, question, expert_outputs,
        llm_fn, provider, api_key,
        threshold=threshold,
        max_iterations=max_iterations,
        use_llm_critic=use_llm_critic,
    )

    return final, critique.to_dict(), trace.to_dict()
