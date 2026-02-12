#!/usr/bin/env python3
"""
MRARFAI V8.0 — Phase IV: Self-Evolution Layer (自进化层)
=========================================================
借鉴:
  - ADAS (ICLR 2025): 元 Agent 自动设计更好的 Agent
  - SKILLRL: 历史轨迹蒸馏为可复用技能
  - LLM-as-Judge: 53.3% 采用率 (LangChain 2026 调研)
  - Self-evolving AI Agents Survey: 反馈闭环框架

+4 分提升

核心理念: Agent 在使用中变得更好
  1. Reviewer 结构化检查 (硬门控) — 输出必须过关
  2. 自动评估循环 — 无需人工打分
  3. 技能蒸馏 — 好的分析模式自动沉淀
  4. 性能追踪 — 持续监控质量趋势
"""

import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================
# 1. 结构化 Reviewer (硬门控)
# ============================================================

@dataclass
class ReviewCheckItem:
    """审查条目"""
    name: str
    passed: bool
    score: float       # 0-10
    detail: str = ""


@dataclass
class ReviewResult:
    """审查结果"""
    overall_score: float
    passed: bool
    checks: List[ReviewCheckItem] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)  # 阻断性问题
    suggestions: List[str] = field(default_factory=list)
    review_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "score": round(self.overall_score, 1),
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "score": c.score, "detail": c.detail}
                for c in self.checks
            ],
            "blockers": self.blockers,
            "suggestions": self.suggestions,
            "time_ms": round(self.review_time_ms, 1),
        }


class StructuredReviewer:
    """
    结构化审查器 — 硬门控

    vs V7 CriticAgent:
    - V7: LLM 打分 (主观、不稳定)
    - V8: 规则+LLM 混合审查 (确定性 + 灵活性)

    检查清单:
    1. [硬] 数据准确性 — 引用的数字必须存在于上下文
    2. [硬] 回答完整性 — 必须回答用户问题
    3. [软] 可执行性 — 建议是否具体
    4. [软] 格式质量 — 结构是否清晰
    5. [软] 简洁度 — 是否冗余
    """

    # 硬门控阈值
    HARD_GATE_THRESHOLD = 5.0   # 硬门控不过直接拒绝
    PASS_THRESHOLD = 6.5        # 总分阈值

    def review(self, answer: str, question: str,
               context_data: str = "", agent_outputs: Dict[str, str] = None) -> ReviewResult:
        """
        结构化审查

        Returns:
            ReviewResult
        """
        t0 = time.time()
        checks = []
        blockers = []
        suggestions = []

        # Check 1: 数据准确性 [硬门控]
        data_check = self._check_data_accuracy(answer, context_data)
        checks.append(data_check)
        if not data_check.passed:
            blockers.append(f"数据准确性不足: {data_check.detail}")

        # Check 2: 回答完整性 [硬门控]
        completeness = self._check_completeness(answer, question)
        checks.append(completeness)
        if not completeness.passed:
            blockers.append(f"回答不完整: {completeness.detail}")

        # Check 3: 可执行性 [软]
        actionability = self._check_actionability(answer, question)
        checks.append(actionability)
        if not actionability.passed:
            suggestions.append("建议更具体化: 增加时间节点、责任人、预期效果")

        # Check 4: 格式质量 [软]
        format_check = self._check_format(answer)
        checks.append(format_check)

        # Check 5: 简洁度 [软]
        conciseness = self._check_conciseness(answer)
        checks.append(conciseness)
        if not conciseness.passed:
            suggestions.append("回答可以更简洁，删除重复内容")

        # 总分
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        overall = sum(c.score * w for c, w in zip(checks, weights))

        # 硬门控: 任何阻断性问题都不通过
        passed = len(blockers) == 0 and overall >= self.PASS_THRESHOLD

        elapsed = (time.time() - t0) * 1000

        return ReviewResult(
            overall_score=overall,
            passed=passed,
            checks=checks,
            blockers=blockers,
            suggestions=suggestions,
            review_time_ms=elapsed,
        )

    def _check_data_accuracy(self, answer: str, context: str) -> ReviewCheckItem:
        """检查数据准确性"""
        # 提取回答中的数字
        answer_numbers = set(re.findall(r'\d+\.?\d*', answer))
        if not answer_numbers:
            return ReviewCheckItem("数据准确性", True, 7.0, "无数字引用")

        # 检查关键数字是否在上下文中
        context_numbers = set(re.findall(r'\d+\.?\d*', context))
        if not context_numbers:
            return ReviewCheckItem("数据准确性", True, 6.0, "上下文无数字基准")

        # 大数字验证 (>100 的数字更需要验证)
        big_numbers = [n for n in answer_numbers if float(n) > 100]
        verified = sum(1 for n in big_numbers if n in context_numbers)
        total_big = max(len(big_numbers), 1)

        accuracy_rate = verified / total_big
        score = accuracy_rate * 10
        passed = score >= self.HARD_GATE_THRESHOLD

        return ReviewCheckItem(
            "数据准确性", passed, score,
            f"大数字验证率: {accuracy_rate:.0%} ({verified}/{total_big})"
        )

    def _check_completeness(self, answer: str, question: str) -> ReviewCheckItem:
        """检查回答完整性"""
        q = question.lower()

        # 检查问题中的关键词是否被回答
        key_topics = []
        topic_patterns = {
            "客户": ['客户', '品牌', '厂商'],
            "风险": ['风险', '预警', '流失'],
            "增长": ['增长', '机会', '潜力'],
            "趋势": ['趋势', '变化', '走势'],
            "建议": ['建议', '策略', '方案'],
        }

        for topic, keywords in topic_patterns.items():
            if any(kw in q for kw in keywords):
                key_topics.append(topic)

        if not key_topics:
            return ReviewCheckItem("回答完整性", True, 7.0, "通用问题")

        covered = sum(1 for t in key_topics if t in answer or any(
            kw in answer for kw in topic_patterns[t]
        ))
        coverage = covered / max(len(key_topics), 1)
        score = coverage * 10
        passed = score >= self.HARD_GATE_THRESHOLD

        return ReviewCheckItem(
            "回答完整性", passed, score,
            f"话题覆盖: {covered}/{len(key_topics)} ({', '.join(key_topics)})"
        )

    def _check_actionability(self, answer: str, question: str) -> ReviewCheckItem:
        """检查可执行性"""
        # 只有涉及建议的问题才检查
        q = question.lower()
        needs_action = any(kw in q for kw in [
            '建议', '怎么办', '策略', '方案', '应该', 'CEO', '报告', '全面'
        ])

        if not needs_action:
            return ReviewCheckItem("可执行性", True, 7.0, "无需行动建议")

        # 检查是否有具体建议
        action_signals = [
            '建议', '方案', '行动', '步骤', '优先', '立即',
            '应该', '需要', '可以', '计划', '安排', '重点',
        ]
        action_count = sum(1 for s in action_signals if s in answer)
        score = min(action_count * 1.5, 10)
        passed = score >= 5.0

        return ReviewCheckItem(
            "可执行性", passed, score,
            f"行动信号: {action_count}个"
        )

    def _check_format(self, answer: str) -> ReviewCheckItem:
        """检查格式质量"""
        score = 5.0

        # 有结构 (标题/分段)
        if any(c in answer for c in ['#', '##', '**', '📊', '🔴', '💡', '⚠️']):
            score += 1.5

        # 有换行分段
        paragraphs = [p for p in answer.split('\n') if p.strip()]
        if len(paragraphs) >= 3:
            score += 1.0

        # 适当长度 (200-800字)
        length = len(answer)
        if 200 <= length <= 800:
            score += 1.5
        elif length < 50:
            score -= 2.0
        elif length > 1500:
            score -= 1.0

        score = max(0, min(10, score))
        return ReviewCheckItem("格式质量", score >= 5.0, score,
                              f"{len(answer)}字, {len(paragraphs)}段")

    def _check_conciseness(self, answer: str) -> ReviewCheckItem:
        """检查简洁度"""
        score = 7.0

        # 重复检测
        sentences = re.split(r'[。！？\n]', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if sentences:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.8:
                score -= 3.0

        # 过长惩罚
        if len(answer) > 1200:
            score -= 1.5
        elif len(answer) > 2000:
            score -= 3.0

        score = max(0, min(10, score))
        return ReviewCheckItem("简洁度", score >= 5.0, score,
                              f"唯一句比例: {len(set(sentences))}/{len(sentences)}" if sentences else "")


# ============================================================
# 2. 自动评估循环
# ============================================================

@dataclass
class EvalMetric:
    """评估指标"""
    name: str
    value: float
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)


class AutoEvalLoop:
    """
    自动评估循环

    借鉴 LangChain State of Agents 2026:
    - LLM-as-Judge: 53.3% 采用率
    - Human Review: 59.8%
    - Automated: 最佳实践

    功能:
    1. 每次分析自动评分
    2. 追踪质量趋势
    3. 异常检测 (质量突然下降)
    4. 定期报告
    """

    def __init__(self):
        self.history: List[EvalMetric] = []
        self.reviewer = StructuredReviewer()
        self.running_avg = 0.0
        self.eval_count = 0

    def evaluate(self, answer: str, question: str,
                 context: str = "", agent_outputs: Dict[str, str] = None,
                 metadata: Dict = None) -> Dict:
        """
        自动评估一次分析结果

        Returns:
            {
                "review": ReviewResult,
                "trend": "improving|stable|declining",
                "avg_score": float,
                "alert": str or None,
            }
        """
        # Reviewer 审查
        review = self.reviewer.review(answer, question, context, agent_outputs)

        # 更新统计
        self.eval_count += 1
        old_avg = self.running_avg
        self.running_avg = (
            old_avg * (self.eval_count - 1) + review.overall_score
        ) / self.eval_count

        # 记录
        metric = EvalMetric(
            name="auto_review",
            value=review.overall_score,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.history.append(metric)

        # 趋势分析
        trend = self._analyze_trend()

        # 异常检测
        alert = None
        if review.overall_score < old_avg - 2.0 and self.eval_count > 5:
            alert = f"⚠️ 质量异常下降: {review.overall_score:.1f} (均值 {old_avg:.1f})"

        return {
            "review": review.to_dict(),
            "trend": trend,
            "avg_score": round(self.running_avg, 2),
            "eval_count": self.eval_count,
            "alert": alert,
        }

    def _analyze_trend(self) -> str:
        """分析质量趋势"""
        if len(self.history) < 5:
            return "insufficient_data"

        recent = [h.value for h in self.history[-5:]]
        older = [h.value for h in self.history[-10:-5]] if len(self.history) >= 10 else recent

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg + 0.5:
            return "improving"
        elif recent_avg < older_avg - 0.5:
            return "declining"
        return "stable"

    def get_report(self) -> Dict:
        """获取评估报告"""
        if not self.history:
            return {"status": "no_data"}

        scores = [h.value for h in self.history]
        return {
            "total_evals": self.eval_count,
            "avg_score": round(self.running_avg, 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
            "trend": self._analyze_trend(),
            "pass_rate": f"{sum(1 for s in scores if s >= 6.5) / len(scores):.0%}",
            "recent_5": [round(s, 1) for s in scores[-5:]],
        }


# ============================================================
# 3. 技能蒸馏器 (SKILLRL)
# ============================================================

@dataclass
class DistilledSkill:
    """蒸馏后的技能"""
    skill_id: str
    name: str                    # 技能名称
    pattern: str                 # 触发模式
    strategy: str                # 分析策略
    source_questions: List[str]  # 来源问题
    success_rate: float = 0.0   # 成功率
    usage_count: int = 0
    created_at: float = 0.0


class SkillDistiller:
    """
    技能蒸馏器 — SKILLRL 启发

    从历史成功的分析轨迹中提取可复用的分析模式。

    流程:
    1. 收集高分分析轨迹
    2. 识别共同模式
    3. 蒸馏为技能
    4. 在新查询中自动应用

    示例:
    - "客户ABC分析" 轨迹 → 蒸馏为 "ABC分级分析技能"
    - "风险预警" 轨迹 → 蒸馏为 "多维风险扫描技能"
    """

    def __init__(self):
        self.skills: Dict[str, DistilledSkill] = {}
        self.trajectories: List[Dict] = []
        self._init_default_skills()

    def _init_default_skills(self):
        """初始化默认技能"""
        defaults = [
            DistilledSkill(
                skill_id="sk_abc",
                name="ABC分级分析",
                pattern="客户|分级|ABC|分类|等级",
                strategy="1.按金额降序排列 2.计算ABC占比 3.对比去年变动 4.识别升降级客户",
                source_questions=["客户分级情况"],
                success_rate=0.85,
                created_at=time.time(),
            ),
            DistilledSkill(
                skill_id="sk_risk",
                name="多维风险扫描",
                pattern="风险|预警|流失|异常|下滑",
                strategy="1.扫描>30%断崖客户 2.检查HHI集中度 3.量化风险金额 4.按紧急度排序",
                source_questions=["风险分析"],
                success_rate=0.80,
                created_at=time.time(),
            ),
            DistilledSkill(
                skill_id="sk_growth",
                name="增长机会识别",
                pattern="增长|机会|潜力|提升|扩大",
                strategy="1.对标行业增长率 2.识别低份额高潜客户 3.品类交叉分析 4.TAM计算",
                source_questions=["增长机会"],
                success_rate=0.75,
                created_at=time.time(),
            ),
            DistilledSkill(
                skill_id="sk_ceo",
                name="CEO级综合报告",
                pattern="CEO|总结|全面|概览|报告|综合",
                strategy="1.核心数字(3句话) 2.客户健康(ABC变动) 3.风险预警(Top3) 4.增长机会 5.行动项",
                source_questions=["CEO报告"],
                success_rate=0.90,
                created_at=time.time(),
            ),
        ]
        for skill in defaults:
            self.skills[skill.skill_id] = skill

    def record_trajectory(self, question: str, answer: str,
                          agents_used: List[str], score: float,
                          expert_outputs: Dict[str, str] = None):
        """记录分析轨迹"""
        self.trajectories.append({
            "question": question,
            "answer_preview": answer[:200],
            "agents": agents_used,
            "score": score,
            "expert_outputs": {k: v[:100] for k, v in (expert_outputs or {}).items()},
            "timestamp": time.time(),
        })

    def match_skills(self, question: str) -> List[DistilledSkill]:
        """匹配相关技能"""
        matched = []
        q = question.lower()
        for skill in self.skills.values():
            pattern_words = skill.pattern.split('|')
            if any(pw in q for pw in pattern_words):
                matched.append(skill)

        # 按成功率排序
        matched.sort(key=lambda s: s.success_rate, reverse=True)
        return matched[:3]

    def distill(self, min_score: float = 7.0, min_count: int = 3) -> List[DistilledSkill]:
        """
        从高分轨迹蒸馏新技能

        条件: 评分 >= min_score 且 类似问题 >= min_count
        """
        # 按问题类型分组
        groups = defaultdict(list)
        for traj in self.trajectories:
            if traj["score"] >= min_score:
                qtype = self._classify_question(traj["question"])
                groups[qtype].append(traj)

        new_skills = []
        for qtype, trajs in groups.items():
            if len(trajs) >= min_count:
                # 提取共同模式
                common_agents = self._find_common(
                    [t["agents"] for t in trajs]
                )
                # 创建新技能
                skill_id = f"sk_learned_{qtype}_{int(time.time())}"
                skill = DistilledSkill(
                    skill_id=skill_id,
                    name=f"学习: {qtype}分析",
                    pattern=qtype,
                    strategy=f"Agent组合: {','.join(common_agents)}",
                    source_questions=[t["question"][:50] for t in trajs[:3]],
                    success_rate=sum(t["score"] for t in trajs) / len(trajs) / 10,
                    created_at=time.time(),
                )
                self.skills[skill_id] = skill
                new_skills.append(skill)

        return new_skills

    def _classify_question(self, question: str) -> str:
        """简单问题分类"""
        q = question.lower()
        if any(kw in q for kw in ['风险', '预警', '流失']):
            return "risk"
        if any(kw in q for kw in ['增长', '机会', '策略']):
            return "growth"
        if any(kw in q for kw in ['CEO', '总结', '全面']):
            return "overview"
        return "analysis"

    def _find_common(self, lists: List[List[str]]) -> List[str]:
        """找出最常出现的元素"""
        counts = defaultdict(int)
        for lst in lists:
            for item in lst:
                counts[item] += 1
        return [k for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)]

    def get_stats(self) -> Dict:
        """技能统计"""
        return {
            "total_skills": len(self.skills),
            "learned_skills": sum(1 for s in self.skills.values() if s.skill_id.startswith("sk_learned")),
            "total_trajectories": len(self.trajectories),
            "avg_trajectory_score": (
                sum(t["score"] for t in self.trajectories) /
                max(len(self.trajectories), 1)
            ),
            "skills": [
                {
                    "name": s.name,
                    "success_rate": f"{s.success_rate:.0%}",
                    "pattern": s.pattern[:30],
                }
                for s in sorted(self.skills.values(),
                               key=lambda x: x.success_rate, reverse=True)
            ],
        }


# ============================================================
# 4. 全局实例
# ============================================================

_reviewer: Optional[StructuredReviewer] = None
_eval_loop: Optional[AutoEvalLoop] = None
_distiller: Optional[SkillDistiller] = None


def get_reviewer() -> StructuredReviewer:
    global _reviewer
    if _reviewer is None:
        _reviewer = StructuredReviewer()
    return _reviewer


def get_eval_loop() -> AutoEvalLoop:
    global _eval_loop
    if _eval_loop is None:
        _eval_loop = AutoEvalLoop()
    return _eval_loop


def get_distiller() -> SkillDistiller:
    global _distiller
    if _distiller is None:
        _distiller = SkillDistiller()
    return _distiller
