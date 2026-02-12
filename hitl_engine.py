#!/usr/bin/env python3
"""
MRARFAI v3.3 — Enhanced Human-in-the-Loop Engine
==================================================
Google Agent Pattern #7: Human-in-the-Loop

置信度分级决策:
  HIGH (>0.8)   → 自动执行，事后展示
  MEDIUM (0.5-0.8) → 展示草稿，请求确认
  LOW (<0.5)    → 暂停执行，请求人工输入

触发维度:
  1. 风险等级 — 高风险客户/金额异常
  2. 数据置信度 — 数据缺失/矛盾/时效性
  3. 决策影响 — 战略建议/资源分配/客户处理
  4. 历史准确率 — 过去相似问题的反馈评分
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 数据模型
# ============================================================

class ConfidenceLevel(Enum):
    HIGH = "high"         # 自动执行
    MEDIUM = "medium"     # 请求确认
    LOW = "low"           # 暂停等待人工

    @staticmethod
    def from_score(score: float) -> "ConfidenceLevel":
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


class ActionType(Enum):
    AUTO_EXECUTE = "auto_execute"       # 自动执行
    AWAIT_CONFIRMATION = "await_confirm" # 等待确认
    AWAIT_INPUT = "await_input"         # 等待人工输入
    ESCALATE = "escalate"               # 升级到管理层


@dataclass
class HITLTrigger:
    """HITL触发事件"""
    trigger_id: str
    category: str            # risk / data_quality / decision_impact / accuracy
    severity: str            # critical / warning / info
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence_impact: float = 0.0  # 负数=降低置信度
    action_required: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.trigger_id,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "confidence_impact": self.confidence_impact,
            "action_required": self.action_required,
        }


@dataclass
class HITLDecision:
    """HITL决策结果"""
    confidence_score: float
    confidence_level: ConfidenceLevel
    action: ActionType
    triggers: List[HITLTrigger]
    reasoning: List[str]
    requires_approval: bool
    approval_items: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "confidence_score": round(self.confidence_score, 3),
            "confidence_level": self.confidence_level.value,
            "action": self.action.value,
            "triggers": [t.to_dict() for t in self.triggers],
            "reasoning": self.reasoning,
            "requires_approval": self.requires_approval,
            "approval_items": self.approval_items,
        }


@dataclass
class ApprovalRequest:
    """审批请求"""
    request_id: str
    question: str
    proposed_answer: str
    triggers: List[dict]
    confidence_score: float
    created_at: str = ""
    status: str = "pending"   # pending / approved / rejected / modified
    approver_notes: str = ""
    resolved_at: str = ""

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "question": self.question[:100],
            "proposed_answer": self.proposed_answer[:200],
            "triggers": self.triggers,
            "confidence": self.confidence_score,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "notes": self.approver_notes,
        }


# ============================================================
# 置信度评估引擎
# ============================================================

class ConfidenceScorer:
    """
    多维度置信度评分
    
    评分维度及权重:
    - data_quality (0.3): 数据完整性和时效性
    - risk_level (0.25): 风险发现的严重程度  
    - decision_impact (0.2): 建议的影响范围
    - historical_accuracy (0.15): 历史相似问题的准确率
    - answer_quality (0.1): 报告质量评分
    """

    WEIGHTS = {
        "data_quality": 0.30,
        "risk_level": 0.25,
        "decision_impact": 0.20,
        "historical_accuracy": 0.15,
        "answer_quality": 0.10,
    }

    @staticmethod
    def score(
        question: str,
        answer: str,
        expert_outputs: Dict[str, str],
        context_data: str,
        critique_score: float = None,
        health_scores: List[dict] = None,
        historical_avg_score: float = None,
    ) -> Tuple[float, List[HITLTrigger]]:
        """
        计算综合置信度分数
        
        Returns:
            (confidence_score, triggers)
        """
        scores = {}
        triggers = []
        trigger_idx = 0

        # ① 数据质量
        dq_score, dq_triggers = ConfidenceScorer._assess_data_quality(
            context_data, question, trigger_idx
        )
        scores["data_quality"] = dq_score
        triggers.extend(dq_triggers)
        trigger_idx += len(dq_triggers)

        # ② 风险等级
        rl_score, rl_triggers = ConfidenceScorer._assess_risk_level(
            expert_outputs, health_scores, trigger_idx
        )
        scores["risk_level"] = rl_score
        triggers.extend(rl_triggers)
        trigger_idx += len(rl_triggers)

        # ③ 决策影响
        di_score, di_triggers = ConfidenceScorer._assess_decision_impact(
            question, answer, trigger_idx
        )
        scores["decision_impact"] = di_score
        triggers.extend(di_triggers)
        trigger_idx += len(di_triggers)

        # ④ 历史准确率
        if historical_avg_score is not None:
            scores["historical_accuracy"] = min(1.0, historical_avg_score / 10)
        else:
            scores["historical_accuracy"] = 0.7  # 默认中等

        # ⑤ 报告质量
        if critique_score is not None:
            scores["answer_quality"] = min(1.0, critique_score / 10)
        else:
            scores["answer_quality"] = 0.7  # 默认

        # 加权计算
        final = sum(
            scores.get(dim, 0.7) * weight
            for dim, weight in ConfidenceScorer.WEIGHTS.items()
        )

        return round(final, 3), triggers

    @staticmethod
    def _assess_data_quality(context_data: str, question: str,
                              idx_start: int) -> Tuple[float, List[HITLTrigger]]:
        """评估数据质量"""
        score = 0.9
        triggers = []

        # 数据量过少
        if len(context_data) < 100:
            score -= 0.3
            triggers.append(HITLTrigger(
                trigger_id=f"hitl_{idx_start + len(triggers)}",
                category="data_quality",
                severity="warning",
                message="数据量不足，可能影响分析准确性",
                confidence_impact=-0.3,
                action_required="确认数据范围是否完整",
            ))

        # 包含"无数据"/"未找到"等标记
        no_data_markers = ["无数据", "未找到", "暂无", "数据缺失", "N/A"]
        for marker in no_data_markers:
            if marker in context_data:
                score -= 0.2
                triggers.append(HITLTrigger(
                    trigger_id=f"hitl_{idx_start + len(triggers)}",
                    category="data_quality",
                    severity="warning",
                    message=f"数据中包含'{marker}'标记，部分数据可能缺失",
                    confidence_impact=-0.2,
                ))
                break

        return max(0, min(1, score)), triggers

    @staticmethod
    def _assess_risk_level(expert_outputs: Dict[str, str],
                            health_scores: List[dict] = None,
                            idx_start: int = 0) -> Tuple[float, List[HITLTrigger]]:
        """评估风险等级"""
        score = 0.9
        triggers = []

        # 检查风控专家输出
        risk_output = ""
        for name, output in expert_outputs.items():
            if "风控" in name or "risk" in name.lower():
                risk_output = output
                break

        if "[HIGH_RISK_ALERT]" in risk_output:
            score -= 0.4
            # 提取风险客户
            customers = re.findall(r'[：:]\s*(.+?)[\n,，]', risk_output[:300])
            triggers.append(HITLTrigger(
                trigger_id=f"hitl_{idx_start + len(triggers)}",
                category="risk",
                severity="critical",
                message="风控专家发现高风险客户",
                details={"customers": customers[:3]},
                confidence_impact=-0.4,
                action_required="确认是否需要立即启动客户挽留计划",
            ))

        if "高风险" in risk_output or "断崖" in risk_output:
            if score > 0.7:
                score -= 0.2
            triggers.append(HITLTrigger(
                trigger_id=f"hitl_{idx_start + len(triggers)}",
                category="risk",
                severity="warning",
                message="存在潜在高风险信号",
                confidence_impact=-0.2,
            ))

        # 检查健康评分
        if health_scores:
            critical_count = sum(
                1 for h in health_scores
                if h.get("score", 100) < 40
            )
            if critical_count > 0:
                score -= 0.15 * critical_count
                triggers.append(HITLTrigger(
                    trigger_id=f"hitl_{idx_start + len(triggers)}",
                    category="risk",
                    severity="warning",
                    message=f"{critical_count}个客户健康评分低于40",
                    details={"critical_count": critical_count},
                    confidence_impact=-0.15 * critical_count,
                ))

        return max(0, min(1, score)), triggers

    @staticmethod
    def _assess_decision_impact(question: str, answer: str,
                                 idx_start: int = 0) -> Tuple[float, List[HITLTrigger]]:
        """评估决策影响"""
        score = 0.9
        triggers = []

        # 高影响关键词
        high_impact_keywords = {
            "战略": 0.15, "裁员": 0.3, "预算": 0.2, "投资": 0.2,
            "合同": 0.25, "终止": 0.25, "重组": 0.2,
            "放弃": 0.2, "停止": 0.15, "全面": 0.1,
        }

        for kw, impact in high_impact_keywords.items():
            if kw in question or kw in answer:
                score -= impact
                triggers.append(HITLTrigger(
                    trigger_id=f"hitl_{idx_start + len(triggers)}",
                    category="decision_impact",
                    severity="warning" if impact < 0.2 else "critical",
                    message=f"涉及'{kw}'类高影响决策",
                    confidence_impact=-impact,
                    action_required=f"确认'{kw}'相关建议是否需要管理层审批",
                ))
                break  # 只取最高影响的

        # 金额级别检测
        amounts = re.findall(r'(\d+(?:\.\d+)?)\s*[万亿]', answer)
        for amt_str in amounts:
            amt = float(amt_str)
            if "亿" in answer[answer.index(amt_str):answer.index(amt_str)+10]:
                amt *= 10000
            if amt >= 1000:  # 超过1000万
                score -= 0.15
                triggers.append(HITLTrigger(
                    trigger_id=f"hitl_{idx_start + len(triggers)}",
                    category="decision_impact",
                    severity="warning",
                    message=f"涉及大额资金（{amt_str}万+）",
                    confidence_impact=-0.15,
                    action_required="确认金额敏感建议",
                ))
                break

        return max(0, min(1, score)), triggers


# ============================================================
# HITL决策引擎
# ============================================================

class HITLEngine:
    """
    Human-in-the-Loop 决策引擎
    
    核心逻辑:
    1. 收集所有维度的置信度信号
    2. 计算综合置信度分数
    3. 根据分数决定执行策略
    4. 生成审批请求（如需要）
    """

    # 审批队列（内存，可升级为SQLite）
    _pending_approvals: Dict[str, ApprovalRequest] = {}

    @staticmethod
    def evaluate(
        question: str,
        answer: str,
        expert_outputs: Dict[str, str],
        context_data: str = "",
        critique_score: float = None,
        health_scores: List[dict] = None,
        historical_avg_score: float = None,
    ) -> HITLDecision:
        """
        评估是否需要人工介入
        
        Returns:
            HITLDecision
        """
        # 计算置信度
        conf_score, triggers = ConfidenceScorer.score(
            question, answer, expert_outputs, context_data,
            critique_score, health_scores, historical_avg_score,
        )

        level = ConfidenceLevel.from_score(conf_score)
        reasoning = []

        # 决定行动
        if level == ConfidenceLevel.HIGH:
            action = ActionType.AUTO_EXECUTE
            reasoning.append(f"置信度 {conf_score:.2f} ≥ 0.8，自动执行")
            requires_approval = False
        elif level == ConfidenceLevel.MEDIUM:
            action = ActionType.AWAIT_CONFIRMATION
            reasoning.append(f"置信度 {conf_score:.2f} 在 0.5-0.8 区间，需要确认")
            requires_approval = True
        else:
            action = ActionType.AWAIT_INPUT
            reasoning.append(f"置信度 {conf_score:.2f} < 0.5，需要人工审查")
            requires_approval = True

        # 关键触发的reasoning
        critical_triggers = [t for t in triggers if t.severity == "critical"]
        if critical_triggers:
            reasoning.append(
                f"发现 {len(critical_triggers)} 个关键风险: "
                + "; ".join(t.message for t in critical_triggers[:2])
            )
            # critical触发强制升级
            if len(critical_triggers) >= 2:
                action = ActionType.ESCALATE
                requires_approval = True
                reasoning.append("多个关键风险触发管理层升级")

        # 生成审批项
        approval_items = []
        if requires_approval:
            for t in triggers:
                if t.severity in ("critical", "warning") and t.action_required:
                    approval_items.append({
                        "trigger_id": t.trigger_id,
                        "item": t.action_required,
                        "severity": t.severity,
                        "status": "pending",
                    })

        return HITLDecision(
            confidence_score=conf_score,
            confidence_level=level,
            action=action,
            triggers=triggers,
            reasoning=reasoning,
            requires_approval=requires_approval,
            approval_items=approval_items,
        )

    @staticmethod
    def create_approval_request(
        question: str,
        answer: str,
        decision: HITLDecision,
    ) -> ApprovalRequest:
        """创建审批请求"""
        import uuid
        req = ApprovalRequest(
            request_id=str(uuid.uuid4())[:8],
            question=question,
            proposed_answer=answer,
            triggers=[t.to_dict() for t in decision.triggers if t.severity != "info"],
            confidence_score=decision.confidence_score,
            created_at=datetime.now().isoformat(),
        )
        HITLEngine._pending_approvals[req.request_id] = req
        return req

    @staticmethod
    def resolve_approval(request_id: str, status: str,
                          notes: str = "") -> Optional[ApprovalRequest]:
        """处理审批"""
        req = HITLEngine._pending_approvals.get(request_id)
        if not req:
            return None
        req.status = status
        req.approver_notes = notes
        req.resolved_at = datetime.now().isoformat()
        return req

    @staticmethod
    def get_pending_approvals() -> List[ApprovalRequest]:
        """获取待审批列表"""
        return [
            r for r in HITLEngine._pending_approvals.values()
            if r.status == "pending"
        ]

    @staticmethod
    def clear_resolved(max_age_hours: int = 24):
        """清理已处理的审批"""
        cutoff = datetime.now()
        to_remove = []
        for rid, req in HITLEngine._pending_approvals.items():
            if req.status != "pending" and req.resolved_at:
                try:
                    resolved = datetime.fromisoformat(req.resolved_at)
                    if (cutoff - resolved).total_seconds() > max_age_hours * 3600:
                        to_remove.append(rid)
                except ValueError:
                    pass
        for rid in to_remove:
            del HITLEngine._pending_approvals[rid]


# ============================================================
# 快捷API
# ============================================================

def evaluate_hitl(
    question: str,
    answer: str,
    expert_outputs: Dict[str, str],
    context_data: str = "",
    critique_score: float = None,
    health_scores: List[dict] = None,
    enabled: bool = True,
) -> Optional[dict]:
    """
    快捷入口 — 评估HITL决策
    
    Returns:
        HITLDecision dict or None (if disabled)
    """
    if not enabled:
        return None

    decision = HITLEngine.evaluate(
        question, answer, expert_outputs, context_data,
        critique_score, health_scores,
    )

    return decision.to_dict()
