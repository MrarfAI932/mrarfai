#!/usr/bin/env python3
"""
MRARFAI V9.0 — Multi-Dimensional Evaluation Framework
=========================================================
综合七篇论文的评估维度:
  1. RLM (Recursive LM)     → 递归分析质量 + 数据覆盖率
  2. AWM (Agent World Model) → SQL 状态验证 + 环境一致性
  3. EnCompass               → 搜索效率 (accuracy / LLM calls)
  4. Reasoning Chains        → 推理结构性 + 模板遵循度
  5. Memory Survey           → 记忆利用率 + 检索质量
  6. LatentLens              → 可解释性 + 意图映射清晰度
  7. AgentSkiller            → All-or-Nothing 端到端评分

升级亮点 vs v8.0 evals.py:
  - 7维评估 (原3维)
  - AWM 合成环境自动生成测试用例 (原手工50个 → 1000+)
  - 权重可调评分系统
  - 实时评估仪表盘数据输出
  - 与现有 evals.py EvalCase/EvalResult 兼容

运行方式:
  python evals_v9.py                        # 运行全量评估
  python evals_v9.py --dimension rlm        # 仅 RLM 维度
  python evals_v9.py --awm-generate 100     # 生成100个合成用例
  python evals_v9.py --report               # 生成完整报告
  python evals_v9.py --compare v8           # 对比 v8 基线

兼容:
  v1.0 evals.py — EvalCase / EvalResult / EvalReport 数据结构
  v9.0 awm_env_factory.py — 合成环境 + SQL 验证
  v9.0 rlm_engine.py — 递归分析评估
  v9.0 search_engine.py — 搜索效率指标
  v9.0 reasoning_templates.py — 推理质量评估
  v9.0 memory_v9.py — 记忆系统评估
  v9.0 interpretability_layer.py — 可解释性评估
"""

import json
import time
import hashlib
import logging
import statistics
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

logger = logging.getLogger("mrarfai.evals_v9")


# ╔══════════════════════════════════════════╗
# ║  Part 1: 数据结构 (兼容 evals.py v1)     ║
# ╚══════════════════════════════════════════╝

class EvalDimension(Enum):
    """七维评估维度 — 每维对应一篇论文"""
    RLM             = "rlm"
    AWM             = "awm"
    ENCOMPASS       = "encompass"
    REASONING       = "reasoning"
    MEMORY          = "memory"
    INTERPRETABILITY = "interpretability"
    E2E             = "e2e"


@dataclass
class DimensionConfig:
    """单个评估维度配置"""
    dimension: EvalDimension
    weight: float
    description: str
    scoring_method: str
    passing_threshold: float = 0.7
    paper_ref: str = ""


@dataclass
class V9EvalCase:
    """V9.0 评估用例 (向后兼容 EvalCase)"""
    id: str
    name: str
    category: str
    input_data: Dict[str, Any]
    expected: Dict[str, Any]
    dimensions: List[EvalDimension] = field(default_factory=lambda: list(EvalDimension))
    difficulty: str = "medium"
    source: str = "manual"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_legacy(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "category": self.category,
            "input_data": self.input_data, "expected": self.expected, "tags": self.tags,
        }


@dataclass
class DimensionScore:
    """单维度评分结果"""
    dimension: EvalDimension
    score: float
    weight: float
    weighted_score: float
    passed: bool
    details: str = ""
    sub_scores: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0


@dataclass
class V9EvalResult:
    """V9.0 多维评估结果"""
    case_id: str
    total_score: float
    passed: bool
    dimension_scores: Dict[str, DimensionScore] = field(default_factory=dict)
    elapsed_ms: float = 0
    timestamp: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_legacy(self) -> Dict:
        return {
            "case_id": self.case_id, "passed": self.passed,
            "score": self.total_score, "details": self.summary(),
            "elapsed_ms": self.elapsed_ms, "errors": self.errors,
        }

    def summary(self) -> str:
        lines = [f"Total: {self.total_score:.3f} ({'PASS' if self.passed else 'FAIL'})"]
        for dim_name, ds in self.dimension_scores.items():
            flag = "✅" if ds.passed else "❌"
            lines.append(f"  {flag} {dim_name}: {ds.score:.3f} (w={ds.weight})")
        return "\n".join(lines)


@dataclass
class V9EvalReport:
    """V9.0 综合评估报告"""
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    avg_score: float = 0.0
    dimension_averages: Dict[str, float] = field(default_factory=dict)
    results: List[V9EvalResult] = field(default_factory=list)
    version: str = "9.0"
    run_timestamp: str = ""
    config_snapshot: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.run_timestamp:
            self.run_timestamp = datetime.now().isoformat()

    def pass_rate(self) -> float:
        return self.passed_cases / max(self.total_cases, 1)

    def to_dict(self) -> Dict:
        return {
            "version": self.version, "run_timestamp": self.run_timestamp,
            "total_cases": self.total_cases, "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": f"{self.pass_rate():.1%}",
            "avg_score": round(self.avg_score, 4),
            "dimension_averages": {k: round(v, 4) for k, v in self.dimension_averages.items()},
            "config": self.config_snapshot,
        }


# ╔══════════════════════════════════════════╗
# ║  Part 2: 维度评估器 (每篇论文一个)        ║
# ╚══════════════════════════════════════════╝

class DimensionEvaluator(ABC):
    @property
    @abstractmethod
    def dimension(self) -> EvalDimension: ...

    @abstractmethod
    def evaluate(self, test_case: V9EvalCase,
                 agent_output: Dict[str, Any]) -> DimensionScore: ...


class RLMEvaluator(DimensionEvaluator):
    """RLM 维度 — 递归分析质量 (arXiv:2512.24601)
    子指标: data_coverage, recursion_depth, merge_quality, cost_efficiency"""
    dimension = EvalDimension.RLM

    def evaluate(self, test_case, agent_output):
        t0 = time.time()
        ss = {}
        total_rows = test_case.input_data.get("total_rows", 1)
        analyzed = agent_output.get("rlm_analyzed_rows", 0)
        ss["data_coverage"] = min(analyzed / max(total_rows, 1), 1.0)

        depth = agent_output.get("rlm_recursion_depth", 0)
        if depth == 0: ss["recursion_depth"] = 0.0
        elif 2 <= depth <= 4: ss["recursion_depth"] = 1.0
        elif depth == 1: ss["recursion_depth"] = 0.6
        elif depth == 5: ss["recursion_depth"] = 0.8
        else: ss["recursion_depth"] = max(0.3, 1.0 - abs(depth - 3) * 0.15)

        exp_keys = set(test_case.expected.get("required_merge_fields", []))
        merged = set(agent_output.get("rlm_merged_fields", []))
        ss["merge_quality"] = (len(merged & exp_keys) / len(exp_keys)) if exp_keys else (1.0 if agent_output.get("rlm_merge_complete") else 0.5)

        tokens = agent_output.get("rlm_total_tokens", 0)
        baseline = test_case.metadata.get("baseline_tokens", tokens * 1.5)
        ss["cost_efficiency"] = min(1.0, max(0.0, 1.5 - tokens / max(baseline, 1))) if baseline > 0 and tokens > 0 else 0.5

        score = ss["data_coverage"] * 0.35 + ss["recursion_depth"] * 0.20 + ss["merge_quality"] * 0.30 + ss["cost_efficiency"] * 0.15
        return DimensionScore(dimension=EvalDimension.RLM, score=round(score, 4), weight=0.0,
            weighted_score=0.0, passed=score >= 0.7,
            details=f"cov={ss['data_coverage']:.2f} depth={depth} merge={ss['merge_quality']:.2f}",
            sub_scores=ss, elapsed_ms=(time.time() - t0) * 1000)


class AWMEvaluator(DimensionEvaluator):
    """AWM 维度 — SQL 状态验证 + 环境一致性 (Snowflake)
    子指标: sql_correctness, state_consistency, tool_usage, schema_awareness"""
    dimension = EvalDimension.AWM

    def evaluate(self, test_case, agent_output):
        t0 = time.time()
        ss = {}
        sql_res = agent_output.get("sql_results", [])
        sql_exp = test_case.expected.get("sql_expected", [])
        if sql_exp:
            correct = sum(1 for exp in sql_exp if any(self._sql_match(exp, r) for r in sql_res))
            ss["sql_correctness"] = correct / len(sql_exp)
        else:
            ss["sql_correctness"] = 0.0 if agent_output.get("sql_errors") else 1.0

        exp_state = test_case.expected.get("db_state", {})
        act_state = agent_output.get("db_state", {})
        ss["state_consistency"] = (sum(1 for k, v in exp_state.items() if act_state.get(k) == v) / len(exp_state)) if exp_state else 1.0

        exp_tools = test_case.expected.get("tool_sequence", [])
        act_tools = agent_output.get("tool_calls", [])
        ss["tool_usage"] = self._lcs_sim(exp_tools, act_tools) if exp_tools else 0.8

        t_accessed = set(agent_output.get("tables_accessed", []))
        t_expected = set(test_case.expected.get("relevant_tables", []))
        if t_expected:
            p = len(t_accessed & t_expected) / max(len(t_accessed), 1)
            r = len(t_accessed & t_expected) / len(t_expected)
            ss["schema_awareness"] = 2 * p * r / max(p + r, 0.001)
        else:
            ss["schema_awareness"] = 0.8

        score = ss["sql_correctness"] * 0.35 + ss["state_consistency"] * 0.30 + ss["tool_usage"] * 0.20 + ss["schema_awareness"] * 0.15
        return DimensionScore(dimension=EvalDimension.AWM, score=round(score, 4), weight=0.0,
            weighted_score=0.0, passed=score >= 0.7,
            details=f"sql={ss['sql_correctness']:.2f} state={ss['state_consistency']:.2f}",
            sub_scores=ss, elapsed_ms=(time.time() - t0) * 1000)

    @staticmethod
    def _sql_match(exp, act):
        if exp.get("row_count") is not None and act.get("row_count") != exp["row_count"]: return False
        if exp.get("columns") and set(exp["columns"]) != set(act.get("columns", [])): return False
        if exp.get("checksum") and act.get("checksum") != exp["checksum"]: return False
        return True

    @staticmethod
    def _lcs_sim(a, b):
        if not a: return 1.0
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n] / m


class EnCompassEvaluator(DimensionEvaluator):
    """EnCompass 维度 — 搜索效率 (MIT CSAIL, NeurIPS 2025)
    子指标: accuracy, budget_utilization, search_depth, pruning_quality"""
    dimension = EvalDimension.ENCOMPASS

    def evaluate(self, test_case, agent_output):
        t0 = time.time()
        ss = {}
        exp_ans = test_case.expected.get("answer")
        act_ans = agent_output.get("answer")
        if exp_ans is not None:
            if act_ans == exp_ans: ss["accuracy"] = 1.0
            elif isinstance(exp_ans, (int, float)) and isinstance(act_ans, (int, float)):
                tol = test_case.metadata.get("tolerance", 0.05)
                diff = abs(act_ans - exp_ans) / max(abs(exp_ans), 1e-9)
                ss["accuracy"] = max(0.0, 1.0 - diff / tol) if diff < tol else 0.0
            else: ss["accuracy"] = 0.0
        else: ss["accuracy"] = 0.5

        calls = agent_output.get("search_llm_calls", 1)
        budget = test_case.metadata.get("search_budget", 50)
        util = calls / max(budget, 1)
        ss["budget_utilization"] = max(0.3, 1.0 - util * 0.5) if ss["accuracy"] >= 0.9 else ss["accuracy"] * (1.0 - util * 0.3)

        max_d = agent_output.get("search_max_depth", 0)
        opt_d = test_case.metadata.get("optimal_search_depth", 3)
        r = max_d / max(opt_d, 1)
        ss["search_depth"] = 1.0 if 0.8 <= r <= 1.3 else max(0.2, 1.0 - abs(r - 1.0) * 0.5)

        tot_br = agent_output.get("search_total_branches", 0)
        pru_br = agent_output.get("search_pruned_branches", 0)
        if tot_br > 0:
            pr = pru_br / tot_br
            ss["pruning_quality"] = 1.0 if 0.4 <= pr <= 0.85 else (0.5 + pr if pr < 0.4 else max(0.3, 1.0 - (pr - 0.85) * 2))
        else: ss["pruning_quality"] = 0.5

        score = ss["accuracy"] * 0.40 + ss["budget_utilization"] * 0.25 + ss["search_depth"] * 0.15 + ss["pruning_quality"] * 0.20
        return DimensionScore(dimension=EvalDimension.ENCOMPASS, score=round(score, 4), weight=0.0,
            weighted_score=0.0, passed=score >= 0.65,
            details=f"acc={ss['accuracy']:.2f} calls={calls}/{budget}",
            sub_scores=ss, elapsed_ms=(time.time() - t0) * 1000)


class ReasoningEvaluator(DimensionEvaluator):
    """Reasoning 维度 — 推理链质量 (arXiv:2602.09276)
    子指标: template_adherence, step_coherence, conclusion_validity, reasoning_efficiency"""
    dimension = EvalDimension.REASONING
    TEMPLATES = {"diagnostic", "trend_analysis", "comparative", "risk_assessment",
                 "root_cause", "prediction", "anomaly_investigation"}

    def evaluate(self, test_case, agent_output):
        t0 = time.time()
        ss = {}
        trace = agent_output.get("reasoning_trace", {})
        steps = trace.get("steps", [])
        tmpl = trace.get("template", "")
        exp_tmpl = test_case.expected.get("reasoning_template", "")

        ss["template_adherence"] = (1.0 if tmpl == exp_tmpl else 0.5 if tmpl in self.TEMPLATES else 0.0) if exp_tmpl else (1.0 if tmpl in self.TEMPLATES else 0.6)

        if len(steps) >= 2:
            coh = 0
            for i in range(1, len(steps)):
                pw = set(str(steps[i-1].get("output", "")).lower().split()) - {"的", "和", "是", "了", "在"}
                cw = set(str(steps[i].get("input", "")).lower().split())
                if len(pw & cw) >= min(3, len(pw) * 0.3): coh += 1
            ss["step_coherence"] = coh / max(len(steps) - 1, 1)
        else: ss["step_coherence"] = 0.5 if steps else 0.0

        conc = agent_output.get("conclusion", "")
        must = test_case.expected.get("conclusion_must_contain", [])
        ss["conclusion_validity"] = (sum(1 for k in must if k.lower() in conc.lower()) / len(must)) if must else (0.8 if len(conc) > 20 else 0.3)

        n = len(steps)
        opt = {"easy": 3, "medium": 5, "hard": 7, "extreme": 10}.get(test_case.difficulty, 5)
        ss["reasoning_efficiency"] = 0.0 if n == 0 else (1.0 if abs(n - opt) <= 1 else max(0.2, 1.0 - abs(n - opt) / opt * 0.4))

        score = ss["template_adherence"] * 0.20 + ss["step_coherence"] * 0.30 + ss["conclusion_validity"] * 0.35 + ss["reasoning_efficiency"] * 0.15
        return DimensionScore(dimension=EvalDimension.REASONING, score=round(score, 4), weight=0.0,
            weighted_score=0.0, passed=score >= 0.65,
            details=f"tmpl={tmpl or 'none'} steps={n} coh={ss['step_coherence']:.2f}",
            sub_scores=ss, elapsed_ms=(time.time() - t0) * 1000)


class MemoryEvaluator(DimensionEvaluator):
    """Memory 维度 — 记忆系统质量 (Memory Survey 2026)
    子指标: retrieval_relevance, formation_quality, cross_agent_sharing, memory_evolution"""
    dimension = EvalDimension.MEMORY

    def evaluate(self, test_case, agent_output):
        t0 = time.time()
        ss = {}
        ops = agent_output.get("memory_operations", {})

        ret = ops.get("retrieved_memories", [])
        rel_ids = set(test_case.expected.get("relevant_memory_ids", []))
        if rel_ids:
            rids = set(m.get("id", "") for m in ret)
            p = len(rids & rel_ids) / max(len(rids), 1)
            r = len(rids & rel_ids) / len(rel_ids)
            ss["retrieval_relevance"] = 2 * p * r / max(p + r, 0.001)
        else: ss["retrieval_relevance"] = 0.7 if ret else 0.3

        formed = ops.get("formed_memories", [])
        if formed:
            qs = [min(0.4 * (m.get("abstraction_level", 0) >= 2) + 0.3 * bool(m.get("key_entities")) + 0.3 * bool(m.get("temporal_context")), 1.0) for m in formed]
            ss["formation_quality"] = statistics.mean(qs)
        else: ss["formation_quality"] = 0.3

        shared = ops.get("cross_agent_shares", 0)
        exp_shares = test_case.expected.get("expected_memory_shares", 0)
        ss["cross_agent_sharing"] = min(shared / exp_shares, 1.0) if exp_shares > 0 else 0.7

        upd = ops.get("memory_updates", 0)
        cons = ops.get("memory_consolidations", 0)
        ss["memory_evolution"] = min(0.5 * min(upd / 3, 1.0) + 0.5 * min(cons / 2, 1.0), 1.0) if upd > 0 or cons > 0 else 0.4

        score = ss["retrieval_relevance"] * 0.35 + ss["formation_quality"] * 0.25 + ss["cross_agent_sharing"] * 0.20 + ss["memory_evolution"] * 0.20
        return DimensionScore(dimension=EvalDimension.MEMORY, score=round(score, 4), weight=0.0,
            weighted_score=0.0, passed=score >= 0.6,
            details=f"ret={ss['retrieval_relevance']:.2f} form={ss['formation_quality']:.2f}",
            sub_scores=ss, elapsed_ms=(time.time() - t0) * 1000)


class InterpretabilityEvaluator(DimensionEvaluator):
    """Interpretability 维度 — 可解释性 (LatentLens, arXiv:2602.07715)
    子指标: intent_clarity, trace_readability, attribution_accuracy, confidence_calibration"""
    dimension = EvalDimension.INTERPRETABILITY

    def evaluate(self, test_case, agent_output):
        t0 = time.time()
        ss = {}
        expl = agent_output.get("explanation", {})

        intent = expl.get("intent", {})
        ss["intent_clarity"] = min(0.3 * bool(intent.get("category")) + 0.3 * (intent.get("confidence", 0) > 0.7) + 0.2 * bool(intent.get("sub_intents")) + 0.2 * bool(intent.get("decomposition_reasoning")), 1.0) if intent else 0.0

        steps = expl.get("process_steps", [])
        ss["trace_readability"] = (sum(1 for s in steps if len(str(s.get("description", ""))) > 10) / len(steps)) if steps else 0.0

        attrs = expl.get("attributions", [])
        exp_src = test_case.expected.get("attribution_sources", [])
        if exp_src and attrs:
            a_src = {a.get("source", "") for a in attrs}
            ss["attribution_accuracy"] = len(a_src & set(exp_src)) / len(exp_src)
        else: ss["attribution_accuracy"] = 0.6 if attrs else 0.0

        conf = agent_output.get("confidence", 0.5)
        correct = 1.0 if agent_output.get("is_correct") else 0.0
        ss["confidence_calibration"] = max(0.0, 1.0 - abs(conf - correct))

        score = ss["intent_clarity"] * 0.30 + ss["trace_readability"] * 0.25 + ss["attribution_accuracy"] * 0.25 + ss["confidence_calibration"] * 0.20
        return DimensionScore(dimension=EvalDimension.INTERPRETABILITY, score=round(score, 4), weight=0.0,
            weighted_score=0.0, passed=score >= 0.55,
            details=f"intent={ss['intent_clarity']:.2f} trace={ss['trace_readability']:.2f}",
            sub_scores=ss, elapsed_ms=(time.time() - t0) * 1000)


class E2EEvaluator(DimensionEvaluator):
    """E2E 维度 — AgentSkiller All-or-Nothing
    子指标: action_sequence_correct, final_answer_correct, no_hallucination, task_completion"""
    dimension = EvalDimension.E2E

    def evaluate(self, test_case, agent_output):
        t0 = time.time()
        ss = {}
        exp_act = test_case.expected.get("action_sequence", [])
        act_act = agent_output.get("actions", [])
        if exp_act:
            ok = all(any(a.get("type") == e.get("type") and a.get("target") == e.get("target") for a in act_act) for e in exp_act)
            ss["action_sequence_correct"] = 1.0 if ok else 0.0
        else: ss["action_sequence_correct"] = 0.5

        exp_ans = test_case.expected.get("final_answer")
        act_ans = agent_output.get("final_answer")
        ss["final_answer_correct"] = 1.0 if exp_ans is not None and self._match(exp_ans, act_ans) else (0.5 if exp_ans is None else 0.0)

        ss["no_hallucination"] = max(0.0, 1.0 - agent_output.get("unsupported_claims", 0) * 0.25) if not agent_output.get("hallucination_detected") else 0.0

        req = test_case.expected.get("required_outputs", [])
        ss["task_completion"] = (sum(1 for r in req if r in agent_output) / len(req)) if req else (1.0 if agent_output.get("task_complete") else 0.5)

        all_pass = all(v >= 0.9 for v in ss.values())
        score = ss["action_sequence_correct"] * 0.30 + ss["final_answer_correct"] * 0.30 + ss["no_hallucination"] * 0.20 + ss["task_completion"] * 0.20
        return DimensionScore(dimension=EvalDimension.E2E, score=round(score, 4), weight=0.0,
            weighted_score=0.0, passed=all_pass,
            details=f"aon={all_pass} act={ss['action_sequence_correct']:.0f} ans={ss['final_answer_correct']:.0f}",
            sub_scores=ss, elapsed_ms=(time.time() - t0) * 1000)

    @staticmethod
    def _match(exp, act):
        if exp == act: return True
        if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
            return abs(exp - act) / max(abs(exp), 1e-9) < 0.01
        if isinstance(exp, str) and isinstance(act, str):
            return exp.strip().lower() == act.strip().lower()
        return False


# ╔══════════════════════════════════════════╗
# ║  Part 3: V9 评估框架主类                  ║
# ╚══════════════════════════════════════════╝

DEFAULT_DIMENSION_CONFIGS = {
    EvalDimension.RLM: DimensionConfig(EvalDimension.RLM, 0.15, "递归分析质量", "data_coverage+recursion", paper_ref="arXiv:2512.24601"),
    EvalDimension.AWM: DimensionConfig(EvalDimension.AWM, 0.20, "SQL状态验证+环境一致性", "awm_sql_verification", paper_ref="Snowflake AWM"),
    EvalDimension.ENCOMPASS: DimensionConfig(EvalDimension.ENCOMPASS, 0.15, "搜索效率", "accuracy_per_llm_call", paper_ref="MIT CSAIL EnCompass"),
    EvalDimension.REASONING: DimensionConfig(EvalDimension.REASONING, 0.15, "推理结构性", "template+conclusion", paper_ref="arXiv:2602.09276"),
    EvalDimension.MEMORY: DimensionConfig(EvalDimension.MEMORY, 0.10, "记忆利用率", "retrieval+formation", paper_ref="Memory Survey 2026"),
    EvalDimension.INTERPRETABILITY: DimensionConfig(EvalDimension.INTERPRETABILITY, 0.10, "可解释性", "trace+intent", paper_ref="arXiv:2602.07715"),
    EvalDimension.E2E: DimensionConfig(EvalDimension.E2E, 0.15, "All-or-Nothing端到端", "all_or_nothing", passing_threshold=0.9, paper_ref="AgentSkiller"),
}


class V9EvaluationFramework:
    """MRARFAI V9.0 多维评估框架 — 7维评估 + AWM合成 + 权重可调"""

    def __init__(self, dimension_configs=None, passing_threshold=0.70, strict_mode=False):
        self.configs = dimension_configs or DEFAULT_DIMENSION_CONFIGS.copy()
        self.passing_threshold = passing_threshold
        self.strict_mode = strict_mode

        total_w = sum(c.weight for c in self.configs.values())
        if abs(total_w - 1.0) > 0.01:
            for cfg in self.configs.values():
                cfg.weight /= total_w

        self._evaluators = {
            EvalDimension.RLM: RLMEvaluator(), EvalDimension.AWM: AWMEvaluator(),
            EvalDimension.ENCOMPASS: EnCompassEvaluator(), EvalDimension.REASONING: ReasoningEvaluator(),
            EvalDimension.MEMORY: MemoryEvaluator(), EvalDimension.INTERPRETABILITY: InterpretabilityEvaluator(),
            EvalDimension.E2E: E2EEvaluator(),
        }

    def evaluate(self, test_case, agent_output):
        t0 = time.time()
        dim_scores = {}
        errors = []
        dims = test_case.dimensions or list(self.configs.keys())

        for dim in dims:
            if dim not in self.configs: continue
            ev = self._evaluators.get(dim)
            if not ev:
                errors.append(f"No evaluator for {dim.value}"); continue
            try:
                ds = ev.evaluate(test_case, agent_output)
                ds.weight = self.configs[dim].weight
                ds.weighted_score = round(ds.score * ds.weight, 4)
                dim_scores[dim.value] = ds
            except Exception as e:
                errors.append(f"{dim.value}: {e}")
                dim_scores[dim.value] = DimensionScore(
                    dimension=dim, score=0.0, weight=self.configs[dim].weight,
                    weighted_score=0.0, passed=False, details=f"ERROR: {e}")

        total = sum(ds.weighted_score for ds in dim_scores.values())
        passed = all(ds.passed for ds in dim_scores.values()) if self.strict_mode else total >= self.passing_threshold

        return V9EvalResult(case_id=test_case.id, total_score=round(total, 4), passed=passed,
            dimension_scores=dim_scores, elapsed_ms=(time.time() - t0) * 1000, errors=errors,
            metadata={"difficulty": test_case.difficulty, "source": test_case.source, "strict_mode": self.strict_mode})

    def evaluate_batch(self, test_cases, agent_outputs, progress_cb=None):
        if len(test_cases) != len(agent_outputs):
            raise ValueError(f"Cases ({len(test_cases)}) != Outputs ({len(agent_outputs)})")

        results = []
        for i, (tc, ao) in enumerate(zip(test_cases, agent_outputs)):
            r = self.evaluate(tc, ao)
            results.append(r)
            if progress_cb: progress_cb(i + 1, len(test_cases), r)

        passed = sum(1 for r in results if r.passed)
        scores = [r.total_score for r in results]
        dim_agg = {}
        for r in results:
            for dn, ds in r.dimension_scores.items():
                dim_agg.setdefault(dn, []).append(ds.score)

        return V9EvalReport(
            total_cases=len(results), passed_cases=passed, failed_cases=len(results) - passed,
            avg_score=statistics.mean(scores) if scores else 0.0,
            dimension_averages={k: statistics.mean(v) for k, v in dim_agg.items()},
            results=results,
            config_snapshot={"passing_threshold": self.passing_threshold, "strict_mode": self.strict_mode,
                "weights": {d.value: c.weight for d, c in self.configs.items()}})


# ╔══════════════════════════════════════════╗
# ║  Part 4: AWM 合成测试用例生成器            ║
# ╚══════════════════════════════════════════╝

class AWMTestCaseGenerator:
    """从 AWM 合成环境自动生成 V9 评估用例 (手工50 → 1000+)"""

    SCENARIO_SEEDS = [
        {"domain": "brand_analysis", "query_template": "分析{brand}在{region}的出货趋势", "difficulty": "easy", "dimensions": ["rlm", "awm", "reasoning"]},
        {"domain": "anomaly_detection", "query_template": "检测{month}的异常订单", "difficulty": "medium", "dimensions": ["awm", "encompass", "interpretability"]},
        {"domain": "revenue_forecast", "query_template": "预测{brand}下季度的收入", "difficulty": "hard", "dimensions": ["rlm", "reasoning", "memory"]},
        {"domain": "customer_risk", "query_template": "{customer}的应收账款风险评估", "difficulty": "medium", "dimensions": ["reasoning", "interpretability", "e2e"]},
        {"domain": "product_mix", "query_template": "优化{category}的产品组合策略", "difficulty": "hard", "dimensions": ["rlm", "encompass", "reasoning", "e2e"]},
        {"domain": "supply_chain", "query_template": "分析{region}的供应链瓶颈", "difficulty": "hard", "dimensions": ["awm", "memory", "reasoning"]},
        {"domain": "cross_brand_compare", "query_template": "对比{brand}和{brand2}的市场表现", "difficulty": "medium", "dimensions": ["rlm", "reasoning", "interpretability"]},
        {"domain": "seasonal_pattern", "query_template": "分析{category}的季节性出货模式", "difficulty": "easy", "dimensions": ["rlm", "memory"]},
        {"domain": "pricing_strategy", "query_template": "评估{brand}的定价策略有效性", "difficulty": "hard", "dimensions": ["reasoning", "encompass", "e2e"]},
        {"domain": "new_market", "query_template": "评估进入{region}市场的可行性", "difficulty": "extreme", "dimensions": ["rlm", "awm", "reasoning", "memory", "e2e"]},
        {"domain": "quality_issue", "query_template": "追踪{product}的质量问题根因", "difficulty": "medium", "dimensions": ["awm", "reasoning", "interpretability"]},
        {"domain": "multi_table", "query_template": "综合分析{brand}的出货-收入-客户关联", "difficulty": "extreme", "dimensions": ["rlm", "awm", "encompass", "reasoning", "memory", "interpretability", "e2e"]},
    ]
    BRANDS = ["Samsung", "Xiaomi", "OPPO", "vivo", "Transsion", "Motorola", "Nokia", "Huawei"]
    REGIONS = ["东南亚", "南亚", "拉美", "非洲", "中东", "欧洲"]
    CATEGORIES = ["智能手机", "功能机", "平板", "IoT设备", "可穿戴"]
    MONTHS = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1"]
    CUSTOMERS = ["Client-A", "Client-B", "Client-C", "Client-D", "Client-E"]

    def __init__(self, seed=42):
        self._counter = 0; self._rng = seed

    def _rand(self):
        self._rng = (self._rng * 1103515245 + 12345) & 0x7FFFFFFF
        return self._rng / 0x7FFFFFFF

    def _pick(self, lst):
        return lst[int(self._rand() * len(lst)) % len(lst)]

    def generate(self, num_cases=100):
        cases = []
        for _ in range(num_cases):
            s = self._pick(self.SCENARIO_SEEDS)
            brand = self._pick(self.BRANDS)
            brand2 = self._pick([b for b in self.BRANDS if b != brand])
            region, cat, month, cust = self._pick(self.REGIONS), self._pick(self.CATEGORIES), self._pick(self.MONTHS), self._pick(self.CUSTOMERS)
            query = s["query_template"].format(brand=brand, brand2=brand2, region=region, category=cat, month=month, customer=cust, product=f"{brand}-{cat}")
            self._counter += 1
            cases.append(V9EvalCase(
                id=f"AWM-{self._counter:04d}", name=f"{s['domain']}: {query[:50]}",
                category=s["domain"],
                input_data={"query": query, "brand": brand, "region": region, "category": cat, "month": month, "total_rows": int(self._rand() * 5000) + 100},
                expected=self._gen_expected(s["domain"], brand),
                dimensions=[EvalDimension(d) for d in s["dimensions"]],
                difficulty=s["difficulty"], source="awm_synthetic", tags=[s["domain"], brand, region]))
        return cases

    def _gen_expected(self, domain, brand):
        base = {"relevant_tables": ["shipments", "revenue"], "required_merge_fields": ["brand", "region", "total_units"]}
        if domain == "anomaly_detection":
            base["sql_expected"] = [{"columns": ["month", "anomaly_type", "severity"]}]
        elif domain == "revenue_forecast":
            base["reasoning_template"] = "prediction"
            base["conclusion_must_contain"] = [brand, "预测"]
        elif domain == "customer_risk":
            base["reasoning_template"] = "risk_assessment"
            base["required_outputs"] = ["risk_score", "risk_factors", "recommendation"]
        elif domain == "multi_table":
            base["relevant_tables"] = ["shipments", "revenue", "customers", "products"]
            base["reasoning_template"] = "diagnostic"
            base["required_outputs"] = ["summary", "charts", "recommendations"]
            base["expected_memory_shares"] = 2
        else:
            base["conclusion_must_contain"] = [brand]
        return base


# ╔══════════════════════════════════════════╗
# ║  Part 5: 基线对比 + 报告生成              ║
# ╚══════════════════════════════════════════╝

class BaselineComparator:
    def __init__(self, path="eval_baseline_v8.json"):
        self.path = path; self.baseline = None
        if os.path.exists(path):
            try: self.baseline = json.load(open(path))
            except: pass

    def save_baseline(self, report):
        data = {"version": report.version, "timestamp": report.run_timestamp,
                "avg_score": report.avg_score, "pass_rate": report.pass_rate(),
                "dimension_averages": report.dimension_averages, "total_cases": report.total_cases}
        json.dump(data, open(self.path, "w"), indent=2, ensure_ascii=False)
        self.baseline = data

    def compare(self, report):
        if not self.baseline: return {"status": "no_baseline", "message": "无基线，当前结果将作为首次基线"}
        bl = self.baseline
        result = {"status": "compared", "baseline_version": bl.get("version", "?"),
            "overall": {"baseline": bl.get("avg_score", 0), "current": report.avg_score,
                "delta": round(report.avg_score - bl.get("avg_score", 0), 4)},
            "regressions": [], "improvements": []}
        for dim, cur in report.dimension_averages.items():
            bl_v = bl.get("dimension_averages", {}).get(dim, 0)
            d = cur - bl_v
            if d < -0.05: result["regressions"].append(dim)
            elif d > 0.05: result["improvements"].append(dim)
        return result


class ReportGenerator:
    @staticmethod
    def to_markdown(report, comparison=None):
        lines = [f"# MRARFAI V{report.version} 评估报告", f"**运行时间**: {report.run_timestamp}", "",
            "## 总体结果", f"| 指标 | 值 |", f"|------|------|",
            f"| 总用例 | {report.total_cases} |", f"| 通过 | {report.passed_cases} |",
            f"| 失败 | {report.failed_cases} |", f"| 通过率 | {report.pass_rate():.1%} |",
            f"| 平均分 | {report.avg_score:.4f} |", "", "## 维度分析",
            "| 维度 | 平均分 | 状态 |", "|------|--------|------|"]
        for dn, avg in sorted(report.dimension_averages.items()):
            st = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
            lines.append(f"| {dn} | {avg:.4f} | {st} |")
        if comparison and comparison.get("status") == "compared":
            lines += ["", "## 基线对比", f"总分变化: {comparison['overall']['delta']:+.4f}"]
            if comparison.get("improvements"): lines.append(f"**改进**: {', '.join(comparison['improvements'])}")
            if comparison.get("regressions"): lines.append(f"**回归**: {', '.join(comparison['regressions'])}")
        return "\n".join(lines)

    @staticmethod
    def to_dashboard_data(report):
        return {
            "summary": {"total": report.total_cases, "passed": report.passed_cases,
                "pass_rate": report.pass_rate(), "avg_score": report.avg_score},
            "radar_chart": {"dimensions": list(report.dimension_averages.keys()),
                "scores": list(report.dimension_averages.values())},
            "difficulty_breakdown": _group_by(report.results,
                key_fn=lambda r: r.metadata.get("difficulty", "unknown"),
                agg_fn=lambda rs: {"count": len(rs), "avg": statistics.mean(r.total_score for r in rs),
                    "pass_rate": sum(1 for r in rs if r.passed) / max(len(rs), 1)}),
        }


def _group_by(items, key_fn, agg_fn):
    g = {}
    for i in items: g.setdefault(key_fn(i), []).append(i)
    return {k: agg_fn(v) for k, v in g.items()}


# ╔══════════════════════════════════════════╗
# ║  Part 6: 快速验证 + CLI                   ║
# ╚══════════════════════════════════════════╝

def run_quick_validation():
    """快速验证 — 5个内置用例 + mock 输出"""
    fw = V9EvaluationFramework()
    cases = [
        V9EvalCase("QV-001", "基础品牌查询", "brand_analysis",
            {"query": "Samsung东南亚出货量", "total_rows": 1200},
            {"relevant_tables": ["shipments"], "conclusion_must_contain": ["Samsung"]}, difficulty="easy"),
        V9EvalCase("QV-002", "异常检测", "anomaly_detection",
            {"query": "检测2024-Q3异常", "total_rows": 3000},
            {"sql_expected": [{"columns": ["month", "anomaly_type", "severity"]}], "reasoning_template": "anomaly_investigation"}, difficulty="medium"),
        V9EvalCase("QV-003", "收入预测", "revenue_forecast",
            {"query": "预测Xiaomi收入", "total_rows": 5000},
            {"reasoning_template": "prediction", "conclusion_must_contain": ["Xiaomi", "预测"], "required_outputs": ["forecast"]}, difficulty="hard"),
        V9EvalCase("QV-004", "多表综合", "multi_table",
            {"query": "OPPO出货-收入-客户", "total_rows": 8000},
            {"relevant_tables": ["shipments", "revenue", "customers"], "reasoning_template": "diagnostic",
             "required_outputs": ["summary", "charts", "recommendations"],
             "action_sequence": [{"type": "query", "target": "shipments"}, {"type": "query", "target": "revenue"},
                 {"type": "query", "target": "customers"}, {"type": "analyze", "target": "correlation"}],
             "expected_memory_shares": 2, "attribution_sources": ["shipments", "revenue", "customers"]},
            dimensions=list(EvalDimension), difficulty="extreme"),
        V9EvalCase("QV-005", "风险评估", "customer_risk",
            {"query": "Client-A风险", "total_rows": 500},
            {"reasoning_template": "risk_assessment", "final_answer": "风险等级: 中",
             "required_outputs": ["risk_score", "risk_factors", "recommendation"]}, difficulty="medium"),
    ]
    mocks = [
        {"rlm_analyzed_rows": 1100, "rlm_recursion_depth": 2, "rlm_merge_complete": True, "rlm_total_tokens": 5000,
         "tables_accessed": ["shipments"], "conclusion": "Samsung东南亚出货量稳步增长",
         "reasoning_trace": {"template": "trend_analysis", "steps": [{"input": "查询", "output": "Samsung数据"}, {"input": "Samsung数据", "output": "趋势"}]},
         "memory_operations": {"retrieved_memories": [{"id": "m1"}]},
         "explanation": {"intent": {"category": "analysis", "confidence": 0.9}}, "task_complete": True},
        {"rlm_analyzed_rows": 2800, "rlm_recursion_depth": 3,
         "sql_results": [{"columns": ["month", "anomaly_type", "severity"], "row_count": 5}],
         "tables_accessed": ["shipments", "anomaly_labels"], "conclusion": "发现3个异常",
         "reasoning_trace": {"template": "anomaly_investigation", "steps": [{"input": "筛选", "output": "异常列表"}, {"input": "异常列表", "output": "根因"}, {"input": "根因", "output": "影响"}]},
         "explanation": {"intent": {"category": "detection", "confidence": 0.85},
             "process_steps": [{"description": "异常检测: 数据清洗→统计检验→标注"}],
             "attributions": [{"source": "shipments"}, {"source": "anomaly_labels"}]},
         "is_correct": True, "confidence": 0.85},
        {"rlm_analyzed_rows": 4500, "rlm_recursion_depth": 4,
         "rlm_merged_fields": ["brand", "region", "total_units", "revenue"], "rlm_total_tokens": 15000,
         "conclusion": "Xiaomi预测下季度收入环比增长8-12%",
         "reasoning_trace": {"template": "prediction", "steps": [{"input": "历史", "output": "Xiaomi趋势"}, {"input": "Xiaomi趋势", "output": "季节性"}, {"input": "季节性", "output": "模型"}, {"input": "模型", "output": "Xiaomi预测"}, {"input": "Xiaomi预测", "output": "置信度"}]},
         "forecast": {"q_next": 1250000}, "memory_operations": {"retrieved_memories": [{"id": "mem-xiaomi-history"}],
             "formed_memories": [{"abstraction_level": 3, "key_entities": ["Xiaomi"], "temporal_context": "Q4"}], "memory_updates": 2}},
        {"rlm_analyzed_rows": 7200, "rlm_recursion_depth": 4, "rlm_merged_fields": ["brand", "region", "total_units"],
         "sql_results": [{"columns": ["brand", "total_units", "total_revenue"], "row_count": 12}, {"columns": ["customer_id", "order_count"], "row_count": 45}],
         "tables_accessed": ["shipments", "revenue", "customers"],
         "actions": [{"type": "query", "target": "shipments"}, {"type": "query", "target": "revenue"}, {"type": "query", "target": "customers"}, {"type": "analyze", "target": "correlation"}],
         "final_answer": "OPPO综合分析报告", "summary": True, "charts": True, "recommendations": True, "task_complete": True,
         "conclusion": "OPPO出货-收入-客户三维正相关",
         "reasoning_trace": {"template": "diagnostic", "steps": [{"input": "多表", "output": "出货汇总"}, {"input": "出货汇总", "output": "收入关联"}, {"input": "收入关联", "output": "客户集中度"}, {"input": "客户集中度", "output": "关联性"}, {"input": "关联性", "output": "OPPO诊断报告"}]},
         "memory_operations": {"retrieved_memories": [{"id": "m1"}, {"id": "m2"}], "cross_agent_shares": 2,
             "formed_memories": [{"abstraction_level": 2, "key_entities": ["OPPO"], "temporal_context": "2024"}],
             "memory_updates": 1, "memory_consolidations": 1},
         "explanation": {"intent": {"category": "diagnostic", "confidence": 0.92, "sub_intents": ["correlation"], "decomposition_reasoning": "三表关联"},
             "process_steps": [{"description": "步骤1: 查询出货收入客户三表"}, {"description": "步骤2: 皮尔逊相关"}, {"description": "步骤3: HHI指数"}],
             "attributions": [{"source": "shipments"}, {"source": "revenue"}, {"source": "customers"}]},
         "is_correct": True, "confidence": 0.88},
        {"rlm_analyzed_rows": 450, "rlm_recursion_depth": 2,
         "conclusion": "Client-A风险等级: 中", "risk_score": 0.45, "risk_factors": ["账期延长"], "recommendation": "加强监控",
         "final_answer": "风险等级: 中", "task_complete": True,
         "reasoning_trace": {"template": "risk_assessment", "steps": [{"input": "客户", "output": "Client-A记录"}, {"input": "Client-A记录", "output": "逾期分析"}, {"input": "逾期分析", "output": "风险因子"}, {"input": "风险因子", "output": "评分"}]},
         "explanation": {"intent": {"category": "risk_assessment", "confidence": 0.88}}},
    ]
    return fw.evaluate_batch(cases, mocks)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MRARFAI V9.0 评估框架")
    parser.add_argument("--quick", action="store_true", help="快速验证")
    parser.add_argument("--awm-generate", type=int, metavar="N", help="生成N个AWM合成用例")
    parser.add_argument("--report", action="store_true", help="Markdown报告")
    parser.add_argument("--save-baseline", action="store_true", help="保存基线")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    parser.add_argument("--strict", action="store_true", help="严格模式")
    args = parser.parse_args()

    if args.awm_generate:
        gen = AWMTestCaseGenerator()
        cases = gen.generate(args.awm_generate)
        print(f"✅ 生成 {len(cases)} 个 AWM 合成评估用例")
        if args.json: print(json.dumps([c.to_legacy() for c in cases], indent=2, ensure_ascii=False))
        return

    print("=" * 60)
    print("MRARFAI V9.0 Evaluation Framework — Quick Validation")
    print("=" * 60)
    report = run_quick_validation()
    print(f"\n总用例: {report.total_cases}")
    print(f"通过: {report.passed_cases}/{report.total_cases} ({report.pass_rate():.1%})")
    print(f"平均分: {report.avg_score:.4f}")
    print("\n维度平均分:")
    for dim, avg in sorted(report.dimension_averages.items()):
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {dim:20s} {bar} {avg:.4f}")

    if args.report: print("\n" + ReportGenerator.to_markdown(report))
    if args.json: print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    if args.save_baseline:
        BaselineComparator().save_baseline(report)
        print("\n✅ 已保存为基线")


if __name__ == "__main__":
    main()
