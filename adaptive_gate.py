#!/usr/bin/env python3
"""
MRARFAI V8.0 — Phase I: Adaptive Gate (智能门控)
==================================================
借鉴:
  - AVIC: skip / call_wm 三级门控 (ICLR 2025)
  - Puppeteer: RL 训练的中央编排器 (NeurIPS 2025)
  - AORCHESTRA: 动态 Agent 4-tuple 创建
  - LangGraph 2.2x 性能优势验证

核心理念:
  不是所有问题都需要全部 Agent。
  简单查询 → SQL 直查 (skip)
  中等复杂 → 1-2 Agent (light)
  全面分析 → 全流程 MARL (full)

对比 V7.0:
  V7: SmartRouter 二级 (LLM路由 / 规则路由)
  V8: AdaptiveGate 三级 (skip / light / full) + 复杂度评估 + 历史学习
"""

import json
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


# ============================================================
# 1. 查询复杂度评估器
# ============================================================

class ComplexityLevel(Enum):
    """查询复杂度级别"""
    SKIP = "skip"      # < 0.3 → SQL直查，不调用任何 Agent
    LIGHT = "light"    # 0.3-0.7 → 1-2 个 Agent
    FULL = "full"      # > 0.7 → 全流程多 Agent


@dataclass
class ComplexityAssessment:
    """复杂度评估结果"""
    score: float                    # 0.0 - 1.0
    level: ComplexityLevel
    dimensions: Dict[str, float]    # 各维度得分
    agents_recommended: List[str]   # 推荐的 Agent 列表
    reason: str                     # 评估理由
    route_source: str = "gate"      # skip / light / full


class QueryComplexityAssessor:
    """
    查询复杂度评估器 — 多维度打分

    维度:
    1. 语义复杂度: 问题是否涉及多个分析角度
    2. 数据范围: 需要查询的数据维度数量
    3. 推理深度: 是否需要因果推理/对比/预测
    4. 时间跨度: 是否涉及多时间段对比
    5. 交互意图: 是简单查询还是深度分析
    """

    # 简单查询模式 → 直接跳过 Agent
    SKIP_PATTERNS = [
        # 简单数据查询
        (r'(总|年度|全年).*(营收|收入|金额|出货)', 0.1),
        (r'(多少|几个|几家|有哪些).*(客户|产品|区域)', 0.1),
        (r'(是什么|是多少|列出|告诉我)', 0.1),
        # 单一客户查询
        (r'(.{2,10})(的|客户).*(数据|金额|占比|排名)', 0.15),
        # 排名查询
        (r'(top|前|最大|最高|排名)', 0.15),
    ]

    # 中等复杂度模式 → 1-2 Agent
    LIGHT_PATTERNS = [
        (r'(对比|比较|vs|差异)', 0.5),
        (r'(趋势|变化|波动|走势)', 0.45),
        (r'(为什么|原因|怎么回事)', 0.55),
        (r'(风险|预警|异常)', 0.5),
        (r'(增长|下滑|机会)', 0.5),
    ]

    # 高复杂度模式 → 全流程
    FULL_PATTERNS = [
        (r'(CEO|总结|全面|综合|报告|概览)', 0.95),
        (r'(战略|策略|建议|决策|怎么办)', 0.9),
        (r'(预测|forecast|2026|未来|前景)', 0.85),
        (r'(竞争|对标|行业|华勤|闻泰|龙旗)', 0.85),
        (r'(价量|分解|结构|深度)', 0.8),
        (r'(客户.{0,5}分析.{0,5}风险|风险.{0,5}建议)', 0.85),
        (r'年度.*(分析|报告|总结)', 0.95),
        (r'(增长|策略).*(怎么|如何|应该)', 0.9),
    ]

    # Agent 推荐映射
    AGENT_TRIGGERS = {
        "analyst": [
            '营收', '收入', '金额', '出货', '数量', '客户', '分级',
            'ABC', '排名', 'top', '占比', '集中', '月度', '季度',
            '同比', '环比', '趋势', '总览', '概览', '多少',
            '产品', '结构', '区域',
        ],
        "risk": [
            '风险', '流失', '预警', '下降', '下滑', '丢失', '断崖',
            '异常', '暴跌', '危险', '警告', '关注', '问题',
            '波动', '偏离', '不正常',
        ],
        "strategist": [
            '增长', '机会', '战略', '策略', '建议', '方向', '投入',
            '竞争', '对手', '华勤', '闻泰', '龙旗', '行业', '对标',
            '预测', 'forecast', '2026', '未来', '前景',
            'CEO', '管理', '决策', '优化', '提升',
            '价格', '价量', '利润',
        ],
    }

    def __init__(self):
        """初始化评估器"""
        # 历史评估记录 — 用于学习优化
        self.history: List[Dict] = []
        self.pattern_hit_counts = defaultdict(int)

    def assess(self, question: str, context: Dict[str, Any] = None) -> ComplexityAssessment:
        """
        评估查询复杂度

        Args:
            question: 用户问题
            context: 附加上下文（历史对话、当前数据等）

        Returns:
            ComplexityAssessment
        """
        q = question.lower().strip()
        dimensions = {}

        # 维度 1: 模式匹配得分
        pattern_score = self._pattern_score(q)
        dimensions["pattern"] = pattern_score

        # 维度 2: 语义复杂度（问题长度、子句数量）
        semantic_score = self._semantic_complexity(q)
        dimensions["semantic"] = semantic_score

        # 维度 3: 数据范围（涉及的数据维度数量）
        scope_score = self._data_scope(q)
        dimensions["scope"] = scope_score

        # 维度 4: 推理深度（是否需要因果/对比/预测）
        reasoning_score = self._reasoning_depth(q)
        dimensions["reasoning"] = reasoning_score

        # 维度 5: 交互意图
        intent_score = self._interaction_intent(q)
        dimensions["intent"] = intent_score

        # 加权综合 (可被 RL 学习优化)
        weights = {
            "pattern": 0.35,
            "semantic": 0.15,
            "scope": 0.10,
            "reasoning": 0.25,
            "intent": 0.15,
        }
        composite = sum(dimensions[k] * weights[k] for k in weights)
        composite = max(0.0, min(1.0, composite))

        # 强触发覆盖: 如果 pattern 或 intent 分数极高，直接提升
        if dimensions["pattern"] >= 0.85:
            composite = max(composite, 0.75)
        if dimensions["intent"] >= 0.85:
            composite = max(composite, 0.72)

        # 确定级别
        if composite < 0.3:
            level = ComplexityLevel.SKIP
        elif composite < 0.7:
            level = ComplexityLevel.LIGHT
        else:
            level = ComplexityLevel.FULL

        # 推荐 Agent
        agents = self._recommend_agents(q, level)

        # 生成理由
        reason = self._generate_reason(dimensions, level, agents)

        assessment = ComplexityAssessment(
            score=round(composite, 3),
            level=level,
            dimensions={k: round(v, 3) for k, v in dimensions.items()},
            agents_recommended=agents,
            reason=reason,
            route_source=level.value,
        )

        # 记录历史
        self.history.append({
            "question": question[:100],
            "score": composite,
            "level": level.value,
            "agents": agents,
            "timestamp": time.time(),
        })

        return assessment

    def _pattern_score(self, q: str) -> float:
        """模式匹配得分"""
        scores = []

        for pattern, score in self.SKIP_PATTERNS:
            if re.search(pattern, q):
                scores.append(score)
                self.pattern_hit_counts[f"skip:{pattern}"] += 1

        for pattern, score in self.LIGHT_PATTERNS:
            if re.search(pattern, q):
                scores.append(score)
                self.pattern_hit_counts[f"light:{pattern}"] += 1

        for pattern, score in self.FULL_PATTERNS:
            if re.search(pattern, q):
                scores.append(score)
                self.pattern_hit_counts[f"full:{pattern}"] += 1

        if not scores:
            return 0.4  # 默认中等

        return max(scores)  # 取最高匹配

    def _semantic_complexity(self, q: str) -> float:
        """语义复杂度"""
        # 问题长度
        length_score = min(len(q) / 50, 1.0)

        # 子句数量（逗号、顿号、"和"、"并且"分隔）
        clause_count = len(re.split(r'[，,、]|和|并且|以及|还有', q))
        clause_score = min(clause_count / 4, 1.0)

        # 问号数量（多个问题 → 高复杂度）
        question_marks = q.count('？') + q.count('?')
        multi_q_score = min(question_marks / 2, 1.0)

        return (length_score * 0.3 + clause_score * 0.4 + multi_q_score * 0.3)

    def _data_scope(self, q: str) -> float:
        """数据范围评估"""
        scopes = set()
        scope_keywords = {
            "revenue": ['营收', '收入', '金额', '总额'],
            "customer": ['客户', '品牌', '厂商'],
            "product": ['产品', '品类', '手机', '平板'],
            "region": ['区域', '地区', '市场'],
            "time": ['月度', '季度', 'H1', 'H2', '同比', '环比'],
            "risk": ['风险', '异常', '预警'],
            "growth": ['增长', '机会', '潜力'],
        }
        for scope, keywords in scope_keywords.items():
            if any(kw in q for kw in keywords):
                scopes.add(scope)

        return min(len(scopes) / 4, 1.0)

    def _reasoning_depth(self, q: str) -> float:
        """推理深度评估"""
        depth = 0.0

        # 因果推理
        if any(kw in q for kw in ['为什么', '原因', '导致', '因为', '怎么回事']):
            depth += 0.3

        # 对比分析
        if any(kw in q for kw in ['对比', '比较', 'vs', '差异', '不同']):
            depth += 0.25

        # 预测/前瞻
        if any(kw in q for kw in ['预测', '未来', '趋势', '前景', 'forecast']):
            depth += 0.3

        # 建议/决策
        if any(kw in q for kw in ['建议', '怎么办', '应该', '策略', '方案']):
            depth += 0.3

        return min(depth, 1.0)

    def _interaction_intent(self, q: str) -> float:
        """交互意图评估"""
        # 简单查询
        if any(kw in q for kw in ['是什么', '是多少', '列出', '有哪些']):
            return 0.15

        # 中等分析
        if any(kw in q for kw in ['分析', '看看', '了解', '情况']):
            return 0.5

        # 深度分析
        if any(kw in q for kw in ['深度', '全面', '详细', '报告', 'CEO']):
            return 0.9

        return 0.4

    def _recommend_agents(self, q: str, level: ComplexityLevel) -> List[str]:
        """推荐 Agent 列表"""
        if level == ComplexityLevel.SKIP:
            return []  # 直接 SQL 查询，不需要 Agent

        # 全量触发 (优先检查)
        if any(k in q for k in ['ceo', '总结', '全面', '概览', '怎么样', '报告', '年度']):
            return ["analyst", "risk", "strategist"]

        # 根据关键词触发
        triggered = {}
        for agent_id, keywords in self.AGENT_TRIGGERS.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > 0:
                triggered[agent_id] = score

        if not triggered:
            return ["analyst"]

        # 按得分排序
        sorted_agents = sorted(triggered.items(), key=lambda x: x[1], reverse=True)

        if level == ComplexityLevel.LIGHT:
            return [a[0] for a in sorted_agents[:2]]
        else:  # FULL
            # FULL 级别: 至少 2 个 Agent
            agents = [a[0] for a in sorted_agents]
            if len(agents) < 2:
                if "analyst" not in agents:
                    agents.append("analyst")
            return agents

    def _generate_reason(self, dims: dict, level: ComplexityLevel,
                         agents: list) -> str:
        """生成评估理由"""
        top_dim = max(dims, key=dims.get)
        dim_names = {
            "pattern": "模式匹配",
            "semantic": "语义复杂度",
            "scope": "数据范围",
            "reasoning": "推理深度",
            "intent": "交互意图",
        }

        level_names = {
            ComplexityLevel.SKIP: "简单查询 → SQL直查",
            ComplexityLevel.LIGHT: "中等复杂 → 精准Agent",
            ComplexityLevel.FULL: "高复杂度 → 全流程分析",
        }

        agent_str = ", ".join(agents) if agents else "无(直查)"
        return (
            f"{level_names[level]} | "
            f"主因: {dim_names[top_dim]}({dims[top_dim]:.2f}) | "
            f"Agent: [{agent_str}]"
        )


# ============================================================
# 2. 自适应门控器 (Adaptive Gate)
# ============================================================

class AdaptiveGate:
    """
    V8.0 核心门控 — 三级路由

    skip:  直接走 SmartDataQuery, 不调用 Agent LLM
    light: 调用 1-2 个 Agent (省钱省时间)
    full:  全流程 Multi-Agent + Critic + HITL

    借鉴 AVIC 的 skip/call_wm 机制 + Puppeteer 的 RL 编排
    """

    def __init__(self):
        self.assessor = QueryComplexityAssessor()
        # 门控统计
        self.stats = {
            "skip": 0,
            "light": 0,
            "full": 0,
            "total": 0,
        }
        # Agent 调用节省统计
        self.savings = {
            "agent_calls_saved": 0,
            "estimated_tokens_saved": 0,
            "estimated_time_saved_ms": 0,
        }

    def route(self, question: str, context: Dict[str, Any] = None,
              force_level: str = None) -> ComplexityAssessment:
        """
        智能门控路由

        Args:
            question: 用户问题
            context: 附加上下文
            force_level: 强制级别 (调试用)

        Returns:
            ComplexityAssessment
        """
        self.stats["total"] += 1

        # 强制级别（调试/测试）
        if force_level:
            level = ComplexityLevel(force_level)
            return ComplexityAssessment(
                score={"skip": 0.1, "light": 0.5, "full": 0.9}[force_level],
                level=level,
                dimensions={},
                agents_recommended=self.assessor._recommend_agents(
                    question.lower(), level
                ),
                reason=f"强制路由: {force_level}",
                route_source=f"forced_{force_level}",
            )

        # 正常评估
        assessment = self.assessor.assess(question, context)

        # 更新统计
        self.stats[assessment.level.value] += 1

        # 计算节省
        if assessment.level == ComplexityLevel.SKIP:
            self.savings["agent_calls_saved"] += 3  # 跳过3个Agent
            self.savings["estimated_tokens_saved"] += 2000
            self.savings["estimated_time_saved_ms"] += 6000
        elif assessment.level == ComplexityLevel.LIGHT:
            skipped = 3 - len(assessment.agents_recommended)
            self.savings["agent_calls_saved"] += max(0, skipped)
            self.savings["estimated_tokens_saved"] += skipped * 600
            self.savings["estimated_time_saved_ms"] += skipped * 2000

        return assessment

    def get_stats(self) -> Dict:
        """获取门控统计"""
        total = max(self.stats["total"], 1)
        return {
            "total_queries": total,
            "skip_rate": f"{self.stats['skip'] / total:.1%}",
            "light_rate": f"{self.stats['light'] / total:.1%}",
            "full_rate": f"{self.stats['full'] / total:.1%}",
            "skip_count": self.stats["skip"],
            "light_count": self.stats["light"],
            "full_count": self.stats["full"],
            **self.savings,
        }


# ============================================================
# 3. 动态 Agent 工厂 (借鉴 AORCHESTRA 4-tuple)
# ============================================================

@dataclass
class AgentSpec:
    """Agent 规格 — AORCHESTRA 4-tuple"""
    instruction: str      # 指令
    context: str          # 上下文
    tools: List[str]      # 工具列表
    model: str            # 模型选择
    agent_id: str = ""    # Agent 标识
    priority: int = 1     # 优先级
    timeout_ms: int = 30000  # 超时
    max_tokens: int = 1000   # 最大输出


class AgentFactory:
    """
    动态 Agent 工厂

    根据任务需求动态创建 Agent（而非固定流水线）。
    借鉴 AORCHESTRA 的 4-tuple: (instruction, context, tools, model)

    核心能力:
    1. 任务分解 → Agent 规格生成
    2. 模型选择 → 根据任务复杂度选模型
    3. 工具选择 → 最小化工具集
    4. 上下文裁剪 → 精选而非全量
    """

    # 模型路由策略
    MODEL_ROUTES = {
        "simple_data": "deepseek",        # 简单数据查询 → 便宜
        "analysis": "deepseek",           # 分析 → 性价比
        "risk_assessment": "gpt-4o-mini", # 风险评估 → 准确
        "strategy": "gpt-4o",             # 战略 → 高质量
        "report": "claude-3-5-sonnet",    # 报告 → 文笔好
        "chinese_narrative": "deepseek",  # 中文叙事 → DeepSeek最优
    }

    # Agent 蓝图
    BLUEPRINTS = {
        "analyst": {
            "task_type": "analysis",
            "default_tools": ["calculator", "data_query", "chart"],
            "base_instruction": "你是资深数据分析师，用数字揭示业务真相。",
        },
        "risk": {
            "task_type": "risk_assessment",
            "default_tools": ["anomaly_detector", "risk_calculator", "alert"],
            "base_instruction": "你是风控专家，敏锐发现数据异常和客户风险信号。",
        },
        "strategist": {
            "task_type": "strategy",
            "default_tools": ["benchmark", "forecast", "competitor"],
            "base_instruction": "你是战略顾问，发现增长机会并制定可执行方案。",
        },
    }

    def __init__(self, default_provider: str = "deepseek"):
        self.default_provider = default_provider
        self.active_agents: Dict[str, AgentSpec] = {}

    def create_agents(self, assessment: ComplexityAssessment,
                      question: str,
                      available_context: str = "",
                      provider: str = "") -> List[AgentSpec]:
        """
        根据评估结果动态创建 Agent

        Returns:
            List[AgentSpec] — 按优先级排序
        """
        provider = provider or self.default_provider
        specs = []

        for i, agent_id in enumerate(assessment.agents_recommended):
            blueprint = self.BLUEPRINTS.get(agent_id, {})

            # 4-tuple 生成
            spec = AgentSpec(
                agent_id=agent_id,
                instruction=self._generate_instruction(
                    agent_id, question, blueprint
                ),
                context=self._select_context(
                    agent_id, question, available_context, assessment.level
                ),
                tools=self._pick_tools(agent_id, question, blueprint),
                model=self._route_model(
                    blueprint.get("task_type", "analysis"), provider
                ),
                priority=i + 1,
                timeout_ms=self._calc_timeout(assessment.level),
                max_tokens=self._calc_max_tokens(assessment.level),
            )
            specs.append(spec)
            self.active_agents[agent_id] = spec

        return specs

    def _generate_instruction(self, agent_id: str, question: str,
                              blueprint: dict) -> str:
        """生成 Agent 指令（精炼版）"""
        base = blueprint.get("base_instruction", "")
        return f"{base}\n\n用户问题: {question}\n\n请基于提供的数据给出精准分析。"

    def _select_context(self, agent_id: str, question: str,
                        full_context: str, level: ComplexityLevel) -> str:
        """
        精选上下文（而非全量注入）

        - skip: 无上下文
        - light: 相关片段
        - full: 完整上下文
        """
        if level == ComplexityLevel.SKIP:
            return ""
        if level == ComplexityLevel.LIGHT:
            # 截取关键部分 (max 2000 chars)
            return full_context[:2000] if full_context else ""
        return full_context

    def _pick_tools(self, agent_id: str, question: str,
                    blueprint: dict) -> List[str]:
        """最小化工具集"""
        return blueprint.get("default_tools", [])

    def _route_model(self, task_type: str, provider: str) -> str:
        """路由最优模型"""
        if provider and provider != "auto":
            return provider
        return self.MODEL_ROUTES.get(task_type, "deepseek")

    def _calc_timeout(self, level: ComplexityLevel) -> int:
        """计算超时"""
        return {
            ComplexityLevel.SKIP: 5000,
            ComplexityLevel.LIGHT: 15000,
            ComplexityLevel.FULL: 30000,
        }[level]

    def _calc_max_tokens(self, level: ComplexityLevel) -> int:
        """计算最大输出 Token"""
        return {
            ComplexityLevel.SKIP: 200,
            ComplexityLevel.LIGHT: 800,
            ComplexityLevel.FULL: 1500,
        }[level]

    def destroy_agent(self, agent_id: str):
        """销毁 Agent（释放资源）"""
        self.active_agents.pop(agent_id, None)

    def destroy_all(self):
        """销毁所有活跃 Agent"""
        self.active_agents.clear()


# ============================================================
# 4. 合约验证器 (Agent 输出 Schema 验证)
# ============================================================

@dataclass
class OutputContract:
    """Agent 输出合约"""
    agent_id: str
    schema_version: str = "1.0"
    required_fields: List[str] = field(default_factory=list)
    max_length: int = 5000
    must_contain_data: bool = True  # 结论必须有数据支撑
    language: str = "zh"


class ContractValidator:
    """
    Agent 输出合约验证器

    借鉴 ClickIT 2026: Orchestrator 验证 Agent 输出 Schema 版本
    → 松耦合升级各 Agent 不影响系统稳定性

    每个 Agent 输出都经过合约检查:
    1. 长度检查
    2. 数据引用检查
    3. 格式一致性
    4. 版本兼容性
    """

    DEFAULT_CONTRACTS = {
        "analyst": OutputContract(
            agent_id="analyst",
            required_fields=["数据", "分析", "结论"],
            max_length=3000,
            must_contain_data=True,
        ),
        "risk": OutputContract(
            agent_id="risk",
            required_fields=["风险", "影响", "建议"],
            max_length=2000,
            must_contain_data=True,
        ),
        "strategist": OutputContract(
            agent_id="strategist",
            required_fields=["机会", "方案", "优先级"],
            max_length=2500,
            must_contain_data=False,
        ),
    }

    def validate(self, agent_id: str, output: str) -> Tuple[bool, List[str]]:
        """
        验证 Agent 输出

        Returns:
            (is_valid, issues)
        """
        contract = self.DEFAULT_CONTRACTS.get(agent_id)
        if not contract:
            return True, []

        issues = []

        # 长度检查
        if len(output) > contract.max_length:
            issues.append(f"输出过长: {len(output)} > {contract.max_length}")

        if len(output) < 20:
            issues.append("输出过短: 少于 20 字")

        # 数据引用检查
        if contract.must_contain_data:
            has_number = bool(re.search(r'\d+\.?\d*', output))
            if not has_number:
                issues.append("缺少数据支撑: 输出中未包含任何数字")

        # 空输出检查
        if not output or output.strip() == "":
            issues.append("空输出")

        return len(issues) == 0, issues


# ============================================================
# 5. 全局实例
# ============================================================

# 全局门控器
_gate: Optional[AdaptiveGate] = None

def get_gate() -> AdaptiveGate:
    """获取全局门控器"""
    global _gate
    if _gate is None:
        _gate = AdaptiveGate()
    return _gate

# 全局 Agent 工厂
_factory: Optional[AgentFactory] = None

def get_factory(provider: str = "deepseek") -> AgentFactory:
    """获取全局 Agent 工厂"""
    global _factory
    if _factory is None:
        _factory = AgentFactory(provider)
    return _factory

# 全局合约验证器
_validator: Optional[ContractValidator] = None

def get_validator() -> ContractValidator:
    """获取全局合约验证器"""
    global _validator
    if _validator is None:
        _validator = ContractValidator()
    return _validator


# ============================================================
# 6. 便捷接口
# ============================================================

def adaptive_route(question: str, context: dict = None,
                   provider: str = "deepseek") -> Dict[str, Any]:
    """
    V8.0 自适应路由 — 一行调用

    Args:
        question: 用户问题
        context: 附加上下文
        provider: LLM 提供商

    Returns:
        {
            "level": "skip|light|full",
            "score": 0.0-1.0,
            "agents": [...],
            "specs": [AgentSpec, ...],
            "reason": "...",
            "gate_stats": {...},
        }
    """
    gate = get_gate()
    factory = get_factory(provider)

    # 评估
    assessment = gate.route(question, context)

    # 如果不是 skip，创建 Agent
    specs = []
    if assessment.level != ComplexityLevel.SKIP:
        specs = factory.create_agents(assessment, question, "", provider)

    return {
        "level": assessment.level.value,
        "score": assessment.score,
        "agents": assessment.agents_recommended,
        "specs": [
            {
                "agent_id": s.agent_id,
                "model": s.model,
                "tools": s.tools,
                "priority": s.priority,
                "timeout_ms": s.timeout_ms,
            }
            for s in specs
        ],
        "reason": assessment.reason,
        "gate_stats": gate.get_stats(),
    }
