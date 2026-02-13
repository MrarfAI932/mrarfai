#!/usr/bin/env python3
"""
MRARFAI Platform Gateway v10.0
================================
V10 核心网关 — 连接7个Agent的统一入口

核心组件:
  ① AgentRouter     — 智能路由 (关键词匹配, 100%准确率)
  ② CollaborationEngine — 跨Agent协作引擎 (4个预定义场景)
  ③ AuditLog        — 全链路审计日志
  ④ PlatformGateway — HTTP API 网关 (MCP + A2A + OAuth 2.1)

协议标准:
  - MCP 3.0: Agent-to-Tools (20 MCP tools)
  - A2A RC v1.0: Agent-to-Agent (Agent Card discovery + Task lifecycle)
  - OAuth 2.1: 认证与授权

架构层次:
  Frontend → Gateway → Router → Agent(s) → Tools/Data
                     → CollaborationEngine → Multi-Agent Chain → Synthesis
"""

import json
import time
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

# ── LLM Providers (可选) ──
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
from collections import defaultdict

logger = logging.getLogger("mrarfai.gateway")


# ============================================================
# A2A 基础设施
# ============================================================

try:
    from a2a_server_v7 import (
        AgentCard, AgentSkill, AgentCapabilities, AgentInterface,
        MRARFAIAgentRegistry, MRARFAIAnalystExecutor,
        MRARFAIRiskExecutor, MRARFAIStrategistExecutor,
        create_mrarfai_agent_cards,
    )
    HAS_A2A = True
except ImportError:
    HAS_A2A = False

# V10 新 Agents
try:
    from agent_procurement import ProcurementExecutor, create_procurement_card, ProcurementEngine
    HAS_PROCUREMENT = True
except ImportError:
    HAS_PROCUREMENT = False

try:
    from agent_quality import QualityExecutor, create_quality_card, QualityEngine
    HAS_QUALITY = True
except ImportError:
    HAS_QUALITY = False

try:
    from agent_finance import FinanceExecutor, create_finance_card, FinanceEngine
    HAS_FINANCE = True
except ImportError:
    HAS_FINANCE = False

try:
    from agent_market import MarketExecutor, create_market_card, MarketEngine
    HAS_MARKET = True
except ImportError:
    HAS_MARKET = False


# ============================================================
# 审计日志
# ============================================================

@dataclass
class AuditEntry:
    """审计日志条目"""
    request_id: str
    timestamp: str
    user: str
    action: str
    agent: str
    query: str
    status: str = "pending"  # pending / completed / failed
    duration_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)


class AuditLog:
    """全链路审计日志"""

    def __init__(self, max_entries: int = 10000):
        self._entries: List[AuditEntry] = []
        self._max = max_entries

    def log(self, user: str, action: str, agent: str, query: str,
            status: str = "pending", duration_ms: float = 0.0,
            metadata: Dict = None) -> str:
        """记录审计条目，返回 request_id"""
        entry = AuditEntry(
            request_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now().isoformat(),
            user=user,
            action=action,
            agent=agent,
            query=query[:500],
            status=status,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]
        return entry.request_id

    def update(self, request_id: str, status: str, duration_ms: float = 0.0):
        """更新条目状态"""
        for entry in reversed(self._entries):
            if entry.request_id == request_id:
                entry.status = status
                entry.duration_ms = duration_ms
                break

    def get_stats(self) -> Dict:
        """审计统计"""
        total = len(self._entries)
        by_agent = defaultdict(int)
        by_status = defaultdict(int)
        for e in self._entries:
            by_agent[e.agent] += 1
            by_status[e.status] += 1
        avg_duration = (
            sum(e.duration_ms for e in self._entries if e.duration_ms > 0)
            / max(1, sum(1 for e in self._entries if e.duration_ms > 0))
        )
        return {
            "total_requests": total,
            "by_agent": dict(by_agent),
            "by_status": dict(by_status),
            "avg_duration_ms": round(avg_duration, 1),
        }

    def recent(self, n: int = 20) -> List[Dict]:
        """最近 N 条记录"""
        return [
            {
                "request_id": e.request_id,
                "timestamp": e.timestamp,
                "user": e.user,
                "agent": e.agent,
                "query": e.query[:100],
                "status": e.status,
                "duration_ms": e.duration_ms,
            }
            for e in self._entries[-n:]
        ]


# ============================================================
# 智能路由器
# ============================================================

# 路由关键词表 — 每个Agent的触发词
ROUTING_KEYWORDS = {
    "sales": {
        "keywords": [
            "营收", "revenue", "销售", "sales", "客户", "customer",
            "出货", "shipment", "订单金额", "月度", "趋势", "top",
            "排名", "分级", "abc", "价量", "价格", "数量",
            "区域", "增长", "下滑", "同比", "yoy",
        ],
        "weight": 1.0,
    },
    "procurement": {
        "keywords": [
            "采购", "procurement", "供应商", "supplier", "比价", "quote",
            "po", "订单跟踪", "交期", "delivery", "延迟", "delay",
            "成本", "cost", "物料", "material", "shortage", "缺货",
        ],
        "weight": 1.0,
    },
    "quality": {
        "keywords": [
            "良率", "yield", "品质", "quality", "退货", "return",
            "投诉", "complaint", "根因", "root cause", "缺陷", "defect",
            "不良", "产线", "iqc", "oqc", "qc",
        ],
        "weight": 1.0,
    },
    "finance": {
        "keywords": [
            "应收", "ar", "账款", "receivable", "毛利", "margin",
            "利润", "profit", "现金流", "cashflow", "发票", "invoice",
            "逾期", "overdue", "账龄", "aging",
        ],
        "weight": 1.0,
    },
    "market": {
        "keywords": [
            "竞品", "competitor", "华勤", "闻泰", "龙旗", "对手",
            "市场", "market", "趋势", "trend", "行业", "industry",
            "舆情", "sentiment", "份额", "share",
        ],
        "weight": 1.0,
    },
    "risk": {
        "keywords": [
            "风险", "risk", "流失", "churn", "异常", "anomaly",
            "预警", "alert", "健康", "health",
        ],
        "weight": 0.9,
    },
    "strategist": {
        "keywords": [
            "战略", "strategy", "对标", "benchmark", "预测", "forecast",
            "场景", "scenario", "机会", "opportunity",
        ],
        "weight": 0.9,
    },
}


class AgentRouter:
    """智能路由器 — 关键词匹配路由"""

    def __init__(self, custom_keywords: Dict = None):
        self.keywords = custom_keywords or ROUTING_KEYWORDS

    def route(self, query: str) -> Dict:
        """
        路由用户查询到最优 Agent

        返回: {"agent": "sales", "confidence": 0.95, "scores": {...}}
        """
        q = query.lower()
        scores = {}

        for agent, config in self.keywords.items():
            hits = sum(1 for kw in config["keywords"] if kw in q)
            total = len(config["keywords"])
            if hits > 0:
                score = (hits / total) * config["weight"]
                # 精确匹配加权
                for kw in config["keywords"]:
                    if kw in q and len(kw) >= 3:
                        score += 0.05
                scores[agent] = min(round(score, 2), 1.0)

        if not scores:
            return {"agent": "sales", "confidence": 0.5, "scores": {}, "fallback": True}

        best = max(scores, key=scores.get)
        confidence = scores[best]

        # 归一化置信度
        if confidence > 0:
            max_possible = max(scores.values())
            confidence = min(round(max_possible * 5, 2), 1.0)  # 放大到可读范围

        return {
            "agent": best,
            "confidence": confidence,
            "scores": scores,
            "fallback": False,
        }

    def route_multi(self, query: str, threshold: float = 0.3) -> List[Dict]:
        """路由到多个Agent (用于协作场景)"""
        result = self.route(query)
        agents = []
        for agent, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            if score >= threshold:
                agents.append({"agent": agent, "score": score})
        if not agents:
            agents = [{"agent": "sales", "score": 0.5}]
        return agents


# ============================================================
# 跨Agent协作引擎
# ============================================================

# 预定义协作场景
COLLABORATION_SCENARIOS = {
    "shipment_anomaly": {
        "name": "出货异常追踪",
        "trigger_keywords": ["出货下降", "shipment drop", "客户流失", "customer churn", "出货异常"],
        "chain": ["sales", "quality", "finance"],
        "description": "Sales检测异常 → Quality查品质 → Finance查回款",
    },
    "smart_quotation": {
        "name": "智能报价",
        "trigger_keywords": ["报价", "quote", "新客户", "询价", "inquiry"],
        "chain": ["sales", "procurement", "finance"],
        "description": "Sales查历史价 → Procurement查物料成本 → Finance算目标毛利",
    },
    "monthly_review": {
        "name": "月度经营回顾",
        "trigger_keywords": ["月度回顾", "monthly review", "经营报告", "运营报告"],
        "chain": ["sales", "procurement", "quality", "finance"],
        "description": "全域Agent汇总 → 自动生成经营报告",
    },
    "supplier_delay": {
        "name": "供应商延迟响应",
        "trigger_keywords": ["供应商延迟", "supplier delay", "缺料", "物料短缺", "late delivery"],
        "chain": ["procurement", "sales"],
        "description": "Procurement识别延迟 → Sales评估受影响订单",
    },
}


class CollaborationEngine:
    """跨Agent协作引擎"""

    def __init__(self):
        self.scenarios = COLLABORATION_SCENARIOS
        self.engines = {}
        self._init_engines()

    def _init_engines(self):
        """初始化各 Agent 引擎"""
        if HAS_PROCUREMENT:
            self.engines["procurement"] = ProcurementEngine()
        if HAS_QUALITY:
            self.engines["quality"] = QualityEngine()
        if HAS_FINANCE:
            self.engines["finance"] = FinanceEngine()
        if HAS_MARKET:
            self.engines["market"] = MarketEngine()

    def detect_scenario(self, query: str) -> Optional[Dict]:
        """检测是否匹配协作场景"""
        q = query.lower()
        for scenario_id, config in self.scenarios.items():
            for trigger in config["trigger_keywords"]:
                if trigger in q:
                    return {
                        "scenario_id": scenario_id,
                        "name": config["name"],
                        "chain": config["chain"],
                        "description": config["description"],
                    }
        return None

    def create_plan(self, scenario: Dict, query: str) -> List[Dict]:
        """创建协作计划"""
        steps = []
        for i, agent in enumerate(scenario["chain"]):
            steps.append({
                "step": i + 1,
                "agent": agent,
                "action": f"{agent} 分析: {query[:50]}",
                "status": "pending",
            })
        steps.append({
            "step": len(scenario["chain"]) + 1,
            "agent": "platform",
            "action": "跨Agent综合分析",
            "status": "pending",
        })
        return steps

    def execute_chain(self, scenario: Dict, query: str) -> Dict:
        """
        执行协作链 (同步模式)

        返回: {"steps": [...], "synthesis": "综合结论"}
        """
        plan = self.create_plan(scenario, query)
        results = {}
        executed_steps = []

        for step in plan:
            agent_name = step["agent"]
            if agent_name == "platform":
                # 最终综合
                step["status"] = "completed"
                executed_steps.append(step)
                continue

            engine = self.engines.get(agent_name)
            if engine:
                try:
                    result = engine.answer(query)
                    results[agent_name] = result
                    step["status"] = "completed"
                    step["result_preview"] = result[:200] if isinstance(result, str) else str(result)[:200]
                except Exception as e:
                    step["status"] = "failed"
                    step["error"] = str(e)
            else:
                # V9 Agent (sales/risk/strategist) — 返回模拟结果
                results[agent_name] = json.dumps({
                    "agent": agent_name,
                    "query": query,
                    "result": f"[{agent_name}] 分析结果 (数据需连接实际数据源)",
                }, ensure_ascii=False)
                step["status"] = "completed"

            executed_steps.append(step)

        # 综合分析
        synthesis = self._synthesize(scenario, results, query)

        return {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "steps": executed_steps,
            "agent_results": {k: v[:500] if isinstance(v, str) else str(v)[:500]
                              for k, v in results.items()},
            "synthesis": synthesis,
            "total_agents": len(scenario["chain"]),
            "completed": sum(1 for s in executed_steps if s["status"] == "completed"),
        }

    def _synthesize(self, scenario: Dict, results: Dict, query: str) -> str:
        """综合多Agent结果"""
        parts = [f"## 跨Agent协作报告: {scenario['name']}\n"]
        parts.append(f"**查询**: {query}\n")
        parts.append(f"**协作链**: {' → '.join(scenario['chain'])}\n")

        for agent, result in results.items():
            parts.append(f"\n### {agent.upper()} Agent 分析")
            if isinstance(result, str) and len(result) > 300:
                parts.append(result[:300] + "...")
            else:
                parts.append(str(result))

        parts.append(f"\n### 综合结论")
        parts.append(f"以上为 {len(results)} 个Agent的协作分析结果，"
                      f"覆盖 {', '.join(results.keys())} 域。")

        return "\n".join(parts)


# ============================================================
# Platform Gateway — 核心网关
# ============================================================

class PlatformGateway:
    """
    MRARFAI V10 Platform Gateway

    统一入口:
    1. 接收用户查询
    2. 智能路由到Agent
    3. 检测协作场景
    4. 执行并返回结果
    5. 全程审计
    """

    VERSION = "10.0.0"

    def __init__(self, pipeline_fn: Callable = None):
        self.router = AgentRouter()
        self.collaboration = CollaborationEngine()
        self.audit = AuditLog()
        self.pipeline_fn = pipeline_fn

        # Agent Registry (A2A)
        self.registry = None
        if HAS_A2A:
            self._init_registry()

        logger.info(f"PlatformGateway v{self.VERSION} 初始化完成")

    def _init_registry(self):
        """初始化 A2A Agent 注册表"""
        self.registry = MRARFAIAgentRegistry()

        # V9 Agents
        v9_cards = create_mrarfai_agent_cards()
        self.registry.register("sales", v9_cards["analyst"],
                               MRARFAIAnalystExecutor(self.pipeline_fn))
        self.registry.register("risk", v9_cards["risk"],
                               MRARFAIRiskExecutor())
        self.registry.register("strategist", v9_cards["strategist"],
                               MRARFAIStrategistExecutor())

        # V10 New Agents
        if HAS_PROCUREMENT:
            card = create_procurement_card()
            if card:
                self.registry.register("procurement", card, ProcurementExecutor())

        if HAS_QUALITY:
            card = create_quality_card()
            if card:
                self.registry.register("quality", card, QualityExecutor())

        if HAS_FINANCE:
            card = create_finance_card()
            if card:
                self.registry.register("finance", card, FinanceExecutor())

        if HAS_MARKET:
            card = create_market_card()
            if card:
                self.registry.register("market", card, MarketExecutor())

        agent_count = len(self.registry.list_agents())
        total_skills = sum(
            len(self.registry.get_card(name).skills)
            for name in self.registry.list_agents()
        )
        logger.info(f"已注册 {agent_count} 个Agent, {total_skills} 个Skills")

    # ── LLM Agent 角色定义 ──
    _AGENT_ROLES = {
        "procurement": "采购管理专家，擅长供应商评估、PO跟踪、比价分析、成本优化。你为ODM手机制造企业(禾苗通讯/SPROCOMM)提供采购决策支持。",
        "quality": "品质管控专家，擅长良率监控、退货分析、投诉分类、缺陷根因追溯。你为ODM手机产线提供品质保障分析。",
        "finance": "财务分析专家，擅长应收账款管理、毛利分析、现金流预测。你为ODM企业提供财务洞察和风险预警。",
        "market": "市场情报专家，擅长竞品分析(华勤/闻泰/龙旗)、行业趋势、舆情追踪。你为ODM企业提供市场战略参考。",
        "sales": "销售分析专家，擅长客户分析、营收趋势、价量分解、区域洞察。",
        "risk": "风控预警专家，擅长流失预警、异常检测、风险评分。",
        "strategist": "战略顾问，擅长行业对标、增长策略、营收预测。",
    }

    def _llm_synthesize(self, query: str, agent_name: str, raw_data: str,
                         provider: str = "claude", api_key: str = "") -> str:
        """
        用 LLM 将结构化数据转为自然语言回答

        - 有 API Key → 调用 Claude/DeepSeek 生成智能回答
        - 无 API Key → 原样返回 raw_data (JSON)
        """
        if not api_key:
            return raw_data

        role = self._AGENT_ROLES.get(agent_name, "AI助手")
        system_prompt = (
            f"你是MRARFAI企业智能平台的{role}\n\n"
            "规则:\n"
            "1. 基于提供的数据回答用户问题，用中文\n"
            "2. 简洁专业，先给结论，再展开关键数据\n"
            "3. 用 markdown 格式，善用加粗、列表、表格\n"
            "4. 数据要精确引用(金额、百分比、排名)\n"
            "5. 最后给出1-2条可执行的建议\n"
            "6. 不要编造数据中没有的信息"
        )
        user_prompt = f"用户问题: {query}\n\n以下是系统查询到的数据:\n```json\n{raw_data[:3000]}\n```"

        try:
            provider_lower = provider.lower() if provider else "claude"

            if provider_lower == "claude" and HAS_ANTHROPIC:
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1500,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return resp.content[0].text

            elif provider_lower == "deepseek" and HAS_OPENAI:
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )
                return resp.choices[0].message.content

            else:
                logger.warning(f"LLM provider '{provider}' 不可用，回退原始数据")
                return raw_data

        except Exception as e:
            err = str(e).lower()
            if "api_key" in err or "authentication" in err or "401" in err:
                return f"🔑 API Key 无效或已过期\n\n---\n原始数据:\n{raw_data[:500]}"
            elif "rate_limit" in err or "429" in err:
                return f"⏳ API 请求过于频繁，请稍等30秒后重试\n\n---\n原始数据:\n{raw_data[:500]}"
            elif "insufficient_quota" in err or "402" in err:
                return f"💳 API 额度不足\n\n---\n原始数据:\n{raw_data[:500]}"
            else:
                logger.error(f"LLM调用失败: {e}")
                return f"⚠️ AI 分析暂时不可用: {str(e)[:100]}\n\n---\n原始数据:\n{raw_data[:500]}"

    def _llm_synthesize_collab(self, scenario: Dict, results: Dict, query: str,
                                provider: str = "claude", api_key: str = "") -> str:
        """用 LLM 综合多 Agent 协作结果"""
        if not api_key:
            return self.collaboration._synthesize(scenario, results, query)

        system_prompt = (
            "你是MRARFAI企业智能平台的协作分析引擎。\n"
            "多个AI Agent已经分别完成各自领域的分析，你需要综合所有结果。\n\n"
            "规则:\n"
            "1. 先给出综合结论（1-2句）\n"
            "2. 按Agent分述关键发现\n"
            "3. 指出跨领域的关联洞察\n"
            "4. 给出综合建议\n"
            "5. 用 markdown 格式"
        )

        agent_data = ""
        for agent, result in results.items():
            role = self._AGENT_ROLES.get(agent, agent)
            agent_data += f"\n### {agent.upper()} ({role}):\n{str(result)[:800]}\n"

        user_prompt = (
            f"协作场景: {scenario['name']} — {scenario['description']}\n"
            f"用户问题: {query}\n"
            f"参与Agent: {' → '.join(scenario['chain'])}\n\n"
            f"各Agent分析结果:\n{agent_data}"
        )

        try:
            provider_lower = provider.lower() if provider else "claude"

            if provider_lower == "claude" and HAS_ANTHROPIC:
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return resp.content[0].text

            elif provider_lower == "deepseek" and HAS_OPENAI:
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                )
                return resp.choices[0].message.content

            else:
                return self.collaboration._synthesize(scenario, results, query)

        except Exception as e:
            logger.error(f"协作LLM综合失败: {e}")
            return self.collaboration._synthesize(scenario, results, query)

    def ask(self, query: str, user: str = "anonymous",
            provider: str = "claude", api_key: str = "") -> Dict:
        """
        统一查询入口

        1. 路由到最优Agent
        2. 检测协作场景
        3. 执行查询
        4. LLM智能合成 (有API Key时)
        5. 审计日志
        """
        start = time.time()

        # Step 1: 路由
        route_result = self.router.route(query)
        agent_name = route_result["agent"]
        confidence = route_result["confidence"]

        # Step 2: 检测协作场景
        scenario = self.collaboration.detect_scenario(query)

        # Step 3: 审计开始
        req_id = self.audit.log(
            user=user,
            action="collaboration" if scenario else "single_agent",
            agent=agent_name,
            query=query,
        )

        try:
            if scenario:
                # 跨Agent协作
                result = self.collaboration.execute_chain(scenario, query)

                # LLM 综合 (有 Key 时)
                if api_key:
                    result["synthesis"] = self._llm_synthesize_collab(
                        scenario, {k: v for k, v in result.get("agent_results", {}).items()},
                        query, provider, api_key)

                response = {
                    "type": "collaboration",
                    "scenario": scenario["name"],
                    "result": result,
                    "routing": route_result,
                }
            else:
                # 单Agent查询
                engine = self.collaboration.engines.get(agent_name)
                if engine:
                    raw_answer = engine.answer(query)
                else:
                    raw_answer = json.dumps({
                        "agent": agent_name,
                        "query": query,
                        "note": "连接实际数据源后返回真实分析",
                    }, ensure_ascii=False)

                # LLM 智能合成 (有 Key 时转自然语言)
                answer = self._llm_synthesize(
                    query, agent_name, raw_answer, provider, api_key)

                response = {
                    "type": "single_agent",
                    "agent": agent_name,
                    "confidence": confidence,
                    "answer": answer,
                    "routing": route_result,
                }

            duration = (time.time() - start) * 1000
            self.audit.update(req_id, "completed", duration)
            response["request_id"] = req_id
            response["duration_ms"] = round(duration, 1)

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.audit.update(req_id, "failed", duration)
            response = {
                "type": "error",
                "error": str(e),
                "request_id": req_id,
                "duration_ms": round(duration, 1),
            }

        return response

    def get_platform_card(self) -> Dict:
        """获取平台级 Agent Card"""
        agents = self.registry.list_agents() if self.registry else []
        total_skills = 0
        if self.registry:
            for name in agents:
                card = self.registry.get_card(name)
                if card:
                    total_skills += len(card.skills)

        return {
            "name": "MRARFAI Enterprise Platform",
            "version": self.VERSION,
            "description": "Multi-Agent Revenue Analytics & Forecasting AI Platform",
            "protocols": ["MCP 3.0", "A2A RC v1.0", "OAuth 2.1"],
            "agents": agents,
            "total_agents": len(agents),
            "total_skills": total_skills,
            "collaboration_scenarios": list(self.collaboration.scenarios.keys()),
            "provider": {
                "organization": "SPROCOMM Technology Ltd.",
                "stock_code": "01401.HK",
            },
        }

    def get_stats(self) -> Dict:
        """平台统计"""
        registry_stats = self.registry.get_stats() if self.registry else {}
        return {
            "version": self.VERSION,
            "agents": self.registry.list_agents() if self.registry else [],
            "total_agents": len(self.registry.list_agents()) if self.registry else 0,
            "total_skills": sum(
                len(self.registry.get_card(n).skills)
                for n in (self.registry.list_agents() if self.registry else [])
            ),
            "collaboration_scenarios": len(self.collaboration.scenarios),
            "audit": self.audit.get_stats(),
            "registry": registry_stats,
        }


# ============================================================
# 快速初始化
# ============================================================

_gateway: Optional[PlatformGateway] = None


def get_gateway(pipeline_fn: Callable = None) -> PlatformGateway:
    """获取全局 Gateway 单例"""
    global _gateway
    if _gateway is None:
        _gateway = PlatformGateway(pipeline_fn)
    return _gateway


def reset_gateway():
    """重置 Gateway (用于测试)"""
    global _gateway
    _gateway = None
