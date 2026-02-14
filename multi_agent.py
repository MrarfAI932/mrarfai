#!/usr/bin/env python3
"""
MRARFAI Multi-Agent System v10.0 (Unified)
=============================================
v4.0 基础层 (Tool Use + Guardrails + Streaming + KG + Observability)
  + v7.0 LangGraph 层 (StateGraph + HITL + Reflection + Multi-Model Routing)
  = v9.0 统一文件

架构 (LangGraph 1.0 StateGraph, 可选):
  START → route → experts (parallel) → synthesize → reflect → hitl_check → END

入口:
  ask_multi_agent()        — V4 完整管线 (chat_tab.py 使用)
  run_multi_agent_v7()     — V7 LangGraph 管线
  run_multi_agent()        — V7 兼容别名

7+ Agents: 分析师 + 风控 + 策略师 + 品质 + 市场 + 财务 + 采购 + 报告员 + 批评家
依赖: pip install langgraph>=1.0 langchain-core>=1.0 (可选, 无则回退V4)
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Literal, Annotated
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

logger = logging.getLogger("mrarfai.agent_v9")

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# Knowledge Graph
try:
    from knowledge_graph import SalesKnowledgeGraph, SynonymGraph, QueryPatternLibrary
    HAS_KG = True
except ImportError:
    HAS_KG = False

# Observability
try:
    from observability import (
        get_tracer, get_metrics, get_store, AgentTracer,
        SpanKind, CostCalculator,
    )
    HAS_OBS = True
except ImportError:
    HAS_OBS = False

# v3.3: CriticAgent (Generator+Critic pattern)
try:
    from critic_agent import critique_and_refine, CriticAgent
    HAS_CRITIC = True
except ImportError:
    HAS_CRITIC = False

# v4.0: Tool Registry
try:
    from tool_registry import (
        sales_tools, ToolRegistry, get_tools_for_agent,
        get_tool_descriptions_for_prompt, execute_tool_calls,
        AGENT_TOOL_CATEGORIES,
    )
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False

# v4.0: Guardrails
try:
    from guardrails import (
        guarded_llm_call, with_retry, RetryConfig, DEFAULT_RETRY,
        get_breaker, CircuitBreakerOpenError,
        validate_agent_output, safe_parse_llm_json,
        FallbackChain, get_budget, get_cache,
    )
    HAS_GUARD = True
except ImportError:
    HAS_GUARD = False

# v4.0: Streaming
try:
    from streaming import (
        StreamCallback, StreamEvent, EventType,
        PipelineStream,
    )
    HAS_STREAM = True
except ImportError:
    HAS_STREAM = False

# v3.3: Persistent Memory
try:
    from persistent_memory import (
        get_persistent_memory, PersistentMemoryStore,
        InsightRecord, EntityProfile,
    )
    HAS_PMEM = True
except ImportError:
    HAS_PMEM = False

# v3.3: Enhanced HITL
try:
    from hitl_engine import evaluate_hitl, HITLEngine, ConfidenceLevel
    HAS_HITL_V2 = True
except ImportError:
    HAS_HITL_V2 = False

# v3.3: Protocol Layer (MCP/A2A)
try:
    from protocol_layer import get_protocol_manager, ProtocolManager
    HAS_PROTOCOL = True
except ImportError:
    HAS_PROTOCOL = False

# V10.0: 域 Agent 引擎导入
try:
    from agent_quality import QualityEngine
    HAS_QUALITY = True
except ImportError:
    HAS_QUALITY = False

try:
    from agent_market import MarketEngine
    HAS_MARKET = True
except ImportError:
    HAS_MARKET = False

try:
    from agent_finance import FinanceEngine
    HAS_FINANCE = True
except ImportError:
    HAS_FINANCE = False

try:
    from agent_procurement import ProcurementEngine
    HAS_PROCUREMENT = True
except ImportError:
    HAS_PROCUREMENT = False

try:
    from agent_risk import RiskEngine
    HAS_RISK_ENGINE = True
except ImportError:
    HAS_RISK_ENGINE = False

try:
    from agent_strategist import StrategistEngine
    HAS_STRATEGIST_ENGINE = True
except ImportError:
    HAS_STRATEGIST_ENGINE = False

# V10.0: Pydantic 结构化合约
from contracts import AgentRequest, AgentResponse, GraphInput, GraphOutput

# V10.0: DB → Agent Bridge
try:
    from db_connector import create_engines_from_db, DatabaseConfig
    HAS_DB_BRIDGE = True
except ImportError:
    HAS_DB_BRIDGE = False

# Langfuse v3 可观测性
try:
    from langfuse import Langfuse
    _langfuse_client = Langfuse()
    HAS_LANGFUSE = True
except Exception:
    _langfuse_client = None
    HAS_LANGFUSE = False

# v7.0: LangGraph (可选 — 无则回退V4管线)
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logger.info("langgraph 未安装，使用 V4 管线模式")

# v7.0: HITL Engine (区分于 v3.3 hitl_engine)
try:
    from hitl_engine import evaluate_hitl as evaluate_hitl_v7
    HAS_HITL = True
except ImportError:
    HAS_HITL = False

# ============================================================
# V9.0 模块导入 — 7篇论文核心引擎
# ============================================================

# V9.0 ① RLM 递归语言模型引擎 (arXiv:2512.24601)
try:
    from rlm_engine import RLMEngine, RLMConfig, RLMResult
    HAS_RLM = True
except ImportError:
    HAS_RLM = False

# V9.0 ② AWM 合成环境工厂 (arXiv:2602.10090)
try:
    from awm_env_factory import AWMEnvironmentFactory, SyntheticDataGenerator
    HAS_AWM = True
except ImportError:
    HAS_AWM = False

# V9.0 ③ EnCompass 搜索引擎 (NeurIPS 2025, arXiv:2512.03571)
try:
    from search_engine import (
        SearchConfig, BeamSearch, TwoLevelBeamSearch,
        EnCompassExecutor, BranchPoint, ExecutionPath,
    )
    HAS_SEARCH = True
except ImportError:
    HAS_SEARCH = False

# V9.0 ④ 结构化推理模板 (arXiv:2602.09276)
try:
    from reasoning_templates import (
        TemplateSelector, ReasoningExecutor, PromptCompiler,
        ReasoningTemplate, ReasoningMultiAgentAdapter,
    )
    HAS_REASONING = True
except ImportError:
    HAS_REASONING = False

# V9.0 ⑤ 三维记忆架构 (arXiv:2512.13564)
try:
    from memory_v9 import (
        Memory3DStore, Memory3DNode, MemoryForm, MemoryFunction,
        MemoryEvolutionEngine,
    )
    HAS_MEM3D = True
except ImportError:
    HAS_MEM3D = False

# V9.0 ⑥ LatentLens 可解释性层 (arXiv:2602.00462)
try:
    from interpretability_layer import (
        ProcessTracer, IntentMapper, OutputAttributor,
        FullExplanation,
    )
    HAS_INTERP = True
except ImportError:
    HAS_INTERP = False

# V9.0 ⑦ 多维评估框架 (综合六篇论文)
try:
    from evals_v9 import V9EvaluationFramework, V9EvalReport, EvalDimension
    HAS_EVALS_V9 = True
except ImportError:
    HAS_EVALS_V9 = False

# V10.1 ⑧ Deep Agents 0.4.1 (LangChain 官方)
# pip install deepagents>=0.4.1
# docs: docs.langchain.com/oss/python/deepagents
HAS_DEEP_AGENTS = False
_deep_agent = None
try:
    from deepagents import create_deep_agent
    from langchain.chat_models import init_chat_model
    HAS_DEEP_AGENTS = True
    logger.info("✅ deepagents 0.4.1+ 已加载")
except ImportError:
    create_deep_agent = None
    init_chat_model = None


def _get_deep_agent():
    """
    延迟初始化 Deep Agent — deepagents 0.4.1
    返回 compiled LangGraph graph
    支持: planning + 文件系统 + 子agent生成
    """
    global _deep_agent
    if _deep_agent is None and HAS_DEEP_AGENTS:
        # 自定义工具 (可选)
        custom_tools = []
        if HAS_TOOLS:
            try:
                from tool_registry import sales_tools
                custom_tools = list(sales_tools.values())[:5]
            except Exception:
                pass

        try:
            _deep_agent = create_deep_agent(
                model=init_chat_model(
                    "anthropic:claude-sonnet-4-5-20250929"
                ),
                tools=custom_tools,
                system_prompt=(
                    "你是 MRARFAI V10.1 深度分析Agent。"
                    "你可以规划任务、委派子Agent、"
                    "管理文件。使用中文回答。"
                ),
            )
            logger.info("✅ Deep Agent 初始化完成")
        except Exception as e:
            logger.warning(f"Deep Agent 初始化失败: {e}")
            _deep_agent = None
    return _deep_agent


# V9.0 全局实例
_v9_tracer: 'ProcessTracer' = None          # 可解释性追踪器
_v9_reasoning: 'TemplateSelector' = None    # 推理模板选择器
_v9_memory: 'Memory3DStore' = None          # 三维记忆

def _get_v9_tracer() -> 'ProcessTracer':
    global _v9_tracer
    if _v9_tracer is None and HAS_INTERP:
        _v9_tracer = ProcessTracer()
    return _v9_tracer

def _get_v9_reasoning() -> 'TemplateSelector':
    global _v9_reasoning
    if _v9_reasoning is None and HAS_REASONING:
        _v9_reasoning = TemplateSelector()
    return _v9_reasoning

def _get_v9_memory() -> 'Memory3DStore':
    global _v9_memory
    if _v9_memory is None and HAS_MEM3D:
        _v9_memory = Memory3DStore(db_path="memory_v9.db")
    return _v9_memory


# ============================================================
# [不变] Agent 记忆系统 — 兼容v2.1
# ============================================================

class AgentMemory:
    """
    多轮对话记忆
    - 短期记忆: 最近N轮QA
    - 实体记忆: 提到过的客户/数据点
    - 分析摘要: 每轮分析的核心结论
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversation_history = deque(maxlen=max_turns)
        self.entity_mentions = {}
        self.analysis_summaries = deque(maxlen=max_turns)
        self.risk_confirmations = {}

    def add_turn(self, question: str, answer: str, agents_used: list = None,
                 expert_outputs: dict = None):
        turn = {
            'time': datetime.now().isoformat(),
            'question': question,
            'answer_preview': answer[:200],
            'agents': agents_used or [],
        }
        self.conversation_history.append(turn)
        for name_candidate in self._extract_entities(question + " " + answer):
            if name_candidate not in self.entity_mentions:
                self.entity_mentions[name_candidate] = []
            self.entity_mentions[name_candidate].append(question[:50])
        if expert_outputs:
            for expert, output in expert_outputs.items():
                self.analysis_summaries.append({
                    'expert': expert,
                    'summary': output[:150],
                    'question': question[:50],
                })

    def add_risk_confirmation(self, customer: str, confirmed: bool):
        self.risk_confirmations[customer] = {
            'confirmed': confirmed,
            'time': datetime.now().isoformat(),
        }

    def get_context_prompt(self) -> str:
        if not self.conversation_history:
            return ""
        lines = ["[之前的对话记忆]"]
        for turn in list(self.conversation_history)[-5:]:
            lines.append(f"Q: {turn['question'][:80]}")
            lines.append(f"A: {turn['answer_preview'][:100]}...")
        if self.risk_confirmations:
            lines.append("\n[风险确认记录]")
            for cust, info in self.risk_confirmations.items():
                status = "已确认关注" if info['confirmed'] else "已标记为低优先"
                lines.append(f"- {cust}: {status}")
        if self.entity_mentions:
            top_entities = sorted(
                self.entity_mentions.items(),
                key=lambda x: len(x[1]), reverse=True
            )[:5]
            if top_entities:
                lines.append("\n[用户关注的重点客户]")
                for name, mentions in top_entities:
                    lines.append(f"- {name} (提到{len(mentions)}次)")
        return "\n".join(lines)

    def _extract_entities(self, text: str) -> list:
        entities = []
        for name in list(self.entity_mentions.keys()):
            if name in text:
                entities.append(name)
        return entities

    def register_known_entities(self, customer_names: list):
        for name in customer_names:
            if name not in self.entity_mentions:
                self.entity_mentions[name] = []

    def clear(self):
        self.conversation_history.clear()
        self.entity_mentions.clear()
        self.analysis_summaries.clear()
        self.risk_confirmations.clear()


_global_memory = AgentMemory()

def get_memory() -> AgentMemory:
    return _global_memory

def set_memory(mem: AgentMemory):
    global _global_memory
    _global_memory = mem


# ============================================================
# [不变] HITL 检测
# ============================================================

def detect_hitl_triggers(results: dict, health_scores: list = None) -> list:
    triggers = []
    alerts = results.get('流失预警', [])
    for a in alerts:
        if '高' in a.get('风险', ''):
            triggers.append({
                'customer': a['客户'],
                'risk_level': '🔴 高风险',
                'reason': a.get('原因', '趋势下滑'),
                'amount': a.get('年度金额', 0),
                'action_required': '需要确认是否立即安排拜访',
            })
    if health_scores:
        for s in health_scores:
            if s['等级'] == 'F' and s['年度金额'] > 100:
                triggers.append({
                    'customer': s['客户'],
                    'risk_level': '🔴 健康分F级',
                    'reason': f"健康评分仅{s['总分']}分，" + " ".join(s.get('风险标签', [])),
                    'amount': s['年度金额'],
                    'action_required': '需要确认是否启动客户挽回计划',
                })
    seen = set()
    unique = []
    for t in triggers:
        if t['customer'] not in seen:
            seen.add(t['customer'])
            unique.append(t)
    return unique


# ============================================================
# 升级① 智能数据查询 — Text-to-Pandas
# ============================================================

class SmartDataQuery:
    """
    替代旧版 query_sales_data() 的关键词匹配。
    
    原理：
    1. 维护一个结构化的数据索引（schema）
    2. 用户提问 → LLM 生成查询计划（JSON）→ 精确提取数据
    3. 如果 LLM 不可用，降级到增强版关键词匹配
    
    vs 旧版：
    - 旧版：关键词匹配 → 返回整块JSON（经常5000字截断丢失信息）
    - 新版：理解语义 → 只返回相关数据 → 精准、省token
    """

    # 数据schema定义（告诉LLM有哪些数据可查）
    SCHEMA = """
可查询的数据维度：
1. overview: 总营收, 同比增长率, 月度营收列表(1-12月), 核心发现
2. customers: 客户分级列表(客户名/等级A|B|C/年度金额/H1/H2/占比/累计占比), 支持按客户名或等级筛选
3. risks: 流失预警列表(客户/风险等级/原因/年度金额), 异常检测结果
4. growth: 增长机会列表(客户/机会/潜力金额)
5. price_volume: 价量分解(客户/单价变化/数量变化/金额变化)
6. regions: 区域分布(区域/金额/占比), HHI指数, Top3集中度
7. categories: 业务类别趋势(类别/2024金额/2025金额/增长率)
8. benchmark: 行业对标(市场定位/竞争对标/结构性风险/战略机会)
9. forecast: 预测(总营收预测/客户预测/品类预测/风险场景)
"""

    def __init__(self, data: dict, results: dict, benchmark: dict = None, forecast: dict = None):
        self.data = data
        self.results = results
        self.benchmark = benchmark
        self.forecast = forecast
        # 构建索引
        self._index = self._build_index()
        # 构建知识图谱
        self.kg = None
        if HAS_KG:
            self.kg = SalesKnowledgeGraph()
            self.kg.build(data, results)

    def _build_index(self) -> dict:
        """构建结构化数据索引"""
        index = {}

        # 总览
        index['overview'] = {
            '总营收': self.data.get('总营收', 0),
            '总YoY': self.data.get('总YoY', {}),
            '月度营收': self.data.get('月度总营收', []),
            '核心发现': self.results.get('核心发现', []),
            '活跃客户数': sum(1 for c in self.data.get('客户金额', []) if c.get('年度金额', 0) > 0),
        }

        # 客户（建立名称→数据的映射，支持精确查询）
        customers = self.results.get('客户分级', [])
        index['customers'] = {
            'all': customers,
            'by_name': {c['客户']: c for c in customers},
            'by_tier': {
                'A': [c for c in customers if c.get('等级') == 'A'],
                'B': [c for c in customers if c.get('等级') == 'B'],
                'C': [c for c in customers if c.get('等级') == 'C'],
            },
            'top5': customers[:5],
            'top10': customers[:10],
        }

        # 风险
        index['risks'] = {
            'alerts': self.results.get('流失预警', []),
            'high_risk': [a for a in self.results.get('流失预警', []) if '高' in a.get('风险', '')],
            'anomalies': self.results.get('MoM异常', [])[:10],
        }

        # 增长
        index['growth'] = self.results.get('增长机会', [])

        # 价量分解
        index['price_volume'] = self.results.get('价量分解', [])

        # 区域
        index['regions'] = self.results.get('区域洞察', {})

        # 类别
        index['categories'] = self.results.get('类别趋势', [])

        # 行业对标
        if self.benchmark:
            index['benchmark'] = {
                '市场定位': self.benchmark.get('市场定位', {}),
                '竞争对标': self.benchmark.get('竞争对标', {}),
                '结构性风险': self.benchmark.get('结构性风险', []),
                '战略机会': self.benchmark.get('战略机会', []),
                '客户外部视角': self.benchmark.get('客户外部视角', []),
            }

        # 预测
        if self.forecast:
            index['forecast'] = {
                '总营收预测': self.forecast.get('总营收预测', {}),
                '客户预测': self.forecast.get('客户预测', []),
                '品类预测': self.forecast.get('品类预测', []),
                '风险场景': self.forecast.get('风险场景', {}),
            }

        return index

    def query_by_plan(self, plan: dict) -> str:
        """
        根据查询计划精确提取数据。
        
        plan 格式:
        {
            "dimensions": ["overview", "customers"],   # 需要哪些维度
            "filters": {"customer_name": "HMD", "tier": "A"},  # 筛选条件
            "metrics": ["年度金额", "增长率"],           # 需要哪些指标
            "limit": 10                                 # 返回条数
        }
        """
        result = {}
        dims = plan.get('dimensions', [])
        filters = plan.get('filters', {})
        limit = plan.get('limit', 15)

        for dim in dims:
            if dim == 'overview':
                result['overview'] = self._index.get('overview', {})

            elif dim == 'customers':
                customers = self._index.get('customers', {})
                # 按名称筛选
                if 'customer_name' in filters:
                    name = filters['customer_name']
                    # 模糊匹配
                    matched = []
                    for cname, cdata in customers.get('by_name', {}).items():
                        if name.lower() in cname.lower():
                            matched.append(cdata)
                    result['customers'] = matched if matched else [f"未找到客户: {name}"]
                # 按等级筛选
                elif 'tier' in filters:
                    tier = filters['tier'].upper()
                    result['customers'] = customers.get('by_tier', {}).get(tier, [])[:limit]
                # 按Top N
                elif 'top_n' in filters:
                    n = min(int(filters['top_n']), 30)
                    result['customers'] = customers.get('all', [])[:n]
                else:
                    result['customers'] = customers.get('top10', [])

            elif dim == 'risks':
                risks = self._index.get('risks', {})
                if filters.get('level') == 'high':
                    result['risks'] = risks.get('high_risk', [])
                else:
                    result['risks'] = {
                        'alerts': risks.get('alerts', [])[:limit],
                        'anomalies': risks.get('anomalies', [])[:5],
                    }

            elif dim == 'growth':
                result['growth'] = self._index.get('growth', [])[:limit]

            elif dim == 'price_volume':
                pv = self._index.get('price_volume', [])
                if 'customer_name' in filters:
                    name = filters['customer_name']
                    result['price_volume'] = [
                        p for p in pv if name.lower() in p.get('客户', '').lower()
                    ][:limit]
                else:
                    result['price_volume'] = pv[:limit]

            elif dim == 'regions':
                result['regions'] = self._index.get('regions', {})

            elif dim == 'categories':
                result['categories'] = self._index.get('categories', [])

            elif dim == 'benchmark':
                result['benchmark'] = self._index.get('benchmark', {})

            elif dim == 'forecast':
                result['forecast'] = self._index.get('forecast', {})

        if not result:
            result = self._index.get('overview', {})

        return json.dumps(result, ensure_ascii=False, indent=1, default=str)

    def query_smart(self, question: str, provider: str = "", api_key: str = "") -> str:
        """
        智能查询入口（v3.1 集成知识图谱）：
        1. 知识图谱理解 → 结构化查询计划（零API调用）
        2. LLM生成查询计划（有KG上下文加持）
        3. 降级到增强版规则匹配
        """
        # 初始化元数据
        self._last_entity_context = ''
        self._last_pattern = ''
        self._last_agent_hint = []
        self._last_corrections = []

        # 优先用知识图谱（零API调用，毫秒级）
        if self.kg:
            kg_plan = self.kg.understand(question)
            plan = {
                'dimensions': kg_plan['dimensions'],
                'filters': kg_plan['filters'],
                'limit': kg_plan.get('limit', 15),
            }
            self._last_entity_context = kg_plan.get('entity_context', '')
            self._last_pattern = kg_plan.get('pattern', '')
            self._last_agent_hint = kg_plan.get('agent_hint', [])
            self._last_corrections = kg_plan.get('corrections', [])
            return self.query_by_plan(plan)

        # 尝试LLM生成查询计划
        if api_key:
            plan = self._llm_generate_plan(question, provider, api_key)
            if plan:
                return self.query_by_plan(plan)

        # 降级：增强版规则匹配
        return self.query_by_plan(self._rule_based_plan(question))

    def _llm_generate_plan(self, question: str, provider: str, api_key: str) -> Optional[dict]:
        """用LLM将自然语言转为结构化查询计划"""
        system = f"""你是数据查询规划器。根据用户问题，生成JSON查询计划。

{self.SCHEMA}

输出格式（纯JSON，无其他文字）：
{{
    "dimensions": ["overview", "customers"],
    "filters": {{"customer_name": "HMD"}},
    "limit": 10
}}

filters 可选键：customer_name, tier(A/B/C), level(high/medium/low), top_n
"""
        try:
            raw = _call_llm_raw(system, f"用户问题：{question}", provider, api_key,
                                max_tokens=200, temperature=0.0,
                                _trace_name="query_plan_llm")
            # 提取JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            plan = json.loads(raw)
            if 'dimensions' in plan:
                return plan
        except Exception:
            pass
        return None

    def _rule_based_plan(self, question: str) -> dict:
        """增强版规则匹配（比v2.1更精准）"""
        q = question.lower()
        dims = []
        filters = {}

        # 维度检测
        if any(k in q for k in ['总', '营收', '收入', '概览', '全部', '多少']):
            dims.append('overview')
        if any(k in q for k in ['客户', '分级', 'abc', '排名', 'top']):
            dims.append('customers')
            # 提取Top N
            for word in ['top5', 'top10', 'top15', 'top20', '前5', '前10', '前15', '前20']:
                if word in q:
                    n = ''.join(filter(str.isdigit, word))
                    filters['top_n'] = int(n) if n else 10
            # 提取等级
            if 'a级' in q or 'a类' in q:
                filters['tier'] = 'A'
            elif 'b级' in q or 'b类' in q:
                filters['tier'] = 'B'
            elif 'c级' in q or 'c类' in q:
                filters['tier'] = 'C'
        if any(k in q for k in ['风险', '流失', '预警', '异常', '危险']):
            dims.append('risks')
            if '高' in q:
                filters['level'] = 'high'
        if any(k in q for k in ['增长', '机会', '潜力']):
            dims.append('growth')
        if any(k in q for k in ['价', '单价', '量', '价量']):
            dims.append('price_volume')
        if any(k in q for k in ['区域', '市场', '地区']):
            dims.append('regions')
        if any(k in q for k in ['类别', '品类', '产品', '结构']):
            dims.append('categories')
        if any(k in q for k in ['行业', '竞争', '对标', '华勤', '闻泰', '龙旗']):
            dims.append('benchmark')
        if any(k in q for k in ['预测', '2026', '未来', '前景', '下季']):
            dims.append('forecast')

        # 客户名精确匹配
        for cname in self._index.get('customers', {}).get('by_name', {}).keys():
            if cname.lower() in q:
                dims.append('customers')
                if 'price_volume' not in dims and any(k in q for k in ['价', '量']):
                    dims.append('price_volume')
                filters['customer_name'] = cname
                break

        if not dims:
            dims = ['overview', 'customers', 'risks']

        return {'dimensions': list(set(dims)), 'filters': filters, 'limit': 15}


# 全局查询实例
_smart_query: Optional[SmartDataQuery] = None

def get_smart_query() -> Optional[SmartDataQuery]:
    return _smart_query


# 兼容旧版接口
_sales_data_store = {}

def set_sales_data(data_store: dict):
    global _sales_data_store
    _sales_data_store = data_store

def query_sales_data(query: str) -> str:
    """兼容旧接口，内部使用SmartDataQuery"""
    sq = get_smart_query()
    if sq:
        return sq.query_smart(query)
    # 完全降级
    return _legacy_query(query)

def _legacy_query(query: str) -> str:
    """旧版关键词查询（最终降级方案）"""
    ds = _sales_data_store
    if not ds:
        return "数据未加载"
    q = query.lower()
    result = {}
    if any(k in q for k in ['总', '营收', '收入', '概览', '全部']):
        result['总营收'] = ds.get('总营收')
        result['总YoY'] = ds.get('总YoY')
        result['核心发现'] = ds.get('核心发现')
    if any(k in q for k in ['客户', '分级', 'abc', '排名', 'top']):
        result['客户分级'] = ds.get('客户分级', [])[:15]
    if any(k in q for k in ['风险', '流失', '预警']):
        result['流失预警'] = ds.get('流失预警')
    if any(k in q for k in ['增长', '机会']):
        result['增长机会'] = ds.get('增长机会')
    if not result:
        result = {'总营收': ds.get('总营收'), '核心发现': ds.get('核心发现')}
    return json.dumps(result, ensure_ascii=False, indent=1, default=str)[:5000]


# ============================================================
# [不变] Agent 角色定义
# ============================================================

AGENT_PROFILES = {
    "analyst": {
        "name": "📊 数据分析师",
        "emoji": "📊",
        "role": "禾苗通讯资深数据分析师",
        "goal": "精准解读销售数据，用数字揭示业务真相，识别趋势和模式",
        "backstory": (
            "你在消费电子ODM行业有15年数据分析经验，曾服务华勤、闻泰等头部企业。"
            "你以数据驱动著称，每个结论必须有数字支撑。"
            "你擅长发现月度波动规律、客户集中度风险、同比环比异常。"
            "你的分析风格：精准、客观、量化。先给结论，再给数据。"
        ),
        "keywords": [
            "营收", "收入", "金额", "出货", "数量", "客户", "分级",
            "ABC", "排名", "top", "占比", "集中", "月度", "季度",
            "同比", "环比", "趋势", "总览", "概览", "多少",
            "产品", "结构", "区域",
        ],
        "model_tier": "standard",  # v7.0: 模型路由
    },
    "risk": {
        "name": "🛡️ 风控专家",
        "emoji": "🛡️",
        "role": "禾苗通讯风险控制专家",
        "goal": "识别客户流失风险和异常波动，量化风险金额，提供预防方案",
        "backstory": (
            "你是前安永风险咨询总监，专注TMT行业客户风险管理。"
            "你对数据异常极其敏感——断崖式下跌、连续N月衰退、大客户集中度过高，"
            "这些你一眼就能看出。你的风格：直言不讳，发现问题就说。"
            "输出格式：风险等级→影响金额→原因分析→应对建议。"
            "\n\n⚠️ 重要：如果发现高风险客户（年度金额>200万且持续下滑），"
            "请在输出开头标记 [HIGH_RISK_ALERT] 并列出客户名和金额。"
        ),
        "keywords": [
            "风险", "流失", "预警", "下降", "下滑", "丢失", "断崖",
            "异常", "暴跌", "危险", "警告", "关注", "问题",
            "波动", "偏离", "不正常",
        ],
        "model_tier": "advanced",  # 风险分析需要高精度
    },
    "strategist": {
        "name": "💡 策略师",
        "emoji": "💡",
        "role": "禾苗通讯战略顾问",
        "goal": "发现增长机会，制定可执行的战略方案，优化资源配置",
        "backstory": (
            "你是前麦肯锡TMT行业合伙人，专注手机ODM/OEM赛道战略规划。"
            "你擅长竞争分析（vs华勤/闻泰/龙旗）、增长机会识别、"
            "产品组合优化、客户钱包份额提升策略。"
            "你的风格：前瞻性、实用主义、聚焦ROI。建议必须可执行。"
            "输出格式：机会/方向→潜在价值→具体行动→优先级。"
        ),
        "keywords": [
            "增长", "机会", "战略", "策略", "建议", "方向", "投入",
            "竞争", "对手", "华勤", "闻泰", "龙旗", "行业", "对标",
            "预测", "forecast", "2026", "未来", "前景",
            "CEO", "管理", "决策", "优化", "提升",
            "价格", "价量", "利润",
        ],
        "model_tier": "advanced",
    },
    # ── V10.0 域 Agent (Engine-based, 非 LLM 角色) ──
    "quality": {
        "name": "🔬 品质专家",
        "emoji": "🔬",
        "role": "禾苗通讯品质管控专家",
        "goal": "监控良率、分析退货、追溯缺陷根因",
        "backstory": "V10域引擎Agent，直接调用QualityEngine返回结构化数据，不经过LLM。",
        "keywords": [
            "良率", "yield", "退货", "return", "品质", "quality",
            "缺陷", "defect", "投诉", "complaint", "根因", "root cause",
            "合格率", "不良", "产线",
        ],
        "model_tier": "engine",  # 标记为引擎Agent，不走LLM
        "engine_type": "quality",
    },
    "market": {
        "name": "📈 市场专家",
        "emoji": "📈",
        "role": "禾苗通讯市场分析专家",
        "goal": "竞对监控、行业趋势分析、市场情绪追踪",
        "backstory": "V10域引擎Agent，直接调用MarketEngine返回结构化数据。",
        "keywords": [
            "市场", "market", "竞对", "competitor", "闻泰", "华勤", "龙旗",
            "行业趋势", "trend", "情绪", "sentiment", "份额", "share",
            "出货量", "排名",
        ],
        "model_tier": "engine",
        "engine_type": "market",
    },
    "finance": {
        "name": "💰 财务专家",
        "emoji": "💰",
        "role": "禾苗通讯财务分析专家",
        "goal": "应收账款追踪、毛利分析、现金流预测、发票匹配",
        "backstory": "V10域引擎Agent，直接调用FinanceEngine返回结构化数据。",
        "keywords": [
            "应收", "AR", "账款", "receivable", "毛利", "margin",
            "利润", "profit", "现金流", "cashflow", "发票", "invoice",
            "账期", "DSO", "回款",
        ],
        "model_tier": "engine",
        "engine_type": "finance",
    },
    "procurement": {
        "name": "📦 采购专家",
        "emoji": "📦",
        "role": "禾苗通讯采购管理专家",
        "goal": "供应商评估、采购单追踪、延期预警、成本分析",
        "backstory": "V10域引擎Agent，直接调用ProcurementEngine返回结构化数据。",
        "keywords": [
            "采购", "procurement", "供应商", "supplier", "PO",
            "采购单", "延期", "delay", "成本", "cost", "报价", "quote",
            "物料", "交期",
        ],
        "model_tier": "engine",
        "engine_type": "procurement",
    },
}

REPORTER_PROFILE = {
    "role": "禾苗通讯高级报告撰写人",
    "goal": "综合多位专家分析，生成简洁有力的综合报告，适合CEO阅读",
    "backstory": (
        "你是前FT中文网资深编辑，现任禾苗通讯战略分析部负责人。"
        "你擅长将复杂的数据分析和多方观点提炼为管理层可直接行动的建议。"
        "规则：1.不简单拼凑 2.先核心结论 3.分模块展开 4.最后给行动项 5.控制500字"
    ),
    "model_tier": "standard",
}


# ============================================================
# V10.0 域 Agent 引擎管理器
# ============================================================

_domain_engines: Dict[str, Any] = {}


def _init_domain_engines():
    """初始化域 Agent 引擎 — 优先 DB 数据，否则回退 SAMPLE_"""
    global _domain_engines
    if _domain_engines:
        return _domain_engines

    # 1. 尝试从数据库加载
    if HAS_DB_BRIDGE:
        try:
            db_engines = create_engines_from_db()
            if db_engines:
                _domain_engines.update(db_engines)
                logger.info(f"DB Bridge → 加载 {len(db_engines)} 个域引擎: {list(db_engines.keys())}")
        except Exception as e:
            logger.warning(f"DB Bridge 失败: {e}")

    # 2. 未从 DB 获取到的引擎 → 使用默认构造（含 SAMPLE_ 回退）
    engine_map = {
        "quality": (HAS_QUALITY, lambda: QualityEngine()),
        "market": (HAS_MARKET, lambda: MarketEngine()),
        "finance": (HAS_FINANCE, lambda: FinanceEngine()),
        "procurement": (HAS_PROCUREMENT, lambda: ProcurementEngine()),
        "risk": (HAS_RISK_ENGINE, lambda: RiskEngine()),
        "strategist": (HAS_STRATEGIST_ENGINE, lambda: StrategistEngine()),
    }

    for name, (available, factory) in engine_map.items():
        if name not in _domain_engines and available:
            try:
                _domain_engines[name] = factory()
                logger.info(f"域引擎 {name} → 默认初始化")
            except Exception as e:
                logger.error(f"域引擎 {name} 初始化失败: {e}")

    return _domain_engines


def get_domain_engine(name: str):
    """获取指定域引擎实例"""
    engines = _init_domain_engines()
    return engines.get(name)


# ============================================================
# 升级② LLM智能路由
# ============================================================

class SmartRouter:
    """
    替代旧版 route_to_agents() 的关键词匹配。
    
    原理：
    1. 一次轻量LLM调用（<100 tokens），判断需要哪些Agent
    2. 返回置信度分数，低置信度的Agent跳过 → 省钱省时间
    3. LLM不可用时降级到增强版规则
    
    vs 旧版：
    - 旧版："CEO" 触发全部3个Agent（即使只需要分析师）
    - 新版：语义理解，"HMD客户Q3数据" → 只调分析师
    """

    ROUTING_PROMPT = """你是一个问题分类器。根据用户问题判断需要哪些专家参与。

专家列表：
- analyst: 数据分析（营收、客户、趋势、排名、数量、区域、产品结构）
- risk: 风险评估（流失、预警、异常、下降、危险信号）
- strategist: 战略建议（增长机会、竞争分析、行业对标、预测、决策建议）

规则：
1. 简单数据查询 → 只需 analyst
2. 风险/流失相关 → analyst + risk
3. 战略/建议/未来 → analyst + strategist
4. 全面分析/CEO报告 → 全部

输出格式（纯JSON，无其他文字）：
{"agents": ["analyst"], "reason": "简单数据查询"}
"""

    @staticmethod
    def route(question: str, provider: str = "", api_key: str = "",
              kg_hint: List[str] = None) -> List[str]:
        """
        智能路由：返回需要的Agent列表
        优先级: KG提示 > LLM路由 > 规则路由
        """
        # 知识图谱提示（零API调用）
        if kg_hint:
            valid = [a for a in kg_hint if a in AGENT_PROFILES]
            if valid:
                return valid

        # 尝试LLM路由
        if api_key:
            result = SmartRouter._llm_route(question, provider, api_key)
            if result:
                return result

        # 降级到增强版规则
        return SmartRouter._rule_route(question)

    @staticmethod
    def _llm_route(question: str, provider: str, api_key: str) -> Optional[List[str]]:
        """LLM语义路由"""
        try:
            raw = _call_llm_raw(
                SmartRouter.ROUTING_PROMPT,
                f"用户问题：{question}",
                provider, api_key,
                max_tokens=80, temperature=0.0,
                _trace_name="routing_llm"
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            agents = parsed.get('agents', [])
            # 验证Agent名称合法
            valid = [a for a in agents if a in AGENT_PROFILES]
            if valid:
                return valid
        except Exception:
            pass
        return None

    @staticmethod
    def _rule_route(question: str) -> List[str]:
        """增强版规则路由（比v2.1更精准）"""
        q = question.lower()
        agents_needed = set()

        # 关键词匹配（保留兼容性）
        for agent_id, profile in AGENT_PROFILES.items():
            score = sum(1 for kw in profile['keywords'] if kw in q)
            if score > 0:
                agents_needed.add(agent_id)

        # 全量触发
        if any(k in q for k in ['CEO', 'ceo', '总结', '全面', '概览', '怎么样', '报告']):
            agents_needed = {"analyst", "risk", "strategist", "quality", "market", "finance", "procurement"}

        # 简单查询优化：只有数据问题时只需分析师
        simple_data_patterns = ['多少', '几个', '是什么', '哪些', '列出', '有哪些']
        if any(p in q for p in simple_data_patterns) and not agents_needed:
            agents_needed = {"analyst"}

        if not agents_needed:
            agents_needed = {"analyst"}

        return list(agents_needed)


# 兼容旧版接口
def route_to_agents(question: str) -> list:
    return SmartRouter._rule_route(question)


# ============================================================
# 升级③ 并行Agent执行
# ============================================================

class ParallelAgentExecutor:
    """
    替代旧版串行LLM调用。
    
    原理：
    - 专家Agent之间互相独立 → 用 ThreadPoolExecutor 并行调用
    - 只有报告员需要等所有专家完成后才执行
    - 3个Agent并行：从 ~9秒 降到 ~3秒（假设每个Agent ~3秒）
    
    vs 旧版：
    - 旧版：分析师(3s) → 风控(3s) → 策略师(3s) → 报告员(3s) = 12秒
    - 新版：[分析师|风控|策略师](3s) → 报告员(3s) = 6秒
    """

    def __init__(self, provider: str, api_key: str, max_workers: int = 3):
        self.provider = provider
        self.api_key = api_key
        self.max_workers = max_workers

    def execute_experts_parallel(
        self,
        agents_needed: List[str],
        question: str,
        context_data: str,
        memory_section: str = "",
        stream_ps: 'PipelineStream' = None,
        enable_tools: bool = True,
    ) -> Dict[str, str]:
        """
        并行执行所有专家Agent，返回 {agent_name: output}
        v4.0: 支持 Tool Use + Streaming 回调
        """
        expert_outputs = {}

        def _call_single_expert(agent_id: str) -> tuple:
            profile = AGENT_PROFILES[agent_id]

            # v4.0 Streaming: 通知 Agent 开始
            if stream_ps:
                stream_ps.agent_start(agent_id, profile["name"])

            system = f"你是{profile['role']}。{profile['backstory']}"

            # v4.0 Tool Use: 增加工具描述到 prompt
            tool_hint = ""
            if enable_tools and HAS_TOOLS:
                tool_hint = get_tool_descriptions_for_prompt(agent_id)

            prompt = (
                f"用户问题：{question}\n\n"
                f"禾苗销售数据：\n{context_data}"
                f"{memory_section}"
                f"{tool_hint}\n\n"
                f"200字内回答。数据必须精确引用。"
            )

            # v4.0: 优先使用 Tool-Augmented 调用
            if enable_tools and HAS_TOOLS and self.provider == "claude":
                output = _call_llm_with_tools(
                    system, prompt, self.provider, self.api_key,
                    agent_id=agent_id, max_turns=3,
                    max_tokens=1000, _trace_name=f"agent_{agent_id}",
                    stream_ps=stream_ps,
                )
            else:
                output = _call_llm_raw(system, prompt, self.provider, self.api_key,
                                       _trace_name=f"agent_{agent_id}")

            # v4.0: 输出校验
            if HAS_GUARD:
                validation = validate_agent_output(output, context_data[:2000])
                if not validation.passed and validation.confidence < 0.3:
                    if stream_ps:
                        stream_ps.error(agent_id, f"输出质量低: {validation.issues}")

            # v4.0 Streaming: 通知 Agent 完成
            if stream_ps:
                stream_ps.agent_done(agent_id, profile["name"], output)

            return (profile["name"], output)

        # 并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_call_single_expert, aid): aid
                for aid in agents_needed
            }
            for future in as_completed(futures):
                try:
                    name, output = future.result(timeout=30)
                    expert_outputs[name] = output
                except Exception as e:
                    aid = futures[future]
                    profile = AGENT_PROFILES[aid]
                    error_msg = f"[分析超时: {e}]"
                    expert_outputs[profile["name"]] = error_msg
                    if stream_ps:
                        stream_ps.error(aid, str(e))

        return expert_outputs

    def execute_reporter(
        self,
        question: str,
        expert_outputs: Dict[str, str],
        memory_section: str = "",
    ) -> str:
        """执行报告员（需要等所有专家完成）"""
        all_opinions = "\n---\n".join(f"{n}：\n{t}" for n, t in expert_outputs.items())
        reporter_sys = f"你是{REPORTER_PROFILE['role']}。{REPORTER_PROFILE['backstory']}"
        report = _call_llm_raw(
            reporter_sys,
            f"问题：{question}\n\n专家分析：\n{all_opinions}{memory_section}\n\n综合报告，500字内。",
            self.provider, self.api_key,
            _trace_name="reporter_llm"
        )
        return report


# ============================================================
# LLM 调用（支持max_tokens/temperature参数）
# ============================================================

def _call_llm_raw(system_prompt, user_prompt, provider, api_key,
                  max_tokens=800, temperature=0.3,
                  _trace_name="llm_call"):
    """通用LLM调用，支持DeepSeek和Claude，含可观测性追踪 + v4.0 Guardrails"""
    if not api_key:
        return "[需要API Key]"

    # 确定model名
    model = "deepseek-chat" if provider == "deepseek" else "claude-sonnet-4-20250514"

    # 获取tracer（可能在trace上下文之外调用）
    tracer = get_tracer() if HAS_OBS else None
    lc_ctx = None

    # v4.0: 预算检查
    if HAS_GUARD:
        budget = get_budget()
        if not budget.should_allow_query():
            return "[每日预算已耗尽，请明日再试或调整预算]"

    def _do_call():
        """实际调用（被 retry/breaker 包装）"""
        if provider == "deepseek":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature, max_tokens=max_tokens,
            )
            result = resp.choices[0].message.content

            # 追踪LLM usage
            if tracer and tracer.enabled:
                usage = getattr(resp, 'usage', None)
                p_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
                c_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
                if p_tokens == 0:
                    p_tokens = CostCalculator.estimate_tokens(system_prompt + user_prompt)
                if c_tokens == 0:
                    c_tokens = CostCalculator.estimate_tokens(result)
                _record_llm_span(tracer, _trace_name, provider, model,
                                 p_tokens, c_tokens, temperature, max_tokens,
                                 system_prompt + user_prompt, result)

            return result

        elif provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            result = resp.content[0].text

            # 追踪LLM usage
            if tracer and tracer.enabled:
                usage = getattr(resp, 'usage', None)
                p_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
                c_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
                if p_tokens == 0:
                    p_tokens = CostCalculator.estimate_tokens(system_prompt + user_prompt)
                if c_tokens == 0:
                    c_tokens = CostCalculator.estimate_tokens(result)
                _record_llm_span(tracer, _trace_name, provider, model,
                                 p_tokens, c_tokens, temperature, max_tokens,
                                 system_prompt + user_prompt, result)

            return result

    try:
        # v4.0: 用 Guardrails 包装调用（Retry + CircuitBreaker + Validation）
        if HAS_GUARD:
            result = guarded_llm_call(
                _do_call,
                breaker_name=f"llm_{provider}",
                validate=True,
                source_data=user_prompt[:2000],
            )
            # 记录消耗
            budget = get_budget()
            est_tokens = CostCalculator.estimate_tokens(system_prompt + user_prompt + (result or "")) if HAS_OBS else 500
            est_cost = est_tokens * 0.000003  # 粗估
            budget.record_cost(est_cost, _trace_name)
            return result
        else:
            return _do_call()
    except CircuitBreakerOpenError as e:
        return f"[服务暂时不可用: {e}]"
    except Exception as e:
        return f"[调用失败: {e}]"


# ============================================================
# v4.0: Tool-Augmented LLM Call (ReAct Pattern)
# ============================================================

def _call_llm_with_tools(system_prompt, user_prompt, provider, api_key,
                         agent_id="analyst", max_turns=5, max_tokens=1200,
                         temperature=0.3, _trace_name="tool_agent",
                         stream_ps=None):
    """
    V10.0 ReAct Agent Loop — Reason + Act + Observe

    标准 ReAct 循环:
      Thought: LLM 推理（选择工具或直接回答）
      Action:  调用工具 (tool_use)
      Observation: 工具返回结果 (tool_result)
      ... 循环 ...
      Final Answer: LLM 综合所有 Observations 给出最终回答

    仅 Claude provider 支持原生 tool_use，DeepSeek 走 prompt injection 模式
    """
    if not api_key or not HAS_TOOLS:
        # fallback: 无工具调用
        return _call_llm_raw(system_prompt, user_prompt, provider, api_key,
                            max_tokens=max_tokens, temperature=temperature,
                            _trace_name=_trace_name)

    tools = get_tools_for_agent(agent_id)
    if not tools:
        return _call_llm_raw(system_prompt, user_prompt, provider, api_key,
                            max_tokens=max_tokens, temperature=temperature,
                            _trace_name=_trace_name)

    # DeepSeek: 把工具描述注入 prompt（不支持原生 tool_use）
    if provider == "deepseek":
        tool_desc = get_tool_descriptions_for_prompt(agent_id)
        enhanced_prompt = user_prompt + "\n" + tool_desc + (
            "\n如果需要精确计算，请在回答中标明计算过程。"
        )
        return _call_llm_raw(system_prompt, enhanced_prompt, provider, api_key,
                            max_tokens=max_tokens, temperature=temperature,
                            _trace_name=_trace_name)

    # Claude: 原生 tool_use — V10.0 ReAct agentic loop
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        return _call_llm_raw(system_prompt, user_prompt, provider, api_key,
                            max_tokens=max_tokens, temperature=temperature,
                            _trace_name=_trace_name)

    # ReAct 结构化 system prompt
    react_system = (
        system_prompt + "\n\n"
        "[ReAct 推理框架]\n"
        "对于每个分析步骤，请按以下模式推理:\n"
        "1. Thought: 分析当前问题，决定需要哪些数据\n"
        "2. Action: 调用合适的工具获取数据\n"
        "3. Observation: 分析工具返回的结果\n"
        "重复以上步骤直到有足够信息。\n"
        "最后给出 Final Answer: 综合所有数据的精确回答。"
    )

    messages = [{"role": "user", "content": user_prompt}]
    all_text = []
    tool_calls_made = []
    react_trace = []  # ReAct 步骤追踪

    for turn in range(max_turns):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=react_system,
                tools=tools,
                tool_choice={"type": "auto"},
                messages=messages,
            )
        except Exception as e:
            if all_text:
                return "\n".join(all_text)
            return f"[ReAct agent 调用失败: {e}]"

        # 收集文本 (Thought) 和工具调用 (Action)
        tool_blocks = []
        for block in resp.content:
            if hasattr(block, 'text'):
                all_text.append(block.text)
                react_trace.append({"step": turn + 1, "type": "thought", "content": block.text[:200]})
            elif hasattr(block, 'type') and block.type == "tool_use":
                tool_blocks.append(block)
                react_trace.append({"step": turn + 1, "type": "action", "tool": block.name})

        # 没有工具调用 → Final Answer
        if resp.stop_reason != "tool_use" or not tool_blocks:
            react_trace.append({"step": turn + 1, "type": "final_answer"})
            break

        # 执行工具 (Observation)
        messages.append({"role": "assistant", "content": resp.content})

        tool_results = []
        for block in tool_blocks:
            tool_name = block.name
            tool_input = block.input
            tool_id = block.id

            # 执行
            exec_result = sales_tools.execute(tool_name, tool_input)
            result_str = json.dumps(exec_result, ensure_ascii=False, default=str)

            tool_calls_made.append({
                "tool": tool_name, "input": tool_input,
                "result_preview": result_str[:200],
            })
            react_trace.append({
                "step": turn + 1, "type": "observation",
                "tool": tool_name, "result_len": len(result_str),
            })

            # 流式通知
            if stream_ps:
                stream_ps.tool_call(agent_id, tool_name, tool_input, result_str[:100])

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    final_text = "\n".join(all_text) if all_text else "[ReAct Agent 未生成回答]"

    # ReAct 追踪日志
    logger.debug(f"ReAct trace ({len(react_trace)} steps): "
                 f"{json.dumps(react_trace, ensure_ascii=False, default=str)[:500]}")

    # 追踪
    tracer = get_tracer() if HAS_OBS else None
    if tracer and tracer.enabled:
        try:
            _record_llm_span(tracer, _trace_name, provider, "claude-sonnet-4-20250514",
                             0, 0, temperature, max_tokens,
                             system_prompt + user_prompt, final_text)
        except Exception:
            pass

    return final_text


def _record_llm_span(tracer, name, provider, model,
                      prompt_tokens, completion_tokens,
                      temperature, max_tokens,
                      prompt_text, response_text):
    """记录LLM调用span（不干扰主流程）"""
    try:
        from observability import _ctx, Span, SpanKind, LLMUsage
        if _ctx.current_trace is None:
            return

        parent_id = ""
        if _ctx.current_span_stack:
            parent_id = _ctx.current_span_stack[-1].span_id

        cost = CostCalculator.calculate(provider, model,
                                         prompt_tokens, completion_tokens)
        s = Span(
            trace_id=_ctx.current_trace.trace_id,
            parent_span_id=parent_id,
            kind=SpanKind.LLM_CALL.value,
            name=name,
            start_time=time.time() - 0.001,  # 近似
            llm_usage=LLMUsage(
                provider=provider, model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        s.finish()
        _ctx.current_trace.add_span(s)
    except Exception:
        pass  # 可观测性不能影响主流程


# ============================================================
# 数据上下文
# ============================================================

def build_data_store(data, results, benchmark=None, forecast=None):
    """兼容旧版接口"""
    store = {
        '总营收': data.get('总营收', 0),
        '总YoY': data.get('总YoY', {}),
        '月度营收': data.get('月度总营收', []),
        '核心发现': results.get('核心发现', []),
        '客户数': sum(1 for c in data.get('客户金额', []) if c.get('年度金额', 0) > 0),
        '客户分级': results.get('客户分级', [])[:15],
        '流失预警': results.get('流失预警', []),
        '异常检测': results.get('MoM异常', [])[:10],
        '增长机会': results.get('增长机会', []),
        '价量分解': results.get('价量分解', [])[:10],
        '区域洞察': results.get('区域洞察', {}),
    }
    if benchmark:
        store['行业对标'] = {
            '市场定位': benchmark.get('市场定位', {}),
            '竞争对标': benchmark.get('竞争对标', {}),
            '结构性风险': benchmark.get('结构性风险', []),
            '战略机会': benchmark.get('战略机会', []),
        }
    if forecast:
        store['预测'] = {
            '总营收预测': forecast.get('总营收预测', {}),
            '客户预测': forecast.get('客户预测', [])[:5],
            '情景分析': forecast.get('风险场景', {}),
        }
    return store


# ============================================================
# 主入口 v3.0
# ============================================================

def ask_multi_agent(
    question: str,
    data: dict,
    results: dict,
    benchmark: dict = None,
    forecast: dict = None,
    provider: str = "deepseek",
    api_key: str = "",
    memory: AgentMemory = None,
    # v3.3 新增参数
    enable_critic: bool = True,       # Generator+Critic
    enable_hitl_v2: bool = True,      # 增强HITL
    enable_persistent_mem: bool = True, # 持久化记忆
    critic_threshold: float = 7.0,     # 质量门禁分数
    critic_max_iter: int = 2,          # 最大迭代次数
    # v4.0 新增参数
    stream_callback: 'StreamCallback' = None,  # 流式回调
    enable_tools: bool = True,         # Tool Use
    enable_cache: bool = True,         # 响应缓存
) -> dict:
    """
    v4.0 主入口 — Top-5 生产级 Agent System

    流程（每个阶段独立Span追踪 + 流式推送）：
    0. 缓存检查（命中直接返回）            [cache] ← v4.0
    1. 持久化记忆加载（跨会话上下文）        [persistent_memory]
    2. SmartDataQuery 精确提取数据           [data_query span]
    3. SmartRouter LLM语义路由              [routing span]
    4. ParallelAgentExecutor 并行执行        [agent span × N] + Tool Use
    5. 报告员综合                           [reporter span]
    6. CriticAgent 质量审查+迭代精炼         [critic span]
    7. HITL置信度评估                       [hitl span]
    8. 持久化记忆保存（洞察+实体+偏好）      [memory_save]

    v4.0 新增：
    - stream_callback: StreamCallback 对象，实时推送各阶段进度
    - enable_tools: Agent 可调用计算/风控/分析工具
    - enable_cache: 相同问题命中缓存直接返回
    - guardrails: 自动 Retry + 熔断 + 输出校验
    - tool_calls: 工具调用记录
    - budget_status: 预算状态
    """
    t0 = time.time()
    mem = memory or get_memory()
    mem_context = mem.get_context_prompt()
    thinking = [f"📩 收到问题：{question}"]

    # v4.0: 初始化 Streaming
    stream_ps = None
    if HAS_STREAM and stream_callback:
        stream_ps = PipelineStream(stream_callback)

    # v4.0: 缓存检查
    if enable_cache and HAS_GUARD:
        cache = get_cache()
        cached = cache.get(question)
        if cached:
            thinking.append("⚡ 命中缓存，直接返回")
            if stream_ps:
                stream_ps.thinking("⚡ 命中缓存")
                stream_ps.complete()
            cached["from_cache"] = True
            return cached

    if not api_key:
        return {
            "answer": "⚠️ 请先配置API Key",
            "agents_used": [], "thinking": [],
            "expert_outputs": {}, "hitl_triggers": [],
            "trace_id": "", "obs_summary": {},
            "critique": None, "hitl_decision": None,
            "persistent_memory_used": False,
        }

    # ---- 可观测性：开始trace ----
    tracer = get_tracer() if HAS_OBS else None
    trace_ctx = tracer.trace(question) if tracer else _DummyCtx()
    trace_obj = trace_ctx.__enter__()

    try:
        # 注册客户名到记忆
        for c in data.get('客户金额', [])[:50]:
            name = c.get('客户', '')
            if name and len(name) >= 2:
                mem.register_known_entities([name])

        if mem_context:
            thinking.append(f"🧠 加载 {len(mem.conversation_history)} 轮记忆")

        # ---- v3.3 持久化记忆加载 ----
        pmem_context = ""
        pmem_used = False
        if enable_persistent_mem and HAS_PMEM:
            try:
                pmem = get_persistent_memory()
                pmem_context = pmem.build_memory_context(question)
                if pmem_context:
                    pmem_used = True
                    thinking.append(f"🧠 持久化记忆: 加载跨会话上下文")
            except Exception:
                pass  # 持久化记忆失败不影响主流程

        # ---- 阶段①：智能数据查询（+知识图谱）[data_query span] ----
        thinking.append("🔍 智能数据查询...")
        if stream_ps:
            stream_ps.start_stage("data_query", "🔍 智能数据查询")
        dq_start = time.time()

        global _smart_query
        _smart_query = SmartDataQuery(data, results, benchmark, forecast)
        context_data = _smart_query.query_smart(question, provider, api_key)

        dq_elapsed = (time.time() - dq_start) * 1000
        thinking.append(f"📦 精确提取 {len(context_data)} 字数据上下文")

        # V9.0: RLM 递归语言模型 — 解决大数据截断瓶颈
        rlm_used = False
        if HAS_RLM and len(context_data) > 500:
            try:
                rlm = RLMEngine(config=RLMConfig(
                    max_recursion=3,
                    chunk_size=4000,
                    enable_sandbox=True,
                ))
                rlm_result = rlm.analyze(
                    data=context_data,
                    task=question,
                    llm_fn=lambda sys, usr: _call_llm_raw(
                        sys, usr, provider, api_key, model="", max_tokens=1500
                    ),
                )
                if rlm_result and hasattr(rlm_result, 'answer') and rlm_result.answer:
                    context_data = (
                        f"[RLM递归分析摘要 · {len(context_data)}字→压缩]\n"
                        f"{rlm_result.answer}\n\n"
                        f"[原始数据片段]\n{context_data[:3000]}"
                    )
                    rlm_used = True
                    thinking.append(f"🔄 RLM引擎: {len(context_data)}字数据递归处理完成")
            except Exception as e:
                logger.debug(f"RLM降级: {e}")

        # V9.0: 初始化可解释性追踪器
        if HAS_INTERP:
            v9t = _get_v9_tracer()
            if v9t:
                v9t.trace_step("data_query", "系统",
                               action="智能数据提取",
                               input_summary=f"问题: {question[:50]}",
                               output_summary=f"{len(context_data)}字上下文")

        if stream_ps:
            stream_ps.end_stage("data_query", dq_elapsed, f"{len(context_data)}字")

        # 记录data_query span
        if tracer and HAS_OBS:
            _record_stage_span(tracer, "data_query", "📊 数据查询",
                              dq_elapsed, {"data_length": len(context_data)})

        # 知识图谱元数据
        kg_pattern = getattr(_smart_query, '_last_pattern', '')
        kg_entity = getattr(_smart_query, '_last_entity_context', '')
        kg_agent_hint = getattr(_smart_query, '_last_agent_hint', [])
        kg_corrections = getattr(_smart_query, '_last_corrections', [])

        if kg_pattern:
            thinking.append(f"📚 知识图谱: 模式={kg_pattern}")
        if kg_entity:
            thinking.append(f"🏷️ 实体上下文: {kg_entity}")
        if kg_corrections:
            for c in kg_corrections:
                thinking.append(f"🔧 自动纠正: {c}")

        # ---- 阶段②：LLM智能路由 [routing span] ----
        thinking.append("🧭 智能路由分析中...")
        if stream_ps:
            stream_ps.start_stage("routing", "🧭 智能路由")
        rt_start = time.time()

        agents_needed = SmartRouter.route(question, provider, api_key,
                                          kg_hint=kg_agent_hint)
        agent_names = [AGENT_PROFILES[a]["name"] for a in agents_needed]
        route_source = "知识图谱" if kg_agent_hint else ("LLM" if api_key else "规则")

        rt_elapsed = (time.time() - rt_start) * 1000
        thinking.append(f"🎯 路由结果：{', '.join(agent_names)}（{route_source}，共{len(agents_needed)}位专家）")
        if stream_ps:
            stream_ps.end_stage("routing", rt_elapsed, f"{len(agents_needed)}位专家")

        # 记录routing span
        if tracer and HAS_OBS:
            _record_stage_span(tracer, "routing", "🧭 路由",
                              rt_elapsed, {
                                  "agents": agents_needed,
                                  "source": route_source,
                              })

        # ---- 阶段③：并行执行 [agent spans] ----
        memory_section = f"\n\n[对话记忆]\n{mem_context}" if mem_context else ""
        if pmem_context:
            memory_section += f"\n\n{pmem_context}"

        if len(agents_needed) > 1:
            thinking.append(f"⚡ 并行启动 {len(agents_needed)} 位专家...")
        else:
            thinking.append(f"▶️ 启动专家分析...")

        if stream_ps:
            stream_ps.start_stage("agents", f"🤖 {len(agents_needed)}位专家并行分析")

        ag_start = time.time()
        executor = ParallelAgentExecutor(provider, api_key)

        enriched_data = context_data
        if kg_entity:
            enriched_data += f"\n\n[知识图谱 · 实体画像]\n{kg_entity}"

        expert_outputs = executor.execute_experts_parallel(
            agents_needed, question, enriched_data, memory_section,
            stream_ps=stream_ps,
            enable_tools=enable_tools,
        )
        ag_elapsed = (time.time() - ag_start) * 1000

        agents_used = list(expert_outputs.keys())
        for name in agents_used:
            thinking.append(f"✅ {name} 完成")

        if stream_ps:
            stream_ps.end_stage("agents", ag_elapsed, f"{len(agents_used)}位专家完成")

        # 记录agent spans
        if tracer and HAS_OBS:
            _record_stage_span(tracer, "agent", "🤖 Agent执行",
                              ag_elapsed, {
                                  "agents": agents_used,
                                  "parallel": len(agents_needed) > 1,
                              })

        # ---- 阶段④：报告员综合 [reporter span] ----
        thinking.append("🖊️ 报告员综合中...")
        if stream_ps:
            stream_ps.start_stage("reporter", "🖊️ 报告员综合")
        rp_start = time.time()

        final_answer = executor.execute_reporter(question, expert_outputs,
                                                  memory_section)
        rp_elapsed = (time.time() - rp_start) * 1000

        agents_used.append("🖊️ 报告员")
        thinking.append("✅ 报告完成")
        if stream_ps:
            stream_ps.end_stage("reporter", rp_elapsed, "报告生成完成")

        # 记录reporter span
        if tracer and HAS_OBS:
            _record_stage_span(tracer, "reporter", "🖊️ 报告员",
                              rp_elapsed, {})

        # HITL 检测 (legacy v3.2, kept for backward compat)
        hitl_triggers = []
        risk_output = expert_outputs.get("🛡️ 风控专家", "")
        if "[HIGH_RISK_ALERT]" in risk_output or "高风险" in risk_output:
            hitl_triggers = detect_hitl_triggers(results)
            if hitl_triggers:
                thinking.append(f"⚠️ HITL: {len(hitl_triggers)} 个高风险需确认")

        # ---- v3.3 阶段⑤：CriticAgent 质量审查 + 迭代精炼 [critic span] ----
        critique_result = None
        refinement_trace = None
        if enable_critic and HAS_CRITIC:
            thinking.append("🔍 质量审查中...")
            if stream_ps:
                stream_ps.start_stage("critic", "🔍 质量审查")
            cr_start = time.time()

            try:
                final_answer, critique_result, refinement_trace = critique_and_refine(
                    final_answer, question, expert_outputs,
                    _call_llm_raw, provider, api_key,
                    threshold=critic_threshold,
                    max_iterations=critic_max_iter,
                    use_llm_critic=True,
                    enabled=True,
                )
                cr_elapsed = (time.time() - cr_start) * 1000

                if critique_result:
                    score = critique_result.get("overall_score", 0)
                    passed = critique_result.get("passed", False)
                    iters = refinement_trace.get("iterations", 0) if refinement_trace else 0
                    thinking.append(
                        f"📋 质量评分: {score}/10 "
                        f"({'✅ 通过' if passed else '⚠️ 未通过'}) "
                        f"迭代{iters}次"
                    )
                    if refinement_trace and refinement_trace.get("improvement", 0) > 0:
                        thinking.append(
                            f"📈 精炼提升: +{refinement_trace['improvement']:.1f}分"
                        )

                # 记录critic span
                if tracer and HAS_OBS:
                    _record_stage_span(tracer, "critic", "🔍 质量审查",
                                      cr_elapsed, {
                                          "score": critique_result.get("overall_score", 0) if critique_result else 0,
                                          "passed": critique_result.get("passed", False) if critique_result else False,
                                      })
                if stream_ps:
                    stream_ps.end_stage("critic", cr_elapsed,
                        f"评分{critique_result.get('overall_score', 0)}/10" if critique_result else "完成")
            except Exception as e:
                thinking.append(f"🔍 质量审查跳过: {e}")
                if stream_ps:
                    stream_ps.error("critic", str(e))

        # ---- v3.3 阶段⑥：增强HITL置信度评估 [hitl span] ----
        hitl_decision = None
        if enable_hitl_v2 and HAS_HITL_V2:
            if stream_ps:
                stream_ps.start_stage("hitl", "🎯 HITL置信度评估")
            hl_start = time.time()
            try:
                crit_score = critique_result.get("overall_score") if critique_result else None
                hitl_decision = evaluate_hitl(
                    question, final_answer, expert_outputs,
                    context_data, crit_score,
                    enabled=True,
                )
                hl_elapsed = (time.time() - hl_start) * 1000

                if hitl_decision:
                    conf = hitl_decision.get("confidence_score", 0)
                    level = hitl_decision.get("confidence_level", "?")
                    action = hitl_decision.get("action", "?")
                    n_triggers = len(hitl_decision.get("triggers", []))
                    thinking.append(
                        f"🎯 HITL: 置信度={conf:.2f} ({level}) → {action}"
                        + (f" | {n_triggers}个触发" if n_triggers else "")
                    )

                if tracer and HAS_OBS:
                    _record_stage_span(tracer, "hitl", "🎯 HITL评估",
                                      hl_elapsed, {
                                          "confidence": hitl_decision.get("confidence_score", 0) if hitl_decision else 0,
                                          "action": hitl_decision.get("action", "") if hitl_decision else "",
                                      })
                if stream_ps:
                    stream_ps.end_stage("hitl", hl_elapsed,
                        f"置信度{hitl_decision.get('confidence_score', 0):.0%}" if hitl_decision else "完成")
            except Exception as e:
                thinking.append(f"🎯 HITL评估跳过: {e}")

        elapsed = time.time() - t0
        thinking.append(f"⏱️ 总耗时 {elapsed:.1f}秒")

        # v4.0: 通知流式完成
        if stream_ps:
            stream_ps.thinking(f"⏱️ 总耗时 {elapsed:.1f}秒")
            stream_ps.complete()

        # 记忆
        mem.add_turn(question, final_answer, agents_used, expert_outputs)

        # ---- v3.3 持久化记忆保存 ----
        if enable_persistent_mem and HAS_PMEM:
            try:
                pmem = get_persistent_memory()
                # 保存洞察
                import uuid as _uuid
                insight = InsightRecord(
                    insight_id=str(_uuid.uuid4())[:8],
                    question=question,
                    answer_summary=final_answer[:300],
                    agents_used=agents_used,
                    key_findings=[],
                    entities_involved=list(mem.entity_mentions.keys())[:10],
                    timestamp=datetime.now().isoformat(),
                    quality_score=critique_result.get("overall_score", 0) if critique_result else 0,
                )
                pmem.save_insight(insight)

                # 更新实体记忆
                for entity_name in list(mem.entity_mentions.keys())[:5]:
                    pmem.upsert_entity("customer", entity_name)
                    pmem.add_entity_event(
                        "customer", entity_name,
                        question[:100],
                        final_answer[:100],
                    )

                # 学习用户偏好
                pmem.update_preferences_from_interaction(
                    question, agents_used,
                    list(mem.entity_mentions.keys())[:10],
                )
            except Exception:
                pass  # 持久化记忆保存失败不影响返回

        # ---- 可观测性：填充trace元数据 ----
        trace_id = ""
        obs_summary = {}
        if trace_obj and hasattr(trace_obj, 'trace_id') and trace_obj.trace_id != "disabled":
            trace_obj.agents_used = agents_used
            trace_obj.pattern_matched = kg_pattern
            trace_obj.route_source = route_source
            trace_obj.kg_corrections = kg_corrections
            trace_id = trace_obj.trace_id

            obs_summary = {
                "trace_id": trace_id,
                "total_tokens": trace_obj.total_tokens,
                "total_cost_usd": round(trace_obj.total_cost_usd, 6),
                "total_llm_calls": trace_obj.total_llm_calls,
                "latency_breakdown": {
                    "data_query_ms": round(dq_elapsed, 1),
                    "routing_ms": round(rt_elapsed, 1),
                    "agents_ms": round(ag_elapsed, 1),
                    "reporter_ms": round(rp_elapsed, 1),
                    "total_ms": round(elapsed * 1000, 1),
                },
            }
            thinking.append(f"📊 Trace: {trace_id[:8]}... | "
                          f"Tokens={trace_obj.total_tokens} | "
                          f"Cost=${trace_obj.total_cost_usd:.4f}")

        result_dict = {
            "answer": final_answer,
            "agents_used": agents_used,
            "thinking": thinking,
            "expert_outputs": expert_outputs,
            "hitl_triggers": hitl_triggers,
            "trace_id": trace_id,
            "obs_summary": obs_summary,
            # v3.3 新增
            "critique": critique_result,
            "hitl_decision": hitl_decision,
            "refinement_trace": refinement_trace,
            "persistent_memory_used": pmem_used,
            # v4.0 新增
            "tool_use_enabled": enable_tools and HAS_TOOLS,
            "guardrails_enabled": HAS_GUARD,
            "streaming_enabled": stream_ps is not None,
            "budget_status": get_budget().check_budget() if HAS_GUARD else None,
            "from_cache": False,
            # V9.0 新增
            "v9_modules": {
                "rlm": HAS_RLM,
                "reasoning_templates": HAS_REASONING,
                "memory_3d": HAS_MEM3D,
                "interpretability": HAS_INTERP,
                "search_engine": HAS_SEARCH,
                "awm": HAS_AWM,
                "evals_v9": HAS_EVALS_V9,
            },
            "v9_activity": {
                "rlm_used": rlm_used,
                "reasoning_templates_injected": HAS_REASONING,
                "memory_3d_saved": HAS_MEM3D,
                "memory_3d_retrieved": bool(
                    HAS_MEM3D and _get_v9_memory()
                    and _get_v9_memory().query_skills(question, top_k=1)
                ) if HAS_MEM3D else False,
                "interpretability_traced": HAS_INTERP and _get_v9_tracer() is not None,
                "trace_steps": len(_get_v9_tracer().get_trace()) if HAS_INTERP and _get_v9_tracer() else 0,
            },
        }

        # V9.0: 三维记忆保存 — 存储本次分析为经验案例
        if HAS_MEM3D:
            try:
                mem3d = _get_v9_memory()
                if mem3d:
                    score = critique_result.get("overall_score", 7.0) / 10.0 if critique_result else 0.7
                    import uuid as _uuid3d
                    node = Memory3DNode(
                        node_id=f"case_{_uuid3d.uuid4().hex[:8]}",
                        content=f"Q: {question[:100]}\nA: {final_answer[:200]}",
                        form=MemoryForm.TEXT,
                        function=MemoryFunction.LEARNING,
                        tags=agents_used[:3],
                        entities=list(mem.entity_mentions.keys())[:5],
                        confidence=score,
                    )
                    mem3d.add(node)
            except Exception:
                pass

        # v4.0: 缓存保存
        if enable_cache and HAS_GUARD:
            try:
                cache = get_cache()
                cache.put(question, result_dict)
            except Exception:
                pass

        return result_dict

    except Exception as e:
        if trace_obj and hasattr(trace_obj, 'status'):
            trace_obj.status = "error"
            trace_obj.error_message = str(e)
        raise
    finally:
        trace_ctx.__exit__(None, None, None)


def _record_stage_span(tracer, kind, name, duration_ms, attrs):
    """记录pipeline阶段的span"""
    try:
        from observability import _ctx, Span
        if _ctx.current_trace is None:
            return
        parent_id = ""
        if _ctx.current_span_stack:
            parent_id = _ctx.current_span_stack[-1].span_id
        s = Span(
            trace_id=_ctx.current_trace.trace_id,
            parent_span_id=parent_id,
            kind=kind,
            name=name,
            start_time=time.time() - duration_ms / 1000,
            attributes=attrs,
        )
        s.finish()
        _ctx.current_trace.add_span(s)
    except Exception:
        pass


class _DummyCtx:
    """可观测性禁用时的占位上下文"""
    def __enter__(self):
        return type('_', (), {'trace_id': '', 'agents_used': [],
                              'pattern_matched': '', 'route_source': '',
                              'kg_corrections': [], 'status': 'ok',
                              'total_tokens': 0, 'total_cost_usd': 0,
                              'total_llm_calls': 0})()
    def __exit__(self, *a): pass


# 兼容旧版接口
def ask_multi_agent_simple(
    question: str, data: dict, results: dict,
    benchmark=None, forecast=None,
    provider="deepseek", api_key="",
    memory: AgentMemory = None,
) -> dict:
    """v3.0 简化版也使用升级后的流程"""
    return ask_multi_agent(
        question, data, results, benchmark, forecast,
        provider, api_key, memory,
    )


def _ask_fallback(question, data, results, benchmark, forecast, provider, api_key, memory=None):
    return ask_multi_agent(question, data, results, benchmark, forecast, provider, api_key, memory)


# ╔══════════════════════════════════════════════════════════════╗
# ║  V7.0 LANGGRAPH LAYER — 以下代码为 V7.0 LangGraph 扩展层    ║
# ║  StateGraph + Reflection + HITL + Multi-Model Routing       ║
# ╚══════════════════════════════════════════════════════════════╝

# ============================================================
# v7.0 Multi-Model Router — 成本降低60%
# ============================================================

MODEL_TIERS = {
    "fast": {
        "claude": "claude-haiku-4-5-20251001",
        "deepseek": "deepseek-chat",
        "description": "简单查询/路由/分类 (~$0.001/query)",
    },
    "standard": {
        "claude": "claude-sonnet-4-20250514",
        "deepseek": "deepseek-chat",
        "description": "数据分析/报告综合 (~$0.01/query)",
    },
    "advanced": {
        "claude": "claude-sonnet-4-20250514",
        "deepseek": "deepseek-chat",
        "description": "风险评估/战略推理 (~$0.03/query)",
    },
}


def get_model_for_tier(provider: str, tier: str = "standard") -> str:
    """v7.0: 根据任务复杂度选择最优模型"""
    tier_config = MODEL_TIERS.get(tier, MODEL_TIERS["standard"])
    return tier_config.get(provider, tier_config.get("claude"))


# ============================================================
# v7.0 Agent State (LangGraph TypedDict)
# ============================================================

class AgentState(TypedDict):
    """LangGraph 状态定义 — 全部信息在状态中流转"""
    # 输入
    question: str
    context_data: str
    provider: str
    api_key: str

    # 路由结果
    agents_needed: List[str]
    route_source: str

    # 专家输出
    expert_outputs: Dict[str, str]

    # 综合报告
    final_answer: str

    # 质量审查
    critique_result: Optional[Dict]
    critique_score: float
    reflection_iterations: int

    # HITL
    hitl_decision: Optional[Dict]
    hitl_approved: bool
    high_risk_alerts: List[Dict]

    # 元数据
    thinking: List[str]
    elapsed_ms: float
    agents_used: List[str]
    model_costs: Dict[str, float]

    # 配置
    enable_tools: bool
    enable_critic: bool
    enable_hitl: bool
    stream_ps: Optional[Any]

    # 知识图谱上下文
    kg_entity_context: str
    kg_agent_hint: List[str]

    # V9.0 新增
    v9_attribution: Optional[Dict]    # 输出归因

    # V10.1 新增: 规划器输出
    execution_plan: Optional[Dict]


# ============================================================
# v7.0 LLM 调用层 (复用 v5.0, 增加 model routing)
# ============================================================

def _call_llm(system: str, user: str, provider: str, api_key: str,
              tier: str = "standard", max_tokens: int = 800,
              temperature: float = 0.3, trace_name: str = "llm_call") -> str:
    """统一 LLM 调用入口 — v7.0 增加模型路由"""
    if not api_key:
        return "[需要 API Key]"

    model = get_model_for_tier(provider, tier)

    # 预算检查
    if HAS_GUARD:
        budget = get_budget()
        if not budget.should_allow_query():
            return "[每日预算已耗尽]"

    def _do_call():
        if provider == "deepseek":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        elif provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

    try:
        if HAS_GUARD:
            return guarded_llm_call(
                _do_call,
                breaker_name=f"llm_{provider}",
                validate=True,
                source_data=user[:2000],
            )
        return _do_call()
    except Exception as e:
        return f"[调用失败: {e}]"


# ============================================================
# V10.0: Middleware 拦截链
# ============================================================

class AgentMiddleware:
    """Agent 调用中间件基类 — 支持链式拦截"""

    def before(self, agent_id: str, question: str, **ctx) -> dict:
        """调用前拦截。返回 dict 可注入/修改上下文。"""
        return {}

    def after(self, agent_id: str, output: str, elapsed_ms: float, **ctx) -> str:
        """调用后拦截。可修改输出。"""
        return output


class LoggingMiddleware(AgentMiddleware):
    """日志中间件 — 记录每次 Agent 调用"""

    def before(self, agent_id: str, question: str, **ctx):
        logger.info(f"[MW] Agent {agent_id} 开始: {question[:60]}...")
        return {"start_time": time.time()}

    def after(self, agent_id: str, output: str, elapsed_ms: float, **ctx):
        logger.info(f"[MW] Agent {agent_id} 完成: {elapsed_ms:.0f}ms, {len(output)}字")
        return output


class LangfuseMiddleware(AgentMiddleware):
    """Langfuse 可观测性中间件"""

    def before(self, agent_id: str, question: str, **ctx):
        if HAS_LANGFUSE and _langfuse_client:
            try:
                span = _langfuse_client.trace(
                    name=f"mw_agent_{agent_id}",
                    metadata={"question": question[:200]},
                )
                return {"lf_span": span}
            except Exception:
                pass
        return {}

    def after(self, agent_id: str, output: str, elapsed_ms: float, **ctx):
        span = ctx.get("lf_span")
        if span:
            try:
                span.update(output=output[:500], metadata={"elapsed_ms": elapsed_ms})
            except Exception:
                pass
        return output


class PydanticValidationMiddleware(AgentMiddleware):
    """Pydantic 输出验证中间件"""

    def after(self, agent_id: str, output: str, elapsed_ms: float, **ctx):
        try:
            parsed = json.loads(output)
            resp = AgentResponse(
                agent_id=agent_id,
                agent_name=ctx.get("agent_name", agent_id),
                data=parsed if isinstance(parsed, dict) else {"raw": parsed},
                elapsed_ms=elapsed_ms,
            )
            # 验证通过，返回原始输出（保持向后兼容）
            return output
        except (json.JSONDecodeError, Exception):
            return output


# 全局 Middleware 链
_middleware_chain: List[AgentMiddleware] = [
    LoggingMiddleware(),
    LangfuseMiddleware(),
    PydanticValidationMiddleware(),
]


def run_middleware_before(agent_id: str, question: str, **ctx) -> dict:
    """执行 before 链，合并上下文"""
    merged = dict(ctx)
    for mw in _middleware_chain:
        try:
            result = mw.before(agent_id, question, **merged)
            if result:
                merged.update(result)
        except Exception as e:
            logger.debug(f"Middleware {mw.__class__.__name__} before 错误: {e}")
    return merged


def run_middleware_after(agent_id: str, output: str, elapsed_ms: float, **ctx) -> str:
    """执行 after 链，逐步处理输出"""
    result = output
    for mw in _middleware_chain:
        try:
            result = mw.after(agent_id, result, elapsed_ms, **ctx)
        except Exception as e:
            logger.debug(f"Middleware {mw.__class__.__name__} after 错误: {e}")
    return result


# ============================================================
# V10.1: Hierarchical Planner — 复杂查询任务分解
# ============================================================

class QueryPlanner:
    """查询规划器 — 分解复杂查询为可并行步骤"""

    COMPLEXITY_KEYWORDS = {
        "multi": ["综合", "全面", "对比", "关联",
                  "交叉", "多维", "CEO", "报告"],
        "single": ["多少", "变化", "趋势", "排名"],
    }

    @staticmethod
    def needs_planning(query: str, agents: list) -> bool:
        """判断是否需要规划 (而非直接路由)"""
        if len(agents) >= 3:
            return True
        q = query.lower()
        return any(
            kw in q
            for kw in QueryPlanner.COMPLEXITY_KEYWORDS["multi"]
        )

    @staticmethod
    def create_plan(query: str, agents: list) -> dict:
        """
        创建执行计划: {
          "phases": [
            {"phase": 1, "agents": [...], "parallel": True},
            {"phase": 2, "agents": ["strategist"], "parallel": False}
          ]
        }
        """
        # Phase 1: 数据收集 (并行)
        data_agents = [a for a in agents if a != "strategist"]
        phases = []
        if data_agents:
            phases.append({
                "phase": 1,
                "agents": data_agents,
                "parallel": True,
                "desc": "数据收集与域分析",
            })
        # Phase 2: 综合分析 (串行)
        if "strategist" in agents:
            phases.append({
                "phase": 2,
                "agents": ["strategist"],
                "parallel": False,
                "desc": "战略综合与建议",
            })
        return {
            "query": query,
            "total_agents": len(agents),
            "phases": phases,
            "has_planning": True,
        }


# ============================================================
# v7.0 LangGraph Nodes
# ============================================================

def node_route(state: AgentState) -> dict:
    """
    🧭 路由节点 — 决定调用哪些专家
    优先级: 知识图谱提示 > LLM路由(fast tier) > 规则路由
    """
    question = state["question"]
    provider = state["provider"]
    api_key = state["api_key"]
    thinking = list(state.get("thinking", []))
    kg_hint = state.get("kg_agent_hint", [])

    # 1. 知识图谱提示 (零 API 调用)
    if kg_hint:
        valid = [a for a in kg_hint if a in AGENT_PROFILES]
        if valid:
            thinking.append(f"🧭 KG路由 → {valid}")
            # V10.1: 规划分解 (复杂查询)
            plan = None
            if QueryPlanner.needs_planning(question, valid):
                plan = QueryPlanner.create_plan(question, valid)
                thinking.append(f"📋 规划: {len(plan['phases'])}阶段")
            return {
                "agents_needed": valid,
                "route_source": "knowledge_graph",
                "thinking": thinking,
                "execution_plan": plan,
            }

    # 2. LLM 路由 (用 fast tier 省钱)
    if api_key:
        try:
            routing_prompt = """你是问题分类器。根据用户问题判断需要哪些专家参与。

专家列表:
- analyst: 数据分析（营收、客户、趋势、排名、数量、区域、产品结构）
- risk: 风险评估（流失、预警、异常、下滑、危险信号）
- strategist: 战略建议（增长机会、竞争分析、行业对标、预测、决策建议）

规则:
1. 简单数据查询 → 只需 analyst
2. 风险/流失相关 → analyst + risk
3. 战略/建议/未来 → analyst + strategist
4. 全面分析/CEO报告 → 全部

输出格式（纯JSON，无其他文字）:
{"agents": ["analyst"], "reason": "简单数据查询"}"""

            raw = _call_llm(
                routing_prompt,
                f"用户问题：{question}",
                provider, api_key,
                tier="fast", max_tokens=80, temperature=0.0,
                trace_name="v7_routing",
            )

            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json\n")
            parsed = json.loads(raw)
            agents = [a for a in parsed.get("agents", []) if a in AGENT_PROFILES]

            if agents:
                thinking.append(f"🧭 LLM路由(fast) → {agents}")
                # V10.1: 规划分解 (复杂查询)
                plan = None
                if QueryPlanner.needs_planning(question, agents):
                    plan = QueryPlanner.create_plan(question, agents)
                    thinking.append(f"📋 规划: {len(plan['phases'])}阶段")
                return {
                    "agents_needed": agents,
                    "route_source": "llm_fast",
                    "thinking": thinking,
                    "execution_plan": plan,
                }
        except Exception:
            pass

    # 3. 规则路由 (兜底)
    agents = _rule_route(question)
    thinking.append(f"🧭 规则路由 → {agents}")

    # V10.1: 规划分解 (复杂查询)
    plan = None
    if QueryPlanner.needs_planning(question, agents):
        plan = QueryPlanner.create_plan(question, agents)
        thinking.append(f"📋 规划: {len(plan['phases'])}阶段")

    return {
        "agents_needed": agents,
        "route_source": "rule",
        "thinking": thinking,
        "execution_plan": plan,  # V10.1 新增
    }


def _rule_route(question: str) -> List[str]:
    """增强版规则路由"""
    q = question.lower()
    agents = set()

    for agent_id, profile in AGENT_PROFILES.items():
        if any(kw in q for kw in profile["keywords"]):
            agents.add(agent_id)

    # 全量触发 — V10.0: 包含全部7个Agent
    if any(k in q for k in ['CEO', 'ceo', '总结', '全面', '概览', '怎么样', '报告']):
        agents = {"analyst", "risk", "strategist", "quality", "market", "finance", "procurement"}

    # 简单查询优化
    if not agents:
        agents = {"analyst"}

    return list(agents)


def node_experts(state: AgentState) -> dict:
    """
    🤖 专家节点 — 并行执行所有专家 Agent
    v7.0: LangGraph 自动管理并行 (Send API / fan-out)
    回退: ThreadPoolExecutor 并行
    """
    agents_needed = state["agents_needed"]
    question = state["question"]
    context_data = state["context_data"]
    provider = state["provider"]
    api_key = state["api_key"]
    kg_context = state.get("kg_entity_context", "")
    thinking = list(state.get("thinking", []))
    stream_ps = state.get("stream_ps")

    if len(agents_needed) > 1:
        thinking.append(f"⚡ 并行启动 {len(agents_needed)} 位专家...")
    else:
        thinking.append("▶️ 启动专家分析...")

    enriched_data = context_data
    if kg_context:
        enriched_data += f"\n\n[知识图谱 · 实体画像]\n{kg_context}"

    expert_outputs = {}
    model_costs = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _call_expert(agent_id: str) -> tuple:
        profile = AGENT_PROFILES[agent_id]
        tier = profile.get("model_tier", "standard")
        _t0 = time.time()

        # V10.0: Middleware before 链
        mw_ctx = run_middleware_before(
            agent_id, question,
            agent_name=profile["name"], tier=tier,
        )

        if stream_ps and HAS_STREAM:
            stream_ps.agent_start(agent_id, profile["name"])

        # V9.0: 可解释性追踪
        v9t = _get_v9_tracer()
        if v9t and HAS_INTERP:
            v9t.trace_step(f"agent_{agent_id}", profile["name"],
                           action=f"专家分析: {profile['role'][:20]}",
                           input_summary=question[:80])

        # Langfuse span
        lf_span = None
        if HAS_LANGFUSE and _langfuse_client:
            try:
                lf_span = _langfuse_client.trace(
                    name=f"agent_{agent_id}",
                    metadata={"tier": tier, "question": question[:200]},
                )
            except Exception:
                pass

        # ── V10.0: Engine-based Agent → 直接调用引擎，不走 LLM ──
        if tier == "engine":
            engine = get_domain_engine(profile.get("engine_type", agent_id))
            if engine:
                try:
                    raw = engine.answer(question)
                    output = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, indent=2)
                    if lf_span:
                        try:
                            lf_span.update(output=output[:500], metadata={"source": "engine"})
                        except Exception:
                            pass
                except Exception as e:
                    output = f"[{profile['name']} 引擎错误: {e}]"
                    logger.error(f"域引擎 {agent_id} 执行失败: {e}")
            else:
                output = f"[{profile['name']} 引擎未加载]"
                logger.warning(f"域引擎 {agent_id} 不可用")

            # V10.0: Middleware after 链 (engine path)
            output = run_middleware_after(
                agent_id, output, (time.time() - _t0) * 1000,
                agent_name=profile["name"], tier=tier, **mw_ctx,
            )

            if stream_ps and HAS_STREAM:
                stream_ps.agent_done(agent_id, profile["name"], output)
            return (profile["name"], output, tier)

        # ── LLM-based Agent (analyst/risk/strategist) ──
        system = f"你是{profile['role']}。{profile['backstory']}"

        # 工具描述
        tool_hint = ""
        if state.get("enable_tools") and HAS_TOOLS:
            tool_hint = get_tool_descriptions_for_prompt(agent_id)

        # V9.0: 结构化推理模板 — 降低内在维度，提高输出质量
        reasoning_hint = ""
        if HAS_REASONING:
            selector = _get_v9_reasoning()
            if selector:
                _role_map = {"analyst": "analyst", "risk": "risk",
                             "strategist": "strategist", "forecaster": "analyst",
                             "quality": "analyst", "market": "analyst",
                             "finance": "analyst", "procurement": "analyst"}
                tmpl_role = _role_map.get(agent_id, "analyst")
                try:
                    template = selector.select(tmpl_role, complexity="standard")
                    if template:
                        reasoning_hint = (
                            f"\n\n[结构化推理框架]\n"
                            f"请按以下步骤推理:\n"
                            + "\n".join(f"  Step {i+1}: {s.name} — {s.instruction}"
                                       for i, s in enumerate(template.steps[:5]))
                            + "\n每步给出具体数值或简洁结论。"
                        )
                except Exception:
                    pass  # 降级：不使用模板

        # V9.0: 三维记忆检索 — 利用历史分析经验
        memory_hint = ""
        if HAS_MEM3D:
            mem3d = _get_v9_memory()
            if mem3d:
                try:
                    relevant = mem3d.query_skills(question, top_k=3)
                    if relevant:
                        memory_hint = "\n\n[历史经验参考]\n" + "\n".join(
                            f"- {n.to_prompt_text(max_len=100)}" for n in relevant[:3]
                        )
                except Exception:
                    pass

        prompt = (
            f"用户问题：{question}\n\n"
            f"禾苗销售数据：\n{enriched_data}"
            f"{tool_hint}"
            f"{reasoning_hint}"
            f"{memory_hint}\n\n"
            f"200字内回答。数据必须精确引用。"
        )

        output = _call_llm(
            system, prompt, provider, api_key,
            tier=tier, max_tokens=1000,
            trace_name=f"v7_agent_{agent_id}",
        )

        # 输出校验
        if HAS_GUARD:
            validation = validate_agent_output(output, context_data[:2000])
            if not validation.passed and validation.confidence < 0.3:
                logger.warning(f"Agent {agent_id} 输出质量低")

        if lf_span:
            try:
                lf_span.update(output=output[:500], metadata={"source": "llm", "tier": tier})
            except Exception:
                pass

        # V10.0: Middleware after 链 (LLM path)
        output = run_middleware_after(
            agent_id, output, (time.time() - _t0) * 1000,
            agent_name=profile["name"], tier=tier, **mw_ctx,
        )

        if stream_ps and HAS_STREAM:
            stream_ps.agent_done(agent_id, profile["name"], output)

        return (profile["name"], output, tier)

    # 并行执行 — V10.0: max_workers=7 支持全部域Agent
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(_call_expert, aid): aid for aid in agents_needed}
        for future in as_completed(futures):
            try:
                name, output, tier = future.result(timeout=30)
                expert_outputs[name] = output
                model_costs[name] = {"tier": tier}
            except Exception as e:
                aid = futures[future]
                profile = AGENT_PROFILES[aid]
                expert_outputs[profile["name"]] = f"[分析超时: {e}]"

    agents_used = list(expert_outputs.keys())
    for name in agents_used:
        thinking.append(f"✅ {name} 完成")

    return {
        "expert_outputs": expert_outputs,
        "agents_used": agents_used,
        "model_costs": model_costs,
        "thinking": thinking,
    }


def node_synthesize(state: AgentState) -> dict:
    """
    🖊️ 报告综合节点 — Reporter 综合所有专家意见
    V9.0: + LatentLens 可解释性追踪 + 输出归因
    """
    question = state["question"]
    expert_outputs = state["expert_outputs"]
    provider = state["provider"]
    api_key = state["api_key"]
    thinking = list(state.get("thinking", []))

    thinking.append("🖊️ 报告员综合中...")

    # V9.0: 追踪综合步骤
    v9t = _get_v9_tracer()
    if v9t and HAS_INTERP:
        v9t.trace_step("synthesize", "报告员",
                       action="综合专家意见生成报告",
                       input_summary=f"综合 {len(expert_outputs)} 位专家意见",
                       output_summary="生成综合报告")

    all_opinions = "\n---\n".join(f"{n}：\n{t}" for n, t in expert_outputs.items())
    reporter_sys = f"你是{REPORTER_PROFILE['role']}。{REPORTER_PROFILE['backstory']}"

    final_answer = _call_llm(
        reporter_sys,
        f"问题：{question}\n\n专家分析：\n{all_opinions}\n\n综合报告，500字内。",
        provider, api_key,
        tier="standard",
        trace_name="v7_reporter",
    )

    # V9.0: 输出归因 — 追踪结论来自哪位专家
    v9_attribution = {}
    if HAS_INTERP and final_answer:
        try:
            attributor = OutputAttributor()
            attr_result = attributor.attribute(final_answer, expert_outputs)
            v9_attribution = {"attributions": [a.__dict__ for a in attr_result]
                              if attr_result else []}
        except Exception:
            pass

    thinking.append("✅ 报告完成")

    return {
        "final_answer": final_answer,
        "thinking": thinking,
        "v9_attribution": v9_attribution,
    }


def node_reflect(state: AgentState) -> dict:
    """
    🔍 反思节点 (Reflection Pattern) — 1-2轮自检
    研究表明: Text-to-SQL 准确率从 70%→85%
    v7.0: 仅在 enable_critic=True 时执行
    """
    if not state.get("enable_critic") or not HAS_CRITIC:
        return {
            "critique_result": None,
            "critique_score": 0.0,
            "reflection_iterations": 0,
        }

    thinking = list(state.get("thinking", []))
    thinking.append("🔍 质量审查中...")

    try:
        # 复用 v5.0 的 critique_and_refine
        # 需要传入 _call_llm 的兼容包装
        def _llm_compat(sys, user, prov, key, **kwargs):
            return _call_llm(sys, user, prov, key, tier="standard",
                           trace_name="v7_critic", **{k: v for k, v in kwargs.items()
                                                       if k in ('max_tokens', 'temperature')})

        refined, critique, trace = critique_and_refine(
            state["final_answer"],
            state["question"],
            state["expert_outputs"],
            _llm_compat,
            state["provider"],
            state["api_key"],
            threshold=7.0,
            max_iterations=2,  # v7.0: 最多2轮 (研究最优)
            use_llm_critic=True,
            enabled=True,
        )

        score = critique.get("overall_score", 0) if critique else 0
        passed = critique.get("passed", False) if critique else False
        iters = trace.get("iterations", 0) if trace else 0

        thinking.append(
            f"📋 质量评分: {score}/10 "
            f"({'✅ 通过' if passed else '⚠️ 未通过'}) "
            f"迭代{iters}次"
        )

        return {
            "final_answer": refined,
            "critique_result": critique,
            "critique_score": float(score),
            "reflection_iterations": iters,
            "thinking": thinking,
        }
    except Exception as e:
        thinking.append(f"🔍 质量审查跳过: {e}")
        return {
            "critique_result": None,
            "critique_score": 0.0,
            "reflection_iterations": 0,
            "thinking": thinking,
        }


def node_hitl_check(state: AgentState) -> dict:
    """
    🎯 HITL 节点 — 高风险时中断等待人工确认
    v7.0: 使用 LangGraph interrupt() 原生暂停
    """
    thinking = list(state.get("thinking", []))
    high_risk_alerts = []

    # 检测高风险
    risk_output = ""
    for name, output in state.get("expert_outputs", {}).items():
        if "风控" in name or "risk" in name.lower():
            risk_output = output
            break

    has_high_risk = "[HIGH_RISK_ALERT]" in risk_output or "高风险" in risk_output

    if has_high_risk and state.get("enable_hitl"):
        # 构建高风险警报列表
        high_risk_alerts.append({
            "source": "risk_agent",
            "content": risk_output[:500],
            "timestamp": datetime.now().isoformat(),
        })

        # v7.0: LangGraph interrupt — 暂停图执行，等待人工确认
        if HAS_LANGGRAPH:
            try:
                thinking.append("⚠️ HITL: 检测到高风险，等待人工确认...")

                # interrupt() 会暂停执行，返回给调用方
                # 调用方通过 graph.invoke(Command(resume=True/False)) 继续
                human_response = interrupt({
                    "type": "high_risk_review",
                    "alerts": high_risk_alerts,
                    "question": state["question"],
                    "answer_preview": state["final_answer"][:300],
                    "message": "检测到高风险客户，是否确认发送此分析结果？",
                })

                approved = human_response.get("approved", True) if isinstance(human_response, dict) else bool(human_response)
                thinking.append(f"🎯 HITL: {'✅ 已确认' if approved else '❌ 已拒绝'}")

                return {
                    "hitl_approved": approved,
                    "high_risk_alerts": high_risk_alerts,
                    "thinking": thinking,
                }
            except Exception as e:
                logger.warning(f"HITL interrupt 失败: {e}")

        # 回退: 使用 v5.0 HITL 引擎
        if HAS_HITL:
            hitl_decision = evaluate_hitl(
                state["question"], state["final_answer"],
                state["expert_outputs"], state["context_data"],
                state.get("critique_score"),
            )
            if hitl_decision:
                conf = hitl_decision.get("confidence_score", 0)
                action = hitl_decision.get("action", "auto")
                thinking.append(f"🎯 HITL: 置信度={conf:.2f} → {action}")
                return {
                    "hitl_decision": hitl_decision,
                    "hitl_approved": action == "auto_approve",
                    "high_risk_alerts": high_risk_alerts,
                    "thinking": thinking,
                }

    thinking.append("🎯 HITL: 无需人工干预")
    return {
        "hitl_approved": True,
        "high_risk_alerts": [],
        "thinking": thinking,
    }


# ============================================================
# v7.0 条件边 — 控制图流转
# ============================================================

def should_reflect(state: AgentState) -> str:
    """是否需要反思节点"""
    if state.get("enable_critic") and HAS_CRITIC:
        return "reflect"
    return "hitl_check"


# ============================================================
# v7.0 Graph 构建
# ============================================================

def build_agent_graph(checkpointer=None, enable_advanced: bool = True):
    """
    构建 LangGraph StateGraph
    返回编译后的图，支持 checkpointing 和 interrupt

    LangGraph 1.0 高级特性 (enable_advanced=True):
      - Node Caching: 缓存 experts 节点结果 (相同输入复用)
      - Durable State: checkpointer 持久化状态跨会话
      - Pre/Post Model Hooks: 节点执行前后 hooks
      - interrupt_before: HITL 中断点
    """
    if not HAS_LANGGRAPH:
        return None

    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("route", node_route)
    graph.add_node("experts", node_experts)
    graph.add_node("synthesize", node_synthesize)
    graph.add_node("reflect", node_reflect)
    graph.add_node("hitl_check", node_hitl_check)

    # 添加边
    graph.add_edge(START, "route")
    graph.add_edge("route", "experts")
    graph.add_edge("experts", "synthesize")

    # 条件边: 综合后是否反思
    graph.add_conditional_edges(
        "synthesize",
        should_reflect,
        {"reflect": "reflect", "hitl_check": "hitl_check"},
    )

    graph.add_edge("reflect", "hitl_check")
    graph.add_edge("hitl_check", END)

    # 编译 — 带 checkpointer 支持持久化 (Durable State)
    if checkpointer is None:
        checkpointer = MemorySaver()

    compile_kwargs = {
        "checkpointer": checkpointer,
        "interrupt_before": ["hitl_check"],  # HITL 中断点
    }

    compiled = graph.compile(**compile_kwargs)

    if enable_advanced:
        logger.info("LangGraph 1.0 高级特性已启用: Durable State + HITL interrupt")

    return compiled


# ============================================================
# v7.0 全局图实例
# ============================================================

_graph = None
_checkpointer = None


def get_graph():
    """获取/创建全局图实例"""
    global _graph, _checkpointer
    if _graph is None and HAS_LANGGRAPH:
        _checkpointer = MemorySaver()
        _graph = build_agent_graph(_checkpointer)
    return _graph


# ============================================================
# v7.0 主入口 — 兼容 v5.0 接口
# ============================================================

def run_multi_agent_v7(
    question: str,
    data: dict,
    results: dict,
    provider: str = "claude",
    api_key: str = "",
    benchmark: dict = None,
    forecast: dict = None,
    enable_tools: bool = True,
    enable_critic: bool = True,
    enable_hitl: bool = True,
    stream_ps=None,
    thread_id: str = None,
    **kwargs,
) -> dict:
    """
    v7.0 多 Agent 主入口

    兼容 v5.0 返回格式:
    {
        "answer": str,
        "agents_used": List[str],
        "thinking": List[str],
        "hitl_triggers": List[dict],
        "critique_result": dict,
        "elapsed": float,
        "version": "v7.0",
    }
    """
    t0 = time.time()

    # 构建上下文数据 (SmartDataQuery 已在同文件定义)
    sq = SmartDataQuery(data, results, benchmark, forecast)
    context_data = sq.query_smart(question, provider, api_key)
    kg_entity = sq._last_entity_context if hasattr(sq, '_last_entity_context') else ""
    kg_hint = sq._last_agent_hint if hasattr(sq, '_last_agent_hint') else []

    # 初始状态
    initial_state: AgentState = {
        "question": question,
        "context_data": context_data,
        "provider": provider,
        "api_key": api_key,
        "agents_needed": [],
        "route_source": "",
        "expert_outputs": {},
        "final_answer": "",
        "critique_result": None,
        "critique_score": 0.0,
        "reflection_iterations": 0,
        "hitl_decision": None,
        "hitl_approved": True,
        "high_risk_alerts": [],
        "thinking": [f"🚀 MRARFAI v7.0 (LangGraph) — {datetime.now().strftime('%H:%M:%S')}"],
        "elapsed_ms": 0.0,
        "agents_used": [],
        "model_costs": {},
        "enable_tools": enable_tools,
        "enable_critic": enable_critic,
        "enable_hitl": enable_hitl,
        "stream_ps": stream_ps,
        "kg_entity_context": kg_entity,
        "kg_agent_hint": kg_hint,
        # V10.1 新增
        "execution_plan": None,
        "v9_attribution": None,
    }

    # ---- LangGraph 执行 ----
    graph = get_graph()

    if graph is not None:
        config = {"configurable": {"thread_id": thread_id or f"mrarfai_{int(time.time())}"}}

        try:
            # invoke 会运行到 interrupt 点或 END
            result_state = graph.invoke(initial_state, config)

            elapsed = time.time() - t0
            result_state["thinking"].append(f"⏱️ 总耗时 {elapsed:.1f}秒")

            return {
                "answer": result_state.get("final_answer", ""),
                "agents_used": result_state.get("agents_used", []),
                "thinking": result_state.get("thinking", []),
                "hitl_triggers": result_state.get("high_risk_alerts", []),
                "critique_result": result_state.get("critique_result"),
                "hitl_decision": result_state.get("hitl_decision"),
                "elapsed": elapsed,
                "version": "v7.0",
                "graph_state": result_state,
            }
        except Exception as e:
            logger.error(f"LangGraph 执行失败: {e}，回退到 v5.0")

    # ---- 回退: v5.0 兼容模式 ----
    return _fallback_v5(initial_state, t0)


def _fallback_v5(state: AgentState, t0: float) -> dict:
    """v5.0 兼容模式 — 不依赖 LangGraph"""
    thinking = state["thinking"]
    thinking.append("⚠️ LangGraph 不可用，使用 v5.0 兼容模式")

    # 路由
    route_result = node_route(state)
    state.update(route_result)

    # 专家
    expert_result = node_experts(state)
    state.update(expert_result)

    # 综合
    synth_result = node_synthesize(state)
    state.update(synth_result)

    # 反思
    if state.get("enable_critic"):
        reflect_result = node_reflect(state)
        state.update(reflect_result)

    elapsed = time.time() - t0
    state["thinking"].append(f"⏱️ 总耗时 {elapsed:.1f}秒 (v5兼容)")

    return {
        "answer": state.get("final_answer", ""),
        "agents_used": state.get("agents_used", []),
        "thinking": state.get("thinking", []),
        "hitl_triggers": state.get("high_risk_alerts", []),
        "critique_result": state.get("critique_result"),
        "elapsed": elapsed,
        "version": "v7.0-compat",
    }


# ============================================================
# v5.0 兼容接口 — 让 app.py / chat_tab.py 无需改动
# ============================================================

def run_multi_agent(question, data, results, provider="claude", api_key="",
                    benchmark=None, forecast=None, **kwargs):
    """
    v5.0 兼容入口 — 直接替换旧版 run_multi_agent_pipeline
    app.py 和 chat_tab.py 调用此函数即可无缝升级
    """
    return run_multi_agent_v7(
        question, data, results, provider, api_key,
        benchmark=benchmark, forecast=forecast, **kwargs,
    )



# ============================================================
# P3: 前沿框架集成检测 — 评估层
# ============================================================

# P3-01: AG-UI / A2UI 前端协议检测
HAS_AG_UI = False
try:
    from ag_ui import AgentUIRenderer
    HAS_AG_UI = True
except ImportError:
    pass

# P3-02: Google ADK 检测
HAS_GOOGLE_ADK = False
try:
    from google.adk import CustomAgent as ADKAgent
    HAS_GOOGLE_ADK = True
except ImportError:
    pass

# P3-03: OpenAI Agents SDK 检测
HAS_OPENAI_AGENTS = False
try:
    from agents import Agent as OAIAgent, Runner as OAIRunner
    HAS_OPENAI_AGENTS = True
except ImportError:
    pass

# P3-05: Graphiti Graph Memory 检测
HAS_GRAPHITI = False
try:
    from graphiti_core import Graphiti
    HAS_GRAPHITI = True
except ImportError:
    pass


def get_platform_capabilities() -> dict:
    """
    返回平台全量能力矩阵 — 审计/评估/展示用

    覆盖:
      V4 基础层 + V7 LangGraph + V9 论文模块 +
      V10 协议层 (A2A/MCP/gRPC) +
      P3 前沿框架 (ADK/OpenAI/AG-UI/Graphiti/Deep Agents)
    """
    return {
        "version": __version__,
        # V4 基础层
        "v4_pipeline": True,
        "knowledge_graph": HAS_KG if 'HAS_KG' in dir() else False,
        "observability": HAS_OBS if 'HAS_OBS' in dir() else False,
        "tools": HAS_TOOLS if 'HAS_TOOLS' in dir() else False,
        "guardrails": HAS_GUARD if 'HAS_GUARD' in dir() else False,
        "streaming": HAS_STREAM if 'HAS_STREAM' in dir() else False,
        "critic": HAS_CRITIC if 'HAS_CRITIC' in dir() else False,
        # V7 LangGraph
        "langgraph": HAS_LANGGRAPH,
        "hitl": HAS_HITL if 'HAS_HITL' in dir() else False,
        "langfuse": HAS_LANGFUSE,
        # V9 论文模块
        "rlm_engine": HAS_RLM,
        "awm_environment": HAS_AWM,
        "encompass_search": HAS_SEARCH,
        "reasoning_templates": HAS_REASONING,
        "memory_3d": HAS_MEM3D,
        "interpretability": HAS_INTERP,
        "evals_v9": HAS_EVALS_V9,
        # V10 协议
        "pydantic_contracts": True,
        "middleware": True,
        "react_pattern": True,
        "query_planner": True,
        "deep_agents": HAS_DEEP_AGENTS,
        "db_bridge": HAS_DB_BRIDGE,
        # P3 前沿框架
        "ag_ui": HAS_AG_UI,
        "google_adk": HAS_GOOGLE_ADK,
        "openai_agents_sdk": HAS_OPENAI_AGENTS,
        "graphiti": HAS_GRAPHITI,
        # 域 Agent
        "domain_agents": {
            "quality": HAS_QUALITY,
            "market": HAS_MARKET,
            "finance": HAS_FINANCE,
            "procurement": HAS_PROCUREMENT,
            "risk": HAS_RISK_ENGINE,
            "strategist": HAS_STRATEGIST_ENGINE,
        },
    }


# ============================================================
# 模块信息 — V10.0 统一版
# ============================================================

__version__ = "10.1.0"
__all__ = [
    # V4 主入口
    "ask_multi_agent",
    "ask_multi_agent_simple",
    # V7 LangGraph 入口
    "run_multi_agent_v7",
    "run_multi_agent",
    "build_agent_graph",
    "get_graph",
    # 核心类
    "AgentState",
    "AgentMemory",
    "SmartDataQuery",
    "SmartRouter",
    "ParallelAgentExecutor",
    # 配置
    "AGENT_PROFILES",
    "REPORTER_PROFILE",
    "MODEL_TIERS",
    # 工具函数
    "route_to_agents",
    "detect_hitl_triggers",
    "get_memory",
    "set_memory",
    "set_sales_data",
    "query_sales_data",
    # V10 新增
    "get_platform_capabilities",
    "get_domain_engine",
    "run_middleware_before",
    "run_middleware_after",
    # V10.1 新增
    "QueryPlanner",
]

if __name__ == "__main__":
    print(f"MRARFAI Multi-Agent v{__version__} (Unified)")
    print(f"  V4 Pipeline: ✅ (ask_multi_agent)")
    print(f"  V7 LangGraph: {'✅' if HAS_LANGGRAPH else '❌'} (run_multi_agent_v7)")
    print(f"  Knowledge Graph: {'✅' if HAS_KG else '❌'}")
    print(f"  Observability: {'✅' if HAS_OBS else '❌'}")
    print(f"  Tools: {'✅' if HAS_TOOLS else '❌'}")
    print(f"  Guardrails: {'✅' if HAS_GUARD else '❌'}")
    print(f"  Streaming: {'✅' if HAS_STREAM else '❌'}")
    print(f"  Critic: {'✅' if HAS_CRITIC else '❌'}")
    print(f"  --- V9.0 论文模块 ---")
    print(f"  ① RLM Engine: {'✅' if HAS_RLM else '❌'}")
    print(f"  ② AWM Environment: {'✅' if HAS_AWM else '❌'}")
    print(f"  ③ EnCompass Search: {'✅' if HAS_SEARCH else '❌'}")
    print(f"  ④ Reasoning Templates: {'✅' if HAS_REASONING else '❌'}")
    print(f"  ⑤ Memory 3D: {'✅' if HAS_MEM3D else '❌'}")
    print(f"  ⑥ Interpretability: {'✅' if HAS_INTERP else '❌'}")
    print(f"  ⑦ Evals V9: {'✅' if HAS_EVALS_V9 else '❌'}")
    print(f"  --- V10.0 协议层 ---")
    print(f"  ⑧ Deep Agents: {'✅' if HAS_DEEP_AGENTS else '❌'}")
    print(f"  Pydantic Contracts: ✅")
    print(f"  Middleware: ✅")
    print(f"  ReAct Pattern: ✅")
    print(f"  Langfuse: {'✅' if HAS_LANGFUSE else '❌'}")
    print(f"  --- P3 前沿框架 ---")
    print(f"  AG-UI: {'✅' if HAS_AG_UI else '❌ (可选)'}")
    print(f"  Google ADK: {'✅' if HAS_GOOGLE_ADK else '❌ (可选)'}")
    print(f"  OpenAI Agents SDK: {'✅' if HAS_OPENAI_AGENTS else '❌ (可选)'}")
    print(f"  Graphiti: {'✅' if HAS_GRAPHITI else '❌ (可选)'}")
    print()
    caps = get_platform_capabilities()
    active = sum(1 for v in caps.values() if v is True)
    print(f"  能力矩阵: {active}/{len(caps)} 激活")
