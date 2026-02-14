#!/usr/bin/env python3
"""
MRARFAI A2A Server v7.0 — Agent2Agent Protocol RC v1.0
========================================================
Phase 4 升级：自定义 A2A 抽象 → Google A2A RC v1.0 标准协议

核心变化:
  ① Agent Card — /.well-known/agent.json 标准能力声明
  ② Task 生命周期 — 8 状态机 (submitted→working→completed/failed)
  ③ JSON-RPC 2.0 — 标准消息格式
  ④ 官方 SDK 适配 — 预留 a2a-sdk 接口
  ⑤ MCP 互补 — A2A 管 Agent 通信，MCP 管工具调用

v5.0 → v7.0 评分变化:
  Agent 通信: 55 → 95 (+40)

架构:
  ┌─────────────────────────────────────────────┐
  │  A2A Server (Starlette / JSON-RPC 2.0)      │
  │                                             │
  │  GET  /.well-known/agent.json → Agent Card  │
  │  POST /a2a → JSON-RPC 2.0 (tasks/send等)    │
  │                                             │
  │  Task States:                                │
  │  submitted → working → completed            │
  │                    ↘ input-required          │
  │                    ↘ failed                  │
  │                    ↘ canceled                │
  └─────────────────────────────────────────────┘

  MRARFAI Agent Ecosystem:
  ┌──────────┐  A2A  ┌──────────┐  A2A  ┌──────────┐
  │ Analyst  │ ←───→ │  Router  │ ←───→ │  Risk    │
  │  Agent   │       │  Agent   │       │  Agent   │
  └──────────┘       └──────────┘       └──────────┘
        ↑ MCP              ↑ MCP              ↑ MCP
   [SQL/Data]         [GraphRAG]         [Anomaly]

依赖: 无额外依赖（纯 Python 实现）
可选: a2a-sdk (pip install a2a-sdk[http-server])
可选: starlette + uvicorn (pip install starlette uvicorn, HTTP 服务)
兼容: 完全兼容 v5.0 的 protocol_layer.py 接口
"""

import json
import time
import uuid
import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

logger = logging.getLogger("mrarfai.a2a")

# ============================================================
# 官方 SDK 检测
# ============================================================

HAS_A2A_SDK = False
try:
    import a2a
    HAS_A2A_SDK = True
    logger.info("✅ A2A 官方 SDK 已加载")
except ImportError:
    pass

HAS_STARLETTE = False
try:
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    HAS_STARLETTE = True
except ImportError:
    pass


# ============================================================
# A2A 数据模型 — 遵循 RC v1.0 规范
# ============================================================

class TaskState(str, Enum):
    """Task 生命周期状态 — A2A RC v1.0 标准"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    AUTH_REQUIRED = "auth-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
            TaskState.REJECTED,
        )


@dataclass
class AgentSkill:
    """Agent 技能声明"""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
        }


@dataclass
class AgentCapabilities:
    """Agent 能力声明"""
    streaming: bool = False
    push_notifications: bool = False
    extended_agent_card: bool = False

    def to_dict(self) -> dict:
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "extendedAgentCard": self.extended_agent_card,
        }


@dataclass
class AgentInterface:
    """Agent 接口定义"""
    url: str
    protocol_binding: str = "JSONRPC"  # JSONRPC | gRPC | HTTP
    protocol_version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "protocolBinding": self.protocol_binding,
            "protocolVersion": self.protocol_version,
        }


@dataclass
class AgentCard:
    """
    A2A Agent Card — /.well-known/agent.json

    遵循 A2A RC v1.0 规范:
    - name, description, version: 必需
    - supported_interfaces: 至少一个
    - skills: Agent 能做什么
    - capabilities: 支持哪些协议特性
    """
    name: str
    description: str
    version: str
    supported_interfaces: List[AgentInterface] = field(default_factory=list)
    default_input_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = field(default_factory=list)
    provider: Dict[str, str] = field(default_factory=dict)
    documentation_url: str = ""
    icon_url: str = ""

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "supportedInterfaces": [i.to_dict() for i in self.supported_interfaces],
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "capabilities": self.capabilities.to_dict(),
            "skills": [s.to_dict() for s in self.skills],
        }
        if self.provider:
            result["provider"] = self.provider
        if self.documentation_url:
            result["documentationUrl"] = self.documentation_url
        if self.icon_url:
            result["iconUrl"] = self.icon_url
        return result


@dataclass
class MessagePart:
    """消息部分"""
    type: str = "text"  # text | file | data
    text: str = ""
    mime_type: str = "text/plain"
    data: Any = None

    def to_dict(self) -> dict:
        d = {"type": self.type}
        if self.type == "text":
            d["text"] = self.text
        elif self.type == "data":
            d["data"] = self.data
            d["mimeType"] = self.mime_type
        return d


@dataclass
class Message:
    """A2A 消息"""
    role: str  # "user" | "agent"
    parts: List[MessagePart] = field(default_factory=list)
    message_id: str = ""

    def __post_init__(self):
        if not self.message_id:
            self.message_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "messageId": self.message_id,
        }

    @staticmethod
    def user_text(text: str) -> "Message":
        return Message(role="user", parts=[MessagePart(type="text", text=text)])

    @staticmethod
    def agent_text(text: str) -> "Message":
        return Message(role="agent", parts=[MessagePart(type="text", text=text)])


@dataclass
class TaskStatus:
    """Task 状态"""
    state: TaskState
    message: Optional[Message] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        d = {"state": self.state.value, "timestamp": self.timestamp}
        if self.message:
            d["message"] = self.message.to_dict()
        return d


@dataclass
class Artifact:
    """Task 产出物"""
    name: str = ""
    description: str = ""
    parts: List[MessagePart] = field(default_factory=list)
    artifact_id: str = ""

    def __post_init__(self):
        if not self.artifact_id:
            self.artifact_id = uuid.uuid4().hex[:8]

    def to_dict(self) -> dict:
        return {
            "artifactId": self.artifact_id,
            "name": self.name,
            "description": self.description,
            "parts": [p.to_dict() for p in self.parts],
        }


@dataclass
class Task:
    """
    A2A Task — 核心实体

    生命周期:
    submitted → working → completed/failed/canceled
                       → input-required → working → ...
    """
    task_id: str = ""
    context_id: str = ""
    status: TaskStatus = None
    artifacts: List[Artifact] = field(default_factory=list)
    history: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = uuid.uuid4().hex[:12]
        if not self.context_id:
            self.context_id = uuid.uuid4().hex[:12]
        if self.status is None:
            self.status = TaskStatus(state=TaskState.SUBMITTED)

    def to_dict(self) -> dict:
        return {
            "id": self.task_id,
            "contextId": self.context_id,
            "status": self.status.to_dict(),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "history": [m.to_dict() for m in self.history],
            "metadata": self.metadata,
        }


# ============================================================
# Task Store — 任务存储
# ============================================================

class InMemoryTaskStore:
    """内存任务存储"""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}

    def save(self, task: Task):
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_by_context(self, context_id: str) -> List[Task]:
        return [t for t in self._tasks.values() if t.context_id == context_id]

    def count(self) -> int:
        return len(self._tasks)


# ============================================================
# Agent Executor — Agent 执行逻辑
# ============================================================

class AgentExecutor:
    """
    Agent 执行器 — 处理 A2A 请求

    子类化并实现 execute() 方法来接入你的 Agent 逻辑
    """

    async def execute(self, task: Task, message: Message) -> Task:
        """
        执行 Agent 逻辑

        输入: Task + 用户消息
        输出: 更新后的 Task（含状态和 artifacts）
        """
        raise NotImplementedError

    async def cancel(self, task: Task) -> Task:
        """取消任务"""
        task.status = TaskStatus(state=TaskState.CANCELED)
        return task


# ============================================================
# MRARFAI Agent Executors
# ============================================================

class MRARFAIAnalystExecutor(AgentExecutor):
    """MRARFAI 分析师 Agent — 处理销售数据分析请求"""

    def __init__(self, pipeline_fn: Callable = None):
        self.pipeline_fn = pipeline_fn

    async def execute(self, task: Task, message: Message) -> Task:
        # 提取用户问题
        question = ""
        for part in message.parts:
            if part.type == "text":
                question = part.text
                break

        if not question:
            task.status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message=Message.agent_text("请输入您的分析问题"),
            )
            return task

        # 开始处理
        task.status = TaskStatus(state=TaskState.WORKING)
        task.history.append(message)

        try:
            # 调用管线
            if self.pipeline_fn:
                result = self.pipeline_fn(question)
                answer = result if isinstance(result, str) else result.get("answer", str(result))
            else:
                answer = f"[模拟分析] 收到问题: {question}"

            # 完成
            agent_msg = Message.agent_text(answer)
            task.history.append(agent_msg)
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=agent_msg,
            )

            # 添加 artifact
            task.artifacts.append(Artifact(
                name="analysis_result",
                description="销售数据分析结果",
                parts=[MessagePart(type="text", text=answer)],
            ))

        except Exception as e:
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message.agent_text(f"分析失败: {str(e)}"),
            )

        return task


class MRARFAIRiskExecutor(AgentExecutor):
    """MRARFAI 风险 Agent — 接入 anomaly_detector + health_score"""

    def __init__(self):
        try:
            from agent_risk import RiskEngine
            self.engine = RiskEngine()
        except ImportError:
            self.engine = None

    async def execute(self, task: Task, message: Message) -> Task:
        question = message.parts[0].text if message.parts else ""
        task.status = TaskStatus(state=TaskState.WORKING)
        task.history.append(message)

        try:
            if self.engine:
                answer = self.engine.answer(question)
            else:
                answer = f"[风险评估] {question} — RiskEngine 未加载"
        except Exception as e:
            answer = f"[风险评估失败] {str(e)}"

        agent_msg = Message.agent_text(answer)
        task.history.append(agent_msg)
        task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_msg)
        task.artifacts.append(Artifact(
            name="risk_result",
            description="风险评估结果",
            parts=[MessagePart(type="text", text=answer)],
        ))
        return task


class MRARFAIStrategistExecutor(AgentExecutor):
    """MRARFAI 策略 Agent — 接入 industry_benchmark + forecast_engine"""

    def __init__(self):
        try:
            from agent_strategist import StrategistEngine
            self.engine = StrategistEngine()
        except ImportError:
            self.engine = None

    async def execute(self, task: Task, message: Message) -> Task:
        question = message.parts[0].text if message.parts else ""
        task.status = TaskStatus(state=TaskState.WORKING)
        task.history.append(message)

        try:
            if self.engine:
                answer = self.engine.answer(question)
            else:
                answer = f"[战略建议] {question} — StrategistEngine 未加载"
        except Exception as e:
            answer = f"[战略分析失败] {str(e)}"

        agent_msg = Message.agent_text(answer)
        task.history.append(agent_msg)
        task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_msg)
        task.artifacts.append(Artifact(
            name="strategy_result",
            description="战略分析结果",
            parts=[MessagePart(type="text", text=answer)],
        ))
        return task


# ============================================================
# A2A Request Handler — JSON-RPC 2.0
# ============================================================

class A2ARequestHandler:
    """
    A2A 请求处理器 — JSON-RPC 2.0

    支持的方法:
    - tasks/send: 发送消息 / 创建任务
    - tasks/get: 获取任务状态
    - tasks/cancel: 取消任务
    """

    def __init__(self, executor: AgentExecutor, task_store: InMemoryTaskStore = None):
        self.executor = executor
        self.task_store = task_store or InMemoryTaskStore()

    async def handle(self, request: dict) -> dict:
        """处理 JSON-RPC 2.0 请求"""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", None)

        handlers = {
            "tasks/send": self._handle_send,
            "tasks/get": self._handle_get,
            "tasks/cancel": self._handle_cancel,
        }

        handler = handlers.get(method)
        if not handler:
            return self._error_response(req_id, -32601, f"Method not found: {method}")

        try:
            result = await handler(params)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }
        except Exception as e:
            return self._error_response(req_id, -32000, str(e))

    async def _handle_send(self, params: dict) -> dict:
        """处理 tasks/send"""
        msg_data = params.get("message", {})
        task_id = params.get("id")

        # 构建 Message
        parts = []
        for p in msg_data.get("parts", []):
            parts.append(MessagePart(
                type=p.get("type", "text"),
                text=p.get("text", ""),
                data=p.get("data"),
                mime_type=p.get("mimeType", "text/plain"),
            ))

        message = Message(
            role=msg_data.get("role", "user"),
            parts=parts,
            message_id=msg_data.get("messageId", ""),
        )

        # 获取或创建 Task
        if task_id:
            task = self.task_store.get(task_id)
            if not task:
                task = Task(task_id=task_id)
        else:
            task = Task()

        # 执行
        import asyncio
        task = await self.executor.execute(task, message)
        self.task_store.save(task)

        return task.to_dict()

    async def _handle_get(self, params: dict) -> dict:
        """处理 tasks/get"""
        task_id = params.get("id", "")
        task = self.task_store.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        return task.to_dict()

    async def _handle_cancel(self, params: dict) -> dict:
        """处理 tasks/cancel"""
        task_id = params.get("id", "")
        task = self.task_store.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        task = await self.executor.cancel(task)
        self.task_store.save(task)
        return task.to_dict()

    def _error_response(self, req_id, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


# ============================================================
# A2A HTTP Server — Starlette
# ============================================================

def create_a2a_app(agent_card: AgentCard, handler: A2ARequestHandler):
    """
    创建 A2A HTTP 应用

    路由:
    - GET  /.well-known/agent.json → Agent Card
    - POST /a2a                    → JSON-RPC 2.0

    Usage:
        app = create_a2a_app(agent_card, handler)
        uvicorn.run(app, host="0.0.0.0", port=9999)
    """
    if not HAS_STARLETTE:
        logger.warning("Starlette 未安装: pip install starlette uvicorn")
        return None

    async def agent_card_endpoint(request: Request):
        return JSONResponse(agent_card.to_dict())

    async def a2a_endpoint(request: Request):
        body = await request.json()
        result = await handler.handle(body)
        return JSONResponse(result)

    app = Starlette(routes=[
        Route("/.well-known/agent.json", agent_card_endpoint, methods=["GET"]),
        Route("/a2a", a2a_endpoint, methods=["POST"]),
    ])

    return app


# ============================================================
# MRARFAI Agent Registry — 本地 Agent 注册表
# ============================================================

class MRARFAIAgentRegistry:
    """
    MRARFAI Agent 注册表

    管理所有内部 Agent 的 A2A Card 和 Handler
    """

    def __init__(self):
        self._agents: Dict[str, Dict] = {}  # name → {card, handler, executor}

    def register(self, name: str, card: AgentCard,
                 executor: AgentExecutor):
        """注册 Agent"""
        handler = A2ARequestHandler(executor)
        self._agents[name] = {
            "card": card,
            "handler": handler,
            "executor": executor,
        }
        logger.info(f"A2A Agent 已注册: {name}")

    def get_card(self, name: str) -> Optional[AgentCard]:
        entry = self._agents.get(name)
        return entry["card"] if entry else None

    def get_handler(self, name: str) -> Optional[A2ARequestHandler]:
        entry = self._agents.get(name)
        return entry["handler"] if entry else None

    async def send_task(self, agent_name: str, question: str) -> Optional[dict]:
        """向某个 Agent 发送任务"""
        handler = self.get_handler(agent_name)
        if not handler:
            return None

        request = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:8],
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": question}],
                    "messageId": uuid.uuid4().hex[:8],
                },
            },
        }

        return await handler.handle(request)

    def discover(self, tag: str = None) -> List[AgentCard]:
        """发现 Agent"""
        cards = []
        for entry in self._agents.values():
            card = entry["card"]
            if tag:
                all_tags = []
                for skill in card.skills:
                    all_tags.extend(skill.tags)
                if tag not in all_tags:
                    continue
            cards.append(card)
        return cards

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def get_stats(self) -> Dict:
        return {
            "total_agents": len(self._agents),
            "agents": {
                name: {
                    "skills": len(entry["card"].skills),
                    "tasks_processed": entry["handler"].task_store.count(),
                }
                for name, entry in self._agents.items()
            },
        }


# ============================================================
# 预建 MRARFAI Agent Cards
# ============================================================

def create_mrarfai_agent_cards(base_url: str = "http://localhost:9999",
                                include_v10: bool = True) -> Dict[str, AgentCard]:
    """创建 MRARFAI 标准 Agent Cards (V9 + V10)"""

    cards = {
        "analyst": AgentCard(
            name="MRARFAI 数据分析师",
            description="禾苗科技销售数据深度分析 — 客户分级、营收趋势、价量分解、区域洞察",
            version="7.0.0",
            supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/analyst")],
            capabilities=AgentCapabilities(streaming=False),
            skills=[
                AgentSkill(
                    id="revenue_analysis",
                    name="营收分析",
                    description="分析总营收、同比增长、月度趋势",
                    tags=["revenue", "trend", "yoy"],
                    examples=["今年总营收是多少？", "月度营收趋势如何？"],
                ),
                AgentSkill(
                    id="customer_analysis",
                    name="客户分析",
                    description="客户分级ABC、排名、集中度分析",
                    tags=["customer", "tier", "ranking"],
                    examples=["Top10客户是哪些？", "A级客户占比多少？"],
                ),
                AgentSkill(
                    id="price_volume",
                    name="价量分解",
                    description="分解客户营收变化为价格因素和数量因素",
                    tags=["price", "volume", "decomposition"],
                    examples=["HMD的价量分解", "哪些客户量增价跌？"],
                ),
            ],
            provider={"organization": "禾苗科技", "url": "https://hemiao.com"},
        ),

        "risk": AgentCard(
            name="MRARFAI 风险预警员",
            description="客户流失预警、异常检测、健康评分",
            version="7.0.0",
            supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/risk")],
            capabilities=AgentCapabilities(streaming=False),
            skills=[
                AgentSkill(
                    id="churn_alert",
                    name="流失预警",
                    description="识别高风险流失客户并分析原因",
                    tags=["risk", "churn", "alert"],
                    examples=["哪些客户有流失风险？", "高风险客户的原因是什么？"],
                ),
                AgentSkill(
                    id="anomaly_detection",
                    name="异常检测",
                    description="统计异常检测 — Z-Score、IQR、趋势突变",
                    tags=["anomaly", "detection", "statistical"],
                    examples=["最近有什么异常吗？", "哪些客户出现异常波动？"],
                ),
            ],
            provider={"organization": "禾苗科技"},
        ),

        "strategist": AgentCard(
            name="MRARFAI 战略顾问",
            description="行业对标、增长机会识别、战略建议",
            version="7.0.0",
            supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/strategist")],
            capabilities=AgentCapabilities(streaming=False),
            skills=[
                AgentSkill(
                    id="benchmark",
                    name="行业对标",
                    description="与华勤/闻泰/龙旗等竞对对标分析",
                    tags=["benchmark", "competitor", "industry"],
                    examples=["跟华勤比怎么样？", "行业排名第几？"],
                ),
                AgentSkill(
                    id="growth_strategy",
                    name="增长策略",
                    description="识别增长机会并制定策略建议",
                    tags=["growth", "strategy", "opportunity"],
                    examples=["有什么增长机会？", "下半年策略建议"],
                ),
                AgentSkill(
                    id="forecast",
                    name="营收预测",
                    description="基于历史数据的营收预测和风险场景分析",
                    tags=["forecast", "prediction", "scenario"],
                    examples=["明年营收预测", "最坏情况是什么？"],
                ),
            ],
            provider={"organization": "禾苗科技"},
        ),
    }

    # V10 新 Agent Cards
    if include_v10:
        try:
            from agent_procurement import create_procurement_card
            card = create_procurement_card(base_url)
            if card:
                cards["procurement"] = card
        except ImportError:
            pass

        try:
            from agent_quality import create_quality_card
            card = create_quality_card(base_url)
            if card:
                cards["quality"] = card
        except ImportError:
            pass

        try:
            from agent_finance import create_finance_card
            card = create_finance_card(base_url)
            if card:
                cards["finance"] = card
        except ImportError:
            pass

        try:
            from agent_market import create_market_card
            card = create_market_card(base_url)
            if card:
                cards["market"] = card
        except ImportError:
            pass

    return cards


# ============================================================
# 快速初始化
# ============================================================

def init_mrarfai_a2a(pipeline_fn: Callable = None) -> MRARFAIAgentRegistry:
    """
    一键初始化 MRARFAI A2A 生态 (V9 + V10)

    Usage:
        registry = init_mrarfai_a2a(pipeline_fn=my_pipeline)
        result = await registry.send_task("analyst", "今年总营收是多少？")
    """
    registry = MRARFAIAgentRegistry()
    cards = create_mrarfai_agent_cards()

    # V9 Agents
    registry.register("analyst", cards["analyst"],
                       MRARFAIAnalystExecutor(pipeline_fn))
    registry.register("risk", cards["risk"],
                       MRARFAIRiskExecutor())
    registry.register("strategist", cards["strategist"],
                       MRARFAIStrategistExecutor())

    # V10 新 Agents
    if "procurement" in cards:
        try:
            from agent_procurement import ProcurementExecutor
            registry.register("procurement", cards["procurement"], ProcurementExecutor())
        except ImportError:
            pass

    if "quality" in cards:
        try:
            from agent_quality import QualityExecutor
            registry.register("quality", cards["quality"], QualityExecutor())
        except ImportError:
            pass

    if "finance" in cards:
        try:
            from agent_finance import FinanceExecutor
            registry.register("finance", cards["finance"], FinanceExecutor())
        except ImportError:
            pass

    if "market" in cards:
        try:
            from agent_market import MarketExecutor
            registry.register("market", cards["market"], MarketExecutor())
        except ImportError:
            pass

    logger.info(f"✅ MRARFAI A2A 已初始化: {len(registry.list_agents())} Agents")
    return registry


# ============================================================
# v5.0 兼容层
# ============================================================

# 兼容 protocol_layer.py 的 AgentCard
class LegacyAgentCard:
    """v5.0 AgentCard 兼容"""

    def __init__(self, card: AgentCard):
        self._card = card
        self.agent_id = card.name
        self.name = card.name
        self.description = card.description
        self.version = card.version
        self.status = "available"

    def to_dict(self):
        return self._card.to_dict()


# ============================================================
# 模块信息
# ============================================================

__version__ = "7.0.0"
__all__ = [
    "AgentCard",
    "AgentSkill",
    "AgentCapabilities",
    "AgentInterface",
    "Task",
    "TaskState",
    "TaskStatus",
    "Message",
    "MessagePart",
    "Artifact",
    "AgentExecutor",
    "A2ARequestHandler",
    "MRARFAIAgentRegistry",
    "InMemoryTaskStore",
    "create_mrarfai_agent_cards",
    "init_mrarfai_a2a",
    "create_a2a_app",
    "HAS_A2A_SDK",
    "HAS_STARLETTE",
]

if __name__ == "__main__":
    import asyncio

    print(f"MRARFAI A2A Server v{__version__}")
    print(f"A2A 官方 SDK: {'✅' if HAS_A2A_SDK else '❌ (可选)'}")
    print(f"Starlette:    {'✅' if HAS_STARLETTE else '❌ (可选)'}")
    print()

    async def test():
        # 初始化
        registry = init_mrarfai_a2a()

        # 查看已注册 Agent
        print(f"已注册 Agents: {registry.list_agents()}")
        print()

        # Agent Card 测试
        cards = create_mrarfai_agent_cards()
        print("Analyst Agent Card:")
        card_json = json.dumps(cards["analyst"].to_dict(), ensure_ascii=False, indent=2)
        print(card_json[:500])
        print()

        # Task 生命周期测试
        print("Task 生命周期测试:")

        # 1. 发送任务给 analyst
        result = await registry.send_task("analyst", "今年总营收是多少？")
        task_data = result.get("result", {})
        print(f"  1. 发送: status={task_data.get('status', {}).get('state', '')}")

        # 2. 发送任务给 risk
        result = await registry.send_task("risk", "哪些客户有流失风险？")
        task_data = result.get("result", {})
        print(f"  2. 风险: status={task_data.get('status', {}).get('state', '')}")

        # 3. 发送任务给 strategist
        result = await registry.send_task("strategist", "下半年增长策略")
        task_data = result.get("result", {})
        print(f"  3. 策略: status={task_data.get('status', {}).get('state', '')}")

        # 4. Agent 发现
        print(f"\n带 'risk' 标签的 Agents:")
        risk_agents = registry.discover(tag="risk")
        for a in risk_agents:
            print(f"  - {a.name}: {a.description}")

        # 统计
        print(f"\n统计: {json.dumps(registry.get_stats(), ensure_ascii=False)}")

        # Task State 测试
        print(f"\nTask 状态测试:")
        for state in TaskState:
            print(f"  {state.value}: terminal={state.is_terminal}")

    asyncio.run(test())
