#!/usr/bin/env python3
"""
MRARFAI v3.3 — MCP/A2A Protocol Layer
========================================
标准化Agent通信与工具访问协议

两大协议:
  1. MCP (Model Context Protocol) — Agent访问外部工具的标准接口
     - Tool注册/发现/调用
     - 上下文传递
     - 安全与权限

  2. A2A (Agent-to-Agent) — Agent间通信协议  
     - Agent注册/发现
     - 消息传递（请求/响应/事件）
     - 任务委派与状态同步
     - Agent Card（能力声明）

当前: 协议抽象层 + 本地实现
未来: 可对接Google A2A、Anthropic MCP server
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# ============================================================
# MCP — Model Context Protocol
# ============================================================

class ToolType(Enum):
    """工具类型"""
    DATA_QUERY = "data_query"     # 数据查询
    CALCULATION = "calculation"   # 计算
    EXTERNAL_API = "external_api" # 外部API
    FILE_IO = "file_io"          # 文件操作
    NOTIFICATION = "notification" # 通知


@dataclass
class ToolSchema:
    """MCP工具schema — 遵循MCP标准"""
    name: str
    description: str
    tool_type: ToolType
    input_schema: Dict[str, Any]       # JSON Schema
    output_schema: Dict[str, Any] = field(default_factory=dict)
    requires_auth: bool = False
    rate_limit: int = 0  # 0=无限制
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "requires_auth": self.requires_auth,
            "version": self.version,
        }


@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
        }


class MCPToolRegistry:
    """
    MCP工具注册中心
    
    管理所有可用工具，提供发现和调用能力
    """

    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}
        self._handlers: Dict[str, Callable] = {}
        self._call_count: Dict[str, int] = {}

    def register(self, schema: ToolSchema, handler: Callable):
        """注册工具"""
        self._tools[schema.name] = schema
        self._handlers[schema.name] = handler
        self._call_count[schema.name] = 0

    def unregister(self, name: str):
        """注销工具"""
        self._tools.pop(name, None)
        self._handlers.pop(name, None)

    def discover(self, tool_type: ToolType = None) -> List[ToolSchema]:
        """发现可用工具"""
        if tool_type:
            return [t for t in self._tools.values() if t.tool_type == tool_type]
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[ToolSchema]:
        """获取工具schema"""
        return self._tools.get(name)

    def call(self, name: str, params: Dict[str, Any],
             context: Dict[str, Any] = None) -> ToolCallResult:
        """调用工具"""
        if name not in self._handlers:
            return ToolCallResult(
                tool_name=name, success=False,
                error=f"Tool '{name}' not found"
            )

        schema = self._tools[name]
        handler = self._handlers[name]

        # 限流检查
        if schema.rate_limit > 0:
            if self._call_count.get(name, 0) >= schema.rate_limit:
                return ToolCallResult(
                    tool_name=name, success=False,
                    error="Rate limit exceeded"
                )

        t0 = time.time()
        try:
            result = handler(params, context or {})
            self._call_count[name] = self._call_count.get(name, 0) + 1
            return ToolCallResult(
                tool_name=name, success=True, output=result,
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolCallResult(
                tool_name=name, success=False, error=str(e),
                duration_ms=(time.time() - t0) * 1000,
            )

    def get_tools_prompt(self) -> str:
        """生成工具列表prompt（供LLM选择工具）"""
        if not self._tools:
            return ""
        lines = ["[可用工具]"]
        for name, schema in self._tools.items():
            params = ", ".join(schema.input_schema.get("properties", {}).keys())
            lines.append(f"- {name}: {schema.description} (参数: {params})")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        return {
            "total_tools": len(self._tools),
            "call_counts": dict(self._call_count),
        }


# ============================================================
# A2A — Agent-to-Agent Protocol
# ============================================================

class AgentCapability(Enum):
    """Agent能力声明"""
    DATA_ANALYSIS = "data_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY = "strategy"
    REPORTING = "reporting"
    CRITIQUE = "critique"
    ROUTING = "routing"
    DATA_QUERY = "data_query"


@dataclass
class AgentCard:
    """
    Agent Card — A2A标准能力声明
    
    每个Agent声明自己的:
    - 身份（ID、名称）
    - 能力（可处理什么类型的任务）
    - 通信偏好（输入/输出格式）
    - 状态（是否可用）
    """
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    input_format: str = "text"        # text / json / structured
    output_format: str = "text"
    max_concurrent: int = 1
    status: str = "available"          # available / busy / offline
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "input_format": self.input_format,
            "output_format": self.output_format,
            "status": self.status,
            "version": self.version,
        }

    def can_handle(self, capability: AgentCapability) -> bool:
        return capability in self.capabilities


class MessageType(Enum):
    """A2A消息类型"""
    REQUEST = "request"          # 任务请求
    RESPONSE = "response"        # 任务响应
    DELEGATE = "delegate"        # 委派
    STATUS_UPDATE = "status"     # 状态更新
    EVENT = "event"              # 事件通知
    HANDOFF = "handoff"          # 控制权交接


@dataclass
class A2AMessage:
    """A2A消息"""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)  # 共享上下文
    parent_message_id: str = ""  # 关联消息
    timestamp: str = ""
    ttl: int = 30  # 秒

    def to_dict(self) -> dict:
        return {
            "id": self.message_id,
            "type": self.message_type.value,
            "from": self.sender_id,
            "to": self.receiver_id,
            "payload": self.payload,
            "context": self.context,
            "parent": self.parent_message_id,
            "time": self.timestamp,
        }

    @staticmethod
    def create(msg_type: MessageType, sender: str, receiver: str,
               payload: dict, context: dict = None,
               parent_id: str = "") -> "A2AMessage":
        return A2AMessage(
            message_id=str(uuid.uuid4())[:8],
            message_type=msg_type,
            sender_id=sender,
            receiver_id=receiver,
            payload=payload,
            context=context or {},
            parent_message_id=parent_id,
            timestamp=datetime.now().isoformat(),
        )


class A2ARouter:
    """
    A2A消息路由器
    
    管理Agent注册、消息路由、状态追踪
    """

    def __init__(self):
        self._agents: Dict[str, AgentCard] = {}
        self._message_log: List[A2AMessage] = []
        self._handlers: Dict[str, Callable] = {}  # agent_id → handler

    def register_agent(self, card: AgentCard, handler: Callable = None):
        """注册Agent"""
        self._agents[card.agent_id] = card
        if handler:
            self._handlers[card.agent_id] = handler

    def unregister_agent(self, agent_id: str):
        """注销Agent"""
        self._agents.pop(agent_id, None)
        self._handlers.pop(agent_id, None)

    def discover_agents(self, capability: AgentCapability = None) -> List[AgentCard]:
        """发现Agent"""
        if capability:
            return [a for a in self._agents.values() if a.can_handle(capability)]
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> Optional[AgentCard]:
        """获取Agent Card"""
        return self._agents.get(agent_id)

    def send_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """
        发送消息并获取响应
        
        当前实现: 同步直调
        未来: 可升级为异步消息队列
        """
        self._message_log.append(message)

        # 检查接收方
        receiver = self._agents.get(message.receiver_id)
        if not receiver:
            return A2AMessage.create(
                MessageType.RESPONSE, "system", message.sender_id,
                {"error": f"Agent '{message.receiver_id}' not found"},
                parent_id=message.message_id,
            )

        if receiver.status != "available":
            return A2AMessage.create(
                MessageType.RESPONSE, "system", message.sender_id,
                {"error": f"Agent '{message.receiver_id}' is {receiver.status}"},
                parent_id=message.message_id,
            )

        # 调用handler
        handler = self._handlers.get(message.receiver_id)
        if handler:
            try:
                receiver.status = "busy"
                result = handler(message)
                receiver.status = "available"

                response = A2AMessage.create(
                    MessageType.RESPONSE, message.receiver_id, message.sender_id,
                    {"result": result},
                    context=message.context,
                    parent_id=message.message_id,
                )
                self._message_log.append(response)
                return response
            except Exception as e:
                receiver.status = "available"
                return A2AMessage.create(
                    MessageType.RESPONSE, message.receiver_id, message.sender_id,
                    {"error": str(e)},
                    parent_id=message.message_id,
                )

        return None

    def broadcast(self, sender_id: str, capability: AgentCapability,
                   payload: dict, context: dict = None) -> List[A2AMessage]:
        """广播消息给所有具备特定能力的Agent"""
        targets = self.discover_agents(capability)
        responses = []
        for agent in targets:
            if agent.agent_id == sender_id:
                continue
            msg = A2AMessage.create(
                MessageType.REQUEST, sender_id, agent.agent_id,
                payload, context,
            )
            resp = self.send_message(msg)
            if resp:
                responses.append(resp)
        return responses

    def delegate(self, sender_id: str, receiver_id: str,
                  task: dict, context: dict = None) -> Optional[A2AMessage]:
        """委派任务"""
        msg = A2AMessage.create(
            MessageType.DELEGATE, sender_id, receiver_id,
            {"task": task}, context,
        )
        return self.send_message(msg)

    def handoff(self, sender_id: str, receiver_id: str,
                 state: dict, context: dict = None) -> Optional[A2AMessage]:
        """
        控制权交接 — 关键操作
        传递完整状态，确保上下文不丢失
        """
        msg = A2AMessage.create(
            MessageType.HANDOFF, sender_id, receiver_id,
            {"state": state}, context,
        )
        return self.send_message(msg)

    def get_message_log(self, limit: int = 50) -> List[dict]:
        """获取消息日志"""
        return [m.to_dict() for m in self._message_log[-limit:]]

    def get_stats(self) -> dict:
        return {
            "total_agents": len(self._agents),
            "agents": {aid: card.to_dict() for aid, card in self._agents.items()},
            "total_messages": len(self._message_log),
            "messages_by_type": {},
        }


# ============================================================
# MRARFAI Protocol Manager — 统一管理
# ============================================================

class ProtocolManager:
    """
    协议管理器 — 统一MCP和A2A
    
    负责:
    1. 初始化内置工具和Agent
    2. 提供统一的工具/Agent访问入口
    3. 管理跨Agent的上下文传递
    """

    def __init__(self):
        self.mcp = MCPToolRegistry()
        self.a2a = A2ARouter()
        self._shared_context: Dict[str, Any] = {}

    def setup_builtin_tools(self, data_query_fn: Callable = None):
        """注册内置MCP工具"""
        # 数据查询工具
        if data_query_fn:
            self.mcp.register(
                ToolSchema(
                    name="sales_data_query",
                    description="查询禾苗销售数据（客户、产品、区域、月度）",
                    tool_type=ToolType.DATA_QUERY,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "自然语言查询"},
                        },
                        "required": ["query"],
                    },
                ),
                lambda params, ctx: data_query_fn(params.get("query", "")),
            )

        # 计算工具
        self.mcp.register(
            ToolSchema(
                name="calculator",
                description="执行数值计算（同比、环比、占比等）",
                tool_type=ToolType.CALCULATION,
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            ),
            lambda params, ctx: {"result": eval(params.get("expression", "0"))},
        )

    def setup_builtin_agents(self):
        """注册内置A2A Agent Cards"""
        agents = [
            AgentCard(
                agent_id="analyst",
                name="📊 数据分析师",
                description="精准解读销售数据，识别趋势和模式",
                capabilities=[AgentCapability.DATA_ANALYSIS],
            ),
            AgentCard(
                agent_id="risk",
                name="🛡️ 风控专家",
                description="识别客户流失风险和异常波动",
                capabilities=[AgentCapability.RISK_ASSESSMENT],
            ),
            AgentCard(
                agent_id="strategist",
                name="💡 策略师",
                description="发现增长机会，制定可执行战略",
                capabilities=[AgentCapability.STRATEGY],
            ),
            AgentCard(
                agent_id="reporter",
                name="🖊️ 报告员",
                description="综合专家分析，生成CEO可读报告",
                capabilities=[AgentCapability.REPORTING],
            ),
            AgentCard(
                agent_id="critic",
                name="🔍 质量审查",
                description="审查报告质量，提供改进建议",
                capabilities=[AgentCapability.CRITIQUE],
            ),
            AgentCard(
                agent_id="router",
                name="🧭 智能路由",
                description="分析问题意图，路由到合适的Agent",
                capabilities=[AgentCapability.ROUTING],
            ),
        ]
        for card in agents:
            self.a2a.register_agent(card)

    def set_shared_context(self, key: str, value: Any):
        """设置共享上下文"""
        self._shared_context[key] = value

    def get_shared_context(self, key: str = None) -> Any:
        """获取共享上下文"""
        if key:
            return self._shared_context.get(key)
        return dict(self._shared_context)

    def get_status(self) -> dict:
        """获取协议层完整状态"""
        return {
            "mcp": self.mcp.get_stats(),
            "a2a": self.a2a.get_stats(),
            "shared_context_keys": list(self._shared_context.keys()),
        }


# ============================================================
# 全局实例
# ============================================================

_protocol_manager: Optional[ProtocolManager] = None


def get_protocol_manager() -> ProtocolManager:
    """获取全局协议管理器"""
    global _protocol_manager
    if _protocol_manager is None:
        _protocol_manager = ProtocolManager()
        _protocol_manager.setup_builtin_agents()
    return _protocol_manager
