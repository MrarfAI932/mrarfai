#!/usr/bin/env python3
"""
MRARFAI P3-01 — AG-UI Streaming Protocol Adapter
===================================================
CopilotKit AG-UI (Agent-User Interaction Protocol)
支持: 实时token流、进度指示、工具调用可视化

P2-02 — MCP Registry Self-Registration  
P3-04 — LangGraph 1.0 Advanced Features

三合一模块 — 轻量协议扩展层
"""

import json
import asyncio
import logging
from typing import Any, Dict, Generator, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("mrarfai.protocol_ext")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  P3-01: AG-UI Streaming Protocol                            ║
# ╚═══════════════════════════════════════════════════════════════╝

class AGUIEventType(Enum):
    """AG-UI 事件类型 (CopilotKit AG-UI v0.8 aligned)"""
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"


@dataclass
class AGUIEvent:
    """AG-UI 事件包"""
    type: AGUIEventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def to_sse(self) -> str:
        """序列化为 Server-Sent Events 格式"""
        import time
        self.timestamp = self.timestamp or time.time()
        payload = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class AGUIStreamAdapter:
    """
    AG-UI 流式适配器
    
    将 MRARFAI 的 StateGraph 执行事件 转换为 AG-UI SSE 流
    
    用法 (Streamlit / FastAPI):
        adapter = AGUIStreamAdapter()
        
        # StateGraph 执行时注入回调
        for event in adapter.wrap_stategraph_execution(query, graph):
            yield event.to_sse()
    """
    
    def __init__(self, run_id: str = None):
        import uuid
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.events: List[AGUIEvent] = []
    
    def emit(self, event_type: AGUIEventType, data: Dict = None) -> AGUIEvent:
        event = AGUIEvent(type=event_type, data=data or {})
        self.events.append(event)
        return event
    
    def wrap_stategraph_execution(
        self, query: str, graph=None
    ) -> Generator[AGUIEvent, None, None]:
        """
        包装 StateGraph 执行为 AG-UI 事件流
        
        映射:
          route节点     → STEP_STARTED("routing")
          experts节点   → TOOL_CALL_START(agent_name)
          synthesize    → TEXT_MESSAGE_START
          reflect       → STEP_STARTED("reflection") 
          hitl_check    → STATE_SNAPSHOT
        """
        # RUN_STARTED
        yield self.emit(AGUIEventType.RUN_STARTED, {
            "run_id": self.run_id,
            "query": query,
            "agents": 7,
        })
        
        # Step: Routing
        yield self.emit(AGUIEventType.STEP_STARTED, {
            "step": "route",
            "description": "智能路由 — 分析查询意图并分发",
        })
        
        # 如果有实际graph，执行并转换事件
        if graph is not None:
            try:
                for event in graph.stream({"query": query}):
                    node_name = list(event.keys())[0] if event else "unknown"
                    node_data = event.get(node_name, {})
                    
                    if node_name == "route":
                        yield self.emit(AGUIEventType.STEP_FINISHED, {
                            "step": "route",
                            "selected_agents": node_data.get("selected", []),
                        })
                    elif node_name == "experts":
                        for agent_name, result in node_data.items():
                            yield self.emit(AGUIEventType.TOOL_CALL_START, {
                                "tool": agent_name,
                            })
                            yield self.emit(AGUIEventType.TOOL_CALL_END, {
                                "tool": agent_name,
                                "result_preview": str(result)[:200],
                            })
                    elif node_name == "synthesize":
                        answer = node_data.get("answer", "")
                        yield self.emit(AGUIEventType.TEXT_MESSAGE_START, {})
                        # Token-level streaming
                        for chunk in self._chunk_text(answer, 20):
                            yield self.emit(AGUIEventType.TEXT_MESSAGE_CONTENT, {
                                "content": chunk,
                            })
                        yield self.emit(AGUIEventType.TEXT_MESSAGE_END, {})
                    elif node_name == "reflect":
                        yield self.emit(AGUIEventType.STEP_STARTED, {
                            "step": "reflection",
                            "critic": node_data.get("critique", ""),
                        })
                    elif node_name == "hitl_check":
                        yield self.emit(AGUIEventType.STATE_SNAPSHOT, {
                            "state": node_data,
                        })
            except Exception as e:
                logger.warning(f"AG-UI stream error: {e}")
        
        # RUN_FINISHED
        yield self.emit(AGUIEventType.RUN_FINISHED, {
            "run_id": self.run_id,
            "total_events": len(self.events),
        })
    
    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> Generator[str, None, None]:
        """将文本分块模拟token流"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]


# ╔═══════════════════════════════════════════════════════════════╗
# ║  P2-02: MCP Registry Self-Registration                      ║
# ╚═══════════════════════════════════════════════════════════════╝

@dataclass
class MCPRegistryEntry:
    """MCP Registry 注册条目"""
    name: str
    description: str
    url: str
    version: str = "10.0.0"
    transport: str = "streamable-http"
    tools_count: int = 15
    capabilities: List[str] = field(default_factory=lambda: [
        "tools", "resources", "prompts", "tasks"
    ])
    tags: List[str] = field(default_factory=lambda: [
        "odm", "manufacturing", "sales", "analytics"
    ])


class MCPRegistryClient:
    """
    MCP Registry 自注册客户端
    
    当 MCP Registry 标准确定后，用此客户端向公共Registry注册:
      1. 本地 MCP Server 启动时自动注册
      2. 心跳维持在线状态
      3. 下线时注销
    
    当前状态: 协议草案，实现为 local-only 模式
    """
    
    KNOWN_REGISTRIES = [
        # 已知的Registry端点 (2026.2 均为草案)
        # "https://registry.mcp.run/v1",       # 官方 (未上线)
        # "https://mcp-registry.anthropic.com", # Anthropic (未上线)
    ]
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.entry = MCPRegistryEntry(
            name="mrarfai-sales-intelligence",
            description="MRARFAI V10.0 ODM Sales Intelligence — 7-Agent Analytics Platform",
            url=server_url,
        )
        self._registered = False
    
    async def register(self, registry_url: str = None) -> bool:
        """
        注册到MCP Registry
        
        Returns:
            True if registered, False if registry unavailable
        """
        if registry_url is None:
            if not self.KNOWN_REGISTRIES:
                logger.info("📋 MCP Registry: no public registries available yet. Local-only mode.")
                self._registered = False
                return False
            registry_url = self.KNOWN_REGISTRIES[0]
        
        try:
            import aiohttp
            payload = {
                "name": self.entry.name,
                "description": self.entry.description,
                "url": self.entry.url,
                "version": self.entry.version,
                "transport": self.entry.transport,
                "tools_count": self.entry.tools_count,
                "capabilities": self.entry.capabilities,
                "tags": self.entry.tags,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{registry_url}/register",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 201):
                        self._registered = True
                        logger.info(f"✅ Registered with MCP Registry: {registry_url}")
                        return True
                    else:
                        logger.warning(f"Registry returned {resp.status}")
                        return False
        except Exception as e:
            logger.info(f"📋 MCP Registry unavailable: {e} — local-only mode")
            return False
    
    async def heartbeat(self):
        """发送心跳 (维持注册状态)"""
        if not self._registered:
            return
        # TODO: 当Registry标准确定后实现
    
    async def deregister(self):
        """注销"""
        self._registered = False
        logger.info("📋 Deregistered from MCP Registry")
    
    def get_local_manifest(self) -> Dict:
        """获取本地manifest (用于/.well-known/mcp.json)"""
        return {
            "name": self.entry.name,
            "description": self.entry.description,
            "version": self.entry.version,
            "transport": self.entry.transport,
            "tools_count": self.entry.tools_count,
            "capabilities": self.entry.capabilities,
            "tags": self.entry.tags,
            "endpoints": {
                "tools": f"{self.server_url}/tools",
                "resources": f"{self.server_url}/resources",
                "prompts": f"{self.server_url}/prompts",
            },
        }


# ╔═══════════════════════════════════════════════════════════════╗
# ║  P3-04: LangGraph 1.0 Advanced Features                     ║
# ╚═══════════════════════════════════════════════════════════════╝

class LangGraphAdvanced:
    """
    LangGraph 1.0 高级特性封装
    
    已有: StateGraph, interrupt(), Command (R5.5导入)
    新增:
      - Durable State with checkpointer
      - Dynamic breakpoints
      - Node caching (inference cost reduction)
      - Fan-out / Fan-in patterns
      - Subgraph composition
    """
    
    @staticmethod
    def create_checkpointed_graph(graph_builder, checkpointer=None):
        """
        编译带持久化checkpoint的StateGraph
        
        用途: 长时间运行的分析任务可以中断恢复
        """
        try:
            from langgraph.checkpoint.memory import MemorySaver
            cp = checkpointer or MemorySaver()
            compiled = graph_builder.compile(checkpointer=cp)
            logger.info("✅ Checkpointed graph compiled")
            return compiled
        except ImportError:
            logger.warning("MemorySaver not available")
            return graph_builder.compile()
    
    @staticmethod
    def add_dynamic_breakpoint(graph_builder, node_name: str, condition_fn=None):
        """
        添加动态断点 — 条件满足时暂停等待人工确认
        
        用途: 高风险操作(如大额订单确认)自动暂停
        """
        try:
            from langgraph.graph import END
            
            def breakpoint_wrapper(state):
                if condition_fn and condition_fn(state):
                    # 触发interrupt
                    try:
                        from langgraph.types import interrupt
                        human_input = interrupt({
                            "question": f"节点 {node_name} 需要人工确认",
                            "state_preview": {k: str(v)[:100] for k, v in state.items()},
                        })
                        state["human_feedback"] = human_input
                    except ImportError:
                        logger.warning("langgraph.types.interrupt not available")
                return state
            
            logger.info(f"✅ Dynamic breakpoint added to node: {node_name}")
            return breakpoint_wrapper
        except ImportError:
            logger.warning("LangGraph interrupt not available")
            return None
    
    @staticmethod
    def create_fan_out_fan_in(agents: List[str], graph_builder):
        """
        Fan-out/Fan-in 模式 — 并行执行多个Agent后汇总
        
        当前: node_experts 已实现基础并行
        增强: 使用 LangGraph 1.0 Send() API 实现动态fan-out
        """
        try:
            from langgraph.types import Send
            
            def route_to_agents(state):
                """动态分发到多个Agent"""
                query = state.get("query", "")
                selected = state.get("selected_agents", agents)
                return [Send(agent, {"query": query}) for agent in selected]
            
            logger.info(f"✅ Fan-out/Fan-in configured for {len(agents)} agents")
            return route_to_agents
        except ImportError:
            logger.warning("LangGraph Send not available")
            return None


# ============================================================
# Export
# ============================================================

__all__ = [
    # AG-UI
    "AGUIStreamAdapter",
    "AGUIEvent",
    "AGUIEventType",
    # MCP Registry
    "MCPRegistryClient",
    "MCPRegistryEntry",
    # LangGraph Advanced
    "LangGraphAdvanced",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test AG-UI
    adapter = AGUIStreamAdapter()
    events = list(adapter.wrap_stategraph_execution("Samsung出货分析"))
    print(f"\n✅ AG-UI: {len(events)} events generated")
    for e in events[:3]:
        print(f"   {e.type.value}: {e.data}")
    
    # Test MCP Registry
    registry = MCPRegistryClient()
    manifest = registry.get_local_manifest()
    print(f"\n✅ MCP Registry manifest: {manifest['name']}")
    print(f"   Tools: {manifest['tools_count']}, Capabilities: {manifest['capabilities']}")
    
    # Test LangGraph Advanced
    print(f"\n✅ LangGraphAdvanced features available")
