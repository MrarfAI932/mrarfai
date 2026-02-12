#!/usr/bin/env python3
"""
MRARFAI Streaming v4.0
========================
实时渐进式输出：
- StreamCallback: 管道各阶段事件推送
- StreamManager: Streamlit UI 容器管理
- 支持并行 Agent 实时状态更新

事件类型：
  stage_start  — 阶段开始（数据查询/路由/Agent执行/报告/审查）
  stage_end    — 阶段完成
  agent_start  — 单个Agent开始
  agent_done   — 单个Agent完成（含输出）
  tool_call    — Agent调用工具
  thinking     — 推理日志
  token_stream — LLM token流式输出（预留）
  error        — 错误
"""

import time
import threading
import queue
from typing import Callable, Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    AGENT_START = "agent_start"
    AGENT_DONE = "agent_done"
    TOOL_CALL = "tool_call"
    THINKING = "thinking"
    TOKEN = "token_stream"
    ERROR = "error"


@dataclass
class StreamEvent:
    """一个流式事件"""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class StreamCallback:
    """
    流式回调管理器 — multi_agent.py 在关键节点调用 emit()
    chat_tab.py 通过 poll() 或 on_event 消费事件

    用法（生产端）：
        cb = StreamCallback()
        cb.emit(EventType.STAGE_START, {"stage": "routing", "label": "🧭 智能路由"})
        cb.emit(EventType.AGENT_START, {"agent": "analyst", "name": "📊 数据分析师"})
        cb.emit(EventType.AGENT_DONE, {"agent": "analyst", "output": "..."})

    用法（消费端）：
        while not cb.is_complete:
            event = cb.poll(timeout=0.1)
            if event:
                update_ui(event)
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._events: List[StreamEvent] = []
        self._complete = False
        self._lock = threading.Lock()
        self._on_event: Optional[Callable] = None  # 同步回调（可选）

    def emit(self, event_type: EventType, data: dict = None):
        """发射事件"""
        event = StreamEvent(type=event_type, data=data or {})
        with self._lock:
            self._events.append(event)
        self._queue.put(event)
        # 同步回调
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    def poll(self, timeout: float = 0.1) -> Optional[StreamEvent]:
        """轮询获取下一个事件"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> List[StreamEvent]:
        """获取所有待处理事件"""
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def mark_complete(self):
        """标记流结束"""
        self._complete = True
        self.emit(EventType.STAGE_END, {"stage": "complete"})

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def all_events(self) -> List[StreamEvent]:
        with self._lock:
            return list(self._events)

    def set_on_event(self, callback: Callable):
        """设置同步事件回调"""
        self._on_event = callback


# ============================================================
# Pipeline Stage Helpers（简化 emit 调用）
# ============================================================

class PipelineStream:
    """
    Pipeline 流式辅助器 — 封装常用的 emit 模式

    Usage:
        ps = PipelineStream(callback)
        ps.start_stage("data_query", "🔍 智能数据查询")
        ...
        ps.end_stage("data_query", elapsed_ms=123.4, detail="提取了3000字")
        ps.agent_start("analyst", "📊 数据分析师")
        ps.agent_done("analyst", "📊 数据分析师", output="分析结果...")
    """

    def __init__(self, callback: Optional[StreamCallback] = None):
        self.cb = callback
        self._stage_starts: Dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.cb is not None

    def start_stage(self, stage_id: str, label: str):
        if not self.cb:
            return
        self._stage_starts[stage_id] = time.time()
        self.cb.emit(EventType.STAGE_START, {
            "stage": stage_id, "label": label,
        })

    def end_stage(self, stage_id: str, elapsed_ms: float = 0,
                  detail: str = "", extra: dict = None):
        if not self.cb:
            return
        if not elapsed_ms and stage_id in self._stage_starts:
            elapsed_ms = (time.time() - self._stage_starts[stage_id]) * 1000
        data = {"stage": stage_id, "elapsed_ms": round(elapsed_ms, 1), "detail": detail}
        if extra:
            data.update(extra)
        self.cb.emit(EventType.STAGE_END, data)

    def agent_start(self, agent_id: str, agent_name: str):
        if not self.cb:
            return
        self._stage_starts[f"agent_{agent_id}"] = time.time()
        self.cb.emit(EventType.AGENT_START, {
            "agent": agent_id, "name": agent_name,
        })

    def agent_done(self, agent_id: str, agent_name: str,
                   output: str = "", elapsed_ms: float = 0):
        if not self.cb:
            return
        key = f"agent_{agent_id}"
        if not elapsed_ms and key in self._stage_starts:
            elapsed_ms = (time.time() - self._stage_starts[key]) * 1000
        self.cb.emit(EventType.AGENT_DONE, {
            "agent": agent_id, "name": agent_name,
            "output_preview": output[:200] if output else "",
            "output_length": len(output),
            "elapsed_ms": round(elapsed_ms, 1),
        })

    def tool_call(self, agent_id: str, tool_name: str, args: dict = None,
                  result: str = ""):
        if not self.cb:
            return
        self.cb.emit(EventType.TOOL_CALL, {
            "agent": agent_id, "tool": tool_name,
            "args_preview": str(args)[:100] if args else "",
            "result_preview": result[:100],
        })

    def thinking(self, message: str):
        if not self.cb:
            return
        self.cb.emit(EventType.THINKING, {"message": message})

    def error(self, stage: str, message: str):
        if not self.cb:
            return
        self.cb.emit(EventType.ERROR, {"stage": stage, "message": message})

    def complete(self):
        if self.cb:
            self.cb.mark_complete()
