#!/usr/bin/env python3
"""
MRARFAI Langfuse Integration v5.0
====================================
Phase 1 升级：生产级可观测性

与现有 observability.py 并行运行：
  - Langfuse 负责：Dashboard / Eval Loop / 成本对比 / 质量趋势
  - 自建 obs 保留：SQLite 离线分析 / 本地 fallback

集成方式：装饰器 + 上下文管理器，最小侵入
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger("mrarfai.langfuse")

# ============================================================
# Langfuse 初始化（优雅降级）
# ============================================================

HAS_LANGFUSE = False
LANGFUSE_SDK_VERSION = 0  # 0=not installed, 2=v2, 3=v3
_langfuse_client = None

try:
    from langfuse import Langfuse
    HAS_LANGFUSE = True
    # 检测 v3（2025-06-05 GA）: v3 有 get_client / observe 在顶层
    try:
        from langfuse import get_client, observe
        LANGFUSE_SDK_VERSION = 3
    except ImportError:
        # v2: decorators 在子模块
        try:
            from langfuse.decorators import observe, langfuse_context
            LANGFUSE_SDK_VERSION = 2
        except ImportError:
            LANGFUSE_SDK_VERSION = 2  # v2 无 decorators 也能用低级 API
            def observe(*a, **kw):
                def w(fn): return fn
                if a and callable(a[0]): return a[0]
                return w
    logger.info(f"Langfuse SDK v{LANGFUSE_SDK_VERSION} 已加载")
except ImportError:
    logger.info("Langfuse 未安装，使用本地可观测性。pip install langfuse 启用。")
    def observe(*args, **kwargs):
        def wrapper(fn): return fn
        if args and callable(args[0]): return args[0]
        return wrapper


def init_langfuse(
    public_key: str = None,
    secret_key: str = None,
    host: str = None,
) -> Optional['Langfuse']:
    """
    初始化 Langfuse 客户端
    
    优先级：参数 > 环境变量
    环境变量:
      LANGFUSE_PUBLIC_KEY=pk-lf-...
      LANGFUSE_SECRET_KEY=sk-lf-...
      LANGFUSE_HOST=https://cloud.langfuse.com
    """
    global _langfuse_client, HAS_LANGFUSE
    
    if not HAS_LANGFUSE:
        logger.warning("Langfuse 库未安装，跳过初始化")
        return None
    
    try:
        _langfuse_client = Langfuse(
            public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
            host=host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("✅ Langfuse 初始化成功")
        return _langfuse_client
    except Exception as e:
        logger.error(f"Langfuse 初始化失败: {e}")
        HAS_LANGFUSE = False
        return None


def get_langfuse() -> Optional['Langfuse']:
    """获取 Langfuse 客户端（懒初始化）"""
    global _langfuse_client
    if _langfuse_client is None and HAS_LANGFUSE:
        _langfuse_client = init_langfuse()
    return _langfuse_client


# ============================================================
# 核心追踪 — 包装 ask_multi_agent
# ============================================================

class LangfuseTracer:
    """
    Langfuse 追踪器 — 嵌入到现有 pipeline 中
    
    用法:
        tracer = LangfuseTracer()
        with tracer.trace("用户问题") as t:
            t.span("routing", input=question, output=agents)
            t.generation("agent-analyst", model="claude-sonnet-4-20250514", 
                         input=prompt, output=response,
                         usage={"input": 500, "output": 200})
            t.score("correctness", 0.85, "数据引用准确")
    """
    
    def __init__(self):
        self.client = get_langfuse()
        self.enabled = self.client is not None
    
    @contextmanager
    def trace(self, question: str, metadata: Dict = None):
        """创建一个完整的 Trace（对应一次 ask_multi_agent 调用）"""
        if not self.enabled:
            yield _DummyTrace()
            return
        
        trace = self.client.trace(
            name="mrarfai-ask",
            input=question,
            metadata=metadata or {},
            tags=["v5.0", "phase1"],
        )
        
        ctx = _TraceContext(trace, self.client)
        try:
            yield ctx
        except Exception as e:
            trace.update(
                output=f"ERROR: {e}",
                metadata={**(metadata or {}), "error": str(e)},
            )
            raise
        finally:
            # 确保所有数据都发送
            try:
                self.client.flush()
            except Exception:
                pass
    
    def flush(self):
        """强制发送所有待发送数据"""
        if self.client:
            self.client.flush()


class _TraceContext:
    """单次 Trace 的上下文，提供 span/generation/score 方法"""
    
    def __init__(self, trace, client):
        self.trace = trace
        self.client = client
        self.trace_id = trace.id
        self._spans = {}
        self._start_time = time.time()
    
    def span(self, name: str, input: Any = None, output: Any = None,
             metadata: Dict = None, level: str = "DEFAULT") -> 'Span':
        """记录一个处理阶段（routing / data_query / hitl 等）"""
        s = self.trace.span(
            name=name,
            input=_safe_json(input),
            output=_safe_json(output),
            metadata=metadata or {},
            level=level,
        )
        self._spans[name] = s
        return s
    
    def generation(self, name: str, model: str,
                   input: Any = None, output: Any = None,
                   usage: Dict = None, metadata: Dict = None):
        """记录一次 LLM 调用（比 span 多了 model/usage 信息）"""
        gen = self.trace.generation(
            name=name,
            model=model,
            input=_safe_json(input),
            output=_safe_json(output),
            usage=usage or {},
            metadata=metadata or {},
        )
        return gen
    
    def score(self, name: str, value: float, comment: str = ""):
        """给这次 Trace 打分"""
        self.trace.score(
            name=name,
            value=value,
            comment=comment,
        )
    
    def update(self, output: str = None, metadata: Dict = None):
        """更新 Trace 的输出和元数据"""
        update_kwargs = {}
        if output is not None:
            update_kwargs["output"] = output
        if metadata is not None:
            update_kwargs["metadata"] = metadata
        if update_kwargs:
            self.trace.update(**update_kwargs)
    
    @property
    def elapsed_ms(self):
        return (time.time() - self._start_time) * 1000


class _DummyTrace:
    """Langfuse 不可用时的占位对象"""
    trace_id = "local-only"
    
    def span(self, *a, **kw): return self
    def generation(self, *a, **kw): return self
    def score(self, *a, **kw): pass
    def update(self, *a, **kw): pass
    def end(self, *a, **kw): pass
    
    @property
    def elapsed_ms(self): return 0


# ============================================================
# 集成到 ask_multi_agent 的辅助函数
# ============================================================

def trace_pipeline_stage(trace_ctx: _TraceContext, stage_name: str, 
                         duration_ms: float, input_data: Any = None,
                         output_data: Any = None, metadata: Dict = None):
    """记录 pipeline 的一个阶段到 Langfuse"""
    if isinstance(trace_ctx, _DummyTrace):
        return
    
    trace_ctx.span(
        name=stage_name,
        input=input_data,
        output=output_data,
        metadata={
            **(metadata or {}),
            "duration_ms": round(duration_ms, 1),
        },
    )


def trace_llm_call(trace_ctx: _TraceContext, name: str, provider: str,
                   model: str, system_prompt: str, user_prompt: str,
                   response: str, input_tokens: int = 0, 
                   output_tokens: int = 0, duration_ms: float = 0):
    """记录一次 LLM 调用到 Langfuse"""
    if isinstance(trace_ctx, _DummyTrace):
        return
    
    # 计算成本
    cost = _estimate_cost(provider, model, input_tokens, output_tokens)
    
    trace_ctx.generation(
        name=name,
        model=model,
        input=f"[System] {system_prompt[:500]}...\n[User] {user_prompt[:500]}...",
        output=response[:1000],
        usage={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "unit": "TOKENS",
            "totalCost": cost,
        },
        metadata={
            "provider": provider,
            "duration_ms": round(duration_ms, 1),
        },
    )


def trace_scores(trace_ctx: _TraceContext, scores: Dict[str, Any]):
    """批量记录评分到 Langfuse"""
    if isinstance(trace_ctx, _DummyTrace):
        return
    
    for name, data in scores.items():
        if name == "overall":
            continue
        if isinstance(data, dict):
            trace_ctx.score(
                name=f"judge-{name}",
                value=data.get("score", 0),
                comment=data.get("reasoning", ""),
            )
        elif isinstance(data, (int, float)):
            trace_ctx.score(name=name, value=float(data))


# ============================================================
# 成本计算
# ============================================================

COST_TABLE = {
    # provider: {model: (input_per_1k, output_per_1k)}
    "deepseek": {
        "deepseek-chat": (0.00014, 0.00028),
    },
    "claude": {
        "claude-sonnet-4-20250514": (0.003, 0.015),
        "claude-haiku-4-5-20251001": (0.0008, 0.004),
    },
}

def _estimate_cost(provider: str, model: str, 
                   input_tokens: int, output_tokens: int) -> float:
    """估算 LLM 调用成本"""
    costs = COST_TABLE.get(provider, {}).get(model)
    if not costs:
        return 0.0
    return (input_tokens / 1000 * costs[0]) + (output_tokens / 1000 * costs[1])


# ============================================================
# 工具函数
# ============================================================

def _safe_json(obj: Any) -> Any:
    """安全转换为 JSON 可序列化对象"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj[:20]]  # 限制长度
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in list(obj.items())[:50]}
    return str(obj)[:2000]


# ============================================================
# 快速检查
# ============================================================

def check_langfuse_status() -> Dict:
    """检查 Langfuse 连接状态"""
    status = {
        "installed": HAS_LANGFUSE,
        "configured": False,
        "connected": False,
    }
    
    if HAS_LANGFUSE:
        status["configured"] = bool(
            os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
        )
        if status["configured"]:
            try:
                client = get_langfuse()
                status["connected"] = client is not None
            except Exception:
                pass
    
    return status


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MRARFAI Langfuse Integration 状态检查")
    print("=" * 60)
    
    status = check_langfuse_status()
    
    if not status["installed"]:
        print("❌ Langfuse 未安装")
        print("   运行: pip install langfuse")
    elif not status["configured"]:
        print("⚠️  Langfuse 已安装但未配置")
        print("   设置环境变量:")
        print("   export LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("   export LANGFUSE_SECRET_KEY=sk-lf-...")
        print("   export LANGFUSE_HOST=https://cloud.langfuse.com")
    elif not status["connected"]:
        print("⚠️  Langfuse 配置正确但连接失败")
        print("   请检查 API Key 和网络")
    else:
        print("✅ Langfuse 连接正常")
        
        # 发送测试 trace
        tracer = LangfuseTracer()
        with tracer.trace("集成测试") as t:
            t.span("test_stage", input="hello", output="world")
            t.score("test_score", 1.0, "集成测试通过")
            t.update(output="集成测试完成")
        tracer.flush()
        print("✅ 测试 Trace 已发送，登录 Langfuse Dashboard 查看")
