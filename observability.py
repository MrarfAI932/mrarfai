#!/usr/bin/env python3
"""
MRARFAI Observability Engine v1.0
===================================
生产级可观测性系统，参考 OpenTelemetry / LangSmith / Braintrust 设计。

核心能力：
  1. Trace Lifecycle — 完整请求链路追踪（request → route → agents → report）
  2. Span Model     — 每个阶段独立计时，支持嵌套（parent/child span）
  3. LLM Tracking   — 每次LLM调用的token/cost/latency/provider追踪
  4. SQLite Store    — 持久化存储，支持历史查询和趋势分析
  5. Metrics Engine  — 实时聚合指标（P50/P95延迟、成本、吞吐量、错误率）
  6. Quality Signals — 用户反馈、路由准确性、数据完整性评分
  7. Export API      — JSON/CSV导出，兼容外部分析工具

设计原则：
  - 零侵入：通过上下文管理器接入，不改变业务逻辑
  - 低开销：SQLite WAL模式，<5ms额外延迟
  - 可扩展：预留 webhook/metrics push 接口，未来可接 Prometheus/Grafana
  - 兼容性：数据模型对齐 OpenTelemetry Span，未来可直接导出OTLP
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple


# ============================================================
# 1. 数据模型（对齐 OpenTelemetry Span 规范）
# ============================================================

class SpanKind(str, Enum):
    TRACE = "trace"
    ROUTING = "routing"
    DATA_QUERY = "data_query"
    LLM_CALL = "llm_call"
    AGENT = "agent"
    REPORTER = "reporter"
    KG_LOOKUP = "kg_lookup"
    VALIDATION = "validation"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class LLMUsage:
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    temperature: float = 0.0
    max_tokens: int = 0


@dataclass
class Span:
    trace_id: str = ""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str = ""
    kind: str = SpanKind.TRACE.value
    name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = SpanStatus.OK.value
    error_message: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    llm_usage: Optional[LLMUsage] = None

    def finish(self, status: str = SpanStatus.OK.value, error: str = ""):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error_message = error


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    question: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_llm_calls: int = 0
    agents_used: List[str] = field(default_factory=list)
    pattern_matched: str = ""
    route_source: str = ""
    kg_corrections: List[str] = field(default_factory=list)
    status: str = SpanStatus.OK.value
    error_message: str = ""
    user_feedback: Optional[int] = None
    feedback_text: str = ""
    spans: List[Span] = field(default_factory=list)

    def add_span(self, span: Span):
        self.spans.append(span)
        if span.llm_usage:
            self.total_tokens += span.llm_usage.total_tokens
            self.total_cost_usd += span.llm_usage.cost_usd
            self.total_llm_calls += 1

    def finish(self):
        root = [s for s in self.spans if s.kind == SpanKind.TRACE.value]
        if root:
            self.total_duration_ms = root[0].duration_ms
        elif self.spans:
            self.total_duration_ms = sum(s.duration_ms for s in self.spans)


# ============================================================
# 2. 成本计算器
# ============================================================

class CostCalculator:
    # $/1M tokens [input, output]
    PRICING = {
        "deepseek": {
            "deepseek-chat": {"input": 0.14, "output": 0.28},
            "deepseek-reasoner": {"input": 0.55, "output": 2.19},
        },
        "claude": {
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
            "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        },
        "openai": {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        },
    }

    @classmethod
    def calculate(cls, provider: str, model: str,
                  prompt_tokens: int, completion_tokens: int) -> float:
        provider_prices = cls.PRICING.get(provider, {})
        model_prices = provider_prices.get(model, {"input": 1.0, "output": 3.0})
        cost = (prompt_tokens * model_prices["input"] +
                completion_tokens * model_prices["output"]) / 1_000_000
        return round(cost, 8)

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        if not text:
            return 0
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text)
        if total_chars == 0:
            return 0
        chinese_ratio = chinese_chars / total_chars
        chars_per_token = 1.5 * chinese_ratio + 4.0 * (1 - chinese_ratio)
        return max(1, int(total_chars / chars_per_token))

    @classmethod
    def update_pricing(cls, provider: str, model: str,
                       input_price: float, output_price: float):
        if provider not in cls.PRICING:
            cls.PRICING[provider] = {}
        cls.PRICING[provider][model] = {"input": input_price, "output": output_price}


# ============================================================
# 3. SQLite 持久化存储
# ============================================================

class TraceStore:
    """
    SQLite持久化 — WAL模式，支持并发读写
    默认保留90天数据，可配置
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "observability.db"
            )
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id        TEXT PRIMARY KEY,
                question        TEXT,
                timestamp       TEXT,
                total_duration_ms REAL DEFAULT 0,
                total_tokens    INTEGER DEFAULT 0,
                total_cost_usd  REAL DEFAULT 0,
                total_llm_calls INTEGER DEFAULT 0,
                agents_used     TEXT DEFAULT '[]',
                pattern_matched TEXT DEFAULT '',
                route_source    TEXT DEFAULT '',
                kg_corrections  TEXT DEFAULT '[]',
                status          TEXT DEFAULT 'ok',
                error_message   TEXT DEFAULT '',
                user_feedback   INTEGER,
                feedback_text   TEXT DEFAULT '',
                created_at      TEXT DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS spans (
                span_id         TEXT PRIMARY KEY,
                trace_id        TEXT NOT NULL,
                parent_span_id  TEXT DEFAULT '',
                kind            TEXT NOT NULL,
                name            TEXT DEFAULT '',
                start_time      REAL,
                end_time        REAL,
                duration_ms     REAL DEFAULT 0,
                status          TEXT DEFAULT 'ok',
                error_message   TEXT DEFAULT '',
                attributes      TEXT DEFAULT '{}',
                llm_provider    TEXT DEFAULT '',
                llm_model       TEXT DEFAULT '',
                prompt_tokens   INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens    INTEGER DEFAULT 0,
                cost_usd        REAL DEFAULT 0,
                temperature     REAL DEFAULT 0,
                max_tokens_limit INTEGER DEFAULT 0,
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id        TEXT NOT NULL,
                rating          INTEGER CHECK(rating BETWEEN 1 AND 5),
                text            TEXT DEFAULT '',
                created_at      TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            );

            CREATE TABLE IF NOT EXISTS daily_metrics (
                date            TEXT NOT NULL,
                metric_name     TEXT NOT NULL,
                metric_value    REAL DEFAULT 0,
                updated_at      TEXT DEFAULT (datetime('now', 'localtime')),
                PRIMARY KEY (date, metric_name)
            );

            CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
            CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
            CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(kind);
            CREATE INDEX IF NOT EXISTS idx_spans_trace_kind ON spans(trace_id, kind);
            CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id);
            CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_metrics(date);
        """)
        conn.commit()

    def save_trace(self, trace: Trace):
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO traces
                (trace_id, question, timestamp, total_duration_ms, total_tokens,
                 total_cost_usd, total_llm_calls, agents_used, pattern_matched,
                 route_source, kg_corrections, status, error_message,
                 user_feedback, feedback_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.trace_id, trace.question, trace.timestamp,
                trace.total_duration_ms, trace.total_tokens,
                trace.total_cost_usd, trace.total_llm_calls,
                json.dumps(trace.agents_used, ensure_ascii=False),
                trace.pattern_matched, trace.route_source,
                json.dumps(trace.kg_corrections, ensure_ascii=False),
                trace.status, trace.error_message,
                trace.user_feedback, trace.feedback_text,
            ))

            for span in trace.spans:
                llm = span.llm_usage or LLMUsage()
                conn.execute("""
                    INSERT OR REPLACE INTO spans
                    (span_id, trace_id, parent_span_id, kind, name,
                     start_time, end_time, duration_ms, status, error_message,
                     attributes, llm_provider, llm_model, prompt_tokens,
                     completion_tokens, total_tokens, cost_usd,
                     temperature, max_tokens_limit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    span.span_id, span.trace_id, span.parent_span_id,
                    span.kind, span.name,
                    span.start_time, span.end_time, span.duration_ms,
                    span.status, span.error_message,
                    json.dumps(span.attributes, ensure_ascii=False),
                    llm.provider, llm.model,
                    llm.prompt_tokens, llm.completion_tokens, llm.total_tokens,
                    llm.cost_usd, llm.temperature, llm.max_tokens,
                ))

            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[Observability] Save trace error: {e}")

    def save_feedback(self, trace_id: str, rating: int, text: str = ""):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO feedback (trace_id, rating, text) VALUES (?, ?, ?)",
            (trace_id, rating, text)
        )
        conn.execute(
            "UPDATE traces SET user_feedback=?, feedback_text=? WHERE trace_id=?",
            (rating, text, trace_id)
        )
        conn.commit()

    def get_trace(self, trace_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM traces WHERE trace_id=?", (trace_id,)
        ).fetchone()
        if not row:
            return None
        trace = dict(row)
        trace['agents_used'] = json.loads(trace.get('agents_used', '[]'))
        trace['kg_corrections'] = json.loads(trace.get('kg_corrections', '[]'))
        spans = conn.execute(
            "SELECT * FROM spans WHERE trace_id=? ORDER BY start_time",
            (trace_id,)
        ).fetchall()
        trace['spans'] = [dict(s) for s in spans]
        return trace

    def get_recent_traces(self, limit: int = 50, offset: int = 0) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM traces ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d['agents_used'] = json.loads(d.get('agents_used', '[]'))
            results.append(d)
        return results

    def get_traces_by_date_range(self, start_date: str, end_date: str) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM traces WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC",
            (start_date, end_date)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_span_breakdown(self, trace_id: str) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT kind, name, duration_ms, total_tokens, cost_usd, status "
            "FROM spans WHERE trace_id=? ORDER BY start_time",
            (trace_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def cleanup(self, days: int = 90):
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        conn.execute("DELETE FROM spans WHERE trace_id IN "
                     "(SELECT trace_id FROM traces WHERE timestamp < ?)", (cutoff,))
        conn.execute("DELETE FROM feedback WHERE trace_id IN "
                     "(SELECT trace_id FROM traces WHERE timestamp < ?)", (cutoff,))
        conn.execute("DELETE FROM traces WHERE timestamp < ?", (cutoff,))
        conn.commit()

    def get_db_stats(self) -> dict:
        conn = self._get_conn()
        traces = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        spans = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {
            "total_traces": traces,
            "total_spans": spans,
            "total_feedback": feedback,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
        }


# ============================================================
# 4. Trace 上下文（线程安全）
# ============================================================

class _TraceContext(threading.local):
    def __init__(self):
        self.current_trace: Optional[Trace] = None
        self.current_span_stack: List[Span] = []

_ctx = _TraceContext()


# ============================================================
# 5. AgentTracer — 核心 Instrumentation API
# ============================================================

class AgentTracer:
    """
    使用方式：
        tracer = get_tracer()
        with tracer.trace("Nokia风险分析") as t:
            with tracer.span("data_query", "KG查询") as s:
                ...
            with tracer.llm_call("routing", provider="deepseek") as lc:
                result = call_llm(...)
                lc.set_response(result, prompt_tokens=50, completion_tokens=30)
    """

    def __init__(self, store: TraceStore = None, enabled: bool = True):
        self.store = store or _get_default_store()
        self.enabled = enabled

    @contextmanager
    def trace(self, question: str = ""):
        if not self.enabled:
            yield _NoopTrace()
            return

        t = Trace(question=question)
        _ctx.current_trace = t
        _ctx.current_span_stack = []

        root_span = Span(
            trace_id=t.trace_id,
            kind=SpanKind.TRACE.value,
            name="ask_multi_agent",
            start_time=time.time(),
        )
        _ctx.current_span_stack.append(root_span)

        try:
            yield t
        except Exception as e:
            t.status = SpanStatus.ERROR.value
            t.error_message = str(e)
            root_span.finish(SpanStatus.ERROR.value, str(e))
            raise
        finally:
            root_span.finish()
            t.add_span(root_span)
            t.total_duration_ms = root_span.duration_ms
            t.finish()
            self._save(t)
            _ctx.current_trace = None
            _ctx.current_span_stack = []

    @contextmanager
    def span(self, kind: str, name: str = "", **attributes):
        if not self.enabled or not _ctx.current_trace:
            yield _NoopSpan()
            return

        parent_id = ""
        if _ctx.current_span_stack:
            parent_id = _ctx.current_span_stack[-1].span_id

        s = Span(
            trace_id=_ctx.current_trace.trace_id,
            parent_span_id=parent_id,
            kind=kind,
            name=name,
            start_time=time.time(),
            attributes=attributes,
        )
        _ctx.current_span_stack.append(s)

        try:
            yield s
        except Exception as e:
            s.finish(SpanStatus.ERROR.value, str(e))
            raise
        finally:
            s.finish()
            _ctx.current_span_stack.pop()
            _ctx.current_trace.add_span(s)

    @contextmanager
    def llm_call(self, name: str = "llm_call",
                 provider: str = "", model: str = "",
                 temperature: float = 0.3, max_tokens: int = 800):
        if not self.enabled or not _ctx.current_trace:
            yield _NoopLLMCall()
            return

        parent_id = ""
        if _ctx.current_span_stack:
            parent_id = _ctx.current_span_stack[-1].span_id

        s = Span(
            trace_id=_ctx.current_trace.trace_id,
            parent_span_id=parent_id,
            kind=SpanKind.LLM_CALL.value,
            name=name,
            start_time=time.time(),
            llm_usage=LLMUsage(
                provider=provider, model=model,
                temperature=temperature, max_tokens=max_tokens,
            ),
        )

        lc = _LLMCallContext(s, provider, model)
        _ctx.current_span_stack.append(s)

        try:
            yield lc
        except Exception as e:
            s.finish(SpanStatus.ERROR.value, str(e))
            raise
        finally:
            s.finish()
            _ctx.current_span_stack.pop()
            _ctx.current_trace.add_span(s)

    def _save(self, trace: Trace):
        try:
            self.store.save_trace(trace)
        except Exception as e:
            print(f"[Observability] Save error: {e}")

    def save_feedback(self, trace_id: str, rating: int, text: str = ""):
        self.store.save_feedback(trace_id, rating, text)


class _LLMCallContext:
    def __init__(self, span: Span, provider: str, model: str):
        self.span = span
        self.provider = provider
        self.model = model

    def set_response(self, response_text: str = "",
                     prompt_tokens: int = 0, completion_tokens: int = 0,
                     prompt_text: str = ""):
        llm = self.span.llm_usage
        if prompt_tokens == 0 and prompt_text:
            prompt_tokens = CostCalculator.estimate_tokens(prompt_text)
        if completion_tokens == 0 and response_text:
            completion_tokens = CostCalculator.estimate_tokens(response_text)

        llm.prompt_tokens = prompt_tokens
        llm.completion_tokens = completion_tokens
        llm.total_tokens = prompt_tokens + completion_tokens
        llm.cost_usd = CostCalculator.calculate(
            self.provider, self.model, prompt_tokens, completion_tokens
        )


class _NoopTrace:
    trace_id = "disabled"
    question = ""
    agents_used = []
    pattern_matched = ""
    route_source = ""
    kg_corrections = []
    status = "ok"
    def __getattr__(self, name):
        return lambda *a, **kw: None

class _NoopSpan:
    attributes = {}
    def __getattr__(self, name):
        return lambda *a, **kw: None

class _NoopLLMCall:
    def set_response(self, *a, **kw): pass


# ============================================================
# 6. 指标聚合引擎
# ============================================================

class MetricsAggregator:

    def __init__(self, store: TraceStore = None):
        self.store = store or _get_default_store()

    def get_overview(self, days: int = 7) -> dict:
        conn = self.store._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        row = conn.execute("""
            SELECT
                COUNT(*) as total_queries,
                COALESCE(AVG(total_duration_ms), 0) as avg_latency_ms,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(total_cost_usd), 0) as total_cost_usd,
                COALESCE(AVG(total_tokens), 0) as avg_tokens_per_query,
                COALESCE(AVG(total_cost_usd), 0) as avg_cost_per_query,
                COALESCE(SUM(total_llm_calls), 0) as total_llm_calls,
                COALESCE(AVG(total_llm_calls), 0) as avg_llm_calls_per_query,
                SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as error_count,
                COALESCE(AVG(user_feedback), 0) as avg_rating
            FROM traces WHERE timestamp >= ?
        """, (cutoff,)).fetchone()

        return {
            "period_days": days,
            "total_queries": row["total_queries"] or 0,
            "avg_latency_ms": round(row["avg_latency_ms"] or 0, 1),
            "total_tokens": row["total_tokens"] or 0,
            "total_cost_usd": round(row["total_cost_usd"] or 0, 6),
            "avg_tokens_per_query": round(row["avg_tokens_per_query"] or 0, 0),
            "avg_cost_per_query": round(row["avg_cost_per_query"] or 0, 6),
            "total_llm_calls": row["total_llm_calls"] or 0,
            "avg_llm_calls": round(row["avg_llm_calls_per_query"] or 0, 1),
            "error_count": row["error_count"] or 0,
            "error_rate": round((row["error_count"] or 0) / max(1, row["total_queries"] or 1) * 100, 1),
            "avg_rating": round(row["avg_rating"] or 0, 1),
        }

    def get_latency_percentiles(self, days: int = 7) -> dict:
        conn = self.store._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = conn.execute(
            "SELECT total_duration_ms FROM traces WHERE timestamp >= ? "
            "ORDER BY total_duration_ms",
            (cutoff,)
        ).fetchall()

        if not rows:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "min": 0, "count": 0}

        values = [r["total_duration_ms"] for r in rows]
        n = len(values)
        return {
            "p50": round(values[int(n * 0.50)], 1),
            "p90": round(values[int(n * 0.90)] if n > 1 else values[0], 1),
            "p95": round(values[int(n * 0.95)] if n > 1 else values[0], 1),
            "p99": round(values[min(int(n * 0.99), n - 1)], 1),
            "max": round(values[-1], 1),
            "min": round(values[0], 1),
            "count": n,
        }

    def get_latency_by_stage(self, days: int = 7) -> List[dict]:
        conn = self.store._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = conn.execute("""
            SELECT s.kind,
                   COUNT(*) as count,
                   AVG(s.duration_ms) as avg_ms,
                   MAX(s.duration_ms) as max_ms,
                   MIN(s.duration_ms) as min_ms
            FROM spans s
            JOIN traces t ON s.trace_id = t.trace_id
            WHERE t.timestamp >= ?
            GROUP BY s.kind
            ORDER BY avg_ms DESC
        """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def get_cost_breakdown(self, days: int = 7) -> dict:
        conn = self.store._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        by_provider = conn.execute("""
            SELECT s.llm_provider as provider,
                   SUM(s.total_tokens) as tokens,
                   SUM(s.cost_usd) as cost,
                   COUNT(*) as calls
            FROM spans s
            JOIN traces t ON s.trace_id = t.trace_id
            WHERE t.timestamp >= ? AND s.kind = 'llm_call'
            GROUP BY s.llm_provider
        """, (cutoff,)).fetchall()

        by_stage = conn.execute("""
            SELECT s.name as stage,
                   SUM(s.total_tokens) as tokens,
                   SUM(s.cost_usd) as cost,
                   COUNT(*) as calls
            FROM spans s
            JOIN traces t ON s.trace_id = t.trace_id
            WHERE t.timestamp >= ? AND s.kind = 'llm_call'
            GROUP BY s.name
            ORDER BY cost DESC
        """, (cutoff,)).fetchall()

        return {
            "by_provider": [dict(r) for r in by_provider],
            "by_stage": [dict(r) for r in by_stage],
        }

    def get_daily_trend(self, days: int = 30) -> List[dict]:
        conn = self.store._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = conn.execute("""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as queries,
                AVG(total_duration_ms) as avg_latency,
                SUM(total_tokens) as tokens,
                SUM(total_cost_usd) as cost,
                SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors,
                AVG(user_feedback) as avg_rating
            FROM traces
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def get_routing_stats(self, days: int = 7) -> dict:
        conn = self.store._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        by_source = conn.execute("""
            SELECT route_source, COUNT(*) as count
            FROM traces WHERE timestamp >= ?
            GROUP BY route_source
        """, (cutoff,)).fetchall()

        by_pattern = conn.execute("""
            SELECT pattern_matched, COUNT(*) as count
            FROM traces WHERE timestamp >= ? AND pattern_matched != ''
            GROUP BY pattern_matched ORDER BY count DESC
        """, (cutoff,)).fetchall()

        agent_usage = {}
        rows = conn.execute(
            "SELECT agents_used FROM traces WHERE timestamp >= ?",
            (cutoff,)
        ).fetchall()
        for row in rows:
            agents = json.loads(row["agents_used"])
            for a in agents:
                agent_usage[a] = agent_usage.get(a, 0) + 1

        return {
            "by_source": [dict(r) for r in by_source],
            "by_pattern": [dict(r) for r in by_pattern],
            "agent_usage": agent_usage,
        }

    def get_quality_metrics(self, days: int = 7) -> dict:
        conn = self.store._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        feedback_dist = conn.execute("""
            SELECT user_feedback as rating, COUNT(*) as count
            FROM traces
            WHERE timestamp >= ? AND user_feedback IS NOT NULL
            GROUP BY user_feedback ORDER BY rating
        """, (cutoff,)).fetchall()

        kg_corrections_count = conn.execute("""
            SELECT COUNT(*) FROM traces
            WHERE timestamp >= ? AND kg_corrections != '[]'
        """, (cutoff,)).fetchone()[0]

        total = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE timestamp >= ?", (cutoff,)
        ).fetchone()[0]

        return {
            "feedback_distribution": [dict(r) for r in feedback_dist],
            "kg_correction_rate": round(
                kg_corrections_count / max(1, total) * 100, 1
            ),
            "total_with_feedback": sum(r["count"] for r in feedback_dist),
            "total_queries": total,
        }


# ============================================================
# 7. 全局实例管理
# ============================================================

_default_store: Optional[TraceStore] = None
_default_tracer: Optional[AgentTracer] = None

def _get_default_store() -> TraceStore:
    global _default_store
    if _default_store is None:
        _default_store = TraceStore()
    return _default_store

def get_tracer(enabled: bool = True) -> AgentTracer:
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = AgentTracer(enabled=enabled)
    _default_tracer.enabled = enabled
    return _default_tracer

def get_metrics() -> MetricsAggregator:
    return MetricsAggregator()

def get_store() -> TraceStore:
    return _get_default_store()

def set_store_path(path: str):
    global _default_store, _default_tracer
    _default_store = TraceStore(db_path=path)
    _default_tracer = AgentTracer(store=_default_store)


# ============================================================
# 8. 导出工具
# ============================================================

def export_traces_json(days: int = 30, filepath: str = None) -> str:
    store = get_store()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    traces = store.get_traces_by_date_range(cutoff, datetime.now().isoformat())
    for t in traces:
        t['spans'] = store.get_span_breakdown(t['trace_id'])

    if filepath is None:
        filepath = f"traces_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(traces, f, ensure_ascii=False, indent=2, default=str)
    return filepath


def export_traces_csv(days: int = 30, filepath: str = None) -> str:
    import csv
    store = get_store()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    traces = store.get_traces_by_date_range(cutoff, datetime.now().isoformat())

    if filepath is None:
        filepath = f"traces_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    fields = [
        'trace_id', 'question', 'timestamp', 'total_duration_ms',
        'total_tokens', 'total_cost_usd', 'total_llm_calls',
        'agents_used', 'pattern_matched', 'route_source', 'status',
        'user_feedback',
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for t in traces:
            writer.writerow(t)
    return filepath
