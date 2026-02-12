#!/usr/bin/env python3
"""
MRARFAI Guardrails v4.0
=========================
生产级可靠性保障层：

① Retry + Exponential Backoff — 瞬态故障自动重试
② Circuit Breaker — 系统性故障熔断保护
③ Output Validation — 结构校验 + 幻觉检测
④ Fallback Chain — 多级降级（Opus→Sonnet→Haiku→缓存→模板）
⑤ Token Budget — 预算控制 + 分级限流
"""

import time
import json
import hashlib
import threading
import logging
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from enum import Enum
from collections import deque

logger = logging.getLogger("mrarfai.guardrails")


# ============================================================
# ① Retry with Exponential Backoff
# ============================================================

class RetryConfig:
    """重试配置"""
    def __init__(self, max_attempts: int = 4, base_delay: float = 1.0,
                 max_delay: float = 30.0, jitter: bool = True,
                 retryable_errors: tuple = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_errors = retryable_errors or (
            ConnectionError, TimeoutError, OSError,
        )

DEFAULT_RETRY = RetryConfig()


def with_retry(config: RetryConfig = None):
    """
    装饰器：带指数退避的自动重试

    Usage:
        @with_retry()
        def call_llm(...):
            ...
    """
    cfg = config or DEFAULT_RETRY

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(cfg.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()

                    # 判断是否可重试
                    is_retryable = isinstance(e, cfg.retryable_errors)
                    is_rate_limit = "rate" in err_str or "429" in err_str or "overloaded" in err_str
                    is_server_err = "500" in err_str or "502" in err_str or "503" in err_str

                    if not (is_retryable or is_rate_limit or is_server_err):
                        raise  # 不可重试的错误立即抛出

                    if attempt < cfg.max_attempts - 1:
                        delay = min(cfg.base_delay * (2 ** attempt), cfg.max_delay)
                        if cfg.jitter:
                            import random
                            delay = delay * (0.5 + random.random())
                        logger.warning(
                            f"[Retry] {func.__name__} attempt {attempt+1}/{cfg.max_attempts} "
                            f"failed: {e}. Retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)

            raise last_error
        return wrapper
    return decorator


# ============================================================
# ② Circuit Breaker
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 试探


class CircuitBreaker:
    """
    熔断器：连续失败达阈值后短路，避免雪崩

    - CLOSED: 正常通过，记录失败次数
    - OPEN: 直接拒绝，等待 reset_timeout
    - HALF_OPEN: 放行1次试探，成功→CLOSED，失败→OPEN
    """

    def __init__(self, name: str = "default", fail_max: int = 5,
                 reset_timeout: float = 60.0):
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._last_fail_time = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_fail_time > self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器执行函数"""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"熔断器 [{self.name}] 已打开 — "
                f"连续{self._fail_count}次失败，"
                f"等待{self.reset_timeout}s后重试"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self._fail_count = 0
            self._state = CircuitState.CLOSED

    def _on_failure(self):
        with self._lock:
            self._fail_count += 1
            self._last_fail_time = time.time()
            if self._fail_count >= self.fail_max:
                self._state = CircuitState.OPEN
                logger.error(
                    f"[CircuitBreaker] {self.name} OPENED after "
                    f"{self._fail_count} consecutive failures"
                )

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "fail_count": self._fail_count,
            "fail_max": self.fail_max,
        }


class CircuitBreakerOpenError(Exception):
    """熔断器打开时抛出"""
    pass


# 全局熔断器实例
_breakers: Dict[str, CircuitBreaker] = {}


def get_breaker(name: str = "llm_api", fail_max: int = 5,
                reset_timeout: float = 60.0) -> CircuitBreaker:
    """获取或创建熔断器"""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, fail_max, reset_timeout)
    return _breakers[name]


# ============================================================
# ③ Output Validation
# ============================================================

class ValidationResult:
    def __init__(self, passed: bool, issues: List[str] = None,
                 confidence: float = 1.0):
        self.passed = passed
        self.issues = issues or []
        self.confidence = confidence

    def __bool__(self):
        return self.passed


def validate_agent_output(output: str, source_data: str = "",
                          min_length: int = 20,
                          max_length: int = 5000) -> ValidationResult:
    """
    验证 Agent 输出质量

    检查：
    1. 长度合理性
    2. 不是错误消息
    3. 包含中文内容（非纯英文/乱码）
    4. 数字引用合理性（可选：对比源数据）
    """
    issues = []
    confidence = 1.0

    # 1. 长度检查
    if len(output) < min_length:
        issues.append(f"输出过短 ({len(output)}字 < {min_length})")
        confidence -= 0.3
    if len(output) > max_length:
        issues.append(f"输出过长 ({len(output)}字 > {max_length})")
        confidence -= 0.1

    # 2. 错误消息检测
    error_patterns = ["[调用失败", "[需要API", "[分析超时", "Error:", "Exception:"]
    if any(p in output for p in error_patterns):
        issues.append("包含错误消息")
        confidence -= 0.5

    # 3. 中文内容检查
    chinese_chars = sum(1 for c in output if '\u4e00' <= c <= '\u9fff')
    if chinese_chars < 5:
        issues.append("中文内容过少，可能输出异常")
        confidence -= 0.2

    # 4. 基础幻觉检测：检查输出中的大数字是否在源数据中出现
    if source_data:
        import re
        # 提取输出中的数字
        output_numbers = set(re.findall(r'\d+\.?\d+', output))
        source_numbers = set(re.findall(r'\d+\.?\d+', source_data))

        # 检查大数字（>100）是否有源可查
        suspicious = []
        for num_str in output_numbers:
            try:
                num = float(num_str)
                if num > 100 and num_str not in source_data:
                    # 允许简单运算结果（如占比、增长率）
                    if num < 10000:
                        continue
                    suspicious.append(num_str)
            except ValueError:
                pass

        if len(suspicious) >= 3:
            issues.append(f"发现{len(suspicious)}个可疑数字无法溯源")
            confidence -= 0.15

    confidence = max(0.0, min(1.0, confidence))
    return ValidationResult(
        passed=confidence >= 0.5,
        issues=issues,
        confidence=round(confidence, 2),
    )


def safe_parse_llm_json(raw: str) -> Optional[dict]:
    """
    安全解析 LLM 返回的 JSON（处理各种格式问题）

    降级链：直接解析 → 去markdown → 正则提取 → None
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()

    # 1. 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. 去 markdown 代码块
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # 3. 正则提取 JSON 对象
    import re
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# ④ Fallback Chain
# ============================================================

class FallbackChain:
    """
    多级降级链 — 主策略失败时自动切换到备选

    Usage:
        chain = FallbackChain("llm_call")
        chain.add(lambda: call_opus(...), "opus")
        chain.add(lambda: call_sonnet(...), "sonnet")
        chain.add(lambda: call_haiku(...), "haiku")
        chain.add(lambda: cached_response(...), "cache")

        result, used_level = chain.execute()
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.handlers: List[tuple] = []  # [(func, label)]

    def add(self, handler: Callable, label: str):
        self.handlers.append((handler, label))
        return self  # chainable

    def execute(self, *args, **kwargs) -> tuple:
        """执行降级链，返回 (result, used_level_label)"""
        errors = []
        for handler, label in self.handlers:
            try:
                result = handler(*args, **kwargs)
                if result and not str(result).startswith("[调用失败"):
                    if len(errors) > 0:
                        logger.warning(
                            f"[Fallback] {self.name}: 降级到 [{label}] "
                            f"(前{len(errors)}级失败)"
                        )
                    return result, label
            except CircuitBreakerOpenError:
                errors.append(f"{label}: 熔断器已打开")
                continue
            except Exception as e:
                errors.append(f"{label}: {e}")
                continue

        # 全部失败
        error_summary = "; ".join(errors[-3:])
        return f"[所有降级策略均失败: {error_summary}]", "exhausted"


# ============================================================
# ⑤ Token Budget Manager
# ============================================================

class TokenBudget:
    """
    Token 预算管理器 — 防止成本失控

    分级策略：
    - <50%: 正常
    - 50-80%: 告警，启用缓存
    - 80-90%: 限流，降级模型
    - >90%: 仅缓存响应
    """

    def __init__(self, daily_budget_usd: float = 10.0,
                 per_query_limit_usd: float = 0.50):
        self.daily_budget = daily_budget_usd
        self.per_query_limit = per_query_limit_usd
        self._spent_today = 0.0
        self._date = time.strftime("%Y-%m-%d")
        self._lock = threading.Lock()
        self._query_log = deque(maxlen=1000)

    def _reset_if_new_day(self):
        today = time.strftime("%Y-%m-%d")
        if today != self._date:
            self._spent_today = 0.0
            self._date = today

    def check_budget(self) -> dict:
        """检查预算状态"""
        with self._lock:
            self._reset_if_new_day()
            usage_pct = (self._spent_today / self.daily_budget * 100) if self.daily_budget > 0 else 0

            if usage_pct >= 90:
                level = "critical"
                action = "cache_only"
            elif usage_pct >= 80:
                level = "warning"
                action = "downgrade_model"
            elif usage_pct >= 50:
                level = "caution"
                action = "enable_cache"
            else:
                level = "normal"
                action = "full"

            return {
                "level": level,
                "action": action,
                "spent_usd": round(self._spent_today, 4),
                "budget_usd": self.daily_budget,
                "usage_pct": round(usage_pct, 1),
                "remaining_usd": round(self.daily_budget - self._spent_today, 4),
            }

    def record_cost(self, cost_usd: float, query: str = ""):
        """记录消耗"""
        with self._lock:
            self._reset_if_new_day()
            self._spent_today += cost_usd
            self._query_log.append({
                "time": time.time(),
                "cost": cost_usd,
                "query": query[:50],
            })

    def get_recommended_model(self, provider: str = "claude") -> str:
        """根据预算推荐模型"""
        budget = self.check_budget()
        if provider == "claude":
            if budget["action"] == "cache_only":
                return "claude-haiku-4-5-20251001"
            elif budget["action"] == "downgrade_model":
                return "claude-haiku-4-5-20251001"
            else:
                return "claude-sonnet-4-20250514"
        return "deepseek-chat"

    def should_allow_query(self) -> bool:
        """是否允许新查询"""
        budget = self.check_budget()
        return budget["action"] != "cache_only"


# 全局实例
_budget = None


def get_budget(daily_usd: float = 10.0) -> TokenBudget:
    global _budget
    if _budget is None:
        _budget = TokenBudget(daily_budget_usd=daily_usd)
    return _budget


# ============================================================
# ⑥ Response Cache（简易缓存层）
# ============================================================

class ResponseCache:
    """LRU 响应缓存 — 相同问题直接返回缓存结果"""

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[str, dict] = {}
        self._access_order = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def _make_key(self, question: str, data_hash: str = "") -> str:
        raw = f"{question.strip().lower()}:{data_hash}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def get(self, question: str, data_hash: str = "") -> Optional[dict]:
        key = self._make_key(question, data_hash)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["time"] < self.ttl:
                    entry["hits"] += 1
                    return entry["result"]
                else:
                    del self._cache[key]
        return None

    def put(self, question: str, result: dict, data_hash: str = ""):
        key = self._make_key(question, data_hash)
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest = self._access_order.popleft()
                self._cache.pop(oldest, None)
            self._cache[key] = {
                "result": result,
                "time": time.time(),
                "hits": 0,
            }
            self._access_order.append(key)

    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": sum(e["hits"] for e in self._cache.values()),
        }


# 全局缓存
_cache = None

def get_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache


# ============================================================
# 集成：guarded_llm_call — 所有保护层合一
# ============================================================

def guarded_llm_call(
    call_func: Callable,
    *args,
    breaker_name: str = "llm_api",
    retry_config: RetryConfig = None,
    validate: bool = True,
    source_data: str = "",
    **kwargs,
) -> str:
    """
    带完整保护的 LLM 调用

    叠加顺序：Retry → CircuitBreaker → Validate

    Usage:
        result = guarded_llm_call(
            _call_llm_raw,
            system_prompt, user_prompt, provider, api_key,
            breaker_name="agent_analyst",
        )
    """
    breaker = get_breaker(breaker_name)
    cfg = retry_config or DEFAULT_RETRY

    last_error = None
    for attempt in range(cfg.max_attempts):
        try:
            # 熔断器保护
            result = breaker.call(call_func, *args, **kwargs)

            # 输出校验
            if validate and result:
                validation = validate_agent_output(result, source_data)
                if not validation.passed:
                    logger.warning(
                        f"[Validation] Output failed: {validation.issues}"
                    )
                    # 校验失败但有结果，附加警告而非重试
                    if validation.confidence >= 0.3:
                        return result

            return result

        except CircuitBreakerOpenError:
            raise  # 熔断器打开，不重试
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            is_retryable = ("rate" in err_str or "429" in err_str or
                           "500" in err_str or "overloaded" in err_str or
                           "connection" in err_str)

            if not is_retryable:
                raise

            if attempt < cfg.max_attempts - 1:
                import random
                delay = min(cfg.base_delay * (2 ** attempt), cfg.max_delay)
                delay *= (0.5 + random.random()) if cfg.jitter else 1
                logger.warning(f"[GuardedLLM] Retry {attempt+1}: {e}, wait {delay:.1f}s")
                time.sleep(delay)

    raise last_error
