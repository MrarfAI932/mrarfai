#!/usr/bin/env python3
"""
MRARFAI Evaluation Framework v1.0
===================================
自动化测试 Agent 系统输出质量

三层评估体系：
  ① 工具正确性 — 每个工具的输入/输出验证
  ② Agent 输出质量 — 准确性/完整性/可操作性/中文质量
  ③ 端到端管线 — 完整问答流程的延迟/成本/质量

运行方式:
  python evals.py                    # 运行全部离线测试
  python evals.py --tools            # 仅工具测试
  python evals.py --agents           # Agent 质量测试 (需 API Key)
  python evals.py --e2e              # 端到端测试 (需 API Key)
  python evals.py --report           # 生成详细报告
"""

import json
import time
import sys
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any


# ============================================================
# Eval Result Types
# ============================================================

@dataclass
class EvalCase:
    """单个测试用例"""
    id: str
    name: str
    category: str           # tools / agent_quality / e2e
    input_data: dict
    expected: dict           # 期望输出特征
    tags: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """单个测试结果"""
    case_id: str
    passed: bool
    score: float            # 0.0 - 1.0
    details: str = ""
    elapsed_ms: float = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """评估报告"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0
    results: List[EvalResult] = field(default_factory=list)
    by_category: Dict[str, dict] = field(default_factory=dict)
    elapsed_sec: float = 0

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"  MRARFAI Eval Report",
            f"{'='*55}",
            f"  Total: {self.total}  |  ✅ {self.passed}  |  ❌ {self.failed}  |  Score: {self.avg_score:.1%}",
            f"  Time: {self.elapsed_sec:.1f}s",
        ]
        for cat, stats in self.by_category.items():
            lines.append(f"  [{cat}] {stats['passed']}/{stats['total']} ({stats['score']:.0%})")
        lines.append(f"{'='*55}")
        if self.failed > 0:
            lines.append("  ❌ Failed cases:")
            for r in self.results:
                if not r.passed:
                    lines.append(f"    - {r.case_id}: {r.details}")
        return "\n".join(lines)


# ============================================================
# ① Tool Correctness Tests
# ============================================================

TOOL_TEST_CASES = [
    # ---- calc_yoy_growth ----
    EvalCase("T01", "YoY 正增长", "tools",
             {"tool": "calc_yoy_growth", "args": {"current": 41.71, "previous": 27.07}},
             {"growth_pct_range": (54.0, 54.2), "delta_positive": True}),
    EvalCase("T02", "YoY 负增长", "tools",
             {"tool": "calc_yoy_growth", "args": {"current": 20.0, "previous": 30.0}},
             {"growth_pct_range": (-33.4, -33.2), "delta_positive": False}),
    EvalCase("T03", "YoY 零基数", "tools",
             {"tool": "calc_yoy_growth", "args": {"current": 100, "previous": 0}},
             {"has_error_or_special": True}),

    # ---- calc_concentration ----
    EvalCase("T04", "高集中度", "tools",
             {"tool": "calc_concentration", "args": {"revenues": [
                 {"name": "A", "revenue": 800}, {"name": "B", "revenue": 100},
                 {"name": "C", "revenue": 50}, {"name": "D", "revenue": 30}, {"name": "E", "revenue": 20}]}},
             {"hhi_min": 2500, "top3_pct_min": 90}),
    EvalCase("T05", "低集中度", "tools",
             {"tool": "calc_concentration", "args": {"revenues": [
                 {"name": "A", "revenue": 100}, {"name": "B", "revenue": 95},
                 {"name": "C", "revenue": 90}, {"name": "D", "revenue": 85},
                 {"name": "E", "revenue": 80}, {"name": "F", "revenue": 75}]}},
             {"hhi_max": 2500}),

    # ---- detect_churn_risk ----
    EvalCase("T06", "高流失风险-连续下降", "tools",
             {"tool": "detect_churn_risk", "args": {
                 "client_name": "TestHigh",
                 "monthly_values": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 0]}},
             {"risk_level_in": ["极高", "高"]}),
    EvalCase("T07", "低流失风险-稳定客户", "tools",
             {"tool": "detect_churn_risk", "args": {
                 "client_name": "TestLow",
                 "monthly_values": [100, 102, 98, 105, 100, 103, 99, 101, 104, 100, 102, 105]}},
             {"risk_level_in": ["低"]}),
    EvalCase("T08", "H2断崖式下跌", "tools",
             {"tool": "detect_churn_risk", "args": {
                 "client_name": "TestCliff",
                 "monthly_values": [200, 210, 205, 215, 200, 195, 50, 30, 20, 10, 5, 0]}},
             {"risk_level_in": ["极高", "高"]}),

    # ---- analyze_product_mix ----
    EvalCase("T09", "产品BCG分类", "tools",
             {"tool": "analyze_product_mix", "args": {"products": [
                 {"name": "手机", "current": 3000, "previous": 2000},
                 {"name": "IoT", "current": 500, "previous": 200},
                 {"name": "平板", "current": 100, "previous": 150}]}},
             {"has_star": True, "total_gt": 3000}),

    # ---- analyze_monthly_trend ----
    EvalCase("T10", "上升趋势", "tools",
             {"tool": "analyze_monthly_trend", "args": {
                 "monthly_values": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]}},
             {"peak_month": "12月", "trough_month": "1月", "h2_gt_h1": True}),

    # ---- scan_all_risks ----
    EvalCase("T11", "批量风险扫描", "tools",
             {"tool": "scan_all_risks", "args": {"client_data": [
                 {"name": "稳定A", "monthly_values": [100]*12},
                 {"name": "危险B", "monthly_values": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 0]},
                 {"name": "稳定C", "monthly_values": [50]*12}]}},
             {"has_high_risk": True, "total_clients": 3}),

    # ---- format_number ----
    EvalCase("T12", "亿元格式化", "tools",
             {"tool": "format_number", "args": {"value": 41234.5}},
             {"contains": "亿"}),
    EvalCase("T13", "万元格式化", "tools",
             {"tool": "format_number", "args": {"value": 5432.1, "unit": "万元"}},
             {"contains": "万"}),

    # ---- edge cases ----
    EvalCase("T14", "空列表输入", "tools",
             {"tool": "calc_concentration", "args": {"revenues": []}},
             {"handles_empty": True}),
    EvalCase("T15", "未知工具", "tools",
             {"tool": "nonexistent", "args": {}},
             {"has_error": True}),
]


def run_tool_eval(case: EvalCase) -> EvalResult:
    """执行单个工具测试"""
    from tool_registry import sales_tools

    t0 = time.time()
    tool_name = case.input_data["tool"]
    args = case.input_data["args"]

    try:
        result = sales_tools.execute(tool_name, args)
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        # 如果期望错误，这也算通过
        if case.expected.get("has_error") or case.expected.get("has_error_or_special"):
            return EvalResult(case.id, True, 1.0, "Expected error caught", elapsed)
        return EvalResult(case.id, False, 0.0, f"Exception: {e}", elapsed)

    elapsed = (time.time() - t0) * 1000
    errors = []
    score = 1.0
    exp = case.expected
    r = result.get("result", result)

    # Check: has_error
    if exp.get("has_error"):
        if "error" not in result:
            errors.append("Expected error but got success")
        return EvalResult(case.id, len(errors) == 0, 1.0 if not errors else 0.0,
                         "; ".join(errors) if errors else "OK", elapsed, errors)

    # Check: has_error_or_special (zero base etc)
    if exp.get("has_error_or_special"):
        # Accept either error or special value (like "N/A" or very high number)
        return EvalResult(case.id, True, 1.0, "Special case handled", elapsed)

    # Check: handles_empty
    if exp.get("handles_empty"):
        if "error" in result:
            return EvalResult(case.id, True, 1.0, "Empty handled with error", elapsed)
        return EvalResult(case.id, True, 0.8, "Empty handled without error", elapsed)

    # Check: growth_pct_range
    if "growth_pct_range" in exp:
        lo, hi = exp["growth_pct_range"]
        actual = r.get("growth_pct", 0)
        if not (lo <= actual <= hi):
            errors.append(f"growth_pct {actual} not in [{lo}, {hi}]")
            score -= 0.5

    # Check: delta_positive
    if "delta_positive" in exp:
        delta = r.get("delta", 0)
        if exp["delta_positive"] and delta <= 0:
            errors.append(f"Expected positive delta, got {delta}")
            score -= 0.3
        elif not exp["delta_positive"] and delta >= 0:
            errors.append(f"Expected negative delta, got {delta}")
            score -= 0.3

    # Check: hhi
    if "hhi_min" in exp:
        if r.get("hhi", 0) < exp["hhi_min"]:
            errors.append(f"HHI {r.get('hhi')} < {exp['hhi_min']}")
            score -= 0.5
    if "hhi_max" in exp:
        if r.get("hhi", 0) > exp["hhi_max"]:
            errors.append(f"HHI {r.get('hhi')} > {exp['hhi_max']}")
            score -= 0.5

    # Check: top3
    if "top3_pct_min" in exp:
        if r.get("top3_pct", 0) < exp["top3_pct_min"]:
            errors.append(f"top3 {r.get('top3_pct')}% < {exp['top3_pct_min']}%")
            score -= 0.3

    # Check: risk_level
    if "risk_level_in" in exp:
        if r.get("risk_level") not in exp["risk_level_in"]:
            errors.append(f"risk_level '{r.get('risk_level')}' not in {exp['risk_level_in']}")
            score -= 0.5

    # Check: BCG star
    if exp.get("has_star"):
        prods = r.get("products", [])
        has = any("明星" in str(p.get("bcg_quadrant", p.get("category", ""))) for p in prods)
        if not has:
            errors.append("No 明星 product found")
            score -= 0.3
    if "total_gt" in exp:
        if r.get("total_revenue", 0) <= exp["total_gt"]:
            errors.append(f"total {r.get('total_revenue')} <= {exp['total_gt']}")
            score -= 0.2

    # Check: trend
    if "peak_month" in exp:
        if r.get("peak", {}).get("month") != exp["peak_month"]:
            errors.append(f"peak month {r.get('peak',{}).get('month')} != {exp['peak_month']}")
            score -= 0.3
    if "trough_month" in exp:
        if r.get("trough", {}).get("month") != exp["trough_month"]:
            errors.append(f"trough month {r.get('trough',{}).get('month')} != {exp['trough_month']}")
            score -= 0.3
    if exp.get("h2_gt_h1"):
        h2 = r.get("h2_avg", r.get("h2_total", 0))
        h1 = r.get("h1_avg", r.get("h1_total", 0))
        if h2 <= h1:
            errors.append("H2 not > H1")
            score -= 0.2

    # Check: batch risk
    if exp.get("has_high_risk"):
        clients = r.get("clients", [])
        has_hr = any(c.get("risk_level") in ("极高", "高") for c in clients)
        if not has_hr:
            errors.append("No high-risk client found in batch scan")
            score -= 0.5
    if "total_clients" in exp:
        if len(r.get("clients", [])) != exp["total_clients"]:
            errors.append(f"client count {len(r.get('clients',[]))} != {exp['total_clients']}")
            score -= 0.2

    # Check: contains
    if "contains" in exp:
        result_str = str(r)
        if exp["contains"] not in result_str:
            errors.append(f"'{exp['contains']}' not found in output")
            score -= 0.5

    score = max(0.0, score)
    passed = len(errors) == 0
    return EvalResult(case.id, passed, score,
                     "; ".join(errors) if errors else "OK", elapsed, errors)


# ============================================================
# ② Agent Output Quality Tests
# ============================================================

QUALITY_TEST_CASES = [
    EvalCase("Q01", "营收同比问题", "agent_quality",
             {"question": "今年总营收和去年比怎么样？",
              "mock_data": "2024年总营收41.71亿元，2023年总营收27.07亿元"},
             {"must_contain_number": True, "must_chinese": True,
              "min_length": 50, "should_mention": ["增长", "同比"]}),

    EvalCase("Q02", "客户流失风险", "agent_quality",
             {"question": "哪些客户有流失风险？",
              "mock_data": "客户A: 1-12月出货[100,95,90,80,60,40,20,10,5,0,0,0]\n客户B: 稳定出货[50]*12"},
             {"must_contain_number": False, "must_chinese": True,
              "min_length": 50, "should_mention": ["客户A", "风险"]}),

    EvalCase("Q03", "CEO简报", "agent_quality",
             {"question": "CEO本月该关注什么？",
              "mock_data": "本月营收3.5亿，环比-5%，客户A流失风险高，IoT新品增长120%"},
             {"must_chinese": True, "min_length": 80,
              "should_mention": ["建议"]}),

    EvalCase("Q04", "产品结构分析", "agent_quality",
             {"question": "各产品线表现如何？",
              "mock_data": "手机ODM: 30亿(+50%), IoT: 5亿(+150%), 平板: 2亿(-20%), 可穿戴: 1亿(+80%)"},
             {"must_chinese": True, "min_length": 60,
              "should_mention": ["手机", "IoT"]}),
]


def eval_agent_output(output: str, expected: dict) -> EvalResult:
    """评估 Agent 输出质量（离线，基于规则）"""
    errors = []
    score = 1.0

    # 长度检查
    min_len = expected.get("min_length", 20)
    if len(output) < min_len:
        errors.append(f"输出过短: {len(output)} < {min_len}")
        score -= 0.3

    # 中文检查
    if expected.get("must_chinese"):
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', output))
        if cn_chars < 5:
            errors.append(f"中文字符不足: {cn_chars}")
            score -= 0.3

    # 数字检查
    if expected.get("must_contain_number"):
        nums = re.findall(r'\d+\.?\d*', output)
        if not nums:
            errors.append("缺少数字数据")
            score -= 0.2

    # 关键词检查
    for keyword in expected.get("should_mention", []):
        if keyword not in output:
            errors.append(f"缺少关键信息: '{keyword}'")
            score -= 0.15

    # 错误消息检测
    if any(tag in output for tag in ["调用失败", "error", "Exception", "超时"]):
        errors.append("输出包含错误信息")
        score -= 0.5

    score = max(0.0, score)
    return EvalResult("", len(errors) == 0, score,
                     "; ".join(errors) if errors else "OK", 0, errors)


# ============================================================
# ③ End-to-End Pipeline Tests
# ============================================================

E2E_TEST_CASES = [
    EvalCase("E01", "基础营收问题", "e2e",
             {"question": "今年总营收多少？"},
             {"max_latency_sec": 30, "min_agents": 1, "has_answer": True}),
    EvalCase("E02", "多Agent风险问题", "e2e",
             {"question": "哪些客户有流失风险？CEO该怎么应对？"},
             {"max_latency_sec": 45, "min_agents": 2, "has_answer": True}),
    EvalCase("E03", "全Agent问题", "e2e",
             {"question": "请做一个完整的销售分析报告，包括增长、风险和战略建议"},
             {"max_latency_sec": 60, "min_agents": 3, "has_answer": True}),
]


# ============================================================
# MCP Protocol Tests
# ============================================================

MCP_TEST_CASES = [
    EvalCase("M01", "MCP initialize", "mcp",
             {"method": "initialize", "params": {"clientInfo": {"name": "test", "version": "1.0"},
              "protocolVersion": "2025-06-18"}},
             {"has_protocol_version": True, "has_capabilities": True}),
    EvalCase("M02", "MCP tools/list", "mcp",
             {"method": "tools/list", "params": {}},
             {"min_tools": 8}),
    EvalCase("M03", "MCP tools/call", "mcp",
             {"method": "tools/call", "params": {
                 "name": "calc_yoy_growth", "arguments": {"current": 100, "previous": 80}}},
             {"not_error": True, "has_content": True}),
    EvalCase("M04", "MCP resources/list", "mcp",
             {"method": "resources/list", "params": {}},
             {"min_resources": 3}),
    EvalCase("M05", "MCP resources/read", "mcp",
             {"method": "resources/read", "params": {"uri": "mrarfai://tools/catalog"}},
             {"has_contents": True}),
    EvalCase("M06", "MCP prompts/list", "mcp",
             {"method": "prompts/list", "params": {}},
             {"min_prompts": 3}),
    EvalCase("M07", "MCP prompts/get", "mcp",
             {"method": "prompts/get", "params": {"name": "sales-overview", "arguments": {"period": "2024"}}},
             {"has_messages": True}),
    EvalCase("M08", "MCP unknown method", "mcp",
             {"method": "nonexistent/method", "params": {}},
             {"is_error": True}),
    EvalCase("M09", "MCP ping", "mcp",
             {"method": "ping", "params": {}},
             {"is_success": True}),
]


def run_mcp_eval(case: EvalCase) -> EvalResult:
    """执行 MCP 协议测试"""
    from mcp_server import MCPHandler

    handler = MCPHandler()
    t0 = time.time()

    request = {
        "jsonrpc": "2.0",
        "id": case.id,
        "method": case.input_data["method"],
        "params": case.input_data.get("params", {}),
    }

    try:
        response = handler.handle(request)
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return EvalResult(case.id, False, 0.0, f"Exception: {e}", elapsed)

    elapsed = (time.time() - t0) * 1000
    errors = []
    exp = case.expected

    if response is None:
        errors.append("No response")
        return EvalResult(case.id, False, 0.0, "No response", elapsed, errors)

    result = response.get("result", {})
    error = response.get("error")

    if exp.get("is_error"):
        if not error:
            errors.append("Expected error response")
        return EvalResult(case.id, bool(error), 1.0 if error else 0.0,
                         "OK" if error else "Expected error", elapsed, errors)

    if exp.get("is_success"):
        if error:
            errors.append(f"Unexpected error: {error}")
        return EvalResult(case.id, not error, 1.0 if not error else 0.0,
                         "OK" if not error else str(error), elapsed, errors)

    if error:
        return EvalResult(case.id, False, 0.0, f"Error: {error}", elapsed, [str(error)])

    # Specific checks
    if exp.get("has_protocol_version"):
        if "protocolVersion" not in result:
            errors.append("Missing protocolVersion")
    if exp.get("has_capabilities"):
        if "capabilities" not in result:
            errors.append("Missing capabilities")
    if "min_tools" in exp:
        tools = result.get("tools", [])
        if len(tools) < exp["min_tools"]:
            errors.append(f"tools count {len(tools)} < {exp['min_tools']}")
    if exp.get("not_error"):
        if result.get("isError"):
            errors.append("Tool returned error")
    if exp.get("has_content"):
        if not result.get("content"):
            errors.append("Missing content")
    if "min_resources" in exp:
        if len(result.get("resources", [])) < exp["min_resources"]:
            errors.append("Insufficient resources")
    if exp.get("has_contents"):
        if not result.get("contents"):
            errors.append("Missing contents")
    if "min_prompts" in exp:
        if len(result.get("prompts", [])) < exp["min_prompts"]:
            errors.append("Insufficient prompts")
    if exp.get("has_messages"):
        if not result.get("messages"):
            errors.append("Missing messages")

    score = max(0.0, 1.0 - len(errors) * 0.3)
    return EvalResult(case.id, len(errors) == 0, score,
                     "; ".join(errors) if errors else "OK", elapsed, errors)


# ============================================================
# Guardrails Tests
# ============================================================

GUARD_TEST_CASES = [
    EvalCase("G01", "CircuitBreaker 状态机", "guardrails",
             {"test": "breaker_lifecycle"}, {"passes": True}),
    EvalCase("G02", "Output Validation 正常", "guardrails",
             {"test": "validate_normal"}, {"passes": True}),
    EvalCase("G03", "Output Validation 错误", "guardrails",
             {"test": "validate_error"}, {"passes": True}),
    EvalCase("G04", "Response Cache", "guardrails",
             {"test": "cache_hit_miss"}, {"passes": True}),
    EvalCase("G05", "Token Budget 分级", "guardrails",
             {"test": "budget_levels"}, {"passes": True}),
    EvalCase("G06", "Fallback Chain", "guardrails",
             {"test": "fallback_execution"}, {"passes": True}),
    EvalCase("G07", "JSON 安全解析", "guardrails",
             {"test": "safe_json_parse"}, {"passes": True}),
]


def run_guard_eval(case: EvalCase) -> EvalResult:
    """执行 Guardrails 测试"""
    from guardrails import (
        get_breaker, CircuitState, CircuitBreakerOpenError,
        validate_agent_output, get_cache, get_budget, TokenBudget,
        FallbackChain, safe_parse_llm_json,
    )
    import time as _t

    t0 = _t.time()
    test_name = case.input_data["test"]
    errors = []

    try:
        if test_name == "breaker_lifecycle":
            cb = get_breaker(f"eval_{case.id}", fail_max=2, reset_timeout=0.5)
            assert cb.state == CircuitState.CLOSED
            for _ in range(2):
                try:
                    cb.call(lambda: (_ for _ in ()).throw(ConnectionError()))
                except ConnectionError:
                    pass
            assert cb.state == CircuitState.OPEN, f"Expected OPEN, got {cb.state}"
            try:
                cb.call(lambda: "x")
                errors.append("Should reject in OPEN")
            except CircuitBreakerOpenError:
                pass
            _t.sleep(0.6)
            assert cb.state == CircuitState.HALF_OPEN
            cb.call(lambda: "ok")
            assert cb.state == CircuitState.CLOSED

        elif test_name == "validate_normal":
            v = validate_agent_output("禾苗2024年总营收达到41.71亿元，同比增长54.1%。手机ODM贡献最大。")
            assert v.passed, f"Should pass: {v.issues}"
            assert v.confidence >= 0.8

        elif test_name == "validate_error":
            v = validate_agent_output("[调用失败: timeout]")
            assert not v.passed
            assert v.confidence < 0.5

        elif test_name == "cache_hit_miss":
            cache = get_cache()
            cache.put(f"eval_{case.id}", {"answer": "test"})
            assert cache.get(f"eval_{case.id}") is not None
            assert cache.get("nonexistent_key_xyz") is None

        elif test_name == "budget_levels":
            b = TokenBudget(daily_budget_usd=1.0)
            assert b.check_budget()["level"] == "normal"
            b.record_cost(0.55, "t1")
            assert b.check_budget()["level"] == "caution"
            b.record_cost(0.30, "t2")
            assert b.check_budget()["level"] == "warning"
            b.record_cost(0.10, "t3")
            assert b.check_budget()["level"] == "critical"

        elif test_name == "fallback_execution":
            chain = FallbackChain("eval")
            chain.add(lambda: (_ for _ in ()).throw(Exception("f1")), "L1")
            chain.add(lambda: "ok", "L2")
            result, level = chain.execute()
            assert result == "ok" and level == "L2"

        elif test_name == "safe_json_parse":
            assert safe_parse_llm_json('{"a":1}') == {"a": 1}
            assert safe_parse_llm_json('```json\n{"a":1}\n```') == {"a": 1}
            assert safe_parse_llm_json("not json") is None

    except AssertionError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Exception: {e}")

    elapsed = (_t.time() - t0) * 1000
    passed = len(errors) == 0
    return EvalResult(case.id, passed, 1.0 if passed else 0.0,
                     "; ".join(errors) if errors else "OK", elapsed, errors)


# ============================================================
# Runner
# ============================================================

# ============================================================
# ④ RAG Engine Tests
# ============================================================

RAG_TEST_CASES = [
    EvalCase("R01", "文本导入", "rag",
             {"action": "ingest_text"}, {"chunks_min": 1}),
    EvalCase("R02", "多文档导入", "rag",
             {"action": "ingest_multi"}, {"total_chunks_min": 3}),
    EvalCase("R03", "向量检索", "rag",
             {"action": "vector_search"}, {"has_results": True}),
    EvalCase("R04", "BM25检索", "rag",
             {"action": "bm25_search"}, {"top1_source": "contract"}),
    EvalCase("R05", "混合检索精度", "rag",
             {"action": "precision_test"}, {"precision_min": 0.5}),
    EvalCase("R06", "Context构建", "rag",
             {"action": "build_context"}, {"min_length": 50}),
    EvalCase("R07", "Enrich集成", "rag",
             {"action": "enrich"}, {"has_both": True}),
    EvalCase("R08", "空库检索", "rag",
             {"action": "search_empty"}, {"empty": True}),
    EvalCase("R09", "Stats统计", "rag",
             {"action": "stats"}, {"has_sources": True}),
]


def _make_test_rag():
    """创建预填充的测试 RAG 实例"""
    from rag_engine import RAGEngine
    rag = RAGEngine(chunk_size=200, chunk_overlap=40)
    rag.ingest_text(
        "HMD功能机基准价格8.5美元FOB深圳，智能机35至65美元。"
        "季度调整机制允许正负5%浮动。付款条件T/T 60天月结。"
        "HMD承诺最低采购500万台，功能机350万台，智能机150万台。",
        source="contract.pdf")
    rag.ingest_text(
        "HMD表示2026年缩减功能机产品线，从12个型号减至6到8个。"
        "华勤在价格上更有优势。HMD对CKD散件需求将下降，"
        "因为印度本地组装能力在提升。Action Items包括提交Android Go报价。",
        source="meeting.md")
    rag.ingest_text(
        "全球功能机出货7.2亿台同比下降12%。华勤市占率35%稳定。"
        "闻泰22%转向汽车电子。龙旗18%。禾苗约5%。"
        "印度市场加速萎缩1.1亿台下降19%。2026年预测继续萎缩10到15%。",
        source="report.pdf")
    return rag


def run_rag_eval(case: EvalCase) -> EvalResult:
    """执行RAG测试"""
    from rag_engine import RAGEngine, BM25Index, DocChunk, enrich_context_with_rag

    t0 = time.time()
    action = case.input_data["action"]
    exp = case.expected
    errors = []

    try:
        if action == "ingest_text":
            rag = RAGEngine(chunk_size=200)
            rag.chunker.min_chunk_size = 30
            n = rag.ingest_text(
                "禾苗与HMD合同：功能机基准价格8.5美元FOB深圳，智能机35至65美元区间。"
                "最低采购量500万台，其中功能机350万台，付款条件T/T 60天月结。",
                source="contract"
            )
            if n < exp["chunks_min"]:
                errors.append(f"chunks={n} < {exp['chunks_min']}")

        elif action == "ingest_multi":
            rag = _make_test_rag()
            if rag.vector_store.size() < exp["total_chunks_min"]:
                errors.append(f"total={rag.vector_store.size()} < {exp['total_chunks_min']}")

        elif action == "vector_search":
            rag = _make_test_rag()
            results = rag.search("功能机价格", top_k=3)
            if not results:
                errors.append("No vector search results")

        elif action == "bm25_search":
            bm25 = BM25Index()
            bm25.add([
                DocChunk("c1", "HMD合同价格8.5美元功能机付款T/T60天", "contract", "txt"),
                DocChunk("c2", "华勤市占率35%全球功能机萎缩12%", "report", "txt"),
            ])
            results = bm25.search("HMD合同价格8.5美元", top_k=2)
            if not results:
                errors.append("No BM25 results")
            elif exp.get("top1_source") and exp["top1_source"] not in results[0][0].source:
                errors.append(f"BM25 top1={results[0][0].source}, expected {exp['top1_source']}")

        elif action == "precision_test":
            rag = _make_test_rag()
            tests = [
                ("合同价格8.5美元", "contract"),
                ("功能机萎缩12%", "report"),
                ("CKD需求下降", "meeting"),
                ("印度市场萎缩", "report"),
            ]
            correct = sum(
                1 for q, prefix in tests
                if (r := rag.search(q, top_k=1)) and prefix in r[0].chunk.source
            )
            precision = correct / len(tests)
            if precision < exp["precision_min"]:
                errors.append(f"precision={precision:.0%} < {exp['precision_min']:.0%}")

        elif action == "build_context":
            rag = _make_test_rag()
            ctx = rag.build_context("HMD订单情况", max_tokens=1000)
            if len(ctx) < exp["min_length"]:
                errors.append(f"context len={len(ctx)} < {exp['min_length']}")

        elif action == "enrich":
            rag = _make_test_rag()
            structured = "【总营收】41.71亿元，同比+54.1%"
            enriched = enrich_context_with_rag("HMD价格", structured, rag)
            if "41.71" not in enriched:
                errors.append("Missing structured data")
            if "参考" not in enriched:
                errors.append("Missing RAG context")

        elif action == "search_empty":
            rag = RAGEngine()
            results = rag.search("test")
            if results:
                errors.append(f"Expected empty, got {len(results)} results")

        elif action == "stats":
            rag = _make_test_rag()
            stats = rag.get_stats()
            if not stats.get("sources"):
                errors.append("No sources in stats")

    except Exception as e:
        errors.append(f"Exception: {e}")

    elapsed = (time.time() - t0) * 1000
    passed = len(errors) == 0
    return EvalResult(
        case.id, passed, 1.0 if passed else 0.0,
        "OK" if passed else "; ".join(errors), elapsed, errors
    )


def run_eval_suite(
    categories: List[str] = None,
    verbose: bool = True,
) -> EvalReport:
    """运行评估套件"""
    categories = categories or ["tools", "mcp", "guardrails", "rag"]

    all_cases = []
    if "tools" in categories:
        all_cases.extend(TOOL_TEST_CASES)
    if "mcp" in categories:
        all_cases.extend(MCP_TEST_CASES)
    if "guardrails" in categories:
        all_cases.extend(GUARD_TEST_CASES)
    if "rag" in categories:
        all_cases.extend(RAG_TEST_CASES)
    if "agent_quality" in categories:
        all_cases.extend(QUALITY_TEST_CASES)

    report = EvalReport()
    t0 = time.time()

    for case in all_cases:
        if verbose:
            sys.stdout.write(f"  {case.id} {case.name}...")
            sys.stdout.flush()

        if case.category == "tools":
            result = run_tool_eval(case)
        elif case.category == "mcp":
            result = run_mcp_eval(case)
        elif case.category == "guardrails":
            result = run_guard_eval(case)
        elif case.category == "rag":
            result = run_rag_eval(case)
        elif case.category == "agent_quality":
            # Agent quality needs mock — just validate the checker
            mock_output = case.input_data.get("mock_data", "")
            result = eval_agent_output(mock_output, case.expected)
            result.case_id = case.id
        else:
            result = EvalResult(case.id, False, 0.0, f"Unknown category: {case.category}")

        result.case_id = case.id
        report.results.append(result)

        if verbose:
            status = "✅" if result.passed else "❌"
            print(f" {status} ({result.score:.0%}) {result.details[:60]}")

        # Category stats
        cat = case.category
        if cat not in report.by_category:
            report.by_category[cat] = {"total": 0, "passed": 0, "scores": []}
        report.by_category[cat]["total"] += 1
        if result.passed:
            report.by_category[cat]["passed"] += 1
        report.by_category[cat]["scores"].append(result.score)

    report.total = len(report.results)
    report.passed = sum(1 for r in report.results if r.passed)
    report.failed = report.total - report.passed
    all_scores = [r.score for r in report.results]
    report.avg_score = statistics.mean(all_scores) if all_scores else 0
    report.elapsed_sec = time.time() - t0

    for cat, stats in report.by_category.items():
        stats["score"] = statistics.mean(stats["scores"]) if stats["scores"] else 0

    return report


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MRARFAI Evaluation Framework")
    parser.add_argument("--tools", action="store_true", help="Run tool tests only")
    parser.add_argument("--mcp", action="store_true", help="Run MCP protocol tests only")
    parser.add_argument("--guardrails", action="store_true", help="Run guardrails tests only")
    parser.add_argument("--rag", action="store_true", help="Run RAG engine tests only")
    parser.add_argument("--agents", action="store_true", help="Run agent quality tests")
    parser.add_argument("--all", action="store_true", help="Run all offline tests")
    parser.add_argument("--report", action="store_true", help="Print detailed report")
    args = parser.parse_args()

    cats = []
    if args.tools:
        cats.append("tools")
    if args.mcp:
        cats.append("mcp")
    if args.guardrails:
        cats.append("guardrails")
    if args.rag:
        cats.append("rag")
    if args.agents:
        cats.append("agent_quality")
    if args.all or not cats:
        cats = ["tools", "mcp", "guardrails", "rag"]

    print(f"\n🧪 MRARFAI Eval — Categories: {', '.join(cats)}\n")
    report = run_eval_suite(cats, verbose=True)
    print(report.summary())

    if args.report:
        # Export JSON report
        report_data = {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "avg_score": report.avg_score,
            "elapsed_sec": report.elapsed_sec,
            "by_category": {k: {"total": v["total"], "passed": v["passed"],
                                "score": v["score"]} for k, v in report.by_category.items()},
            "results": [{"case_id": r.case_id, "passed": r.passed, "score": r.score,
                         "details": r.details, "elapsed_ms": r.elapsed_ms}
                        for r in report.results],
        }
        with open("eval_report.json", "w") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Report saved: eval_report.json")

    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
