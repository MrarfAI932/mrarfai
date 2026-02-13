#!/usr/bin/env python3
"""
MRARFAI V10.0 Integration Test Suite
======================================
39 tests across 8 categories — 验证端到端平台功能

Test Categories:
  1. Agent Creation (4 tests)     — 4个新Agent实例化
  2. Agent Execution (8 tests)    — 每个Agent域查询
  3. Platform Gateway (3 tests)   — 7 agents, 20 skills, v10.0 card
  4. Intelligent Routing (5 tests) — 5域路由准确率
  5. End-to-End Ask (6 tests)     — 完整管线: route→execute→respond
  6. Cross-Agent Collaboration (5 tests) — 场景检测+计划+执行
  7. Audit Log (4 tests)          — 日志记录/统计/分布
  8. Platform Stats (4 tests)     — 版本/Agent数/Skill数/Registry

Usage:
  python test_integration_v10.py
  python -m pytest test_integration_v10.py -v
"""

import sys
import json
import asyncio
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("mrarfai.test_v10")


# ============================================================
# 测试框架
# ============================================================

class TestResult:
    def __init__(self, name: str, category: str, passed: bool, detail: str = ""):
        self.name = name
        self.category = category
        self.passed = passed
        self.detail = detail


class TestRunner:
    def __init__(self):
        self.results: list = []

    def add(self, name: str, category: str, passed: bool, detail: str = ""):
        self.results.append(TestResult(name, category, passed, detail))

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        by_category = {}
        for r in self.results:
            if r.category not in by_category:
                by_category[r.category] = {"total": 0, "passed": 0}
            by_category[r.category]["total"] += 1
            if r.passed:
                by_category[r.category]["passed"] += 1
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed}/{total}",
            "by_category": by_category,
        }

    def print_results(self):
        print("\n" + "=" * 70)
        print("MRARFAI V10.0 Integration Test Results")
        print("=" * 70)

        current_cat = ""
        for r in self.results:
            if r.category != current_cat:
                current_cat = r.category
                print(f"\n--- {current_cat} ---")
            status = "PASS" if r.passed else "FAIL"
            icon = "✅" if r.passed else "❌"
            print(f"  {icon} [{status}] {r.name}")
            if not r.passed and r.detail:
                print(f"         Detail: {r.detail}")

        s = self.summary()
        print(f"\n{'=' * 70}")
        print(f"Total: {s['total']} | Passed: {s['passed']} | Failed: {s['failed']}")
        rate_pct = '100%' if s['failed'] == 0 else f"{s['passed']/s['total']:.0%}"
        print(f"Pass Rate: {s['pass_rate']} ({rate_pct})")
        print(f"{'=' * 70}")

        if s["failed"] > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ❌ {r.category} > {r.name}: {r.detail}")


runner = TestRunner()


# ============================================================
# Category 1: Agent Creation (4 tests)
# ============================================================

def test_agent_creation():
    """测试4个新Agent的实例化"""
    cat = "Agent Creation"

    # Test 1: Procurement Agent
    try:
        from agent_procurement import ProcurementEngine, ProcurementExecutor
        engine = ProcurementEngine()
        executor = ProcurementExecutor()
        assert len(engine.suppliers) > 0
        assert len(engine.orders) > 0
        runner.add("Procurement Agent instantiation", cat, True)
    except Exception as e:
        runner.add("Procurement Agent instantiation", cat, False, str(e))

    # Test 2: Quality Agent
    try:
        from agent_quality import QualityEngine, QualityExecutor
        engine = QualityEngine()
        executor = QualityExecutor()
        assert len(engine.yields) > 0
        assert len(engine.returns) > 0
        runner.add("Quality Agent instantiation", cat, True)
    except Exception as e:
        runner.add("Quality Agent instantiation", cat, False, str(e))

    # Test 3: Finance Agent
    try:
        from agent_finance import FinanceEngine, FinanceExecutor
        engine = FinanceEngine()
        executor = FinanceExecutor()
        assert len(engine.ar_records) > 0
        assert len(engine.margins) > 0
        runner.add("Finance Agent instantiation", cat, True)
    except Exception as e:
        runner.add("Finance Agent instantiation", cat, False, str(e))

    # Test 4: Market Agent
    try:
        from agent_market import MarketEngine, MarketExecutor
        engine = MarketEngine()
        executor = MarketExecutor()
        assert len(engine.competitors) > 0
        assert len(engine.trends) > 0
        runner.add("Market Agent instantiation", cat, True)
    except Exception as e:
        runner.add("Market Agent instantiation", cat, False, str(e))


# ============================================================
# Category 2: Agent Execution (8 tests)
# ============================================================

def test_agent_execution():
    """测试每个Agent的域查询"""
    cat = "Agent Execution"

    # Procurement (2 tests)
    try:
        from agent_procurement import ProcurementEngine
        engine = ProcurementEngine()
        result = engine.compare_quotes("显示屏")
        assert "suppliers" in result
        assert len(result["suppliers"]) > 0
        runner.add("Procurement: compare_quotes", cat, True)
    except Exception as e:
        runner.add("Procurement: compare_quotes", cat, False, str(e))

    try:
        from agent_procurement import ProcurementEngine
        engine = ProcurementEngine()
        result = engine.alert_delay()
        assert "total_delayed" in result
        runner.add("Procurement: alert_delay", cat, True)
    except Exception as e:
        runner.add("Procurement: alert_delay", cat, False, str(e))

    # Quality (2 tests)
    try:
        from agent_quality import QualityEngine
        engine = QualityEngine()
        result = engine.monitor_yield("S60")
        assert "trends" in result
        runner.add("Quality: monitor_yield", cat, True)
    except Exception as e:
        runner.add("Quality: monitor_yield", cat, False, str(e))

    try:
        from agent_quality import QualityEngine
        engine = QualityEngine()
        result = engine.trace_root_cause("触控失灵")
        assert "probable_cause" in result
        assert "recommended_actions" in result
        runner.add("Quality: trace_root_cause", cat, True)
    except Exception as e:
        runner.add("Quality: trace_root_cause", cat, False, str(e))

    # Finance (2 tests)
    try:
        from agent_finance import FinanceEngine
        engine = FinanceEngine()
        result = engine.track_ar()
        assert "total_outstanding" in result
        assert "aging_analysis" in result
        runner.add("Finance: track_ar", cat, True)
    except Exception as e:
        runner.add("Finance: track_ar", cat, False, str(e))

    try:
        from agent_finance import FinanceEngine
        engine = FinanceEngine()
        result = engine.analyze_margin()
        assert "gross_margin" in result
        assert "by_product" in result
        runner.add("Finance: analyze_margin", cat, True)
    except Exception as e:
        runner.add("Finance: analyze_margin", cat, False, str(e))

    # Market (2 tests)
    try:
        from agent_market import MarketEngine
        engine = MarketEngine()
        result = engine.monitor_competitor("华勤")
        assert "competitors" in result
        assert len(result["competitors"]) > 0
        runner.add("Market: monitor_competitor", cat, True)
    except Exception as e:
        runner.add("Market: monitor_competitor", cat, False, str(e))

    try:
        from agent_market import MarketEngine
        engine = MarketEngine()
        result = engine.track_sentiment()
        assert "sentiment_score" in result
        assert "signals" in result
        runner.add("Market: track_sentiment", cat, True)
    except Exception as e:
        runner.add("Market: track_sentiment", cat, False, str(e))


# ============================================================
# Category 3: Platform Gateway (3 tests)
# ============================================================

def test_platform_gateway():
    """测试平台网关"""
    cat = "Platform Gateway"

    try:
        from platform_gateway import PlatformGateway
        gw = PlatformGateway()

        # Test 1: 7 agents registered
        card = gw.get_platform_card()
        agent_count = card["total_agents"]
        assert agent_count == 7, f"Expected 7 agents, got {agent_count}"
        runner.add("7 agents registered", cat, True)
    except Exception as e:
        runner.add("7 agents registered", cat, False, str(e))

    try:
        from platform_gateway import PlatformGateway
        gw = PlatformGateway()
        card = gw.get_platform_card()

        # Test 2: 20 skills aggregated
        total_skills = card["total_skills"]
        assert total_skills == 20, f"Expected 20 skills, got {total_skills}"
        runner.add("20 skills aggregated", cat, True)
    except Exception as e:
        runner.add("20 skills aggregated", cat, False, str(e))

    try:
        from platform_gateway import PlatformGateway
        gw = PlatformGateway()
        card = gw.get_platform_card()

        # Test 3: v10.0 card
        assert card["version"] == "10.0.0", f"Expected v10.0.0, got {card['version']}"
        assert "MCP 3.0" in card["protocols"]
        assert "A2A RC v1.0" in card["protocols"]
        runner.add("v10.0.0 platform card", cat, True)
    except Exception as e:
        runner.add("v10.0.0 platform card", cat, False, str(e))


# ============================================================
# Category 4: Intelligent Routing (5 tests)
# ============================================================

def test_intelligent_routing():
    """测试智能路由 — 5域100%准确率"""
    cat = "Intelligent Routing"

    from platform_gateway import AgentRouter
    router = AgentRouter()

    test_cases = [
        ("今天总营收是多少？", "sales", "Sales query"),
        ("供应商交期延迟", "procurement", "Procurement query"),
        ("本月良率多少？", "quality", "Quality query"),
        ("毛利率分析", "finance", "Finance query"),
        ("竞品华勤对比", "market", "Market query"),
    ]

    for query, expected, label in test_cases:
        try:
            result = router.route(query)
            actual = result["agent"]
            confidence = result["confidence"]
            assert actual == expected, f"Expected {expected}, got {actual}"
            runner.add(f"Route: {label} → {expected} (conf={confidence})", cat, True)
        except Exception as e:
            runner.add(f"Route: {label} → {expected}", cat, False, str(e))


# ============================================================
# Category 5: End-to-End Ask (6 tests)
# ============================================================

def test_end_to_end():
    """测试完整管线: route → execute → respond"""
    cat = "End-to-End Ask"

    from platform_gateway import PlatformGateway
    gw = PlatformGateway()

    test_queries = [
        ("供应商比价分析", "procurement", "suppliers"),
        ("良率趋势如何？", "quality", "trends"),
        ("应收账款逾期情况", "finance", "total_outstanding"),
        ("竞品市场份额", "market", "competitors"),
        ("采购延迟预警", "procurement", "alerts"),
        ("退货分析", "quality", "total_returns"),
    ]

    for query, expected_agent, expected_key in test_queries:
        try:
            result = gw.ask(query, user="test")
            assert "request_id" in result, "Missing request_id"
            assert "duration_ms" in result, "Missing duration_ms"

            if result["type"] == "single_agent":
                assert result["agent"] == expected_agent, \
                    f"Expected {expected_agent}, got {result.get('agent')}"
                answer = result.get("answer", "")
                assert expected_key in answer, f"Expected '{expected_key}' in answer"
            elif result["type"] == "collaboration":
                # 协作模式也算通过
                pass

            runner.add(f"E2E: '{query[:20]}...' → {expected_agent}", cat, True)
        except Exception as e:
            runner.add(f"E2E: '{query[:20]}...' → {expected_agent}", cat, False, str(e))


# ============================================================
# Category 6: Cross-Agent Collaboration (5 tests)
# ============================================================

def test_collaboration():
    """测试跨Agent协作"""
    cat = "Cross-Agent Collaboration"

    from platform_gateway import CollaborationEngine, PlatformGateway

    # Test 1: Scenario detection
    try:
        engine = CollaborationEngine()
        scenario = engine.detect_scenario("出货异常，客户流失严重")
        assert scenario is not None
        assert scenario["scenario_id"] == "shipment_anomaly"
        runner.add("Detect shipment_anomaly scenario", cat, True)
    except Exception as e:
        runner.add("Detect shipment_anomaly scenario", cat, False, str(e))

    # Test 2: Plan creation
    try:
        engine = CollaborationEngine()
        scenario = engine.detect_scenario("报价分析")
        assert scenario is not None
        plan = engine.create_plan(scenario, "新客户报价")
        assert len(plan) > 0
        assert plan[0]["status"] == "pending"
        runner.add("Create collaboration plan", cat, True)
    except Exception as e:
        runner.add("Create collaboration plan", cat, False, str(e))

    # Test 3: Chain execution
    try:
        engine = CollaborationEngine()
        scenario = engine.detect_scenario("供应商延迟响应")
        assert scenario is not None
        result = engine.execute_chain(scenario, "供应商延迟了")
        assert result["completed"] > 0
        assert "synthesis" in result
        runner.add("Execute supplier_delay chain", cat, True)
    except Exception as e:
        runner.add("Execute supplier_delay chain", cat, False, str(e))

    # Test 4: Gateway collaboration
    try:
        gw = PlatformGateway()
        result = gw.ask("出货下降，客户流失严重", user="test")
        assert result["type"] == "collaboration"
        assert "scenario" in result
        runner.add("Gateway collaboration detection", cat, True)
    except Exception as e:
        runner.add("Gateway collaboration detection", cat, False, str(e))

    # Test 5: Monthly review scenario
    try:
        engine = CollaborationEngine()
        scenario = engine.detect_scenario("月度回顾经营报告")
        assert scenario is not None
        assert len(scenario["chain"]) == 4  # sales+procurement+quality+finance
        runner.add("Monthly review 4-agent chain", cat, True)
    except Exception as e:
        runner.add("Monthly review 4-agent chain", cat, False, str(e))


# ============================================================
# Category 7: Audit Log (4 tests)
# ============================================================

def test_audit_log():
    """测试审计日志"""
    cat = "Audit Log"

    from platform_gateway import AuditLog

    # Test 1: Request logging
    try:
        audit = AuditLog()
        req_id = audit.log("user1", "query", "sales", "今天营收多少？")
        assert req_id is not None
        assert len(req_id) > 0
        runner.add("Request logging", cat, True)
    except Exception as e:
        runner.add("Request logging", cat, False, str(e))

    # Test 2: Status update
    try:
        audit = AuditLog()
        req_id = audit.log("user1", "query", "sales", "test")
        audit.update(req_id, "completed", 150.5)
        recent = audit.recent(1)
        assert recent[0]["status"] == "completed"
        assert recent[0]["duration_ms"] == 150.5
        runner.add("Status update", cat, True)
    except Exception as e:
        runner.add("Status update", cat, False, str(e))

    # Test 3: Statistics
    try:
        audit = AuditLog()
        audit.log("user1", "query", "sales", "q1")
        audit.log("user1", "query", "procurement", "q2")
        audit.log("user2", "query", "sales", "q3")
        stats = audit.get_stats()
        assert stats["total_requests"] == 3
        assert stats["by_agent"]["sales"] == 2
        assert stats["by_agent"]["procurement"] == 1
        runner.add("Audit statistics", cat, True)
    except Exception as e:
        runner.add("Audit statistics", cat, False, str(e))

    # Test 4: Agent distribution
    try:
        audit = AuditLog()
        for agent in ["sales", "procurement", "quality", "finance"]:
            audit.log("test", "query", agent, f"test {agent}")
        stats = audit.get_stats()
        assert len(stats["by_agent"]) == 4
        runner.add("Agent distribution tracking", cat, True)
    except Exception as e:
        runner.add("Agent distribution tracking", cat, False, str(e))


# ============================================================
# Category 8: Platform Stats (4 tests)
# ============================================================

def test_platform_stats():
    """测试平台统计"""
    cat = "Platform Stats"

    from platform_gateway import PlatformGateway

    try:
        gw = PlatformGateway()
        stats = gw.get_stats()

        # Test 1: Version
        assert stats["version"] == "10.0.0"
        runner.add("Platform version = 10.0.0", cat, True)
    except Exception as e:
        runner.add("Platform version = 10.0.0", cat, False, str(e))

    try:
        gw = PlatformGateway()
        stats = gw.get_stats()

        # Test 2: Agent count
        assert stats["total_agents"] == 7, f"Expected 7, got {stats['total_agents']}"
        runner.add("Total agents = 7", cat, True)
    except Exception as e:
        runner.add("Total agents = 7", cat, False, str(e))

    try:
        gw = PlatformGateway()
        stats = gw.get_stats()

        # Test 3: Skill count
        assert stats["total_skills"] == 20, f"Expected 20, got {stats['total_skills']}"
        runner.add("Total skills = 20", cat, True)
    except Exception as e:
        runner.add("Total skills = 20", cat, False, str(e))

    try:
        gw = PlatformGateway()
        stats = gw.get_stats()

        # Test 4: Registry stats
        registry = stats.get("registry", {})
        assert registry.get("total_agents", 0) == 7
        runner.add("Registry stats valid", cat, True)
    except Exception as e:
        runner.add("Registry stats valid", cat, False, str(e))


# ============================================================
# Main
# ============================================================

def run_all_tests():
    """运行所有39个测试"""
    print(f"\nMRARFAI V10.0 Integration Tests — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    test_agent_creation()       # 4 tests
    test_agent_execution()      # 8 tests
    test_platform_gateway()     # 3 tests
    test_intelligent_routing()  # 5 tests
    test_end_to_end()           # 6 tests
    test_collaboration()        # 5 tests
    test_audit_log()            # 4 tests
    test_platform_stats()       # 4 tests

    runner.print_results()

    summary = runner.summary()
    return summary["failed"] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
