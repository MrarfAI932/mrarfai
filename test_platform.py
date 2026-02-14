#!/usr/bin/env python3
"""
MRARFAI V10.0 — Platform Integration Test
============================================
测试: Agent注册、路由、执行、跨Agent协作、审计

运行: python test_platform.py
"""

import asyncio
import json
import sys
import os

# 确保import路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2a_server_v7 import (
    AgentCard, AgentSkill, AgentCapabilities, AgentInterface,
    AgentExecutor, Task, TaskState, TaskStatus,
    Message, MessagePart, MRARFAIAgentRegistry,
    MRARFAIAnalystExecutor, MRARFAIRiskExecutor, MRARFAIStrategistExecutor,
    create_mrarfai_agent_cards,
)
from platform_gateway import PlatformGateway, AgentRouter, CollaborationEngine
from agent_procurement import create_procurement_agent, ProcurementExecutor
from agent_quality import create_quality_agent, QualityExecutor
from agent_finance import create_finance_agent, FinanceExecutor
from agent_market import create_market_agent, MarketExecutor


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "✅" if condition else "❌"
        self.results.append((status, name, detail))
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  {status} {name}" + (f" — {detail}" if detail else ""))

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*55}")
        print(f"  RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        print(f"{'='*55}")
        return self.failed == 0


async def main():
    print("🚀 MRARFAI V10.0 Platform Integration Test")
    print("=" * 55)
    t = TestResults()

    # ============================================================
    # TEST 1: Individual Agent Creation
    # ============================================================
    print("\n📦 TEST 1: Agent Creation")

    name_p, card_p, exec_p = create_procurement_agent()
    t.check("Procurement Agent created", name_p == "procurement")
    t.check("Procurement has skills", len(card_p.skills) == 3)

    name_q, card_q, exec_q = create_quality_agent()
    t.check("Quality Agent created", name_q == "quality")

    name_f, card_f, exec_f = create_finance_agent()
    t.check("Finance Agent created", name_f == "finance")

    name_m, card_m, exec_m = create_market_agent()
    t.check("Market Agent created", name_m == "market")

    # ============================================================
    # TEST 2: Individual Agent Execution
    # ============================================================
    print("\n⚡ TEST 2: Agent Execution")

    # Procurement
    task = Task()
    result = await exec_p.execute(task, Message.user_text("有哪些延迟的采购订单？"))
    t.check("Procurement answers delays",
            result.status.state == TaskState.COMPLETED,
            f"state={result.status.state.value}")
    answer = result.status.message.parts[0].text
    t.check("Procurement mentions delay", "延迟" in answer or "delay" in answer.lower())

    # Quality
    task = Task()
    result = await exec_q.execute(task, Message.user_text("良品率报告"))
    t.check("Quality answers yield",
            result.status.state == TaskState.COMPLETED)
    answer = result.status.message.parts[0].text
    t.check("Quality has yield data", "%" in answer)

    # Finance
    task = Task()
    result = await exec_f.execute(task, Message.user_text("应收账款情况"))
    t.check("Finance answers AR",
            result.status.state == TaskState.COMPLETED)
    answer = result.status.message.parts[0].text
    t.check("Finance has AR data", "应收" in answer)

    # Market
    task = Task()
    result = await exec_m.execute(task, Message.user_text("竞品分析"))
    t.check("Market answers competitor",
            result.status.state == TaskState.COMPLETED)
    answer = result.status.message.parts[0].text
    t.check("Market mentions competitors", "华勤" in answer or "闻泰" in answer)

    # ============================================================
    # TEST 3: Platform Gateway
    # ============================================================
    print("\n🏗️ TEST 3: Platform Gateway")

    gateway = PlatformGateway()

    # Register V7 agents
    v7_cards = create_mrarfai_agent_cards()
    gateway.register_agent("sales", v7_cards["analyst"], MRARFAIAnalystExecutor(None))
    gateway.register_agent("risk", v7_cards["risk"], MRARFAIRiskExecutor())
    gateway.register_agent("strategist", v7_cards["strategist"], MRARFAIStrategistExecutor())

    # Register V10 agents
    gateway.register_agent(*create_procurement_agent())
    gateway.register_agent(*create_quality_agent())
    gateway.register_agent(*create_finance_agent())
    gateway.register_agent(*create_market_agent())

    total = len(gateway.registry.list_agents())
    t.check("All 7 agents registered", total == 7, f"got {total}")

    # Platform Card
    card = gateway.get_platform_card()
    t.check("Platform card has skills", len(card["skills"]) > 0,
            f"{len(card['skills'])} skills")
    t.check("Platform card v10", card["version"] == "10.0.0")

    # ============================================================
    # TEST 4: Intelligent Routing
    # ============================================================
    print("\n🧭 TEST 4: Intelligent Routing")

    router = AgentRouter()
    available = gateway.registry.list_agents()

    test_routes = [
        ("今年总营收是多少？", "sales"),
        ("供应商交期延迟", "procurement"),
        ("本月良品率", "quality"),
        ("毛利率分析", "finance"),
        ("竞品华勤对比", "market"),
    ]

    for q, expected in test_routes:
        result = router.route(q, available)
        t.check(f"Route '{q[:15]}' → {expected}",
                result.agent_name == expected,
                f"got {result.agent_name} (conf={result.confidence:.2f})")

    # ============================================================
    # TEST 5: End-to-End Ask
    # ============================================================
    print("\n🎯 TEST 5: End-to-End Ask")

    # Single agent
    result = await gateway.ask("供应商评估报告")
    t.check("Ask routes to procurement",
            result["agent"] == "procurement",
            f"agent={result['agent']}")
    t.check("Ask returns answer", len(result["answer"]) > 10)
    t.check("Ask has metadata", result.get("request_id") is not None)

    result = await gateway.ask("良品率多少？")
    t.check("Ask routes to quality", result["agent"] == "quality")

    result = await gateway.ask("应收账款")
    t.check("Ask routes to finance", result["agent"] == "finance")

    # With target agent
    result = await gateway.ask("今年总营收", target_agent="sales")
    t.check("Target agent override", result["agent"] == "sales")

    # ============================================================
    # TEST 6: Cross-Agent Collaboration
    # ============================================================
    print("\n🔗 TEST 6: Cross-Agent Collaboration")

    collab = CollaborationEngine()
    scenario = collab.detect_scenario("客户A出货异常分析")
    t.check("Detects shipment anomaly scenario",
            scenario == "shipment_anomaly",
            f"got {scenario}")

    scenario = collab.detect_scenario("新客户报价")
    t.check("Detects new customer quote scenario",
            scenario == "new_customer_quote")

    # Full collaboration via gateway
    result = await gateway.ask("客户A出货异常，全链路追踪")
    t.check("Collaboration triggered",
            result.get("collaboration") == True)
    t.check("Collaboration has steps",
            len(result.get("steps", [])) > 0,
            f"{len(result.get('steps', []))} steps")
    t.check("Collaboration has answer",
            len(result.get("answer", "")) > 20)

    # ============================================================
    # TEST 7: Audit Log
    # ============================================================
    print("\n📋 TEST 7: Audit Log")

    stats = gateway.audit.get_stats()
    t.check("Audit logged requests",
            stats["total"] > 0,
            f"{stats['total']} entries")
    t.check("Audit has agent distribution",
            len(stats.get("by_agent", {})) > 0)
    t.check("Audit tracks collaboration",
            "collaboration_rate" in stats)

    recent = gateway.audit.get_recent(5)
    t.check("Audit recent entries", len(recent) > 0)

    # ============================================================
    # TEST 8: Platform Stats
    # ============================================================
    print("\n📊 TEST 8: Platform Stats")

    stats = gateway.get_platform_stats()
    t.check("Stats has version", stats["version"] == "10.0.0")
    t.check("Stats has agents", stats["total_agents"] == 7)
    t.check("Stats has skills", stats["total_skills"] > 0)

    # ============================================================
    # Summary
    # ============================================================
    success = t.summary()

    # Print platform overview
    print("\n📋 PLATFORM OVERVIEW:")
    print(f"  Agents: {', '.join(gateway.registry.list_agents())}")
    print(f"  Skills: {stats['total_skills']}")
    print(f"  Collaboration Scenarios: {len(gateway.collaboration.SCENARIOS)}")
    print(f"  Audit Entries: {gateway.audit.get_stats()['total']}")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
