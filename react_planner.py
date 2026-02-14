#!/usr/bin/env python3
"""
MRARFAI P2-05 — ReAct + Planning 模式集成
============================================
基于 LangGraph 1.0 实现:
  - ReAct Loop: Reason → Act → Observe → Reason...
  - Hierarchical Planning: Plan → Execute → Verify
  - Tool-Augmented Reasoning: 动态工具选择

参考: Yao et al. "ReAct: Synergizing Reasoning and Acting" (2023)
      Wei et al. "Chain-of-Thought Prompting" (2022)
      LangGraph 1.0 create_react_agent()
"""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger("mrarfai.react")

# ============================================================
# LangGraph 1.0 ReAct Agent
# ============================================================
try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    HAS_REACT = True
except ImportError:
    HAS_REACT = False
    logger.info("langgraph.prebuilt not available — ReAct disabled")


# ============================================================
# Planning Layer — 任务分解器
# ============================================================

@dataclass
class PlanStep:
    """计划步骤"""
    step_id: int
    description: str
    agent: str                # 负责的Agent域
    tool: Optional[str]       # 需要的工具
    depends_on: List[int]     # 依赖的步骤
    status: str = "pending"   # pending / running / done / failed
    result: Optional[str] = None


class HierarchicalPlanner:
    """
    层级规划器 — 将复杂问题分解为可执行的子计划 (独立模块)

    注意: multi_agent.py 内置了轻量级 QueryPlanner (返回 dict)，
    本 HierarchicalPlanner 是功能更丰富的替代方案 (返回 PlanStep DAG)。
    两者接口不同:
      - QueryPlanner.create_plan() → dict (phases 列表)
      - HierarchicalPlanner.create_plan() → List[PlanStep] (DAG结构)

    Pipeline:
      1. 问题理解 → 识别涉及的域和工具
      2. 任务分解 → 生成DAG结构的执行计划
      3. 依赖分析 → 确定并行/串行执行顺序
      4. 执行控制 → 按计划调度Agent
    """
    
    # 域 → 关键词映射 (与 AGENT_PROFILES 对齐)
    DOMAIN_KEYWORDS = {
        "analyst": ["营收", "出货", "客户", "分级", "ABC", "趋势", "同比", "环比"],
        "procurement": ["采购", "供应商", "报价", "交期", "成本", "PO"],
        "quality": ["品质", "良率", "退货", "根因", "缺陷", "产线"],
        "finance": ["应收", "毛利", "现金流", "账龄", "发票", "利润"],
        "market": ["竞品", "市场", "行业", "份额", "竞争", "趋势"],
        "risk": ["风险", "异常", "预警", "流失", "健康评分", "波动"],
        "strategist": ["战略", "预测", "对标", "CEO", "建议", "规划"],
    }

    def __init__(self):
        self.plans: Dict[str, List[PlanStep]] = {}

    def analyze_domains(self, query: str) -> List[str]:
        """识别查询涉及的域"""
        q = query.lower()
        domains = []
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                domains.append(domain)
        # 默认至少包含 analyst
        if not domains:
            domains = ["analyst"]
        return domains

    def create_plan(self, query: str) -> List[PlanStep]:
        """
        根据查询创建执行计划
        
        Returns:
            按依赖拓扑排序的PlanStep列表
        """
        domains = self.analyze_domains(query)
        steps = []
        step_id = 0
        
        # Phase 1: 数据收集 (并行)
        data_steps = []
        for domain in domains:
            step = PlanStep(
                step_id=step_id,
                description=f"[{domain}] 数据查询与初步分析",
                agent=domain,
                tool=f"query_{domain}_data",
                depends_on=[],
            )
            steps.append(step)
            data_steps.append(step_id)
            step_id += 1
        
        # Phase 2: 交叉分析 (依赖Phase1)
        if len(domains) > 1:
            cross_step = PlanStep(
                step_id=step_id,
                description="交叉域分析 — 整合多源数据发现隐藏关联",
                agent="strategist",
                tool="cross_domain_analysis",
                depends_on=data_steps,
            )
            steps.append(cross_step)
            step_id += 1
        
        # Phase 3: 综合报告
        synthesize_step = PlanStep(
            step_id=step_id,
            description="综合分析报告生成",
            agent="strategist",
            tool="generate_report",
            depends_on=[s.step_id for s in steps],
        )
        steps.append(synthesize_step)
        
        # 缓存
        plan_key = str(hash(query))[:8]
        self.plans[plan_key] = steps
        
        logger.info(f"📋 Plan created: {len(steps)} steps, {len(domains)} domains")
        return steps

    def get_parallel_groups(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """将计划步骤分组为可并行执行的批次"""
        groups = []
        done_ids = set()
        remaining = list(steps)
        
        while remaining:
            batch = []
            for step in remaining:
                if all(dep in done_ids for dep in step.depends_on):
                    batch.append(step)
            
            if not batch:
                # 防止死循环
                batch = [remaining[0]]
            
            groups.append(batch)
            for s in batch:
                done_ids.add(s.step_id)
                remaining.remove(s)
        
        return groups


# ============================================================
# ReAct Agent — Reason + Act + Observe Loop
# ============================================================

REACT_SYSTEM_PROMPT = """你是 MRARFAI V10.1 ReAct Agent。

你使用 Reason-Act-Observe 循环来回答复杂问题:
1. **Reason**: 分析当前信息，决定下一步行动
2. **Act**: 调用工具获取数据或执行操作
3. **Observe**: 检查工具返回结果
4. 重复直到有足够信息生成最终回答

规则:
- 每次推理明确说明 "我接下来要做什么，为什么"
- 工具调用后必须检验结果是否合理
- 发现异常数据时自动交叉验证
- 最终回答必须数据支撑，包含行动建议
"""


def create_react_sales_agent(tools: List[Any] = None, model=None) -> Any:
    """
    创建基于 LangGraph 1.0 的 ReAct Agent
    
    Args:
        tools: LangChain Tool 列表
        model: LangChain ChatModel
    
    Returns:
        LangGraph compiled graph 或 None
    """
    if not HAS_REACT:
        logger.warning("langgraph.prebuilt not available. Install: pip install langgraph>=1.0")
        return None
    
    if model is None:
        try:
            from langchain.chat_models import init_chat_model
            model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
        except Exception:
            logger.warning("Could not init model for ReAct agent")
            return None
    
    if tools is None:
        tools = []
    
    # LangGraph 1.0 prebuilt ReAct
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=REACT_SYSTEM_PROMPT,
    )
    
    logger.info(f"✅ ReAct Agent created — tools={len(tools)}")
    return agent


# ============================================================
# StateGraph Integration — 规划节点
# ============================================================

def add_planning_nodes(builder, planner: HierarchicalPlanner = None):
    """
    向现有 StateGraph 添加规划+执行节点
    
    用法 (multi_agent.py 集成):
        from react_planner import add_planning_nodes, HierarchicalPlanner
        planner = HierarchicalPlanner()
        add_planning_nodes(builder, planner)
    """
    if planner is None:
        planner = HierarchicalPlanner()
    
    def planning_node(state):
        """规划节点: 分解复杂查询"""
        query = state.get("query", "")
        steps = planner.create_plan(query)
        groups = planner.get_parallel_groups(steps)
        
        return {
            "plan": [
                {
                    "step_id": s.step_id,
                    "description": s.description,
                    "agent": s.agent,
                    "tool": s.tool,
                    "depends_on": s.depends_on,
                }
                for s in steps
            ],
            "parallel_groups": [
                [s.step_id for s in group] for group in groups
            ],
        }
    
    try:
        builder.add_node("plan", planning_node)
        logger.info("✅ Planning node added to StateGraph")
    except Exception as e:
        logger.warning(f"Could not add planning node: {e}")


# ============================================================
# Export
# ============================================================

__all__ = [
    "HierarchicalPlanner",
    "PlanStep",
    "create_react_sales_agent",
    "add_planning_nodes",
    "HAS_REACT",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    planner = HierarchicalPlanner()
    
    # Test 1: 单域查询
    steps = planner.create_plan("Samsung客户今年的出货趋势如何？")
    print(f"\n📋 单域计划: {len(steps)} 步")
    for s in steps:
        print(f"  Step {s.step_id}: [{s.agent}] {s.description} (依赖: {s.depends_on})")
    
    # Test 2: 多域查询
    steps = planner.create_plan("综合分析Samsung的采购成本、品质问题和应收账款风险")
    groups = planner.get_parallel_groups(steps)
    print(f"\n📋 多域计划: {len(steps)} 步, {len(groups)} 并行批次")
    for i, group in enumerate(groups):
        print(f"  Batch {i}: {[s.description for s in group]}")
    
    # Test 3: ReAct
    if HAS_REACT:
        agent = create_react_sales_agent()
        print(f"\n✅ ReAct Agent: {'ready' if agent else 'failed'}")
    else:
        print("\n⬜ ReAct: langgraph.prebuilt not available")
