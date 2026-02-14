#!/usr/bin/env python3
"""
MRARFAI P2-04 — Deep Agents 0.4.1 集成适配器
==============================================
基于 LangChain deepagents 0.4.1 (2026.2):
  - create_deep_agent() → LangGraph compiled graph
  - 子Agent生成 (subagent spawning)
  - 规划工具 (todo_write)
  - 文件系统后端 (filesystem backend)

参考: https://docs.langchain.com/oss/python/deepagents/overview
      https://blog.langchain.com/building-multi-agent-applications-with-deep-agents/

安装: pip install deepagents>=0.4.1
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mrarfai.deep_agent")

# ============================================================
# Safe import — deepagents 0.4.1
# ============================================================
try:
    from deepagents import create_deep_agent
    HAS_DEEP_AGENTS = True
except ImportError:
    HAS_DEEP_AGENTS = False
    logger.info("deepagents not installed — Deep Agent features disabled")

try:
    from langchain.chat_models import init_chat_model
    HAS_LANGCHAIN_CHAT = True
except ImportError:
    HAS_LANGCHAIN_CHAT = False


# ============================================================
# MRARFAI Domain Subagents
# ============================================================

def _create_domain_subagents() -> List[Dict[str, Any]]:
    """创建7个MRARFAI域子Agent定义"""
    domains = [
        {
            "name": "sales-analyst",
            "description": "销售数据分析 — 营收趋势、客户分级、价量分解",
            "system_prompt": (
                "你是禾苗科技资深销售数据分析师。分析时必须包含: "
                "1) 数据支撑 2) 同比环比变化 3) 关键发现 4) 建议行动。"
                "使用中文回答，数字精确到万元。"
            ),
        },
        {
            "name": "procurement-expert",
            "description": "采购分析 — 供应商报价对比、交期追踪、成本优化",
            "system_prompt": "你是禾苗科技采购域专家。分析供应商报价、交付延迟和成本分解。",
        },
        {
            "name": "quality-inspector",
            "description": "品质管控 — 良率监控、退货追踪、根因分析",
            "system_prompt": "你是禾苗科技品质管控专家。分析产线良率、退货原因和改善措施。",
        },
        {
            "name": "finance-analyst",
            "description": "财务分析 — 应收账款、毛利分析、现金流",
            "system_prompt": "你是禾苗科技财务分析师。分析AR账龄、产品毛利和现金流状况。",
        },
        {
            "name": "market-intelligence",
            "description": "市场情报 — 竞品监控、行业趋势、市场份额",
            "system_prompt": "你是禾苗科技市场情报专家。分析竞争对手动态和行业趋势。",
        },
        {
            "name": "risk-controller",
            "description": "风险管控 — 异常检测、客户健康评分、流失预警",
            "system_prompt": "你是禾苗科技风控专家。识别异常波动、评估客户健康度和流失风险。",
        },
        {
            "name": "strategist",
            "description": "战略顾问 — 行业对标、预测建模、CEO备忘录",
            "system_prompt": "你是禾苗科技首席战略顾问。提供行业对标、营收预测和战略建议。",
        },
    ]
    return domains


# ============================================================
# Deep Agent Factory
# ============================================================

MRARFAI_DEEP_SYSTEM_PROMPT = """你是 MRARFAI V10.0 深度分析Agent — 禾苗科技 (SPROCOMM 01401.HK) 的AI首席分析师。

你的能力:
- PLANNING: 使用 todo_write 工具将复杂分析任务分解为子步骤
- DELEGATION: 使用 call_subagent 委派给7个域专家子Agent
- DOCUMENTATION: 使用文件系统保存分析中间结果和最终报告
- SYNTHESIS: 综合多域分析结果，生成CEO级别的决策建议

分析流程:
1. 理解用户问题，创建分析计划 (todo_write)
2. 委派给相关域子Agent进行专项分析
3. 收集各子Agent结果，交叉验证
4. 综合生成结构化分析报告

风格: 数据驱动 · 中文回答 · 先结论后数据 · 包含行动建议
"""


def create_mrarfai_deep_agent(
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    custom_tools: List[Any] = None,
) -> Any:
    """
    创建 MRARFAI Deep Agent — 带子Agent生成 + 规划 + 文件系统
    
    Returns:
        LangGraph compiled graph (可直接 .invoke() 或 .stream())
        如果 deepagents 未安装，返回 None
    """
    if not HAS_DEEP_AGENTS:
        logger.warning("deepagents>=0.4.1 not installed. Install: pip install deepagents>=0.4.1")
        return None
    
    if not HAS_LANGCHAIN_CHAT:
        logger.warning("langchain chat_models not available")
        return None

    # 初始化模型
    model = init_chat_model(model_name)
    
    # 域子Agents
    subagents = _create_domain_subagents()
    
    # 自定义工具 (MCP tools等)
    tools = custom_tools or []
    
    # 创建 Deep Agent (StateBackend 为默认后端，无需显式传入)
    agent = create_deep_agent(
        model=model,
        tools=tools,
        subagents=subagents,
        system_prompt=MRARFAI_DEEP_SYSTEM_PROMPT,
    )
    
    logger.info(f"✅ MRARFAI Deep Agent created — model={model_name}, subagents={len(subagents)}")
    return agent


def run_deep_analysis(query: str, agent=None) -> Dict[str, Any]:
    """
    执行深度分析 — 自动规划 + 子Agent委派 + 结果综合
    
    Args:
        query: 用户问题 (如 "综合分析Samsung客户的风险和机会")
        agent: Deep Agent实例 (如果为None则自动创建)
    
    Returns:
        {
            "answer": str,        # 综合分析结果
            "plan": list,         # 执行计划
            "delegations": list,  # 子Agent调用记录
            "files": list,        # 生成的文件
        }
    """
    if agent is None:
        agent = create_mrarfai_deep_agent()
    
    if agent is None:
        return {
            "answer": "Deep Agents功能不可用。请安装: pip install deepagents>=0.4.1",
            "plan": [],
            "delegations": [],
            "files": [],
        }
    
    # 执行
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    # 提取结果
    messages = result.get("messages", [])
    answer = ""
    delegations = []
    
    for msg in messages:
        if hasattr(msg, "content") and msg.type == "ai":
            answer = msg.content
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc.get("name") == "call_subagent":
                    delegations.append(tc.get("args", {}))
    
    return {
        "answer": answer,
        "plan": [],  # extracted from todo_write calls
        "delegations": delegations,
        "files": [],
    }


# ============================================================
# Integration with multi_agent.py
# ============================================================

def integrate_with_stategraph(state_graph_builder):
    """
    将 Deep Agent 作为 StateGraph 的一个节点集成
    
    用法:
        from deep_agent_adapter import integrate_with_stategraph
        builder = StateGraph(GraphState)
        integrate_with_stategraph(builder)
    """
    if not HAS_DEEP_AGENTS:
        return
    
    def deep_analysis_node(state):
        """StateGraph节点: 深度分析 (复杂多步任务)"""
        query = state.get("query", "")
        result = run_deep_analysis(query)
        return {
            "deep_analysis": result["answer"],
            "deep_delegations": result["delegations"],
        }
    
    state_graph_builder.add_node("deep_analysis", deep_analysis_node)
    logger.info("✅ Deep Agent node integrated into StateGraph")


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if HAS_DEEP_AGENTS:
        print(f"✅ deepagents {__import__('deepagents').__version__} installed")
        agent = create_mrarfai_deep_agent()
        if agent:
            print("✅ MRARFAI Deep Agent created successfully")
            print(f"   Subagents: {len(_create_domain_subagents())}")
    else:
        print("❌ deepagents not installed")
        print("   Install: pip install deepagents>=0.4.1")
