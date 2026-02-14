#!/usr/bin/env python3
"""
MRARFAI MCP Server v7.0
========================
v5.0 自建 JSON-RPC → v7.0 官方 MCP Python SDK

升级要点:
  ① 官方 mcp SDK 替代手写 JSON-RPC 协议层
  ② Streamable HTTP transport (支持 AWS Lambda 部署)
  ③ Tool annotations (readOnlyHint, destructiveHint)
  ④ Structured output (JSON + text 混合返回)
  ⑤ 兼容 Claude Desktop / Cursor / ChatGPT / VS Code

使用方法:
  python mcp_server_v7.py                # stdio 模式 (Claude Desktop)
  python mcp_server_v7.py --http 8080    # HTTP 模式 (远程调用)

Claude Desktop 配置 (claude_desktop_config.json):
  {
    "mcpServers": {
      "mrarfai-sales": {
        "command": "python",
        "args": ["/path/to/mcp_server_v7.py"]
      }
    }
  }
"""

import json
import sys
import logging
import asyncio
from typing import Any

logger = logging.getLogger("mrarfai.mcp_v7")

# ============================================================
# MCP SDK 导入
# ============================================================
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool, TextContent, Resource, ResourceTemplate,
        Prompt, PromptMessage, PromptArgument,
        GetPromptResult, ReadResourceResult,
    )
    HAS_MCP_SDK = True
except ImportError:
    HAS_MCP_SDK = False
    logger.warning("mcp SDK 未安装: pip install mcp>=1.0")

# 业务模块导入
try:
    from tool_registry import sales_tools, AGENT_TOOL_CATEGORIES
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False

# V10.0: 域 Agent 引擎
try:
    from multi_agent import get_domain_engine
    HAS_DOMAIN_ENGINES = True
except ImportError:
    HAS_DOMAIN_ENGINES = False

# V10.0: Pydantic 结构化合约
try:
    from contracts import AgentRequest, AgentResponse
    HAS_CONTRACTS = True
except ImportError:
    HAS_CONTRACTS = False

# ============================================================
# Server 实例
# ============================================================

SERVER_NAME = "mrarfai-sales"
SERVER_VERSION = "10.0.0"


# ============================================================
# V10.0: MCP Tasks — 异步任务管理
# ============================================================

import uuid
from enum import Enum


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MCPTask:
    """MCP 异步任务"""

    def __init__(self, task_type: str, params: dict):
        self.id = str(uuid.uuid4())[:8]
        self.type = task_type
        self.params = params
        self.state = TaskState.PENDING
        self.progress: float = 0.0
        self.result: Any = None
        self.error: str = ""
        self._future = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.id,
            "type": self.type,
            "state": self.state.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }


# 全局任务存储
_task_store: dict[str, MCPTask] = {}


async def _run_task_async(task: MCPTask):
    """后台执行任务"""
    task.state = TaskState.RUNNING
    task.progress = 0.1

    try:
        if task.type == "comprehensive_report":
            # 综合报告 — 调用全部域 Agent
            if HAS_DOMAIN_ENGINES:
                from multi_agent import get_domain_engine
                results = {}
                engines = ["quality", "market", "finance", "procurement", "risk", "strategist"]
                for i, name in enumerate(engines):
                    engine = get_domain_engine(name)
                    if engine:
                        question = task.params.get("question", "综合分析")
                        results[name] = engine.answer(question)
                    task.progress = 0.1 + (0.8 * (i + 1) / len(engines))
                task.result = results
            else:
                task.result = {"error": "域引擎不可用"}

        elif task.type == "domain_analysis":
            # 单域分析
            domain = task.params.get("domain", "quality")
            question = task.params.get("question", "")
            if HAS_DOMAIN_ENGINES:
                from multi_agent import get_domain_engine
                engine = get_domain_engine(domain)
                if engine:
                    task.progress = 0.5
                    task.result = engine.answer(question)
                else:
                    task.result = {"error": f"域引擎 {domain} 不可用"}
            else:
                task.result = {"error": "域引擎模块不可用"}

        task.state = TaskState.COMPLETED
        task.progress = 1.0

    except Exception as e:
        task.state = TaskState.FAILED
        task.error = str(e)
        logger.error(f"Task {task.id} failed: {e}")

if HAS_MCP_SDK:
    server = Server(SERVER_NAME)

    # ============================================================
    # Tools — 销售分析工具
    # ============================================================

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """列出所有可用工具"""
        tools = [
            Tool(
                name="query_sales_data",
                description="查询禾苗通讯销售数据。支持：总营收、客户排名、区域分布、产品结构、月度趋势等",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "自然语言查询，如：'Top 5客户是谁' 或 'Q3营收多少'"
                        },
                        "dimensions": {
                            "type": "array",
                            "items": {"type": "string", "enum": [
                                "overview", "customers", "risks", "growth",
                                "price_volume", "regions", "categories",
                                "benchmark", "forecast",
                            ]},
                            "description": "可选：指定查询维度"
                        },
                    },
                    "required": ["question"],
                },
                # v7.0: Tool annotations
                # annotations={"readOnlyHint": True, "openWorldHint": False},
            ),
            Tool(
                name="analyze_customer",
                description="深度分析指定客户：月度趋势、同比环比、风险评估、健康度评分",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "customer_name": {
                            "type": "string",
                            "description": "客户名称，如 'HMD', 'Samsung'"
                        },
                    },
                    "required": ["customer_name"],
                },
            ),
            Tool(
                name="detect_anomalies",
                description="运行统计异常检测：Z-Score、IQR、趋势断裂、波动率、系统性风险",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "threshold": {
                            "type": "number",
                            "description": "Z-Score 阈值 (默认 2.0)",
                            "default": 2.0,
                        },
                    },
                },
            ),
            Tool(
                name="run_forecast",
                description="运行销售预测：总营收预测、客户级预测、品类预测、风险场景模拟",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "horizon": {
                            "type": "string",
                            "enum": ["Q1", "H1", "FY"],
                            "description": "预测时间范围",
                            "default": "FY",
                        },
                    },
                },
            ),
            Tool(
                name="generate_report",
                description="生成CEO级综合分析报告（PDF），包含全部12维度分析",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["brief", "full"],
                            "default": "brief",
                        },
                    },
                },
            ),
            # ── V10.0: 6个域 Agent 工具 ──
            Tool(
                name="quality_analysis",
                description="品质分析：良率监控、退货分析、缺陷追溯、投诉分类",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "品质相关问题，如 '当前良率是多少' 或 '主要缺陷类型'"},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="market_analysis",
                description="市场分析：竞对监控(华勤/闻泰/龙旗)、行业趋势、市场情绪",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "市场相关问题，如 '华勤最新动态' 或 '行业趋势'"},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="finance_analysis",
                description="财务分析：应收账款追踪、毛利分析、现金流预测、发票匹配",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "财务相关问题，如 '应收账款总额' 或 '毛利最高的产品'"},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="procurement_analysis",
                description="采购分析：供应商比价、采购单追踪、延期预警、成本分析",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "采购相关问题，如 '供应商评级' 或 '延期订单'"},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="risk_analysis",
                description="风控分析：客户流失预警、异常检测、健康评分、综合风险评估",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "风控相关问题，如 '高风险客户' 或 'Samsung异常检测'"},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="strategy_analysis",
                description="战略分析：行业对标、营收预测、战略建议、增长机会识别",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "战略相关问题，如 '行业对标分析' 或 '2026预测'"},
                    },
                    "required": ["question"],
                },
            ),
            # ── V10.0: MCP Tasks 异步原语 ──
            Tool(
                name="create_task",
                description="创建异步任务。支持 comprehensive_report（综合报告）和 domain_analysis（单域分析）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["comprehensive_report", "domain_analysis"],
                            "description": "任务类型",
                        },
                        "question": {"type": "string", "description": "分析问题"},
                        "domain": {
                            "type": "string",
                            "enum": ["quality", "market", "finance", "procurement", "risk", "strategist"],
                            "description": "域（仅 domain_analysis 时需要）",
                        },
                    },
                    "required": ["task_type", "question"],
                },
            ),
            Tool(
                name="get_task",
                description="查询异步任务状态和结果",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "任务ID"},
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="cancel_task",
                description="取消异步任务",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "任务ID"},
                    },
                    "required": ["task_id"],
                },
            ),
        ]
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """执行工具调用"""
        try:
            if name == "query_sales_data":
                from multi_agent import get_smart_query
                sq = get_smart_query()
                if sq:
                    result = sq.query_smart(arguments["question"])
                else:
                    result = json.dumps({"error": "数据未加载"}, ensure_ascii=False)
                return [TextContent(type="text", text=result)]

            elif name == "analyze_customer":
                customer = arguments["customer_name"]
                from multi_agent import get_smart_query
                sq = get_smart_query()
                if sq:
                    plan = {
                        "dimensions": ["customers", "risks", "price_volume"],
                        "filters": {"customer_name": customer},
                        "limit": 10,
                    }
                    result = sq.query_by_plan(plan)
                else:
                    result = json.dumps({"error": "数据未加载"}, ensure_ascii=False)
                return [TextContent(type="text", text=result)]

            elif name == "detect_anomalies":
                threshold = arguments.get("threshold", 2.0)
                from multi_agent import get_smart_query
                sq = get_smart_query()
                if sq:
                    plan = {"dimensions": ["risks"], "filters": {}, "limit": 20}
                    result = sq.query_by_plan(plan)
                else:
                    result = json.dumps({"error": "数据未加载"}, ensure_ascii=False)
                return [TextContent(type="text", text=result)]

            elif name == "run_forecast":
                from multi_agent import get_smart_query
                sq = get_smart_query()
                if sq:
                    plan = {"dimensions": ["forecast"], "filters": {}, "limit": 10}
                    result = sq.query_by_plan(plan)
                else:
                    result = json.dumps({"error": "数据未加载"}, ensure_ascii=False)
                return [TextContent(type="text", text=result)]

            elif name == "generate_report":
                return [TextContent(type="text", text="报告生成功能请通过 Streamlit UI 使用")]

            # ── V10.0: 6个域 Agent 工具 ──
            elif name in ("quality_analysis", "market_analysis", "finance_analysis",
                          "procurement_analysis", "risk_analysis", "strategy_analysis"):
                engine_map = {
                    "quality_analysis": "quality",
                    "market_analysis": "market",
                    "finance_analysis": "finance",
                    "procurement_analysis": "procurement",
                    "risk_analysis": "risk",
                    "strategy_analysis": "strategist",
                }
                engine_name = engine_map[name]
                if HAS_DOMAIN_ENGINES:
                    engine = get_domain_engine(engine_name)
                    if engine:
                        result = engine.answer(arguments["question"])
                        return [TextContent(type="text", text=result)]
                    else:
                        return [TextContent(type="text", text=f"域引擎 {engine_name} 未加载")]
                else:
                    return [TextContent(type="text", text="域引擎模块不可用")]

            # ── V10.0: MCP Tasks 异步原语 ──
            elif name == "create_task":
                task_type = arguments["task_type"]
                params = {
                    "question": arguments.get("question", ""),
                    "domain": arguments.get("domain", ""),
                }
                task = MCPTask(task_type, params)
                _task_store[task.id] = task
                # 启动后台执行
                asyncio.create_task(_run_task_async(task))
                return [TextContent(
                    type="text",
                    text=json.dumps({"task_id": task.id, "state": "pending",
                                     "message": f"任务已创建: {task_type}"},
                                    ensure_ascii=False),
                )]

            elif name == "get_task":
                task_id = arguments["task_id"]
                task = _task_store.get(task_id)
                if not task:
                    return [TextContent(type="text",
                                        text=json.dumps({"error": f"任务 {task_id} 不存在"},
                                                        ensure_ascii=False))]
                return [TextContent(type="text",
                                    text=json.dumps(task.to_dict(), ensure_ascii=False, default=str))]

            elif name == "cancel_task":
                task_id = arguments["task_id"]
                task = _task_store.get(task_id)
                if not task:
                    return [TextContent(type="text",
                                        text=json.dumps({"error": f"任务 {task_id} 不存在"},
                                                        ensure_ascii=False))]
                if task.state in (TaskState.PENDING, TaskState.RUNNING):
                    task.state = TaskState.CANCELLED
                    return [TextContent(type="text",
                                        text=json.dumps({"task_id": task_id, "state": "cancelled"},
                                                        ensure_ascii=False))]
                return [TextContent(type="text",
                                    text=json.dumps({"task_id": task_id, "state": task.state.value,
                                                     "message": "任务已完成，无法取消"},
                                                    ensure_ascii=False))]

            else:
                return [TextContent(type="text", text=f"未知工具: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"工具执行错误: {e}")]

    # ============================================================
    # Resources — 数据资源
    # ============================================================

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="mrarfai://data/overview",
                name="销售总览",
                description="禾苗通讯年度销售总览数据",
                mimeType="application/json",
            ),
            Resource(
                uri="mrarfai://data/customers",
                name="客户列表",
                description="全部客户 ABC 分级数据",
                mimeType="application/json",
            ),
            Resource(
                uri="mrarfai://data/risks",
                name="风险预警",
                description="客户流失预警和异常检测结果",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        from multi_agent import get_smart_query
        sq = get_smart_query()
        if not sq:
            return json.dumps({"error": "数据未加载"}, ensure_ascii=False)

        resource_map = {
            "mrarfai://data/overview": {"dimensions": ["overview"], "filters": {}, "limit": 10},
            "mrarfai://data/customers": {"dimensions": ["customers"], "filters": {}, "limit": 30},
            "mrarfai://data/risks": {"dimensions": ["risks"], "filters": {}, "limit": 20},
        }

        plan = resource_map.get(uri)
        if plan:
            return sq.query_by_plan(plan)
        return json.dumps({"error": f"未知资源: {uri}"}, ensure_ascii=False)

    # ============================================================
    # Prompts — 预设分析模板
    # ============================================================

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name="ceo_report",
                description="生成 CEO 级别的综合分析报告",
                arguments=[
                    PromptArgument(name="focus", description="关注重点", required=False),
                ],
            ),
            Prompt(
                name="risk_alert",
                description="生成风险预警简报",
                arguments=[],
            ),
            Prompt(
                name="growth_opportunities",
                description="识别增长机会和战略建议",
                arguments=[],
            ),
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
        if name == "ceo_report":
            focus = (arguments or {}).get("focus", "")
            return GetPromptResult(
                description="CEO 综合分析报告",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"请生成禾苗通讯CEO级综合分析报告，包含营收总览、客户分析、风险预警、增长机会和行动建议。{f'重点关注：{focus}' if focus else ''}"
                        ),
                    ),
                ],
            )
        elif name == "risk_alert":
            return GetPromptResult(
                description="风险预警简报",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text="请分析禾苗通讯当前面临的所有风险，按严重程度排序，给出应对建议。"
                        ),
                    ),
                ],
            )
        elif name == "growth_opportunities":
            return GetPromptResult(
                description="增长机会分析",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text="请识别禾苗通讯的所有增长机会，评估潜力，给出优先级排序和具体行动计划。"
                        ),
                    ),
                ],
            )


# ============================================================
# V10.0: MCP Registry — 工具发现与注册
# ============================================================

MCP_REGISTRY_MANIFEST = {
    "name": SERVER_NAME,
    "version": SERVER_VERSION,
    "description": "禾苗通讯 AI 决策平台 — 销售分析、品质监控、财务追踪、采购管理、风险预警、战略顾问",
    "author": "禾苗科技 (Sprocomm Technologies)",
    "homepage": "https://github.com/sprocomm/mrarfai",
    "license": "MIT",
    "categories": ["analytics", "business-intelligence", "manufacturing", "ai-agent"],
    "tags": [
        "sales-analysis", "quality-control", "finance", "procurement",
        "risk-management", "strategy", "multi-agent", "enterprise",
        "chinese", "manufacturing", "odm",
    ],
    "transport": ["stdio", "streamable-http"],
    "tools_count": 14,
    "tools_summary": [
        {"name": "query_sales_data", "category": "analytics"},
        {"name": "analyze_customer", "category": "analytics"},
        {"name": "detect_anomalies", "category": "risk"},
        {"name": "run_forecast", "category": "strategy"},
        {"name": "generate_report", "category": "reporting"},
        {"name": "quality_analysis", "category": "quality"},
        {"name": "market_analysis", "category": "market"},
        {"name": "finance_analysis", "category": "finance"},
        {"name": "procurement_analysis", "category": "procurement"},
        {"name": "risk_analysis", "category": "risk"},
        {"name": "strategy_analysis", "category": "strategy"},
        {"name": "create_task", "category": "task-management"},
        {"name": "get_task", "category": "task-management"},
        {"name": "cancel_task", "category": "task-management"},
    ],
    "resources_count": 3,
    "prompts_count": 3,
    "capabilities": {
        "async_tasks": True,
        "domain_agents": 6,
        "pydantic_contracts": True,
        "langfuse_observability": True,
    },
    "installation": {
        "stdio": {
            "command": "python",
            "args": ["mcp_server_v7.py"],
        },
        "http": {
            "command": "python",
            "args": ["mcp_server_v7.py", "--http", "8080"],
        },
    },
    # V10.1: MCP Registry v0.1 新增字段
    "packages": {
        "pypi": {
            "name": "mrarfai",
            "version": "10.1.0",
            "install": "pip install mrarfai",
        },
    },
    "is_verified": False,
    "source_url": "https://github.com/sprocomm/mrarfai",
    "repository": {
        "url": "https://github.com/sprocomm/mrarfai",
        "source": "github",
    },
}


def get_registry_manifest() -> dict:
    """返回 MCP Registry 注册清单 — 用于工具发现"""
    return MCP_REGISTRY_MANIFEST


def register_to_registry(registry_url: str = "https://registry.modelcontextprotocol.io") -> dict:
    """
    注册到 MCP 官方 Registry (v0.1 API Freeze)
    参考: blog.modelcontextprotocol.io/posts/2025-09-08

    调用方式:
        result = register_to_registry()
        # 或自定义 registry
        result = register_to_registry("https://custom-registry.example.com")

    Returns:
        {"status": "registered", ...} 或 {"status": "error", ...}
    """
    import requests as req

    manifest = get_registry_manifest()
    try:
        resp = req.post(
            f"{registry_url}/v0/servers",    # v0.1 API
            json=manifest,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "mrarfai/10.1",
            },
            timeout=15,
        )
        if resp.status_code in (200, 201):
            logger.info(f"✅ MCP Registry 注册成功: {registry_url}")
            return {"status": "registered", "registry": registry_url, "response": resp.json()}
        else:
            logger.warning(f"MCP Registry 注册返回 {resp.status_code}: {resp.text[:200]}")
            return {"status": "error", "code": resp.status_code, "detail": resp.text[:200]}
    except Exception as e:
        logger.warning(f"MCP Registry 注册失败: {e}")
        return {"status": "error", "detail": str(e),
                "note": "可手动提交到 https://registry.modelcontextprotocol.io 或使用 mcp-publisher CLI 工具"}


# ============================================================
# 入口
# ============================================================

async def main():
    if not HAS_MCP_SDK:
        print("❌ 请先安装 MCP SDK: pip install mcp>=1.0")
        sys.exit(1)

    if "--http" in sys.argv:
        # HTTP 模式 (Streamable HTTP)
        port = 8080
        try:
            idx = sys.argv.index("--http")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            pass

        try:
            from mcp.server.streamable_http import streamable_http_server
            async with streamable_http_server(server, host="0.0.0.0", port=port) as (r, w):
                print(f"🚀 MRARFAI MCP v7.0 HTTP @ http://0.0.0.0:{port}")
                await asyncio.Event().wait()
        except ImportError:
            print("❌ Streamable HTTP 需要额外依赖: pip install 'mcp[http]'")

    else:
        # stdio 模式 (Claude Desktop / Cursor)
        print(f"🚀 MRARFAI MCP v7.0 (stdio)", file=sys.stderr)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)


if __name__ == "__main__":
    if "--manifest" in sys.argv:
        # 输出 Registry 清单 (供手动提交)
        print(json.dumps(get_registry_manifest(), indent=2, ensure_ascii=False))
    elif "--register" in sys.argv:
        # 自动注册到 MCP Registry
        url = "https://registry.modelcontextprotocol.io"
        try:
            idx = sys.argv.index("--register")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
                url = sys.argv[idx + 1]
        except (ValueError, IndexError):
            pass
        result = register_to_registry(url)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        asyncio.run(main())
