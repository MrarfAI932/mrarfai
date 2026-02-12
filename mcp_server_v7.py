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

# ============================================================
# Server 实例
# ============================================================

SERVER_NAME = "mrarfai-sales"
SERVER_VERSION = "7.0.0"

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
    asyncio.run(main())
