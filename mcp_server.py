#!/usr/bin/env python3
"""
MRARFAI MCP Server v1.0
========================
将禾苗销售分析工具暴露为 MCP (Model Context Protocol) 标准服务

支持两种运行模式：
  1. stdio 模式 — 直接接入 Claude Desktop / Cursor
  2. HTTP 模式 — Streamable HTTP 供远程客户端调用

遵循 MCP 2025-06-18 规范:
  - JSON-RPC 2.0 协议
  - Tools / Resources / Prompts 三原语
  - readOnly / destructive 工具属性
  - Structured Tool Output

使用方法:
  python mcp_server.py --stdio       # Claude Desktop / Cursor
  python mcp_server.py --http        # HTTP 远程调用
  python mcp_server.py --fastmcp     # FastMCP 框架

  # Claude Desktop 配置 (claude_desktop_config.json):
  {
    "mcpServers": {
      "mrarfai-sales": {
        "command": "python",
        "args": ["/path/to/mcp_server.py", "--stdio"]
      }
    }
  }
"""

import json
import sys
import time
import logging
from typing import Any, Dict, List, Optional

try:
    from tool_registry import sales_tools, AGENT_TOOL_CATEGORIES
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False

logger = logging.getLogger("mrarfai.mcp")

MCP_VERSION = "2025-06-18"
SERVER_NAME = "mrarfai-sales"
SERVER_VERSION = "1.0.0"

SERVER_CAPABILITIES = {
    "tools": {"listChanged": False},
    "resources": {"subscribe": False, "listChanged": False},
    "prompts": {"listChanged": False},
}

# ============================================================
# JSON-RPC 2.0
# ============================================================

def jsonrpc_response(id, result):
    return {"jsonrpc": "2.0", "id": id, "result": result}

def jsonrpc_error(id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}

ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603

# ============================================================
# MCP Resources
# ============================================================

RESOURCES = [
    {
        "uri": "mrarfai://tools/catalog",
        "name": "MRARFAI 工具目录",
        "description": "所有可用销售分析工具的完整列表",
        "mimeType": "application/json",
    },
    {
        "uri": "mrarfai://config/agent-mapping",
        "name": "Agent-Tool 权限映射",
        "description": "各Agent可调用的工具类别配置",
        "mimeType": "application/json",
    },
    {
        "uri": "mrarfai://help/sales-analysis",
        "name": "销售分析指南",
        "description": "禾苗ODM销售数据分析最佳实践",
        "mimeType": "text/plain",
    },
]

def read_resource(uri):
    if uri == "mrarfai://tools/catalog":
        catalog = []
        if HAS_TOOLS:
            for name, td in sales_tools._tools.items():
                catalog.append({
                    "name": name, "category": td.category,
                    "description": td.description, "parameters": td.parameters,
                    "read_only": td.read_only,
                })
        return {"contents": [{"uri": uri, "mimeType": "application/json",
                "text": json.dumps(catalog, ensure_ascii=False, indent=2)}]}

    elif uri == "mrarfai://config/agent-mapping":
        return {"contents": [{"uri": uri, "mimeType": "application/json",
                "text": json.dumps(AGENT_TOOL_CATEGORIES, ensure_ascii=False, indent=2)}]}

    elif uri == "mrarfai://help/sales-analysis":
        guide = (
            "# 禾苗ODM销售分析指南\n\n"
            "## 常用场景\n"
            "1. 同比/环比增长 — calc_yoy_growth / calc_mom_growth\n"
            "2. 客户集中度 — calc_concentration (HHI指数)\n"
            "3. 流失风险 — detect_churn_risk / scan_all_risks\n"
            "4. 产品BCG — analyze_product_mix\n"
            "5. 月度趋势 — analyze_monthly_trend\n\n"
            "数据格式：营收万元，月度1-12月\n"
        )
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": guide}]}
    return None

# ============================================================
# MCP Prompts
# ============================================================

PROMPTS = [
    {
        "name": "sales-overview",
        "description": "生成销售总览报告",
        "arguments": [
            {"name": "period", "description": "分析周期", "required": True},
            {"name": "focus", "description": "重点领域", "required": False},
        ],
    },
    {
        "name": "risk-alert",
        "description": "客户风险预警报告",
        "arguments": [
            {"name": "threshold", "description": "风险阈值(高/中/低)", "required": False},
        ],
    },
    {
        "name": "ceo-briefing",
        "description": "CEO月度简报",
        "arguments": [
            {"name": "month", "description": "月份", "required": True},
        ],
    },
]

def get_prompt(name, arguments=None):
    args = arguments or {}
    templates = {
        "sales-overview": lambda: {
            "description": f"{args.get('period','2024年')}销售总览",
            "messages": [{"role": "user", "content": {"type": "text", "text":
                f"请分析{args.get('period','2024年')}销售数据，重点关注{args.get('focus','整体表现')}。"
                "\n使用 calc_yoy_growth, calc_concentration, analyze_monthly_trend 辅助分析。"
                "\n输出: 关键指标→增长分析→风险提示→行动建议"}}],
        },
        "risk-alert": lambda: {
            "description": f"风险预警（阈值: {args.get('threshold','中')}）",
            "messages": [{"role": "user", "content": {"type": "text", "text":
                f"扫描所有客户，筛选{args.get('threshold','中')}级以上风险。"
                "\n使用 scan_all_risks 批量扫描 + detect_churn_risk 详细分析。"}}],
        },
        "ceo-briefing": lambda: {
            "description": f"CEO {args.get('month','本月')}简报",
            "messages": [{"role": "user", "content": {"type": "text", "text":
                f"CEO本月该关注什么？基于{args.get('month','本月')}数据生成简报。"}}],
        },
    }
    fn = templates.get(name)
    return fn() if fn else None

# ============================================================
# MCP Handler
# ============================================================

class MCPHandler:
    def __init__(self):
        self.initialized = False

    def handle(self, request):
        if request.get("jsonrpc") != "2.0":
            return jsonrpc_error(request.get("id"), ERR_INVALID_REQUEST, "Invalid jsonrpc")

        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        if req_id is None:
            if method == "notifications/initialized":
                self.initialized = True
            return None

        dispatch = {
            "initialize": self._initialize,
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
            "resources/list": lambda p: {"resources": RESOURCES},
            "resources/read": self._resources_read,
            "prompts/list": lambda p: {"prompts": PROMPTS},
            "prompts/get": self._prompts_get,
            "ping": lambda p: {},
        }

        handler = dispatch.get(method)
        if not handler:
            return jsonrpc_error(req_id, ERR_METHOD_NOT_FOUND, f"Unknown: {method}")

        try:
            return jsonrpc_response(req_id, handler(params))
        except Exception as e:
            return jsonrpc_error(req_id, ERR_INTERNAL, str(e))

    def _initialize(self, params):
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": "MRARFAI 禾苗销售智能分析工具集。提供增长率、集中度、风险、趋势等分析工具。",
        }

    def _tools_list(self, params):
        if not HAS_TOOLS:
            return {"tools": []}
        tools = []
        for name, td in sales_tools._tools.items():
            tools.append({
                "name": td.name,
                "description": td.description,
                "inputSchema": {
                    "type": "object",
                    "properties": td.parameters,
                    "required": td.required,
                },
                "annotations": {
                    "readOnlyHint": td.read_only,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            })
        return {"tools": tools}

    def _tools_call(self, params):
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if not HAS_TOOLS or tool_name not in sales_tools:
            return {"content": [{"type": "text", "text": f"未知工具: {tool_name}"}], "isError": True}

        result = sales_tools.execute(tool_name, arguments)
        if "error" in result:
            return {"content": [{"type": "text", "text": result["error"]}], "isError": True}

        result_text = json.dumps(result.get("result", result), ensure_ascii=False, indent=2, default=str)
        return {
            "content": [{"type": "text", "text": result_text}],
            "isError": False,
            "structuredContent": result.get("result", result),
        }

    def _resources_read(self, params):
        r = read_resource(params.get("uri", ""))
        if r is None:
            raise ValueError(f"Not found: {params.get('uri')}")
        return r

    def _prompts_get(self, params):
        r = get_prompt(params.get("name", ""), params.get("arguments", {}))
        if r is None:
            raise ValueError(f"Not found: {params.get('name')}")
        return r

# ============================================================
# Transport: stdio
# ============================================================

def run_stdio():
    handler = MCPHandler()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(jsonrpc_error(None, ERR_PARSE, "Parse error")) + "\n")
            sys.stdout.flush()
            continue
        response = handler.handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()

# ============================================================
# Transport: HTTP
# ============================================================

def run_http(host="0.0.0.0", port=8765):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    handler_instance = MCPHandler()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self._json(jsonrpc_error(None, ERR_PARSE, "Parse error"), 400)
                return
            resp = handler_instance.handle(req)
            if resp:
                self._json(resp)
            else:
                self.send_response(204)
                self.end_headers()

        def do_GET(self):
            self._json({"status": "ok", "server": SERVER_NAME,
                        "version": SERVER_VERSION, "protocol": MCP_VERSION,
                        "tools": len(sales_tools) if HAS_TOOLS else 0})

        def _json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a): pass

    server = HTTPServer((host, port), H)
    print(f"🌿 MRARFAI MCP Server at http://{host}:{port}")
    print(f"   Tools: {len(sales_tools) if HAS_TOOLS else 0} | Protocol: MCP {MCP_VERSION}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

# ============================================================
# FastMCP Integration
# ============================================================

def create_fastmcp_server():
    """pip install fastmcp 后使用此函数"""
    from fastmcp import FastMCP
    mcp = FastMCP(name=SERVER_NAME, instructions="MRARFAI 禾苗销售智能分析工具集")

    if HAS_TOOLS:
        for tool_name, td in sales_tools._tools.items():
            mcp.tool(name=td.name, description=td.description)(td.function)

    @mcp.resource("mrarfai://tools/catalog")
    def tools_catalog():
        return read_resource("mrarfai://tools/catalog")["contents"][0]["text"]

    @mcp.resource("mrarfai://config/agent-mapping")
    def agent_mapping():
        return read_resource("mrarfai://config/agent-mapping")["contents"][0]["text"]

    @mcp.prompt("sales-overview")
    def sales_overview(period: str, focus: str = "整体表现"):
        return get_prompt("sales-overview", {"period": period, "focus": focus})["messages"][0]["content"]["text"]

    return mcp

# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MRARFAI MCP Server")
    parser.add_argument("--stdio", action="store_true", help="stdio mode")
    parser.add_argument("--http", action="store_true", help="HTTP mode")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fastmcp", action="store_true", help="FastMCP mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if args.fastmcp:
        create_fastmcp_server().run()
    elif args.http:
        run_http(port=args.port)
    else:
        run_stdio()

if __name__ == "__main__":
    main()
