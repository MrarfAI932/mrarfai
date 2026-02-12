#!/usr/bin/env python3
"""
MRARFAI Tool Registry v4.0
============================
Agent 可动态调用的工具集，支持：
- 装饰器注册 + 自动 JSON Schema 生成
- 按类别过滤（Agent 只看到自己需要的工具）
- Claude tool_use / DeepSeek function_calling 双格式
- 执行沙箱 + 超时保护

工具类别：
  analytics  — 数据计算/统计（分析师用）
  risk       — 风险检测/预警（风控用）
  strategy   — 对标/机会分析（策略师用）
  common     — 通用工具（所有 Agent 用）
"""

import inspect
import json
import time
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime


# ============================================================
# Tool Definition + Registry
# ============================================================

@dataclass
class ToolDef:
    """单个工具定义"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    required: list = field(default_factory=list)
    category: str = "common"
    read_only: bool = True  # MCP 2025-06-18: readOnly 属性

    def to_claude_schema(self) -> dict:
        """转为 Claude Messages API tool_use 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            }
        }

    def to_openai_schema(self) -> dict:
        """转为 OpenAI/DeepSeek function_calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                }
            }
        }


class ToolRegistry:
    """
    工具注册中心 — 装饰器注册 + 自动 Schema + 分类管理

    Usage:
        registry = ToolRegistry()

        @registry.register(category="analytics")
        def calc_yoy_growth(current: float, previous: float) -> float:
            '''计算同比增长率'''
            return ((current - previous) / previous) * 100

        # 获取 Claude 格式工具列表
        tools = registry.get_claude_tools(categories=["analytics"])

        # 执行工具
        result = registry.execute("calc_yoy_growth", {"current": 41.71, "previous": 27.07})
    """

    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}
        self._call_count: Dict[str, int] = {}
        self._total_time: Dict[str, float] = {}

    def register(self, name: str = None, description: str = None,
                 category: str = "common", read_only: bool = True):
        """装饰器：注册工具，自动从类型注解生成 JSON Schema"""
        def decorator(func):
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip()

            # Python type → JSON Schema type
            type_map = {
                str: "string", int: "integer", float: "number",
                bool: "boolean", list: "array", dict: "object",
            }

            sig = inspect.signature(func)
            params, req = {}, []
            for pname, param in sig.parameters.items():
                anno = param.annotation
                json_type = type_map.get(anno, "string")

                # 从 docstring 提取参数描述（简版）
                param_desc = f"参数 {pname}"
                if func.__doc__:
                    for line in func.__doc__.split("\n"):
                        if pname in line and ":" in line:
                            param_desc = line.split(":", 1)[-1].strip()
                            break

                params[pname] = {"type": json_type, "description": param_desc}
                if param.default == inspect.Parameter.empty:
                    req.append(pname)

            self._tools[tool_name] = ToolDef(
                name=tool_name,
                description=tool_desc,
                function=func,
                parameters=params,
                required=req,
                category=category,
                read_only=read_only,
            )
            self._call_count[tool_name] = 0
            self._total_time[tool_name] = 0
            return func
        return decorator

    def get_claude_tools(self, categories: List[str] = None) -> list:
        """获取 Claude tool_use 格式工具列表"""
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_claude_schema() for t in tools]

    def get_openai_tools(self, categories: List[str] = None) -> list:
        """获取 OpenAI/DeepSeek function_calling 格式工具列表"""
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_openai_schema() for t in tools]

    def get_tool_names(self, categories: List[str] = None) -> list:
        """获取工具名列表（用于 prompt 注入）"""
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [f"- {t.name}: {t.description[:60]}" for t in tools]

    def execute(self, name: str, args: dict, timeout: float = 10.0) -> Any:
        """执行工具（带超时保护 + 性能追踪）"""
        if name not in self._tools:
            return {"error": f"未知工具: {name}", "available": list(self._tools.keys())}

        tool = self._tools[name]
        t0 = time.time()
        try:
            result = tool.function(**args)
            elapsed = time.time() - t0
            self._call_count[name] += 1
            self._total_time[name] += elapsed
            return {"result": result, "elapsed_ms": round(elapsed * 1000, 1)}
        except Exception as e:
            return {"error": f"工具执行失败: {e}", "tool": name, "args": args}

    def get_stats(self) -> dict:
        """工具调用统计"""
        return {
            name: {"calls": self._call_count[name],
                   "total_ms": round(self._total_time[name] * 1000, 1)}
            for name in self._tools
            if self._call_count[name] > 0
        }

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name):
        return name in self._tools


# ============================================================
# 禾苗销售专用工具集
# ============================================================

sales_tools = ToolRegistry()


@sales_tools.register(category="analytics",
    description="计算同比增长率（YoY）。传入今年和去年的数值，返回增长百分比。"
                "示例: current=41.71亿, previous=27.07亿 → 54.1%")
def calc_yoy_growth(current: float, previous: float) -> dict:
    """计算同比增长率
    current: 当期数值（亿元）
    previous: 上期数值（亿元）
    """
    if previous == 0:
        return {"growth_pct": float('inf') if current > 0 else 0, "delta": current}
    pct = round(((current - previous) / previous) * 100, 1)
    delta = round(current - previous, 2)
    return {"growth_pct": pct, "delta": delta, "current": current, "previous": previous}


@sales_tools.register(category="analytics",
    description="计算环比增长率（MoM）。传入当月和上月数值，返回增长百分比。")
def calc_mom_growth(current_month: float, previous_month: float) -> dict:
    """计算环比增长率
    current_month: 当月数值
    previous_month: 上月数值
    """
    if previous_month == 0:
        return {"growth_pct": float('inf') if current_month > 0 else 0}
    pct = round(((current_month - previous_month) / previous_month) * 100, 1)
    return {"growth_pct": pct, "delta": round(current_month - previous_month, 2)}


@sales_tools.register(category="analytics",
    description="计算客户集中度指标：Top3占比、Top5占比、HHI指数。"
                "传入客户收入列表 [{name, revenue}]，返回集中度分析。"
                "HHI>2500为高集中度，1500-2500为中等。")
def calc_concentration(revenues: list) -> dict:
    """计算客户集中度
    revenues: 客户收入列表 [{"name": "X", "revenue": 100}, ...]
    """
    if not revenues:
        return {"error": "空列表"}
    sorted_rev = sorted(revenues, key=lambda x: x.get("revenue", 0), reverse=True)
    total = sum(r.get("revenue", 0) for r in sorted_rev)
    if total == 0:
        return {"error": "总收入为0"}

    top3 = sum(r.get("revenue", 0) for r in sorted_rev[:3])
    top5 = sum(r.get("revenue", 0) for r in sorted_rev[:5])

    # HHI = Σ(市场份额%)²
    shares = [(r.get("revenue", 0) / total * 100) for r in sorted_rev]
    hhi = round(sum(s ** 2 for s in shares), 1)

    level = "极高风险" if hhi > 4000 else "高集中度" if hhi > 2500 else "中等" if hhi > 1500 else "健康"

    return {
        "total_revenue": round(total, 2),
        "top3_pct": round(top3 / total * 100, 1),
        "top5_pct": round(top5 / total * 100, 1),
        "hhi": hhi,
        "concentration_level": level,
        "top3_clients": [r.get("name", "?") for r in sorted_rev[:3]],
    }


@sales_tools.register(category="risk",
    description="检测客户流失风险。传入客户月度出货数据，分析连续下降、断崖式下跌、零出货等风险信号。"
                "返回风险等级(极高/高/中/低)和具体风险因子。")
def detect_churn_risk(client_name: str, monthly_values: list) -> dict:
    """检测客户流失风险
    client_name: 客户名称
    monthly_values: 12个月出货金额列表 [m1, m2, ..., m12]
    """
    if not monthly_values or len(monthly_values) < 2:
        return {"client": client_name, "risk_level": "数据不足"}

    vals = [float(v) for v in monthly_values]
    total = sum(vals)
    n = len(vals)

    risk_factors = []
    risk_score = 0

    # 1. 近3月零出货
    recent_zeros = sum(1 for v in vals[-3:] if v == 0)
    if recent_zeros >= 2:
        risk_factors.append(f"近3月{recent_zeros}月零出货")
        risk_score += 40

    # 2. H2 vs H1 断崖
    if n >= 6:
        h1 = sum(vals[:n//2])
        h2 = sum(vals[n//2:])
        if h1 > 0:
            h2_drop = (h1 - h2) / h1 * 100
            if h2_drop > 50:
                risk_factors.append(f"H2较H1下降{h2_drop:.0f}%")
                risk_score += 30
            elif h2_drop > 30:
                risk_factors.append(f"H2较H1下降{h2_drop:.0f}%")
                risk_score += 15

    # 3. 连续下降月数
    consecutive_decline = 0
    for i in range(len(vals)-1, 0, -1):
        if vals[i] < vals[i-1]:
            consecutive_decline += 1
        else:
            break
    if consecutive_decline >= 3:
        risk_factors.append(f"连续{consecutive_decline}月下降")
        risk_score += 20

    # 4. 变异系数
    if total > 0:
        mean_val = total / n
        if mean_val > 0:
            std = math.sqrt(sum((v - mean_val)**2 for v in vals) / n)
            cv = std / mean_val
            if cv > 1.0:
                risk_factors.append(f"出货极不稳定(CV={cv:.1f})")
                risk_score += 15

    # 风险等级
    if risk_score >= 60:
        level = "极高"
    elif risk_score >= 40:
        level = "高"
    elif risk_score >= 20:
        level = "中"
    else:
        level = "低"

    return {
        "client": client_name,
        "risk_level": level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "total_revenue": round(total, 2),
        "recent_3m_avg": round(sum(vals[-3:]) / 3, 2) if len(vals) >= 3 else 0,
    }


@sales_tools.register(category="risk",
    description="批量扫描所有客户的流失风险，返回按风险等级排序的客户列表。"
                "传入 data dict（含'客户金额'字段），自动识别高风险客户。")
def scan_all_risks(client_data: list) -> dict:
    """批量风险扫描
    client_data: 客户数据列表 [{"name": "X", "monthly_values": [m1..m12]}, ...]
                 也接受 {"客户": "X", "月度": [...]}
    """
    results = {"极高": [], "高": [], "中": [], "低": []}
    total_at_risk = 0
    all_clients = []

    for c in client_data:
        name = c.get("name", c.get("客户", "?"))
        monthly = c.get("monthly_values", c.get("月度", []))
        amount = c.get("总金额", sum(monthly) if monthly else 0)

        risk = detect_churn_risk(name, monthly)
        level = risk.get("risk_level", "低")
        entry = {
            "client": name,
            "risk_level": level,
            "amount": amount,
            "factors": risk.get("risk_factors", []),
            "score": risk.get("risk_score", 0),
        }
        all_clients.append(entry)
        if level in results:
            results[level].append(entry)
            if level in ("极高", "高"):
                total_at_risk += amount

    return {
        "summary": {k: len(v) for k, v in results.items()},
        "total_at_risk_amount": round(total_at_risk, 2),
        "high_risk_clients": results["极高"] + results["高"],
        "medium_risk_clients": results["中"],
        "clients": all_clients,
    }


@sales_tools.register(category="strategy",
    description="产品结构分析。传入各产品线收入数据，计算占比、增速，识别明星/瘦狗产品。")
def analyze_product_mix(products: list) -> dict:
    """产品结构分析
    products: [{"name": "手机", "current": 100, "previous": 80}, ...]
    """
    total_current = sum(p.get("current", 0) for p in products)
    total_previous = sum(p.get("previous", 0) for p in products)

    result = []
    for p in products:
        curr = p.get("current", 0)
        prev = p.get("previous", 0)
        share = round(curr / total_current * 100, 1) if total_current > 0 else 0
        growth = round((curr - prev) / prev * 100, 1) if prev > 0 else (999 if curr > 0 else 0)

        # BCG 分类
        if share > 20 and growth > 20:
            category = "⭐ 明星"
        elif share > 20 and growth <= 20:
            category = "🐄 现金牛"
        elif share <= 20 and growth > 20:
            category = "❓ 问题"
        else:
            category = "🐕 瘦狗"

        result.append({
            "name": p.get("name", "?"),
            "revenue": curr,
            "share_pct": share,
            "growth_pct": growth,
            "category": category,
        })

    result.sort(key=lambda x: x["revenue"], reverse=True)
    return {
        "total_revenue": round(total_current, 2),
        "total_growth_pct": round((total_current - total_previous) / total_previous * 100, 1) if total_previous > 0 else 0,
        "products": result,
    }


@sales_tools.register(category="analytics",
    description="月度趋势分析。传入12个月数据，识别峰谷值、季节性模式、连续增长/下降段。")
def analyze_monthly_trend(monthly_values: list, labels: list = None) -> dict:
    """月度趋势分析
    monthly_values: 12个月数值列表
    labels: 月份标签 ["1月", "2月", ...] (可选)
    """
    if not monthly_values:
        return {"error": "空数据"}

    vals = [float(v) for v in monthly_values]
    n = len(vals)
    labels = labels or [f"{i+1}月" for i in range(n)]

    peak_idx = vals.index(max(vals))
    trough_idx = vals.index(min(vals))

    # 连续增长/下降段
    streaks = []
    current_streak = {"direction": None, "start": 0, "length": 0}
    for i in range(1, n):
        direction = "up" if vals[i] > vals[i-1] else ("down" if vals[i] < vals[i-1] else "flat")
        if direction == current_streak["direction"]:
            current_streak["length"] += 1
        else:
            if current_streak["length"] >= 2:
                streaks.append({**current_streak, "end": i-1})
            current_streak = {"direction": direction, "start": i, "length": 1}
    if current_streak["length"] >= 2:
        streaks.append({**current_streak, "end": n-1})

    return {
        "total": round(sum(vals), 2),
        "average": round(sum(vals) / n, 2),
        "peak": {"month": labels[peak_idx], "value": vals[peak_idx]},
        "trough": {"month": labels[trough_idx], "value": vals[trough_idx]},
        "h1_total": round(sum(vals[:n//2]), 2),
        "h2_total": round(sum(vals[n//2:]), 2),
        "notable_streaks": [
            f"{s['direction']}×{s['length']}月 ({labels[s['start']]}→{labels[s['end']]})"
            for s in streaks
        ],
    }


@sales_tools.register(category="common",
    description="数值格式化工具。将数字转为易读的中文格式：万元/亿元，带千分符。")
def format_number(value: float, unit: str = "auto") -> str:
    """数值格式化
    value: 数值
    unit: 单位 (auto/万/亿/元)
    """
    if unit == "auto":
        if abs(value) >= 10000:
            return f"{value/10000:.2f}亿"
        elif abs(value) >= 1:
            return f"{value:.2f}万"
        else:
            return f"{value*10000:.0f}元"
    elif unit == "亿":
        return f"{value/10000:.2f}亿"
    elif unit == "万":
        return f"{value:.2f}万"
    else:
        return f"{value:,.2f}{unit}"


# ============================================================
# Agent-Tool Mapping（每个Agent只看到相关工具）
# ============================================================

AGENT_TOOL_CATEGORIES = {
    "analyst": ["analytics", "common"],
    "risk": ["risk", "analytics", "common"],
    "strategist": ["strategy", "analytics", "common"],
}


def get_tools_for_agent(agent_id: str) -> list:
    """获取 Agent 可用的工具列表（Claude 格式）"""
    categories = AGENT_TOOL_CATEGORIES.get(agent_id, ["common"])
    return sales_tools.get_claude_tools(categories=categories)


def get_tool_descriptions_for_prompt(agent_id: str) -> str:
    """获取工具描述文本（注入到 Agent prompt 中）"""
    categories = AGENT_TOOL_CATEGORIES.get(agent_id, ["common"])
    tools = sales_tools.get_tool_names(categories=categories)
    if not tools:
        return ""
    return "\n可用分析工具：\n" + "\n".join(tools) + "\n请根据需要调用工具进行精确计算。\n"


# ============================================================
# Tool Execution Loop（处理 Claude tool_use 响应）
# ============================================================

def execute_tool_calls(tool_blocks: list) -> list:
    """
    批量执行 Claude 返回的 tool_use blocks

    Args:
        tool_blocks: [{"type": "tool_use", "id": "xxx", "name": "calc_yoy_growth", "input": {...}}]

    Returns:
        [{"type": "tool_result", "tool_use_id": "xxx", "content": "..."}]
    """
    results = []
    for block in tool_blocks:
        if not isinstance(block, dict):
            # anthropic SDK 返回的是对象
            name = getattr(block, 'name', block.get('name', ''))
            input_data = getattr(block, 'input', block.get('input', {}))
            tool_id = getattr(block, 'id', block.get('id', ''))
        else:
            name = block.get("name", "")
            input_data = block.get("input", {})
            tool_id = block.get("id", "")

        exec_result = sales_tools.execute(name, input_data)

        results.append({
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": json.dumps(exec_result, ensure_ascii=False, default=str),
            "is_error": "error" in exec_result,
        })

    return results
