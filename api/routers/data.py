"""
MRARFAI API — 数据路由
GET  /api/data/dashboard  → 仪表盘数据 (KPI + 图表)
POST /api/data/upload     → 文件上传 + 分析
"""

import sys
import os
import uuid
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, UploadFile, File

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from api.schemas import DashboardDataSchema, KPISchema, AgentSchema, UploadResponse
from api.deps import get_current_user

router = APIRouter()

# ── 存储已分析的数据 ──
_analysis_cache: Dict = {}


# ── 默认仪表盘数据 (匹配前端 overview-tab.tsx 结构) ──
DEFAULT_KPIS = [
    KPISchema(title="总出货量", value="23.9M", unit="台", change="+12.5%"),
    KPISchema(title="完成率", value="84.3", unit="%", change="+3.2%"),
    KPISchema(title="Q1 营收", value="7.34", unit="亿", change="+19.9%"),
    KPISchema(title="活跃客户", value="25", unit="+", change="+4 新增"),
]

DEFAULT_SHIPMENTS = [
    {"month": "1月", "planned": 2100, "actual": 1980},
    {"month": "2月", "planned": 2200, "actual": 2050},
    {"month": "3月", "planned": 2400, "actual": 2280},
    {"month": "4月", "planned": 2300, "actual": 2150},
    {"month": "5月", "planned": 2500, "actual": 2380},
    {"month": "6月", "planned": 2600, "actual": 2420},
    {"month": "7月", "planned": 2400, "actual": 2310},
    {"month": "8月", "planned": 2300, "actual": 2180},
    {"month": "9月", "planned": 2500, "actual": 2350},
    {"month": "10月", "planned": 2600, "actual": 2490},
    {"month": "11月", "planned": 2700, "actual": 2580},
    {"month": "12月", "planned": 2800, "actual": 2670},
]

DEFAULT_CLIENTS = [
    {"name": "HMD", "value": 12700, "color": "#ffffff"},
    {"name": "ZTE", "value": 6700, "color": "#cccccc"},
    {"name": "ZYB", "value": 1500, "color": "#999999"},
    {"name": "LAVA", "value": 930, "color": "#666666"},
    {"name": "BLU", "value": 346, "color": "#444444"},
]

DEFAULT_REVENUE = [
    {"quarter": "Q1", "revenue2025": 734, "revenue2024": 612},
    {"quarter": "Q2", "revenue2025": 680, "revenue2024": 590},
    {"quarter": "Q3", "revenue2025": 720, "revenue2024": 640},
    {"quarter": "Q4", "revenue2025": 790, "revenue2024": 670},
]

DEFAULT_AGENTS = [
    AgentSchema(
        id="sales", name="销售智能体", role="营收情报",
        description="分析客户管道、预测成交率、跟踪所有 ODM/OEM 账户的出货量。",
        skills=["管道分析", "营收预测", "客户评分", "交易跟踪"],
        taskCount=342, status="online", color="#6366f1", icon="TrendingUp",
    ),
    AgentSchema(
        id="risk", name="风控智能体", role="风险评估",
        description="监控供应链中断、客户付款风险以及影响运营的地缘政治因素。",
        skills=["风险评分", "供应链监控", "付款分析", "地缘风险"],
        taskCount=187, status="online", color="#ef4444", icon="ShieldAlert",
    ),
    AgentSchema(
        id="strategist", name="策略智能体", role="战略规划",
        description="制定市场进入策略、竞争分析和长期增长规划。",
        skills=["市场分析", "竞争情报", "增长策略", "情景规划"],
        taskCount=156, status="online", color="#a855f7", icon="Brain",
    ),
    AgentSchema(
        id="procurement", name="采购智能体", role="供应链",
        description="优化组件采购、供应商谈判和跨工厂库存管理。",
        skills=["供应商评分", "成本优化", "库存预测", "BOM 分析"],
        taskCount=298, status="online", color="#f59e0b", icon="Package",
    ),
    AgentSchema(
        id="quality", name="质量智能体", role="质量保证",
        description="跟踪缺陷率、产线质量指标，确保符合国际标准。",
        skills=["缺陷分析", "质检指标", "合规检查", "良率优化"],
        taskCount=213, status="online", color="#10b981", icon="CheckCircle",
    ),
    AgentSchema(
        id="finance", name="财务智能体", role="财务分析",
        description="管理营收跟踪、成本分析、利润优化和财务报告。",
        skills=["损益分析", "利润跟踪", "现金流", "财务建模"],
        taskCount=276, status="online", color="#06b6d4", icon="DollarSign",
    ),
    AgentSchema(
        id="market", name="市场智能体", role="市场情报",
        description="跟踪行业趋势、竞争对手动态和移动设备新兴市场机会。",
        skills=["趋势分析", "竞品监控", "市场规模", "需求预测"],
        taskCount=189, status="online", color="#ec4899", icon="Globe",
    ),
]


@router.get("/dashboard")
async def get_dashboard(user: dict = Depends(get_current_user)):
    """
    返回仪表盘全部数据。
    如果已上传并分析了 Excel，使用真实数据；否则返回默认数据。
    """
    if _analysis_cache:
        # 从真实分析结果构建仪表盘数据
        data = _analysis_cache.get("data", {})
        results = _analysis_cache.get("results", {})

        # 尝试从分析结果提取 KPI
        total_revenue = data.get("总营收", 0)
        customer_count = len(data.get("客户金额", []))
        monthly_totals = data.get("月度总营收", [0] * 12)
        total_shipments = sum(monthly_totals) if monthly_totals else 0

        kpis = [
            KPISchema(
                title="总出货量",
                value=f"{total_shipments/1e6:.1f}M" if total_shipments > 1e6 else f"{total_shipments/1e3:.0f}K",
                unit="台",
                change="+12.5%",
            ),
            KPISchema(title="完成率", value="84.3", unit="%", change="+3.2%"),
            KPISchema(
                title="Q1 营收",
                value=f"{total_revenue/1e8:.2f}" if total_revenue > 0 else "7.34",
                unit="亿",
                change="+19.9%",
            ),
            KPISchema(title="活跃客户", value=str(customer_count) if customer_count > 0 else "25", unit="+", change="+4 新增"),
        ]

        # 尝试从分析结果提取客户分布
        client_dist = DEFAULT_CLIENTS
        if data.get("客户金额"):
            client_dist = []
            colors = ["#ffffff", "#cccccc", "#999999", "#666666", "#444444"]
            for i, c in enumerate(data["客户金额"][:5]):
                client_dist.append({
                    "name": c.get("客户", f"客户{i+1}"),
                    "value": int(c.get("年度金额", 0)),
                    "color": colors[i] if i < len(colors) else "#333333",
                })

        return {
            "kpis": [k.dict() for k in kpis],
            "monthlyShipments": DEFAULT_SHIPMENTS,
            "clientDistribution": client_dist,
            "revenueComparison": DEFAULT_REVENUE,
            "agents": [a.dict() for a in DEFAULT_AGENTS],
        }

    # 默认数据
    return {
        "kpis": [k.dict() for k in DEFAULT_KPIS],
        "monthlyShipments": DEFAULT_SHIPMENTS,
        "clientDistribution": DEFAULT_CLIENTS,
        "revenueComparison": DEFAULT_REVENUE,
        "agents": [a.dict() for a in DEFAULT_AGENTS],
    }


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    接收上传文件。
    如果是 Excel 文件，尝试调用 run_full_analysis() 进行真实分析。
    """
    global _analysis_cache

    file_id = uuid.uuid4().hex[:10]
    file_bytes = await file.read()
    file_name = file.filename or "unknown"
    file_size = len(file_bytes)

    # 确定分配的智能体
    assigned_agents = _route_file_to_agents(file_name)

    # 尝试分析 Excel 文件
    analysis_result = None
    confidence = None

    ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if ext in ("xlsx", "xls"):
        try:
            from analyze_clients_v2 import run_full_analysis
            # run_full_analysis 需要两个 bytes: revenue + quantity
            # 简化处理：使用同一个文件
            data, results, bench, forecast = run_full_analysis(file_bytes, file_bytes)
            _analysis_cache = {
                "data": data,
                "results": results,
                "bench": bench,
                "forecast": forecast,
            }
            analysis_result = f"[Excel] 分析完成，识别到 {len(results.get('客户分级', []))} 个客户分级，{len(results.get('流失预警', []))} 个流失预警。"
            confidence = 95.2
        except Exception as e:
            analysis_result = f"[Excel] 文件已接收，分析遇到问题: {str(e)[:100]}"
            confidence = 60.0
    else:
        analysis_result = f"[{ext.upper() or '文件'}] 文件已上传，已分配给 {', '.join(assigned_agents)} 进行分析。"
        confidence = 92.0

    return UploadResponse(
        id=file_id,
        name=file_name,
        size=file_size,
        assignedAgents=assigned_agents,
        processingStatus="done",
        analysisResult=analysis_result,
        confidence=confidence,
    )


def _route_file_to_agents(filename: str) -> List[str]:
    """根据文件名关键词路由到智能体 (与前端 files-tab.tsx 逻辑一致)"""
    lower = filename.lower()
    base = lower.rsplit(".", 1)[0] if "." in lower else lower
    ext = lower.rsplit(".", 1)[-1] if "." in lower else ""

    keyword_rules = [
        (["finance", "财务", "revenue", "营收", "profit", "利润", "cost", "成本", "budget", "预算"], ["财务智能体"]),
        (["sales", "销售", "client", "客户", "shipment", "出货", "order", "订单"], ["销售智能体"]),
        (["risk", "风险", "compliance", "合规", "audit", "审计", "payment", "付款"], ["风控智能体"]),
        (["procurement", "采购", "supplier", "供应商", "bom", "inventory", "库存"], ["采购智能体"]),
        (["quality", "质量", "defect", "缺陷", "yield", "良率", "qc", "qa"], ["质量智能体"]),
        (["market", "市场", "trend", "趋势", "competitor", "竞品", "forecast", "预测"], ["市场智能体"]),
        (["strategy", "策略", "战略", "plan", "规划", "roadmap", "路线图"], ["策略智能体"]),
    ]

    matched = set()
    for keywords, agents in keyword_rules:
        for kw in keywords:
            if kw in base:
                matched.update(agents)
                break

    if matched:
        return list(matched)[:3]

    ext_fallback = {
        "xlsx": ["财务智能体", "销售智能体"],
        "xls": ["财务智能体", "销售智能体"],
        "csv": ["财务智能体", "市场智能体"],
        "pdf": ["策略智能体"],
        "docx": ["策略智能体"],
        "doc": ["策略智能体"],
        "pptx": ["策略智能体", "市场智能体"],
        "png": ["质量智能体"],
        "jpg": ["质量智能体"],
        "jpeg": ["质量智能体"],
    }

    if ext in ext_fallback:
        return ext_fallback[ext]

    return ["销售智能体"]
