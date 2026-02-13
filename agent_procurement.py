#!/usr/bin/env python3
"""
MRARFAI Procurement Agent v10.0
================================
采购域 Agent — 供应商评估、PO跟踪、成本分析

MCP Tools:
  - compare_quotes: 比价分析
  - track_po: 采购订单跟踪
  - alert_delay: 延迟预警
  - analyze_cost: 成本分析

A2A Skills:
  - supplier_evaluation: 供应商综合评估
  - delivery_prediction: 交期预测
  - cost_optimization: 成本优化建议
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("mrarfai.agent.procurement")

# A2A 基础设施
try:
    from a2a_server_v7 import (
        AgentExecutor, AgentCard, AgentSkill, AgentCapabilities,
        AgentInterface, Task, TaskStatus, TaskState,
        Message, MessagePart, Artifact,
    )
    HAS_A2A = True
except ImportError:
    HAS_A2A = False

# Tool Registry
try:
    from tool_registry import ToolRegistry
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Supplier:
    """供应商"""
    name: str
    category: str = ""  # 电子元器件 / 结构件 / 包装 / 组装
    lead_time_days: int = 14
    quality_score: float = 0.90
    price_index: float = 1.0  # 1.0=行业均价
    on_time_rate: float = 0.92
    defect_rate: float = 0.02
    credit_rating: str = "A"
    country: str = "CN"

    def overall_score(self) -> float:
        """综合评分 (0-100)"""
        return round(
            self.quality_score * 30
            + self.on_time_rate * 25
            + (1 - self.defect_rate) * 20
            + (1 - min(self.price_index, 2) / 2) * 15
            + {"A": 10, "B": 7, "C": 4, "D": 1}.get(self.credit_rating, 5)
        , 1)


@dataclass
class PurchaseOrder:
    """采购订单"""
    po_id: str
    supplier: str
    items: List[Dict] = field(default_factory=list)
    total_amount: float = 0.0
    currency: str = "RMB"
    status: str = "pending"  # pending / confirmed / shipped / received / delayed
    created_at: str = ""
    expected_date: str = ""
    actual_date: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def is_delayed(self) -> bool:
        if self.status == "delayed":
            return True
        if self.expected_date and self.status not in ("received", "canceled"):
            try:
                exp = datetime.fromisoformat(self.expected_date)
                return datetime.now() > exp
            except ValueError:
                pass
        return False


# ============================================================
# 模拟数据 — 供应商 + PO (实际部署接ERP)
# ============================================================

SAMPLE_SUPPLIERS = [
    Supplier("深圳华星光电", "显示屏", 21, 0.95, 1.05, 0.94, 0.015, "A"),
    Supplier("京东方BOE", "显示屏", 18, 0.93, 0.98, 0.91, 0.018, "A"),
    Supplier("舜宇光学", "摄像模组", 14, 0.92, 1.02, 0.89, 0.025, "B"),
    Supplier("欧菲光", "摄像模组", 12, 0.88, 0.95, 0.85, 0.032, "B"),
    Supplier("立讯精密", "连接器", 10, 0.96, 1.10, 0.97, 0.008, "A"),
    Supplier("比亚迪电子", "结构件", 15, 0.91, 0.92, 0.93, 0.020, "A"),
    Supplier("蓝思科技", "玻璃盖板", 14, 0.90, 1.08, 0.88, 0.022, "B"),
    Supplier("德赛电池", "电池", 12, 0.94, 1.00, 0.95, 0.010, "A"),
]

SAMPLE_POS = [
    PurchaseOrder("PO-2025-0891", "深圳华星光电", [{"item": "6.5寸LCD", "qty": 50000}],
                  325.0, "RMB", "shipped", "2025-11-01", "2025-12-05"),
    PurchaseOrder("PO-2025-0923", "舜宇光学", [{"item": "13MP摄像模组", "qty": 80000}],
                  192.0, "RMB", "delayed", "2025-11-10", "2025-12-01"),
    PurchaseOrder("PO-2025-0945", "立讯精密", [{"item": "Type-C连接器", "qty": 200000}],
                  86.0, "RMB", "received", "2025-11-15", "2025-11-28", "2025-11-26"),
    PurchaseOrder("PO-2025-0967", "比亚迪电子", [{"item": "后壳CNC", "qty": 30000}],
                  210.0, "RMB", "confirmed", "2025-12-01", "2026-01-10"),
    PurchaseOrder("PO-2025-0980", "京东方BOE", [{"item": "6.7寸AMOLED", "qty": 20000}],
                  480.0, "RMB", "pending", "2025-12-15", "2026-01-20"),
]


# ============================================================
# 采购 Agent 核心逻辑
# ============================================================

class ProcurementEngine:
    """采购分析引擎"""

    def __init__(self, suppliers: List[Supplier] = None, orders: List[PurchaseOrder] = None):
        self.suppliers = {s.name: s for s in (suppliers or SAMPLE_SUPPLIERS)}
        self.orders = orders or SAMPLE_POS

    @classmethod
    def from_dataframes(cls, suppliers_df=None, orders_df=None):
        """从 DataFrame 创建引擎（用于 Excel 上传）"""
        suppliers = []
        if suppliers_df is not None:
            for _, row in suppliers_df.iterrows():
                try:
                    suppliers.append(Supplier(
                        name=str(row.iloc[0]),
                        category=str(row.iloc[1]) if len(row) > 1 else "",
                        lead_time_days=int(row.iloc[2]) if len(row) > 2 else 14,
                        quality_score=float(row.iloc[3]) if len(row) > 3 else 0.9,
                        price_index=float(row.iloc[4]) if len(row) > 4 else 1.0,
                        on_time_rate=float(row.iloc[5]) if len(row) > 5 else 0.9,
                        defect_rate=float(row.iloc[6]) if len(row) > 6 else 0.02,
                        credit_rating=str(row.iloc[7]) if len(row) > 7 else "B",
                    ))
                except Exception:
                    continue

        orders = []
        if orders_df is not None:
            for _, row in orders_df.iterrows():
                try:
                    orders.append(PurchaseOrder(
                        po_id=str(row.iloc[0]),
                        supplier=str(row.iloc[1]) if len(row) > 1 else "",
                        items=[{"item": str(row.iloc[2]) if len(row) > 2 else "", "qty": int(row.iloc[3]) if len(row) > 3 else 0}],
                        total_amount=float(row.iloc[4]) if len(row) > 4 else 0,
                        currency="RMB",
                        status=str(row.iloc[5]).lower() if len(row) > 5 else "pending",
                        created_at=str(row.iloc[6]) if len(row) > 6 else "",
                        expected_date=str(row.iloc[7]) if len(row) > 7 else "",
                    ))
                except Exception:
                    continue

        return cls(
            suppliers=suppliers if suppliers else None,
            orders=orders if orders else None,
        )

    def compare_quotes(self, category: str = "", top_n: int = 5) -> Dict:
        """比价分析"""
        candidates = [s for s in self.suppliers.values()
                      if not category or category in s.category]
        candidates.sort(key=lambda s: s.overall_score(), reverse=True)
        return {
            "category": category or "全部",
            "suppliers": [
                {
                    "name": s.name,
                    "category": s.category,
                    "overall_score": s.overall_score(),
                    "price_index": s.price_index,
                    "quality_score": f"{s.quality_score:.0%}",
                    "on_time_rate": f"{s.on_time_rate:.0%}",
                    "defect_rate": f"{s.defect_rate:.1%}",
                    "lead_time": f"{s.lead_time_days}天",
                    "credit": s.credit_rating,
                }
                for s in candidates[:top_n]
            ],
            "recommendation": candidates[0].name if candidates else "无",
            "analysis_time": datetime.now().isoformat(),
        }

    def track_po(self, po_id: str = "", supplier: str = "") -> Dict:
        """PO跟踪"""
        results = []
        for po in self.orders:
            if po_id and po_id != po.po_id:
                continue
            if supplier and supplier not in po.supplier:
                continue
            results.append({
                "po_id": po.po_id,
                "supplier": po.supplier,
                "items": po.items,
                "total_amount": f"¥{po.total_amount:.1f}万",
                "status": po.status,
                "expected": po.expected_date,
                "is_delayed": po.is_delayed(),
            })
        return {
            "total_orders": len(results),
            "delayed": sum(1 for r in results if r["is_delayed"]),
            "orders": results,
        }

    def alert_delay(self) -> Dict:
        """延迟预警"""
        delayed = [po for po in self.orders if po.is_delayed()]
        alerts = []
        for po in delayed:
            alerts.append({
                "po_id": po.po_id,
                "supplier": po.supplier,
                "items": po.items,
                "expected": po.expected_date,
                "days_overdue": (datetime.now() - datetime.fromisoformat(po.expected_date)).days
                    if po.expected_date else 0,
                "amount_at_risk": f"¥{po.total_amount:.1f}万",
                "severity": "高" if po.total_amount > 200 else "中",
            })
        return {
            "total_delayed": len(alerts),
            "total_at_risk": f"¥{sum(po.total_amount for po in delayed):.1f}万",
            "alerts": alerts,
        }

    def analyze_cost(self, category: str = "") -> Dict:
        """成本分析"""
        relevant = [po for po in self.orders
                    if not category or any(category in str(item) for item in po.items)]
        total_spend = sum(po.total_amount for po in relevant)
        by_supplier = {}
        for po in relevant:
            by_supplier.setdefault(po.supplier, 0)
            by_supplier[po.supplier] += po.total_amount
        return {
            "period": "2025年度",
            "total_spend": f"¥{total_spend:.1f}万",
            "by_supplier": {k: f"¥{v:.1f}万" for k, v in
                           sorted(by_supplier.items(), key=lambda x: -x[1])},
            "top_supplier": max(by_supplier, key=by_supplier.get) if by_supplier else "无",
            "optimization_suggestions": [
                "显示屏供应商可引入第三方竞价，预估降本3-5%",
                "摄像模组延迟率较高，建议增加备选供应商",
                "连接器立讯精密表现优异，可扩大份额获量价优惠",
            ],
        }

    def answer(self, question: str) -> str:
        """自然语言入口"""
        q = question.lower()
        if any(kw in q for kw in ["比价", "quote", "报价", "供应商评估"]):
            cat = ""
            for c in ["显示屏", "摄像", "连接器", "结构件", "电池", "玻璃"]:
                if c in q:
                    cat = c
                    break
            return json.dumps(self.compare_quotes(cat), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["po", "订单", "跟踪", "track"]):
            return json.dumps(self.track_po(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["延迟", "delay", "逾期", "shortage", "缺货"]):
            return json.dumps(self.alert_delay(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["成本", "cost", "spend", "花费"]):
            return json.dumps(self.analyze_cost(), ensure_ascii=False, indent=2)
        else:
            overview = {
                "agent": "Procurement",
                "suppliers": len(self.suppliers),
                "active_pos": len(self.orders),
                "delayed": sum(1 for po in self.orders if po.is_delayed()),
                "capabilities": ["compare_quotes", "track_po", "alert_delay", "analyze_cost"],
            }
            return json.dumps(overview, ensure_ascii=False, indent=2)


# ============================================================
# A2A Executor
# ============================================================

class ProcurementExecutor(AgentExecutor if HAS_A2A else object):
    """采购 Agent A2A 执行器"""

    def __init__(self):
        self.engine = ProcurementEngine()

    async def execute(self, task: 'Task', message: 'Message') -> 'Task':
        question = message.parts[0].text if message.parts else ""
        task.status = TaskStatus(state=TaskState.WORKING)
        task.history.append(message)

        try:
            answer = self.engine.answer(question)
            agent_msg = Message.agent_text(answer)
            task.history.append(agent_msg)
            task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_msg)
            task.artifacts.append(Artifact(
                name="procurement_result",
                description="采购分析结果",
                parts=[MessagePart(type="text", text=answer)],
            ))
        except Exception as e:
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message.agent_text(f"采购分析失败: {str(e)}"),
            )
        return task


# ============================================================
# Agent Card
# ============================================================

def create_procurement_card(base_url: str = "http://localhost:9999") -> 'AgentCard':
    """创建采购 Agent Card"""
    if not HAS_A2A:
        return None
    return AgentCard(
        name="MRARFAI 采购专员",
        description="采购域智能Agent — 供应商比价评估、PO跟踪、延迟预警、成本优化",
        version="10.0.0",
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/procurement")],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="supplier_evaluation",
                name="供应商评估",
                description="多维度供应商综合评分（质量/交期/价格/信用）",
                tags=["supplier", "evaluation", "procurement"],
                examples=["供应商排名", "显示屏供应商比价"],
            ),
            AgentSkill(
                id="delivery_prediction",
                name="交期预测",
                description="PO跟踪与延迟风险预警",
                tags=["delivery", "delay", "po", "tracking"],
                examples=["有哪些延迟的订单？", "供应商交期表现"],
            ),
            AgentSkill(
                id="cost_optimization",
                name="成本优化",
                description="采购成本分析与优化建议",
                tags=["cost", "optimization", "spend"],
                examples=["采购成本分析", "如何降低物料成本？"],
            ),
        ],
        provider={"organization": "禾苗科技"},
    )
