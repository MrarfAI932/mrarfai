#!/usr/bin/env python3
"""
MRARFAI Finance Agent v10.0
=============================
财务域 Agent — 应收账款跟踪、毛利分析、现金流预测

MCP Tools:
  - track_ar: 应收账款跟踪
  - analyze_margin: 毛利分析
  - forecast_cashflow: 现金流预测
  - match_invoice: 发票匹配

A2A Skills:
  - ar_tracking: 应收账款管理
  - margin_analysis: 毛利率深度分析
  - cashflow_forecast: 现金流预测
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from contracts import (
    FinanceARResponse, ARHighRisk,
    FinanceMarginResponse, FinanceCashflowResponse, CashflowMonth,
    FinanceInvoiceResponse, InvoiceRecord,
)

logger = logging.getLogger("mrarfai.agent.finance")

try:
    from a2a_server_v7 import (
        AgentExecutor, AgentCard, AgentSkill, AgentCapabilities,
        AgentInterface, Task, TaskStatus, TaskState,
        Message, MessagePart, Artifact,
    )
    HAS_A2A = True
except ImportError:
    HAS_A2A = False


# ============================================================
# 数据模型
# ============================================================

@dataclass
class ARRecord:
    """应收账款"""
    customer: str
    invoice_no: str
    amount: float  # 万元
    currency: str = "RMB"
    due_date: str = ""
    status: str = "outstanding"  # outstanding / overdue / paid / disputed
    aging_days: int = 0

    def is_overdue(self) -> bool:
        if self.status == "overdue":
            return True
        if self.due_date:
            try:
                return datetime.now() > datetime.fromisoformat(self.due_date)
            except ValueError:
                pass
        return False


@dataclass
class MarginRecord:
    """毛利记录"""
    product: str
    customer: str
    revenue: float  # 万元
    cogs: float     # 万元
    gross_margin: float = 0.0

    def __post_init__(self):
        if self.revenue > 0 and self.gross_margin == 0:
            self.gross_margin = (self.revenue - self.cogs) / self.revenue


# ============================================================
# 模拟数据 (实际部署对接财务系统)
# ============================================================

SAMPLE_AR = [
    ARRecord("Samsung", "INV-2025-1201", 520.0, "RMB", "2025-11-30", "overdue", 75),
    ARRecord("Samsung", "INV-2025-1245", 680.0, "RMB", "2025-12-15", "overdue", 60),
    ARRecord("Samsung", "INV-2025-1290", 600.0, "RMB", "2026-01-15", "outstanding", 30),
    ARRecord("HMD/Nokia", "INV-2025-1180", 380.0, "USD", "2025-12-20", "outstanding", 55),
    ARRecord("HMD/Nokia", "INV-2025-1220", 420.0, "USD", "2026-01-10", "outstanding", 35),
    ARRecord("Lava", "INV-2025-1150", 210.0, "USD", "2025-12-01", "paid", 0),
    ARRecord("Transsion", "INV-2025-1260", 890.0, "RMB", "2026-01-20", "outstanding", 25),
    ARRecord("BLU", "INV-2025-1100", 150.0, "USD", "2025-11-15", "overdue", 90),
    ARRecord("Motorola/Lenovo", "INV-2025-1300", 1200.0, "USD", "2026-02-01", "outstanding", 13),
]

SAMPLE_MARGINS = [
    MarginRecord("S60 Pro", "Samsung", 2800, 2240),
    MarginRecord("S60 Pro", "HMD/Nokia", 1500, 1185),
    MarginRecord("S60 Pro", "Lava", 800, 624),
    MarginRecord("T80", "Transsion", 1200, 900),
    MarginRecord("T80", "BLU", 450, 346),
    MarginRecord("S60 Pro", "Transsion", 600, 462),
    MarginRecord("Feature Phone", "Lava", 350, 301),
    MarginRecord("S60 Pro", "Motorola/Lenovo", 2200, 1826),
]


# ============================================================
# 财务 Agent 核心逻辑
# ============================================================

class FinanceEngine:
    """财务分析引擎"""

    def __init__(self, ar_records: List[ARRecord] = None, margins: List[MarginRecord] = None):
        self.ar_records = ar_records or SAMPLE_AR
        self.margins = margins or SAMPLE_MARGINS

    @classmethod
    def from_dataframes(cls, ar_df=None, margin_df=None):
        """从 DataFrame 创建引擎（用于 Excel 上传）"""
        ar_records = []
        if ar_df is not None:
            for _, row in ar_df.iterrows():
                try:
                    ar_records.append(ARRecord(
                        customer=str(row.iloc[0]),
                        invoice_no=str(row.iloc[1]) if len(row) > 1 else "",
                        amount=float(row.iloc[2]) if len(row) > 2 else 0,
                        currency=str(row.iloc[3]) if len(row) > 3 else "RMB",
                        due_date=str(row.iloc[4]) if len(row) > 4 else "",
                        status=str(row.iloc[5]).lower() if len(row) > 5 else "outstanding",
                        aging_days=int(row.iloc[6]) if len(row) > 6 else 0,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse AR record {row.iloc[0] if len(row) > 0 else 'unknown'}: {e}")
                    continue

        margins = []
        if margin_df is not None:
            for _, row in margin_df.iterrows():
                try:
                    margins.append(MarginRecord(
                        product=str(row.iloc[0]),
                        customer=str(row.iloc[1]) if len(row) > 1 else "",
                        revenue=float(row.iloc[2]) if len(row) > 2 else 0,
                        cogs=float(row.iloc[3]) if len(row) > 3 else 0,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse margin record {row.iloc[0] if len(row) > 0 else 'unknown'}: {e}")
                    continue

        return cls(
            ar_records=ar_records if ar_records else None,
            margins=margins if margins else None,
        )

    def track_ar(self, customer: str = "") -> Dict:
        """应收账款跟踪"""
        records = self.ar_records
        if customer:
            records = [r for r in records if customer in r.customer]

        total_outstanding = sum(r.amount for r in records if r.status != "paid")
        total_overdue = sum(r.amount for r in records if r.status == "overdue")

        # 账龄分析
        aging = {"0-30天": 0, "31-60天": 0, "61-90天": 0, "90天+": 0}
        for r in records:
            if r.status == "paid":
                continue
            if r.aging_days <= 30:
                aging["0-30天"] += r.amount
            elif r.aging_days <= 60:
                aging["31-60天"] += r.amount
            elif r.aging_days <= 90:
                aging["61-90天"] += r.amount
            else:
                aging["90天+"] += r.amount

        by_customer = {}
        for r in records:
            if r.status == "paid":
                continue
            by_customer.setdefault(r.customer, {"total": 0, "overdue": 0})
            by_customer[r.customer]["total"] += r.amount
            if r.status == "overdue":
                by_customer[r.customer]["overdue"] += r.amount

        return FinanceARResponse(
            total_outstanding=f"¥{total_outstanding:.0f}万",
            total_overdue=f"¥{total_overdue:.0f}万",
            overdue_ratio=f"{total_overdue/total_outstanding:.0%}" if total_outstanding > 0 else "0%",
            aging_analysis={k: f"¥{v:.0f}万" for k, v in aging.items()},
            by_customer={
                k: {kk: f"¥{vv:.0f}万" for kk, vv in v.items()}
                for k, v in sorted(by_customer.items(), key=lambda x: -x[1]["total"])
            },
            high_risk=[
                ARHighRisk(customer=r.customer, invoice=r.invoice_no,
                           amount=f"¥{r.amount:.0f}万", aging=f"{r.aging_days}天")
                for r in records if r.aging_days > 60
            ],
        ).model_dump()

    def analyze_margin(self, product: str = "", customer: str = "") -> Dict:
        """毛利分析"""
        records = self.margins
        if product:
            records = [m for m in records if product in m.product]
        if customer:
            records = [m for m in records if customer in m.customer]

        total_rev = sum(m.revenue for m in records)
        total_cogs = sum(m.cogs for m in records)
        avg_margin = (total_rev - total_cogs) / total_rev if total_rev > 0 else 0

        by_product = {}
        for m in records:
            by_product.setdefault(m.product, {"revenue": 0, "cogs": 0})
            by_product[m.product]["revenue"] += m.revenue
            by_product[m.product]["cogs"] += m.cogs

        by_customer = {}
        for m in records:
            by_customer.setdefault(m.customer, {"revenue": 0, "cogs": 0})
            by_customer[m.customer]["revenue"] += m.revenue
            by_customer[m.customer]["cogs"] += m.cogs

        return FinanceMarginResponse(
            total_revenue=f"¥{total_rev:.0f}万",
            total_cogs=f"¥{total_cogs:.0f}万",
            gross_margin=f"{avg_margin:.1%}",
            by_product={
                k: {"revenue": f"¥{v['revenue']:.0f}万",
                    "margin": f"{(v['revenue']-v['cogs'])/v['revenue']:.1%}" if v['revenue'] > 0 else "N/A"}
                for k, v in sorted(by_product.items(), key=lambda x: -x[1]["revenue"])
            },
            by_customer={
                k: {"revenue": f"¥{v['revenue']:.0f}万",
                    "margin": f"{(v['revenue']-v['cogs'])/v['revenue']:.1%}" if v['revenue'] > 0 else "N/A"}
                for k, v in sorted(by_customer.items(), key=lambda x: -x[1]["revenue"])
            },
        ).model_dump()

    def forecast_cashflow(self, months: int = 3) -> Dict:
        """现金流预测"""
        outstanding_by_month = []
        now = datetime.now()
        for i in range(months):
            month_start = now + timedelta(days=30 * i)
            month_label = month_start.strftime("%Y年%m月")
            expected_in = sum(
                r.amount * (0.8 if r.aging_days > 60 else 0.95)
                for r in self.ar_records
                if r.status != "paid"
            ) / months
            expected_out = sum(m.cogs for m in self.margins) / 12
            outstanding_by_month.append({
                "month": month_label,
                "expected_inflow": f"¥{expected_in:.0f}万",
                "expected_outflow": f"¥{expected_out:.0f}万",
                "net_cashflow": f"¥{expected_in - expected_out:.0f}万",
            })

        return FinanceCashflowResponse(
            forecast_period=f"未来{months}个月",
            monthly_forecast=[CashflowMonth(**m) for m in outstanding_by_month],
            risks=[
                "Samsung应收逾期60天+，影响约¥1,200万回款",
                "BLU逾期90天+，建议启动催收流程",
            ],
            suggestions=[
                "1. 优先催收Samsung逾期款项",
                "2. BLU逾期90天+，考虑计提坏账准备",
                "3. 新客户建议缩短账期至30天",
            ],
        ).model_dump()

    def match_invoice(self, invoice_no: str = "") -> Dict:
        """发票匹配"""
        if invoice_no:
            matched = [r for r in self.ar_records if invoice_no in r.invoice_no]
        else:
            matched = self.ar_records

        return FinanceInvoiceResponse(
            matched=len(matched),
            invoices=[
                InvoiceRecord(
                    invoice_no=r.invoice_no,
                    customer=r.customer,
                    amount=f"¥{r.amount:.0f}万",
                    status=r.status,
                    due_date=r.due_date,
                    aging=f"{r.aging_days}天",
                )
                for r in matched
            ],
        ).model_dump()

    def answer(self, question: str) -> str:
        """自然语言入口"""
        q = question.lower()
        if any(kw in q for kw in ["应收", "ar", "账款", "receivable", "逾期", "overdue"]):
            customer = ""
            for c in ["Samsung", "HMD", "Nokia", "Lava", "Transsion", "BLU"]:
                if c.lower() in q:
                    customer = c
            return json.dumps(self.track_ar(customer), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["毛利", "margin", "利润", "利率"]):
            return json.dumps(self.analyze_margin(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["现金流", "cashflow", "cash flow", "资金"]):
            return json.dumps(self.forecast_cashflow(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["发票", "invoice"]):
            return json.dumps(self.match_invoice(), ensure_ascii=False, indent=2)
        else:
            overview = {
                "agent": "Finance",
                "ar_records": len(self.ar_records),
                "margin_records": len(self.margins),
                "capabilities": ["track_ar", "analyze_margin", "forecast_cashflow", "match_invoice"],
            }
            return json.dumps(overview, ensure_ascii=False, indent=2)


# ============================================================
# A2A Executor
# ============================================================

class FinanceExecutor(AgentExecutor if HAS_A2A else object):
    """财务 Agent A2A 执行器"""

    def __init__(self):
        self.engine = FinanceEngine()

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
                name="finance_result",
                description="财务分析结果",
                parts=[MessagePart(type="text", text=answer)],
            ))
        except Exception as e:
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message.agent_text(f"财务分析失败: {str(e)}"),
            )
        return task


# ============================================================
# Agent Card
# ============================================================

def create_finance_card(base_url: str = "http://localhost:9999") -> 'AgentCard':
    """创建财务 Agent Card"""
    if not HAS_A2A:
        return None
    return AgentCard(
        name="MRARFAI 财务分析师",
        description="财务域智能Agent — 应收跟踪、毛利分析、现金流预测、发票匹配",
        version="10.0.0",
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/finance")],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="ar_tracking",
                name="应收跟踪",
                description="应收账款管理 — 账龄分析、逾期预警、客户欠款排名",
                tags=["ar", "receivable", "overdue", "finance"],
                examples=["应收账款情况", "哪些客户逾期了？"],
            ),
            AgentSkill(
                id="margin_analysis",
                name="毛利分析",
                description="产品/客户维度的毛利率深度分析",
                tags=["margin", "profit", "cogs"],
                examples=["毛利率分析", "哪个产品毛利最高？"],
            ),
            AgentSkill(
                id="cashflow_forecast",
                name="现金流预测",
                description="基于应收/应付的未来现金流预测",
                tags=["cashflow", "forecast", "treasury"],
                examples=["未来3个月现金流预测", "资金缺口风险"],
            ),
        ],
        provider={"organization": "禾苗科技"},
    )
