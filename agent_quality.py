#!/usr/bin/env python3
"""
MRARFAI Quality Control Agent v10.0
=====================================
品质域 Agent — 良率监控、退货分析、根因追溯

MCP Tools:
  - monitor_yield: 良率监控
  - analyze_returns: 退货分析
  - classify_complaints: 投诉分类
  - trace_root_cause: 根因追溯

A2A Skills:
  - yield_monitoring: 良率趋势监控
  - returns_analysis: 退货/售后分析
  - root_cause: 根因追溯链
"""

import json
import logging
try:
    import pandas as pd
except ImportError:
    pd = None
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from contracts import (
    QualityYieldResponse, QualityReturnsResponse, QualityRootCauseResponse,
    QualityComplaintsResponse,
    YieldTrend, DefectCount, YieldAlert,
    ReturnRecord as ReturnRecordContract,
)

logger = logging.getLogger("mrarfai.agent.quality")

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
class YieldRecord:
    """良率记录"""
    line: str           # 产线
    product: str        # 产品
    month: str          # 月份
    total_produced: int = 0
    passed: int = 0
    yield_rate: float = 0.0
    defect_types: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.total_produced > 0 and self.yield_rate == 0:
            self.yield_rate = self.passed / self.total_produced


@dataclass
class ReturnRecord:
    """退货记录"""
    customer: str
    product: str
    quantity: int
    reason: str
    date: str
    severity: str = "中"  # 高/中/低


# ============================================================
# 模拟数据 (实际部署对接 MES/QMS)
# ============================================================

SAMPLE_YIELDS = [
    YieldRecord("SMT-Line1", "S60 Pro", "2025-10", 50000, 48750, 0.975, {"焊接不良": 450, "元件偏移": 300, "其他": 500}),
    YieldRecord("SMT-Line1", "S60 Pro", "2025-11", 52000, 50700, 0.975, {"焊接不良": 480, "元件偏移": 320, "其他": 500}),
    YieldRecord("SMT-Line1", "S60 Pro", "2025-12", 48000, 46320, 0.965, {"焊接不良": 620, "元件偏移": 380, "触控失灵": 480, "其他": 200}),
    YieldRecord("SMT-Line2", "T80", "2025-10", 30000, 29400, 0.980, {"焊接不良": 200, "其他": 400}),
    YieldRecord("SMT-Line2", "T80", "2025-11", 32000, 31200, 0.975, {"焊接不良": 280, "元件偏移": 200, "其他": 320}),
    YieldRecord("SMT-Line2", "T80", "2025-12", 35000, 34300, 0.980, {"焊接不良": 250, "其他": 450}),
    YieldRecord("Assembly-A", "S60 Pro", "2025-12", 46000, 44620, 0.970, {"屏幕安装不良": 520, "螺丝漏装": 310, "外观划伤": 550}),
]

SAMPLE_RETURNS = [
    ReturnRecord("Samsung", "S60 Pro", 45, "触控失灵", "2025-12-05", "高"),
    ReturnRecord("Samsung", "S60 Pro", 12, "屏幕亮线", "2025-12-10", "中"),
    ReturnRecord("HMD/Nokia", "T80", 8, "电池鼓包", "2025-11-20", "高"),
    ReturnRecord("Lava", "S60 Pro", 15, "充电异常", "2025-12-08", "中"),
    ReturnRecord("Transsion", "S60 Pro", 22, "触控失灵", "2025-12-12", "高"),
    ReturnRecord("BLU", "T80", 5, "WiFi断连", "2025-11-28", "低"),
]


# ============================================================
# 品质 Agent 核心逻辑
# ============================================================

class QualityEngine:
    """品质分析引擎"""

    def __init__(self, yields: List[YieldRecord] = None, returns: List[ReturnRecord] = None):
        self.yields = yields or SAMPLE_YIELDS
        self.returns = returns or SAMPLE_RETURNS

    @classmethod
    def from_dataframes(cls, yields_df=None, returns_df=None):
        """从 DataFrame 创建引擎（用于 Excel 上传）"""
        yields = []
        if yields_df is not None:
            for _, row in yields_df.iterrows():
                try:
                    # 前5列: 产线 | 产品 | 月份 | 总产数 | 合格数，后面是缺陷类型
                    defects = {}
                    for col in yields_df.columns[5:]:
                        val = row[col]
                        if pd.notna(val) and val:
                            defects[str(col)] = int(val)
                    total = int(row.iloc[3])
                    passed = int(row.iloc[4])
                    yields.append(YieldRecord(
                        line=str(row.iloc[0]),
                        product=str(row.iloc[1]) if len(row) > 1 else "",
                        month=str(row.iloc[2]) if len(row) > 2 else "",
                        total_produced=total,
                        passed=passed,
                        yield_rate=round(passed / total, 4) if total > 0 else 0,
                        defect_types=defects,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse yield record {row.iloc[0] if len(row) > 0 else 'unknown'}: {e}")
                    continue

        returns = []
        if returns_df is not None:
            for _, row in returns_df.iterrows():
                try:
                    returns.append(ReturnRecord(
                        customer=str(row.iloc[0]),
                        product=str(row.iloc[1]) if len(row) > 1 else "",
                        quantity=int(row.iloc[2]) if len(row) > 2 else 0,
                        reason=str(row.iloc[3]) if len(row) > 3 else "",
                        date=str(row.iloc[4]) if len(row) > 4 else "",
                        severity=str(row.iloc[5]) if len(row) > 5 else "中",
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse return record {row.iloc[0] if len(row) > 0 else 'unknown'}: {e}")
                    continue

        return cls(
            yields=yields if yields else None,
            returns=returns if returns else None,
        )

    def monitor_yield(self, product: str = "", line: str = "") -> Dict:
        """良率监控"""
        records = self.yields
        if product:
            records = [y for y in records if product in y.product]
        if line:
            records = [y for y in records if line in y.line]

        # 按月趋势
        monthly = {}
        for y in records:
            monthly.setdefault(y.month, []).append(y)

        trends = []
        for month, recs in sorted(monthly.items()):
            total_prod = sum(r.total_produced for r in recs)
            total_pass = sum(r.passed for r in recs)
            rate = total_pass / total_prod if total_prod > 0 else 0
            trends.append({
                "month": month,
                "yield_rate": f"{rate:.1%}",
                "total_produced": total_prod,
                "defects": total_prod - total_pass,
            })

        # 缺陷 Top 类型
        defect_agg = {}
        for y in records:
            for dtype, count in y.defect_types.items():
                defect_agg[dtype] = defect_agg.get(dtype, 0) + count

        top_defects = sorted(defect_agg.items(), key=lambda x: -x[1])[:5]

        # 良率预警
        alerts = []
        if len(trends) >= 2:
            latest = float(trends[-1]["yield_rate"].strip("%")) / 100
            prev = float(trends[-2]["yield_rate"].strip("%")) / 100
            if latest < prev - 0.005:
                alerts.append({
                    "type": "良率下降",
                    "detail": f"{trends[-1]['month']}良率{trends[-1]['yield_rate']}，较上月下降{(prev-latest):.1%}",
                    "severity": "高" if latest < 0.97 else "中",
                })

        return QualityYieldResponse(
            filter={"product": product or "全部", "line": line or "全部"},
            trends=[YieldTrend(**t) for t in trends],
            top_defects=[DefectCount(type=d[0], count=d[1]) for d in top_defects],
            alerts=[YieldAlert(**a) for a in alerts],
        ).model_dump()

    def analyze_returns(self, customer: str = "") -> Dict:
        """退货分析"""
        records = self.returns
        if customer:
            records = [r for r in records if customer in r.customer]

        by_reason = {}
        by_customer = {}
        for r in records:
            by_reason[r.reason] = by_reason.get(r.reason, 0) + r.quantity
            by_customer[r.customer] = by_customer.get(r.customer, 0) + r.quantity

        total_qty = sum(r.quantity for r in records)
        high_severity = [r for r in records if r.severity == "高"]

        return QualityReturnsResponse(
            total_returns=total_qty,
            total_cases=len(records),
            high_severity_cases=len(high_severity),
            by_reason={k: v for k, v in sorted(by_reason.items(), key=lambda x: -x[1])},
            by_customer={k: v for k, v in sorted(by_customer.items(), key=lambda x: -x[1])},
            recent_high_severity=[
                ReturnRecordContract(customer=r.customer, product=r.product, qty=r.quantity,
                                     reason=r.reason, date=r.date)
                for r in high_severity
            ],
        ).model_dump()

    def classify_complaints(self) -> Dict:
        """投诉分类"""
        categories = {
            "硬件缺陷": ["触控失灵", "屏幕亮线", "电池鼓包"],
            "功能异常": ["充电异常", "WiFi断连"],
            "外观问题": ["外观划伤"],
            "装配问题": ["屏幕安装不良", "螺丝漏装"],
        }
        result = {}
        for cat, reasons in categories.items():
            matched = [r for r in self.returns if r.reason in reasons]
            result[cat] = {
                "count": sum(r.quantity for r in matched),
                "cases": len(matched),
                "details": [r.reason for r in matched],
            }
        return QualityComplaintsResponse(
            classification=result,
            total=sum(r.quantity for r in self.returns),
        ).model_dump()

    def trace_root_cause(self, defect: str = "触控失灵") -> Dict:
        """根因追溯"""
        # 查找相关退货
        related_returns = [r for r in self.returns if defect in r.reason]
        # 查找相关良率数据
        related_yields = [y for y in self.yields if defect in y.defect_types]

        timeline = []
        for y in sorted(related_yields, key=lambda x: x.month):
            timeline.append({
                "date": y.month,
                "source": "产线",
                "line": y.line,
                "product": y.product,
                "defect_count": y.defect_types.get(defect, 0),
            })
        for r in sorted(related_returns, key=lambda x: x.date):
            timeline.append({
                "date": r.date,
                "source": "客户退货",
                "customer": r.customer,
                "product": r.product,
                "qty": r.quantity,
            })

        return QualityRootCauseResponse(
            defect=defect,
            total_production_defects=sum(y.defect_types.get(defect, 0) for y in related_yields),
            total_customer_returns=sum(r.quantity for r in related_returns),
            affected_lines=list(set(y.line for y in related_yields)),
            affected_customers=list(set(r.customer for r in related_returns)),
            timeline=sorted(timeline, key=lambda x: x["date"]),
            probable_cause=f"12月{defect}缺陷激增，疑似新批次触控IC来料问题",
            recommended_actions=[
                "1. IQC加严触控IC来料检验",
                "2. 联系供应商追溯批次",
                "3. 对在库半成品抽检",
                "4. 通知受影响客户并提供换货方案",
            ],
        ).model_dump()

    def answer(self, question: str) -> str:
        """自然语言入口"""
        q = question.lower()
        if any(kw in q for kw in ["良率", "yield", "产线", "生产"]):
            product = ""
            for p in ["S60", "T80"]:
                if p.lower() in q:
                    product = p
            return json.dumps(self.monitor_yield(product), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["退货", "return", "售后", "退回"]):
            customer = ""
            for c in ["Samsung", "HMD", "Nokia", "Lava", "Transsion"]:
                if c.lower() in q:
                    customer = c
            return json.dumps(self.analyze_returns(customer), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["投诉", "complaint", "分类"]):
            return json.dumps(self.classify_complaints(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["根因", "root cause", "追溯", "trace"]):
            defect = "触控失灵"
            for d in ["触控失灵", "电池鼓包", "充电异常", "WiFi断连"]:
                if d in q:
                    defect = d
            return json.dumps(self.trace_root_cause(defect), ensure_ascii=False, indent=2)
        else:
            overview = {
                "agent": "Quality Control",
                "yield_records": len(self.yields),
                "return_cases": len(self.returns),
                "capabilities": ["monitor_yield", "analyze_returns", "classify_complaints", "trace_root_cause"],
            }
            return json.dumps(overview, ensure_ascii=False, indent=2)


# ============================================================
# A2A Executor
# ============================================================

class QualityExecutor(AgentExecutor if HAS_A2A else object):
    """品质 Agent A2A 执行器"""

    def __init__(self):
        self.engine = QualityEngine()

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
                name="quality_result",
                description="品质分析结果",
                parts=[MessagePart(type="text", text=answer)],
            ))
        except Exception as e:
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message.agent_text(f"品质分析失败: {str(e)}"),
            )
        return task


# ============================================================
# Agent Card
# ============================================================

def create_quality_card(base_url: str = "http://localhost:9999") -> 'AgentCard':
    """创建品质 Agent Card"""
    if not HAS_A2A:
        return None
    return AgentCard(
        name="MRARFAI 品质管控员",
        description="品质域智能Agent — 良率监控、退货分析、投诉分类、根因追溯",
        version="10.0.0",
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/quality")],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="yield_monitoring",
                name="良率监控",
                description="产线良率趋势监控与异常预警",
                tags=["yield", "quality", "production"],
                examples=["本月良率如何？", "S60 Pro产线良率趋势"],
            ),
            AgentSkill(
                id="returns_analysis",
                name="退货分析",
                description="客户退货统计、原因分布、严重度评估",
                tags=["returns", "warranty", "customer"],
                examples=["Samsung退货情况", "高严重度退货有哪些？"],
            ),
            AgentSkill(
                id="root_cause",
                name="根因追溯",
                description="缺陷根因分析 — 从产线到客户的全链路追溯",
                tags=["root_cause", "traceability", "defect"],
                examples=["触控失灵的根因是什么？", "追溯电池问题"],
            ),
        ],
        provider={"organization": "禾苗科技"},
    )
