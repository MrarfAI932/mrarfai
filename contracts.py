#!/usr/bin/env python3
"""
MRARFAI V10.0 — Pydantic 结构化合约
======================================
全部 Agent 输入/输出的类型安全合约层。
LangChain 1.0 / CrewAI 2.x 标准要求。

用法:
  from contracts import AgentRequest, QualityYieldResponse, ...
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================
# 通用 Agent 合约
# ============================================================

class AgentRequest(BaseModel):
    """统一 Agent 请求合约"""
    question: str = Field(..., description="用户查询")
    agent_id: str = Field(default="", description="目标Agent ID")
    context: Optional[str] = Field(default=None, description="附加上下文")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="过滤条件")


class AgentResponse(BaseModel):
    """统一 Agent 响应合约"""
    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent 名称")
    status: str = Field(default="completed", description="执行状态")
    data: Dict[str, Any] = Field(default_factory=dict, description="结构化结果")
    summary: Optional[str] = Field(default=None, description="文本摘要")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    elapsed_ms: Optional[float] = Field(default=None, description="执行耗时(ms)")


# ============================================================
# Quality Agent 合约
# ============================================================

class YieldTrend(BaseModel):
    month: str
    yield_rate: str
    total_produced: int
    defects: int


class DefectCount(BaseModel):
    type: str
    count: int


class YieldAlert(BaseModel):
    type: str
    detail: str
    severity: str


class QualityYieldResponse(BaseModel):
    """良率监控响应"""
    filter: Dict[str, str] = Field(default_factory=dict)
    trends: List[YieldTrend] = Field(default_factory=list)
    top_defects: List[DefectCount] = Field(default_factory=list)
    alerts: List[YieldAlert] = Field(default_factory=list)


class ReturnRecord(BaseModel):
    customer: str
    product: str
    qty: int
    reason: str
    date: str


class QualityReturnsResponse(BaseModel):
    """退货分析响应"""
    total_returns: int = 0
    total_cases: int = 0
    high_severity_cases: int = 0
    by_reason: Dict[str, int] = Field(default_factory=dict)
    by_customer: Dict[str, int] = Field(default_factory=dict)
    recent_high_severity: List[ReturnRecord] = Field(default_factory=list)


class QualityRootCauseResponse(BaseModel):
    """根因追溯响应"""
    defect: str = ""
    total_production_defects: int = 0
    total_customer_returns: int = 0
    affected_lines: List[str] = Field(default_factory=list)
    affected_customers: List[str] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    probable_cause: str = ""
    recommended_actions: List[str] = Field(default_factory=list)


class QualityComplaintsResponse(BaseModel):
    """投诉分类响应"""
    classification: Dict[str, Any] = Field(default_factory=dict)
    total: int = 0


# ============================================================
# Market Agent 合约
# ============================================================

class CompetitorProfile(BaseModel):
    name: str
    stock: str = ""
    revenue_2025: str = ""
    yoy_growth: str = ""
    market_share: str = ""
    main_clients: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class MarketCompetitorResponse(BaseModel):
    """竞对监控响应"""
    competitors: List[CompetitorProfile] = Field(default_factory=list)
    sprocomm_position: Dict[str, str] = Field(default_factory=dict)


class SentimentSignal(BaseModel):
    source: str
    sentiment: str  # "正面" | "中性" | "负面"
    content: str


class MarketSentimentResponse(BaseModel):
    """市场情绪响应"""
    total_signals: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    sentiment_score: float = 0.0
    signals: List[SentimentSignal] = Field(default_factory=list)


class MarketReportResponse(BaseModel):
    """市场报告响应"""
    report_date: str = ""
    market_size: str = ""
    growth_trend: str = ""
    key_drivers: List[str] = Field(default_factory=list)
    sprocomm_position: str = ""


# ============================================================
# Finance Agent 合约
# ============================================================

class ARHighRisk(BaseModel):
    customer: str
    invoice: str
    amount: str
    aging: str


class FinanceARResponse(BaseModel):
    """应收账款响应"""
    total_outstanding: str = ""
    total_overdue: str = ""
    overdue_ratio: str = ""
    aging_analysis: Dict[str, str] = Field(default_factory=dict)
    by_customer: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    high_risk: List[ARHighRisk] = Field(default_factory=list)


class FinanceMarginResponse(BaseModel):
    """毛利分析响应"""
    total_revenue: str = ""
    total_cogs: str = ""
    gross_margin: str = ""
    by_product: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    by_customer: Dict[str, Dict[str, str]] = Field(default_factory=dict)


class CashflowMonth(BaseModel):
    month: str
    expected_inflow: str
    expected_outflow: str
    net_cashflow: str


class FinanceCashflowResponse(BaseModel):
    """现金流预测响应"""
    forecast_period: str = ""
    monthly_forecast: List[CashflowMonth] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class InvoiceRecord(BaseModel):
    invoice_no: str
    customer: str
    amount: str
    status: str
    due_date: str
    aging: str


class FinanceInvoiceResponse(BaseModel):
    """发票匹配响应"""
    matched: int = 0
    invoices: List[InvoiceRecord] = Field(default_factory=list)


# ============================================================
# Procurement Agent 合约
# ============================================================

class SupplierQuote(BaseModel):
    name: str
    category: str = ""
    overall_score: float = 0.0
    price_index: float = 0.0
    quality_score: str = ""
    on_time_rate: str = ""
    defect_rate: str = ""
    lead_time: str = ""
    credit: str = ""


class ProcurementQuoteResponse(BaseModel):
    """供应商比价响应"""
    category: str = ""
    suppliers: List[SupplierQuote] = Field(default_factory=list)
    recommendation: str = ""
    analysis_time: str = ""


class POItem(BaseModel):
    item: str
    qty: int


class PurchaseOrder(BaseModel):
    po_id: str
    supplier: str
    items: List[POItem] = Field(default_factory=list)
    total_amount: str = ""
    status: str = ""
    expected: str = ""
    is_delayed: bool = False


class ProcurementPOResponse(BaseModel):
    """采购单追踪响应"""
    total_orders: int = 0
    delayed: int = 0
    orders: List[PurchaseOrder] = Field(default_factory=list)


class DelayAlert(BaseModel):
    po_id: str
    supplier: str
    items: List[POItem] = Field(default_factory=list)
    expected: str = ""
    days_overdue: int = 0
    amount_at_risk: str = ""
    severity: str = ""


class ProcurementDelayResponse(BaseModel):
    """延期预警响应"""
    total_delayed: int = 0
    total_at_risk: str = ""
    alerts: List[DelayAlert] = Field(default_factory=list)


class ProcurementCostResponse(BaseModel):
    """成本分析响应"""
    period: str = ""
    total_spend: str = ""
    by_supplier: Dict[str, str] = Field(default_factory=dict)
    top_supplier: str = ""
    optimization_suggestions: List[str] = Field(default_factory=list)


# ============================================================
# Risk Agent 合约
# ============================================================

class Anomaly(BaseModel):
    client: str = Field(..., alias="客户")
    type: str
    month: str
    detail: str
    severity: str
    score: float

    model_config = {"populate_by_name": True}


class RiskAnomalyResponse(BaseModel):
    """异常检测响应"""
    total_anomalies: int = 0
    severe: int = 0
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    risk_summary: str = ""


class HealthScore(BaseModel):
    client: str = Field(..., alias="客户")
    total_score: int = Field(..., alias="总分")
    grade: str = Field(..., alias="等级")
    risk_tags: List[str] = Field(default_factory=list, alias="风险标签")
    suggestion: str = Field(default="", alias="建议")

    model_config = {"populate_by_name": True}


class RiskHealthResponse(BaseModel):
    """健康评分响应"""
    avg_score: float = 0.0
    grade_distribution: Dict[str, int] = Field(default_factory=dict)
    scores: List[Dict[str, Any]] = Field(default_factory=list)
    critical_clients: List[Dict[str, Any]] = Field(default_factory=list)


class ChurnAlert(BaseModel):
    client: str = Field(..., alias="客户")
    annual_amount: str = Field(..., alias="年度金额")
    risk_level: str = Field(..., alias="风险")
    reason: str = Field(..., alias="原因")

    model_config = {"populate_by_name": True}


class RiskChurnResponse(BaseModel):
    """流失预警响应"""
    high_risk: int = 0
    medium_risk: int = 0
    low_risk: int = 0
    high_risk_amount: str = ""
    total_monitored: int = 0
    exposure_rate: str = ""
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    action_required: List[str] = Field(default_factory=list)


class RiskAssessmentResponse(BaseModel):
    """综合风险评估响应"""
    overall_risk: str = ""
    anomaly_summary: str = ""
    health_avg: float = 0.0
    churn_high_risk: int = 0
    churn_exposure: str = ""
    top_risks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ============================================================
# Strategist Agent 合约
# ============================================================

class StrategistBenchmarkResponse(BaseModel):
    """行业对标响应"""
    report_date: str = ""
    positioning: Optional[Dict[str, Any]] = None
    competitive: Optional[Dict[str, Any]] = None
    key_insight: str = ""


class StrategistForecastResponse(BaseModel):
    """营收预测响应"""
    forecast: Optional[Dict[str, Any]] = None
    methodology: str = ""
    confidence: str = ""


class StrategistAdviceResponse(BaseModel):
    """战略建议响应"""
    structural_risks: Optional[List[Dict[str, Any]]] = None
    strategic_opportunities: Optional[List[Dict[str, Any]]] = None
    priority_actions: List[str] = Field(default_factory=list)
    kpi_targets_2026: Optional[Dict[str, str]] = None


class StrategistComprehensiveResponse(BaseModel):
    """综合战略响应"""
    positioning: Optional[Dict[str, Any]] = None
    competitive_landscape: Optional[Dict[str, Any]] = None
    risks: List[Dict[str, Any]] = Field(default_factory=list)
    opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    forecast_summary: Optional[Dict[str, Any]] = None
    executive_summary: str = ""


# ============================================================
# StateGraph 状态合约
# ============================================================

class GraphInput(BaseModel):
    """LangGraph StateGraph 输入合约"""
    question: str
    context_data: str = ""
    provider: str = "claude"
    api_key: str = ""
    enable_tools: bool = True
    enable_critic: bool = True
    enable_hitl: bool = True


class GraphOutput(BaseModel):
    """LangGraph StateGraph 输出合约"""
    final_answer: str = ""
    agents_used: List[str] = Field(default_factory=list)
    critique_score: float = 0.0
    hitl_approved: bool = True
    thinking: List[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    high_risk_alerts: List[Dict[str, Any]] = Field(default_factory=list)
