#!/usr/bin/env python3
"""
MRARFAI Market Intelligence Agent v10.0
=========================================
市场域 Agent — 竞品分析、行业趋势、舆情监控

MCP Tools:
  - monitor_competitor: 竞品监控
  - summarize_report: 行业报告摘要
  - track_sentiment: 舆情追踪

A2A Skills:
  - competitor_analysis: 竞品深度分析
  - trend_insight: 行业趋势洞察
  - market_overview: 市场全景
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("mrarfai.agent.market")

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
class Competitor:
    """竞品信息"""
    name: str
    stock_code: str = ""
    revenue_2025: float = 0.0  # 亿元
    yoy_growth: float = 0.0
    main_clients: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    market_share: float = 0.0  # ODM 手机市场份额


# ============================================================
# 模拟数据 (实际部署对接行业数据库)
# ============================================================

COMPETITORS = [
    Competitor(
        "华勤技术", "603296.SH", 850.0, 0.12,
        ["Samsung", "Xiaomi", "OPPO", "Vivo"],
        ["规模最大", "全品类覆盖", "Samsung核心供应商"],
        ["毛利率偏低", "人力成本高"],
        0.28
    ),
    Competitor(
        "闻泰科技", "600745.SH", 620.0, 0.08,
        ["Samsung", "Motorola/Lenovo", "Xiaomi"],
        ["半导体+ODM双轮驱动", "安世半导体并表"],
        ["整合周期长", "手机ODM占比下降"],
        0.18
    ),
    Competitor(
        "龙旗科技", "300750.SZ", 420.0, 0.15,
        ["Xiaomi", "Honor", "Realme"],
        ["5G方案领先", "小米第一ODM"],
        ["客户集中度高", "海外占比低"],
        0.14
    ),
    Competitor(
        "禾苗通讯(SPROCOMM)", "01401.HK", 42.0, 0.18,
        ["Samsung", "HMD/Nokia", "Lava", "Transsion"],
        ["功能机龙头", "Feature Phone+Entry Smartphone", "印度/非洲渠道强"],
        ["规模较小", "高端机型缺失"],
        0.03
    ),
]

INDUSTRY_TRENDS = [
    {
        "trend": "AI手机渗透率加速",
        "detail": "2026年AI手机出货量预计达5.2亿部，渗透率超40%",
        "impact": "ODM需具备AI算法集成能力",
        "source": "IDC 2026Q1",
    },
    {
        "trend": "印度制造本地化",
        "detail": "印度PLI政策推动本地组装占比超65%",
        "impact": "禾苗已有印度合作产线，可加大投入",
        "source": "Counterpoint 2026",
    },
    {
        "trend": "功能机市场缩减",
        "detail": "全球功能机出货量同比下降12%，但非洲仍增长5%",
        "impact": "禾苗需加速Entry Smartphone转型",
        "source": "IDC 2025Q4",
    },
    {
        "trend": "ODM整合趋势",
        "detail": "Top 5 ODM市占率从65%升至72%，小厂加速出清",
        "impact": "禾苗需差异化竞争，避免被边缘化",
        "source": "TrendForce 2026",
    },
]

SENTIMENT_DATA = [
    {"source": "分析师报告", "sentiment": "正面", "content": "禾苗Q3营收增长18%超预期，Samsung订单量价齐升"},
    {"source": "行业新闻", "sentiment": "中性", "content": "华勤拿下Samsung Galaxy A16 ODM大单，竞争加剧"},
    {"source": "供应链消息", "sentiment": "负面", "content": "触控IC供应链短缺，可能影响Q1交付"},
    {"source": "客户反馈", "sentiment": "正面", "content": "Nokia对禾苗T80品质评价提升，考虑扩大合作"},
]


# ============================================================
# 市场 Agent 核心逻辑
# ============================================================

class MarketEngine:
    """市场分析引擎"""

    def __init__(self, competitors=None, trends=None, sentiments=None):
        self.competitors = {c.name: c for c in (competitors or COMPETITORS)}
        self.trends = trends or INDUSTRY_TRENDS
        self.sentiments = sentiments or SENTIMENT_DATA

    @classmethod
    def from_dataframes(cls, competitors_df=None):
        """从 DataFrame 创建引擎（用于 Excel 上传）"""
        competitors = []
        if competitors_df is not None:
            for _, row in competitors_df.iterrows():
                try:
                    competitors.append(Competitor(
                        name=str(row.iloc[0]),
                        stock_code=str(row.iloc[1]) if len(row) > 1 else "",
                        revenue_2025=float(row.iloc[2]) if len(row) > 2 else 0,
                        yoy_growth=float(row.iloc[3]) / 100 if len(row) > 3 else 0,
                        main_clients=[s.strip() for s in str(row.iloc[4]).split(",")] if len(row) > 4 else [],
                        strengths=[s.strip() for s in str(row.iloc[5]).split(",")] if len(row) > 5 else [],
                        weaknesses=[s.strip() for s in str(row.iloc[6]).split(",")] if len(row) > 6 else [],
                        market_share=float(row.iloc[7]) / 100 if len(row) > 7 else 0,
                    ))
                except Exception:
                    continue

        return cls(competitors=competitors if competitors else None)

    def monitor_competitor(self, name: str = "") -> Dict:
        """竞品监控"""
        if name:
            targets = [c for c in COMPETITORS if name in c.name]
        else:
            targets = COMPETITORS

        return {
            "competitors": [
                {
                    "name": c.name,
                    "stock": c.stock_code,
                    "revenue_2025": f"¥{c.revenue_2025:.0f}亿",
                    "yoy_growth": f"{c.yoy_growth:.0%}",
                    "market_share": f"{c.market_share:.0%}",
                    "main_clients": c.main_clients,
                    "strengths": c.strengths,
                    "weaknesses": c.weaknesses,
                }
                for c in targets
            ],
            "sprocomm_position": {
                "rank": "ODM Top 10 (按营收)",
                "niche": "Feature Phone + Entry Smartphone 领域Top 3",
                "competitive_advantage": "印度/非洲渠道 + Samsung/Nokia信任关系",
            },
        }

    def summarize_report(self) -> Dict:
        """行业趋势报告"""
        return {
            "report_date": datetime.now().strftime("%Y年%m月"),
            "market_size": "全球手机ODM市场约2,800亿元(2025)",
            "yoy_growth": "整体+8%，智能手机+12%，功能机-12%",
            "trends": self.trends,
            "key_takeaway": "AI手机+印度本地化是2026两大确定性机会",
        }

    def track_sentiment(self) -> Dict:
        """舆情追踪"""
        positive = sum(1 for s in self.sentiments if s["sentiment"] == "正面")
        negative = sum(1 for s in self.sentiments if s["sentiment"] == "负面")
        return {
            "total_signals": len(self.sentiments),
            "positive": positive,
            "negative": negative,
            "neutral": len(self.sentiments) - positive - negative,
            "sentiment_score": round((positive - negative) / len(self.sentiments), 2),
            "signals": self.sentiments,
        }

    def answer(self, question: str) -> str:
        """自然语言入口"""
        q = question.lower()
        if any(kw in q for kw in ["竞品", "competitor", "华勤", "闻泰", "龙旗", "对手"]):
            name = ""
            for c in ["华勤", "闻泰", "龙旗"]:
                if c in q:
                    name = c
            return json.dumps(self.monitor_competitor(name), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["趋势", "trend", "行业", "报告", "report"]):
            return json.dumps(self.summarize_report(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["舆情", "sentiment", "新闻", "评价"]):
            return json.dumps(self.track_sentiment(), ensure_ascii=False, indent=2)
        else:
            overview = {
                "agent": "Market Intelligence",
                "competitors_tracked": len(self.competitors),
                "active_trends": len(self.trends),
                "sentiment_signals": len(self.sentiments),
                "capabilities": ["monitor_competitor", "summarize_report", "track_sentiment"],
            }
            return json.dumps(overview, ensure_ascii=False, indent=2)


# ============================================================
# A2A Executor
# ============================================================

class MarketExecutor(AgentExecutor if HAS_A2A else object):
    """市场 Agent A2A 执行器"""

    def __init__(self):
        self.engine = MarketEngine()

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
                name="market_result",
                description="市场分析结果",
                parts=[MessagePart(type="text", text=answer)],
            ))
        except Exception as e:
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message.agent_text(f"市场分析失败: {str(e)}"),
            )
        return task


# ============================================================
# Agent Card
# ============================================================

def create_market_card(base_url: str = "http://localhost:9999") -> 'AgentCard':
    """创建市场 Agent Card"""
    if not HAS_A2A:
        return None
    return AgentCard(
        name="MRARFAI 市场情报员",
        description="市场域智能Agent — 竞品监控、行业趋势、舆情追踪",
        version="10.0.0",
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/market")],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="competitor_analysis",
                name="竞品分析",
                description="华勤/闻泰/龙旗等ODM竞对深度分析",
                tags=["competitor", "market", "benchmark"],
                examples=["华勤最新动态", "竞品市场份额对比"],
            ),
            AgentSkill(
                id="trend_insight",
                name="趋势洞察",
                description="AI手机、印度制造、功能机转型等行业趋势",
                tags=["trend", "industry", "insight"],
                examples=["2026行业趋势", "AI手机对ODM的影响"],
            ),
            AgentSkill(
                id="market_overview",
                name="市场全景",
                description="ODM手机市场规模、增速、格局",
                tags=["market", "overview", "size"],
                examples=["ODM市场多大？", "禾苗排名第几？"],
            ),
        ],
        provider={"organization": "禾苗科技"},
    )
