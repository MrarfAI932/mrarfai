#!/usr/bin/env python3
"""
MRARFAI Strategic Advisory Agent v10.0
==========================================
战略域 Agent — 行业对标、营收预测、战略建议

基于:
  - industry_benchmark.py: 竞品对标、市场定位、结构性风险
  - forecast_engine.py: 营收预测、客户预测、风险场景

MCP Tools:
  - benchmark_industry: 行业对标分析
  - forecast_revenue: 营收预测
  - strategic_advice: 战略建议

A2A Skills:
  - strategic_planning: 战略规划
  - market_positioning: 市场定位
  - revenue_forecast: 营收预测
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("mrarfai.agent.strategist")


# ============================================================
# 内置样本数据 (无上传时使用)
# ============================================================

SAMPLE_POSITIONING = {
    "禾苗营收": "42.0亿",
    "行业位次": "第4梯队(华勤920亿>闻泰615亿>龙旗560亿>>禾苗42亿)",
    "规模差距": "仅为华勤的4.6%",
    "差异化优势": "毛利率9.5%(行业均值7.5%),聚焦中低端+功能机利基市场",
    "增速": "+54.1%(华勤+28%,龙旗+22%,闻泰+5%) 增速领先但基数最小",
}

SAMPLE_COMPETITIVE = {
    "营收(亿)": {"华勤": 920, "闻泰": 615, "龙旗": 560, "禾苗": 42},
    "增速": {"禾苗": "+54.1%", "华勤": "+28%", "龙旗": "+22%", "闻泰": "+5%"},
    "毛利率": {"禾苗": "9.5%", "闻泰": "8.2%", "行业均值": "7.5%", "龙旗": "7.1%", "华勤": "6.8%"},
    "核心优势": "禾苗毛利率领先 → 说明差异化定位有效，但规模瓶颈明显",
}

SAMPLE_RISKS = [
    {"风险": "功能机结构性萎缩", "行业": "全球功能机年萎缩约15%",
     "禾苗": "HMD(最大FP客户)下滑42%,功能机是传统核心",
     "建议": "FP维持利润贡献但不应追加投入,加速SP/平板/IoT转型"},
    {"风险": "印度市场过度依赖", "行业": "印度智能机仅+4%,功能机-19%",
     "禾苗": "印度占出货61.7%",
     "建议": "中东/拉美/非洲作为分散化重点"},
    {"风险": "HMD客户持续流失", "行业": "HMD全球-21%,Nokia授权不确定",
     "禾苗": "HMD从第一大降到第三,年度-42%",
     "建议": "主动对接HMD新品线,开发Entry SP替代FP"},
    {"风险": "客户集中度高", "行业": "ODM行业健康线Top3<50%",
     "禾苗": "Top3客户占69.7%",
     "建议": "积极开拓新客户如realme/ITEL/Infinix"},
]

SAMPLE_OPPORTUNITIES = [
    {"机会": "AI手机ODM", "时间窗口": "2026-2027", "投入": "1-2亿研发",
     "潜力": "AI手机渗透率40%+, 均价提升15%",
     "行动": "联合芯片商(紫光/联发科)开发AI方案模板"},
    {"机会": "印度PLI深化", "时间窗口": "2026持续", "投入": "产线升级",
     "潜力": "PLI补贴+本地化率65%",
     "行动": "扩大印度CKD产能,争取Lava/Micromax新项目"},
    {"机会": "非洲Entry SP", "时间窗口": "2026-2028", "投入": "渠道建设",
     "潜力": "非洲功能机→智能机转换期,年增5%",
     "行动": "Transsion合作深化, 开拓TECNO/Infinix供应链"},
    {"机会": "IoT/平板/TWS", "时间窗口": "2026-2028", "投入": "团队+产线",
     "潜力": "IoT ODM市场年增18%",
     "行动": "成立IoT事业部, 试点平板/TWS代工"},
]

SAMPLE_FORECAST = {
    "Q1_2026预测": "¥11,200万",
    "全年2026预测": {
        "乐观(+20%)": "¥50,400万",
        "基准(+10%)": "¥46,200万",
        "悲观(-5%)": "¥39,900万",
    },
    "增长驱动": ["ZTE海外扩张(+40%)", "Lava印度PLI(+15%)", "Transsion非洲(+8%)"],
    "风险因素": ["HMD持续下滑(-20%)", "功能机萎缩(-15%)", "Samsung竞争加剧"],
}


# ============================================================
# Strategist Agent Engine
# ============================================================

class StrategistEngine:
    """战略顾问引擎"""

    def __init__(self, positioning=None, competitive=None, risks=None,
                 opportunities=None, forecast=None):
        self.positioning = positioning or SAMPLE_POSITIONING
        self.competitive = competitive or SAMPLE_COMPETITIVE
        self.risks = risks or SAMPLE_RISKS
        self.opportunities = opportunities or SAMPLE_OPPORTUNITIES
        self.forecast = forecast or SAMPLE_FORECAST

    @classmethod
    def from_pipeline(cls, data: dict, results: dict):
        """从 V9 销售数据管线创建 (有上传数据时)"""
        positioning = None
        competitive = None
        risks = None
        opportunities = None
        forecast = None

        try:
            from industry_benchmark import IndustryBenchmark
            bench = IndustryBenchmark(data, results)
            bench_result = bench.run()
            positioning = bench_result.get("市场定位")
            competitive = bench_result.get("竞争对标")
            risks = bench_result.get("结构性风险")
            opportunities = bench_result.get("战略机会")
        except Exception as e:
            logger.error(f"行业对标失败: {e}")

        try:
            from forecast_engine import ForecastEngine
            fc = ForecastEngine(data, results)
            fc_result = fc.run()
            forecast = {
                "Q1_2026预测": fc_result.get("总营收预测", {}),
                "客户预测": fc_result.get("客户预测", [])[:5],
                "品类预测": fc_result.get("品类预测", [])[:5],
                "风险场景": fc_result.get("风险场景", {}),
            }
        except Exception as e:
            logger.error(f"预测引擎失败: {e}")

        return cls(
            positioning=positioning,
            competitive=competitive,
            risks=risks,
            opportunities=opportunities,
            forecast=forecast,
        )

    def benchmark_industry(self) -> Dict:
        """行业对标分析"""
        return {
            "report_date": datetime.now().strftime("%Y年%m月"),
            "positioning": self.positioning,
            "competitive": self.competitive,
            "key_insight": "禾苗增速领先但规模差距大，毛利率优势说明利基策略有效",
        }

    def forecast_revenue(self) -> Dict:
        """营收预测"""
        return {
            "forecast": self.forecast,
            "methodology": "同比外推(35%) + 季节性因子(35%) + 占比法(30%)",
            "confidence": "基准场景置信度70%",
        }

    def strategic_advice(self) -> Dict:
        """战略建议"""
        return {
            "structural_risks": self.risks,
            "strategic_opportunities": self.opportunities,
            "priority_actions": [
                "🔴 紧急: HMD客户挽留方案 — CEO级沟通",
                "🟡 短期: AI手机方案模板 — 联合芯片商Q2完成",
                "🟢 中期: IoT事业部组建 — Q3试点平板代工",
                "🔵 长期: 中东/拉美渠道 — 2026下半年启动",
            ],
            "kpi_targets_2026": {
                "营收": "46-50亿(+10~20%)",
                "毛利率": "维持>9%",
                "客户集中度": "Top3<60%",
                "新客户": "≥3家新品牌客户",
                "功能机占比": "降至<30%",
            },
        }

    def comprehensive_strategy(self) -> Dict:
        """综合战略分析"""
        return {
            "positioning": self.positioning,
            "competitive_landscape": self.competitive,
            "risks": self.risks[:3],
            "opportunities": self.opportunities[:3],
            "forecast_summary": self.forecast,
            "executive_summary": (
                "禾苗2025年增速+54.1%领跑ODM行业，毛利率9.5%高于行业均值。"
                "但功能机萎缩、HMD流失、印度过度依赖三大结构性风险突出。"
                "2026战略重心: AI手机方案能力 + 非洲/中东市场 + IoT新品类。"
            ),
        }

    def answer(self, question: str) -> str:
        """自然语言入口"""
        q = question.lower()
        if any(kw in q for kw in ["对标", "benchmark", "竞争", "定位", "华勤", "闻泰", "龙旗"]):
            return json.dumps(self.benchmark_industry(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["预测", "forecast", "q1", "2026", "营收预测"]):
            return json.dumps(self.forecast_revenue(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["建议", "advice", "战略", "strategy", "行动", "action"]):
            return json.dumps(self.strategic_advice(), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["机会", "opportunity", "增长", "growth"]):
            return json.dumps({"opportunities": self.opportunities}, ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["风险", "risk", "威胁", "threat"]):
            return json.dumps({"risks": self.risks}, ensure_ascii=False, indent=2)
        else:
            return json.dumps(self.comprehensive_strategy(), ensure_ascii=False, indent=2)
