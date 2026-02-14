#!/usr/bin/env python3
"""
MRARFAI Risk Intelligence Agent v10.0
========================================
风险域 Agent — 异常检测、健康评分、流失预警

基于:
  - anomaly_detector.py: Z-Score/IQR/趋势断裂/波动率/系统性风险
  - health_score.py: 多维度客户健康评分

MCP Tools:
  - detect_anomaly: 异常检测
  - evaluate_health: 健康评分
  - churn_alert: 流失预警

A2A Skills:
  - risk_assessment: 综合风险评估
  - anomaly_scan: 异常扫描
  - health_check: 健康检查
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("mrarfai.agent.risk")


# ============================================================
# 内置样本数据 (无上传时使用)
# ============================================================

SAMPLE_RISK_CLIENTS = [
    {"客户": "Samsung India", "年度金额": 28500, "月度金额": [2800, 2600, 2900, 2750, 2400, 2100, 1800, 1600, 1500, 1400, 1200, 1100],
     "风险": "高", "原因": "连续6个月下滑，环比跌幅加速"},
    {"客户": "HMD/Nokia", "年度金额": 18200, "月度金额": [2200, 2100, 1800, 1500, 1200, 1300, 1400, 1600, 1500, 1400, 1200, 1000],
     "风险": "高", "原因": "HMD全球出货-21%，禾苗降幅-42%远超行业"},
    {"客户": "Transsion", "年度金额": 15800, "月度金额": [1000, 1100, 1200, 1300, 1400, 1500, 1400, 1350, 1300, 1250, 1200, 1150],
     "风险": "中", "原因": "增速放缓，非洲市场竞争加剧"},
    {"客户": "Lava India", "年度金额": 12500, "月度金额": [800, 900, 1000, 1100, 1050, 1100, 1150, 1100, 1050, 1000, 1050, 1100],
     "风险": "低", "原因": "PLI政策利好，CKD需求稳定增长"},
    {"客户": "ZTE", "年度金额": 9800, "月度金额": [600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1200],
     "风险": "低", "原因": "海外扩张驱动持续增长"},
    {"客户": "OPPO印度", "年度金额": 6500, "月度金额": [800, 750, 700, 600, 500, 450, 400, 350, 350, 400, 450, 500],
     "风险": "高", "原因": "H1暴跌后微弱恢复，总量大幅萎缩"},
]

SAMPLE_ANOMALIES = [
    {"客户": "Samsung India", "type": "趋势断裂", "month": "5月", "detail": "📉 连续增长后暴跌 -12.7%", "severity": "🔴", "score": 4.2},
    {"客户": "Samsung India", "type": "波动率异常", "month": "7月", "detail": "环比-14.3%, 历史波动率3.2%, 偏离4.5倍", "severity": "🔴", "score": 4.5},
    {"客户": "HMD/Nokia", "type": "Z-Score", "month": "12月", "detail": "📉 异常低 (z=-2.8)", "severity": "🔴", "score": 2.8},
    {"客户": "OPPO印度", "type": "趋势断裂", "month": "3月", "detail": "📉 增长趋势断裂 -16.7%", "severity": "🟡", "score": 1.7},
    {"客户": "Transsion", "type": "IQR异常", "month": "1月", "detail": "低于下界 (<950)", "severity": "🟡", "score": 2.0},
]

SAMPLE_HEALTH_SCORES = [
    {"客户": "ZTE", "总分": 82, "等级": "A", "营收贡献": 15, "增长趋势": 22, "稳定性": 18, "价格质量": 14, "订单频率": 13, "风险标签": [], "建议": "核心优质客户，加大资源投入"},
    {"客户": "Lava India", "总分": 75, "等级": "B", "营收贡献": 18, "增长趋势": 15, "稳定性": 16, "价格质量": 13, "订单频率": 13, "风险标签": ["印度依赖"], "建议": "稳定合作，关注PLI政策变化"},
    {"客户": "Transsion", "总分": 65, "等级": "B", "营收贡献": 20, "增长趋势": 12, "稳定性": 14, "价格质量": 10, "订单频率": 9, "风险标签": ["增速放缓"], "建议": "加强沟通，争取非洲新项目"},
    {"客户": "Samsung India", "总分": 45, "等级": "D", "营收贡献": 25, "增长趋势": 5, "稳定性": 6, "价格质量": 5, "订单频率": 4, "风险标签": ["持续下滑", "大客户流失"], "建议": "紧急制定挽留方案，CEO级拜访"},
    {"客户": "HMD/Nokia", "总分": 38, "等级": "D", "营收贡献": 22, "增长趋势": 3, "稳定性": 5, "价格质量": 4, "订单频率": 4, "风险标签": ["品牌转型", "ODM减少"], "建议": "主动对接HMD新品线，探索自有品牌合作"},
    {"客户": "OPPO印度", "总分": 32, "等级": "F", "营收贡献": 10, "增长趋势": 4, "稳定性": 5, "价格质量": 6, "订单频率": 7, "风险标签": ["量价齐跌", "竞争流失"], "建议": "评估是否降低该客户资源投入"},
]


# ============================================================
# Risk Agent Engine
# ============================================================

class RiskEngine:
    """风险分析引擎"""

    def __init__(self, risk_clients=None, anomalies=None, health_scores=None):
        self.risk_clients = risk_clients or SAMPLE_RISK_CLIENTS
        self.anomalies = anomalies or SAMPLE_ANOMALIES
        self.health_scores = health_scores or SAMPLE_HEALTH_SCORES

    @classmethod
    def from_pipeline(cls, data: dict, results: dict):
        """从 V9 销售数据管线创建 (有上传数据时)"""
        try:
            from anomaly_detector import run_full_detection
            detection = run_full_detection(data, results)
            anomalies = detection.get("top_anomalies", [])[:20]
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            anomalies = None

        try:
            from health_score import compute_health_scores
            health = compute_health_scores(data, results)
        except Exception as e:
            logger.error(f"健康评分失败: {e}")
            health = None

        # 从 results 提取流失预警
        risk_clients = results.get("流失预警", [])

        return cls(
            risk_clients=risk_clients if risk_clients else None,
            anomalies=anomalies,
            health_scores=health,
        )

    def detect_anomaly(self, client_name: str = "") -> Dict:
        """异常检测"""
        if client_name:
            filtered = [a for a in self.anomalies if client_name in a.get("客户", "")]
        else:
            filtered = self.anomalies

        severe = [a for a in filtered if a.get("severity") == "🔴"]
        return {
            "total_anomalies": len(filtered),
            "severe": len(severe),
            "anomalies": filtered[:15],
            "risk_summary": f"发现 {len(filtered)} 个异常，其中 {len(severe)} 个严重",
        }

    def evaluate_health(self, client_name: str = "") -> Dict:
        """健康评分"""
        if client_name:
            filtered = [s for s in self.health_scores if client_name in s.get("客户", "")]
        else:
            filtered = self.health_scores

        if not filtered:
            return {"message": "无健康评分数据", "scores": []}

        avg = sum(s["总分"] for s in filtered) / len(filtered)
        grade_dist = {}
        for s in filtered:
            g = s["等级"]
            grade_dist[g] = grade_dist.get(g, 0) + 1

        return {
            "avg_score": round(avg, 1),
            "grade_distribution": grade_dist,
            "scores": filtered,
            "critical_clients": [s for s in filtered if s["等级"] in ("D", "F")],
        }

    def churn_alert(self) -> Dict:
        """流失预警"""
        high = [c for c in self.risk_clients if c.get("风险") == "高"]
        medium = [c for c in self.risk_clients if c.get("风险") == "中"]
        low = [c for c in self.risk_clients if c.get("风险") == "低"]

        high_amount = sum(c.get("年度金额", 0) for c in high)
        total_amount = sum(c.get("年度金额", 0) for c in self.risk_clients)

        return {
            "high_risk": len(high),
            "medium_risk": len(medium),
            "low_risk": len(low),
            "high_risk_amount": f"¥{high_amount:,.0f}万",
            "total_monitored": len(self.risk_clients),
            "exposure_rate": f"{(high_amount / max(total_amount, 1)) * 100:.1f}%",
            "alerts": [
                {
                    "客户": c.get("客户", ""),
                    "年度金额": f"¥{c.get('年度金额', 0):,.0f}万",
                    "风险": c.get("风险", ""),
                    "原因": c.get("原因", ""),
                }
                for c in self.risk_clients
            ],
            "action_required": [c.get("客户", "") for c in high],
        }

    def comprehensive_assessment(self) -> Dict:
        """综合风险评估"""
        anomaly = self.detect_anomaly()
        health = self.evaluate_health()
        churn = self.churn_alert()

        # 综合风险等级
        severe_count = anomaly["severe"] + len(health.get("critical_clients", []))
        if severe_count >= 5:
            overall = "🔴 高风险"
        elif severe_count >= 2:
            overall = "🟡 中等风险"
        else:
            overall = "🟢 低风险"

        return {
            "overall_risk": overall,
            "anomaly_summary": anomaly["risk_summary"],
            "health_avg": health.get("avg_score", 0),
            "churn_high_risk": churn["high_risk"],
            "churn_exposure": churn["exposure_rate"],
            "top_risks": [
                f"{c.get('客户', '')} — {c.get('原因', '')}"
                for c in self.risk_clients if c.get("风险") == "高"
            ][:5],
            "recommendations": [
                "紧急拜访高风险客户：" + ", ".join(churn["action_required"][:3]),
                f"关注 {anomaly['severe']} 个严重异常指标",
                f"健康评分均值 {health.get('avg_score', 0):.1f}，D/F级客户需专项跟进",
            ],
        }

    def answer(self, question: str) -> str:
        """自然语言入口"""
        q = question.lower()
        if any(kw in q for kw in ["异常", "anomaly", "波动", "检测", "zscore"]):
            name = ""
            for c in self.risk_clients:
                cn = c.get("客户", "")
                if cn.lower() in q or cn in q:
                    name = cn
                    break
            return json.dumps(self.detect_anomaly(name), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["健康", "health", "评分", "score", "等级"]):
            name = ""
            for s in self.health_scores:
                cn = s.get("客户", "")
                if cn.lower() in q or cn in q:
                    name = cn
                    break
            return json.dumps(self.evaluate_health(name), ensure_ascii=False, indent=2)
        elif any(kw in q for kw in ["流失", "churn", "预警", "alert", "高风险"]):
            return json.dumps(self.churn_alert(), ensure_ascii=False, indent=2)
        else:
            return json.dumps(self.comprehensive_assessment(), ensure_ascii=False, indent=2)
