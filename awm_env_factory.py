#!/usr/bin/env python3
"""
MRARFAI V9.0 — AWM 合成环境工厂
=================================
融合两篇前沿论文:
  ① Agent World Model (Snowflake, arXiv 2602.10090, Feb 2026)
     - 合成环境生成管线: Scenario → Tasks → SQLite DB → MCP API → Verifier
     - 1,000 环境 × 35 工具/环境 × 数据库状态一致性
     - 开源: Snowflake-Labs/agent-world-model
  ② Agent Workflow Memory (ICML 2025, CMU, arXiv 2409.07429)
     - 工作流归纳: 从 Agent 轨迹中提取可复用 routine
     - 滚雪球效应: 简单workflow → 组合成复杂workflow
     - +24.6% / +51.1% 相对成功率 (Mind2Web / WebArena)

MRARFAI 定制:
  - 场景: ODM/OEM 手机平板出货分析（禾苗通讯垂直领域）
  - 数据: 合成 SQLite 数据库（品牌×品类×月度×客户×区域）
  - 工具: 与现有 tool_registry.py / mcp_server.py 对齐
  - 验证: SQL 状态检查作为 eval ground truth
  - 工作流: 分析轨迹 → 归纳可复用分析模板

与 V9.0 其他模块的协同:
  - AWM × RLM: 合成环境作为 RLM REPL 的数据源，递归分析合成数据
  - AWM × EnCompass: 多分支搜索在合成环境中零成本探索
  - AWM × AgentSkiller: DAG 编排在合成环境中验证故障恢复
  - AWM × Memory Survey: 跨合成环境共享 workflow 记忆

使用:
  factory = AWMEnvironmentFactory()
  envs = factory.generate_batch(num_envs=50, domain="odm_shipment")
  
  evaluator = AWMEvalIntegration(factory)
  report = evaluator.run_eval_suite(agent_fn=my_agent)
  
  memory = AWMWorkflowMemory()
  memory.ingest_trajectory(trajectory)
  workflows = memory.get_workflows(query="品牌月度趋势分析")
"""

import json
import os
import time
import uuid
import random
import hashlib
import sqlite3
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger("mrarfai.awm")


# ╔══════════════════════════════════════════╗
# ║  Part 1: 场景定义 (Scenario Seeds)       ║
# ╚══════════════════════════════════════════╝

class ScenarioDomain(Enum):
    ODM_SHIPMENT = "odm_shipment"
    ODM_REVENUE = "odm_revenue"
    SUPPLY_CHAIN = "supply_chain"
    CUSTOMER_PORTFOLIO = "customer_portfolio"
    MARKET_FORECAST = "market_forecast"
    RISK_ASSESSMENT = "risk_assessment"
    COMPETITIVE_INTEL = "competitive_intel"
    FINANCIAL_HEALTH = "financial_health"


@dataclass
class ScenarioSeed:
    domain: ScenarioDomain
    name: str
    description: str
    entities: List[str]
    tools_required: List[str]
    complexity: str = "medium"
    data_volume: str = "medium"
    time_span_months: int = 12
    num_brands: int = 10
    num_categories: int = 5
    num_regions: int = 8


SCENARIO_TEMPLATES: List[ScenarioSeed] = [
    ScenarioSeed(
        domain=ScenarioDomain.ODM_SHIPMENT, name="标准出货量分析",
        description="分析多品牌手机/平板月度出货量趋势、同比环比、品类占比",
        entities=["brand", "category", "month", "quantity"],
        tools_required=["query_shipment", "calc_trend", "detect_anomaly"],
    ),
    ScenarioSeed(
        domain=ScenarioDomain.ODM_REVENUE, name="金额收入分析",
        description="分析品牌×品类×月度收入分布、ASP趋势、毛利变化",
        entities=["brand", "category", "month", "revenue", "asp"],
        tools_required=["query_revenue", "calc_asp", "calc_margin"],
    ),
    ScenarioSeed(
        domain=ScenarioDomain.CUSTOMER_PORTFOLIO, name="客户集中度风险",
        description="Top-N客户集中度、客户流失预警、新客户开发追踪",
        entities=["customer", "brand", "revenue", "order_count", "churn_risk"],
        tools_required=["query_customer", "calc_concentration", "predict_churn"],
        complexity="hard",
    ),
    ScenarioSeed(
        domain=ScenarioDomain.SUPPLY_CHAIN, name="供应链交期分析",
        description="订单交付周期、延迟率、物料缺口预警",
        entities=["order", "sku", "delivery_date", "delay_days"],
        tools_required=["query_orders", "calc_otd", "predict_delay"],
        complexity="hard", data_volume="large",
    ),
    ScenarioSeed(
        domain=ScenarioDomain.MARKET_FORECAST, name="市场预测与定价",
        description="品类市场增速预测、竞争定价分析、市占率估算",
        entities=["market", "category", "growth_rate", "price_band"],
        tools_required=["query_market", "forecast_growth", "analyze_pricing"],
        complexity="extreme",
    ),
    ScenarioSeed(
        domain=ScenarioDomain.RISK_ASSESSMENT, name="多维风险评估",
        description="客户信用、地缘政治、汇率、季节性等多因子风险评估",
        entities=["risk_factor", "score", "weight", "trend"],
        tools_required=["query_risk", "calc_risk_score", "recommend_mitigation"],
        complexity="extreme",
    ),
    ScenarioSeed(
        domain=ScenarioDomain.COMPETITIVE_INTEL, name="竞对动态追踪",
        description="华勤/闻泰/龙旗出货对比、品类策略分析、市场份额变化",
        entities=["competitor", "category", "volume", "share"],
        tools_required=["query_competitor", "compare_share", "detect_strategy"],
        complexity="hard",
    ),
    ScenarioSeed(
        domain=ScenarioDomain.FINANCIAL_HEALTH, name="客户财务健康度",
        description="应收账款、回款周期、坏账风险、客户健康评分",
        entities=["customer", "ar", "dso", "payment_history"],
        tools_required=["query_finance", "calc_dso", "score_health"],
        complexity="hard",
    ),
]


# ╔══════════════════════════════════════════╗
# ║  Part 2: 合成数据生成器                   ║
# ╚══════════════════════════════════════════╝

BRAND_POOLS = {
    "tier1": ["Samsung", "Apple", "Xiaomi", "OPPO", "vivo", "Huawei"],
    "tier2": ["realme", "Motorola", "Nokia", "Honor", "OnePlus", "Google Pixel"],
    "tier3": ["Tecno", "Infinix", "iTel", "ZTE", "Alcatel", "TCL"],
    "emerging": ["Nothing", "Fairphone", "CMF", "POCO", "Redmi"],
    "white_label": ["BrandX", "OEM-Direct", "Private-Label", "ValueTech"],
}

CATEGORY_POOLS = [
    "智能手机_旗舰", "智能手机_中端", "智能手机_入门",
    "平板_高端", "平板_教育", "平板_入门",
    "智能穿戴", "TWS耳机", "智能家居终端",
    "功能手机", "工业终端", "车载终端",
]

REGION_POOLS = [
    "中国大陆", "东南亚", "南亚", "中东非洲", "欧洲",
    "拉美", "北美", "日韩", "独联体", "大洋洲",
]

CUSTOMER_POOLS = [
    "Samsung_Mobile", "Xiaomi_Corp", "OPPO_Global", "vivo_CN",
    "Motorola_NA", "Google_HW", "Nokia_MEA", "realme_APAC",
    "Tecno_Africa", "Honor_EU", "Nothing_Tech", "ZTE_Corp",
    "BrandX_ODM", "ValueTech_WL", "CMF_Direct", "Amazon_PL",
]


@dataclass
class SyntheticDataConfig:
    num_brands: int = 10
    num_categories: int = 5
    num_regions: int = 6
    num_customers: int = 12
    num_months: int = 12
    start_year: int = 2025
    start_month: int = 1
    seasonal_pattern: bool = True
    growth_trend: bool = True
    anomaly_injection: bool = True
    anomaly_rate: float = 0.05
    concentration_risk: bool = True
    missing_data_rate: float = 0.02
    min_quantity: int = 100
    max_quantity: int = 500000
    min_asp: float = 20.0
    max_asp: float = 1200.0


class SyntheticDataGenerator:
    """
    合成数据生成器 — AWM Pipeline Stage 2
    参考 Agent World Model: Scenario → Tasks → DB Schema → Sample Data → Verification
    定制为 ODM/OEM 出货数据格式，与禾苗实际数据结构对齐。
    """

    def __init__(self, config: SyntheticDataConfig = None, seed: int = None):
        self.config = config or SyntheticDataConfig()
        self.rng = random.Random(seed or int(time.time()))
        self._select_pools()

    def _select_pools(self):
        all_brands = []
        for tier_brands in BRAND_POOLS.values():
            all_brands.extend(tier_brands)
        self.rng.shuffle(all_brands)
        self.brands = all_brands[:self.config.num_brands]
        cats = list(CATEGORY_POOLS)
        self.rng.shuffle(cats)
        self.categories = cats[:self.config.num_categories]
        regions = list(REGION_POOLS)
        self.rng.shuffle(regions)
        self.regions = regions[:self.config.num_regions]
        customers = list(CUSTOMER_POOLS)
        self.rng.shuffle(customers)
        self.customers = customers[:self.config.num_customers]

    def _generate_months(self) -> List[str]:
        months = []
        y, m = self.config.start_year, self.config.start_month
        for _ in range(self.config.num_months):
            months.append(f"{y}-{m:02d}")
            m += 1
            if m > 12:
                m = 1
                y += 1
        return months

    def _seasonal_factor(self, month: int) -> float:
        base = 1.0
        if self.config.seasonal_pattern:
            base += 0.15 * math.sin(2 * math.pi * (month - 6) / 12)
            base += 0.10 * math.sin(2 * math.pi * (month - 11) / 12)
            if month in (1, 2):
                base *= 0.85
        return max(0.5, base)

    def _growth_factor(self, month_idx: int) -> float:
        if self.config.growth_trend:
            annual_rate = 0.08 + self.rng.random() * 0.17
            monthly_rate = (1 + annual_rate) ** (1 / 12)
            return monthly_rate ** month_idx
        return 1.0

    def _inject_anomaly(self, value: float) -> Tuple[float, bool]:
        if self.config.anomaly_injection and self.rng.random() < self.config.anomaly_rate:
            r = self.rng.random()
            if r < 0.6:
                return value * (2.0 + self.rng.random() * 3.0), True
            elif r < 0.9:
                return value * (0.1 + self.rng.random() * 0.3), True
            else:
                return 0, True
        return value, False

    def generate_shipment_db(self, db_path: str) -> Dict[str, Any]:
        months = self._generate_months()
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS shipments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT NOT NULL, category TEXT NOT NULL,
                region TEXT NOT NULL, month TEXT NOT NULL,
                quantity INTEGER NOT NULL, is_anomaly INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS revenue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT NOT NULL, category TEXT NOT NULL,
                month TEXT NOT NULL, revenue_usd REAL NOT NULL,
                quantity INTEGER NOT NULL, asp_usd REAL NOT NULL,
                margin_pct REAL DEFAULT 0, is_anomaly INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                tier TEXT DEFAULT 'B', region TEXT,
                credit_score REAL DEFAULT 70.0,
                first_order_date TEXT, is_active INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS customer_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL, brand TEXT, category TEXT,
                month TEXT NOT NULL, order_quantity INTEGER NOT NULL,
                order_value_usd REAL NOT NULL, payment_days INTEGER DEFAULT 30,
                is_anomaly INTEGER DEFAULT 0,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );
            CREATE TABLE IF NOT EXISTS anomaly_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL, record_id INTEGER NOT NULL,
                anomaly_type TEXT NOT NULL, severity TEXT DEFAULT 'medium',
                description TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ship_bm ON shipments(brand, month);
            CREATE INDEX IF NOT EXISTS idx_rev_bm ON revenue(brand, month);
            CREATE INDEX IF NOT EXISTS idx_co_cm ON customer_orders(customer_id, month);
        """)

        anomalies_meta = []
        total_rows = 0

        # 品牌基础出货量
        brand_base = {}
        for brand in self.brands:
            tf = 1.0
            if brand in BRAND_POOLS["tier1"]:
                tf = self.rng.uniform(3.0, 10.0)
            elif brand in BRAND_POOLS["tier2"]:
                tf = self.rng.uniform(1.0, 3.0)
            elif brand in BRAND_POOLS["tier3"]:
                tf = self.rng.uniform(0.5, 1.5)
            brand_base[brand] = tf

        # 生成出货量
        for brand in self.brands:
            for cat in self.categories:
                cf = 1.0
                if "旗舰" in cat: cf = 0.3
                elif "入门" in cat: cf = 2.5
                elif "平板" in cat: cf = 0.4

                for mi, month in enumerate(months):
                    mn = int(month.split("-")[1])
                    for region in self.regions:
                        base = self.config.min_quantity + self.rng.random() * (
                            self.config.max_quantity - self.config.min_quantity) * 0.1
                        base *= brand_base[brand] * cf
                        base *= self._seasonal_factor(mn) * self._growth_factor(mi)
                        base *= 0.5 + self.rng.random() * 1.5
                        qty = int(base)
                        qty_f, is_anom = self._inject_anomaly(float(qty))
                        qty = max(0, int(qty_f))
                        if self.rng.random() < self.config.missing_data_rate:
                            continue
                        cur.execute(
                            "INSERT INTO shipments (brand,category,region,month,quantity,is_anomaly) VALUES (?,?,?,?,?,?)",
                            (brand, cat, region, month, qty, int(is_anom)))
                        rid = cur.lastrowid
                        total_rows += 1
                        if is_anom:
                            at = "spike" if qty_f > base else ("drop" if qty > 0 else "zero")
                            anomalies_meta.append({"table": "shipments", "id": rid, "type": at, "brand": brand, "month": month})
                            cur.execute(
                                "INSERT INTO anomaly_labels (table_name,record_id,anomaly_type,severity,description) VALUES (?,?,?,?,?)",
                                ("shipments", rid, at, "high" if at == "zero" else "medium",
                                 f"{brand}/{cat}/{month}/{region}: {at}"))

        # 生成收入
        for brand in self.brands:
            for cat in self.categories:
                if "旗舰" in cat: asp_b = self.rng.uniform(400, 1200)
                elif "中端" in cat: asp_b = self.rng.uniform(150, 400)
                elif "入门" in cat or "功能" in cat: asp_b = self.rng.uniform(20, 150)
                elif "平板" in cat: asp_b = self.rng.uniform(100, 600)
                else: asp_b = self.rng.uniform(30, 200)

                for mi, month in enumerate(months):
                    cur.execute(
                        "SELECT COALESCE(SUM(quantity),0) FROM shipments WHERE brand=? AND category=? AND month=?",
                        (brand, cat, month))
                    tq = cur.fetchone()[0]
                    if tq == 0: continue
                    asp = asp_b * (1 + self.rng.uniform(-0.03, 0.05) * mi / 12)
                    rev = tq * asp
                    margin = self.rng.uniform(0.03, 0.15)
                    rev_f, is_anom = self._inject_anomaly(rev)
                    cur.execute(
                        "INSERT INTO revenue (brand,category,month,revenue_usd,quantity,asp_usd,margin_pct,is_anomaly) VALUES (?,?,?,?,?,?,?,?)",
                        (brand, cat, month, round(rev_f, 2), tq, round(asp, 2), round(margin, 4), int(is_anom)))
                    if is_anom:
                        cur.execute(
                            "INSERT INTO anomaly_labels (table_name,record_id,anomaly_type,severity,description) VALUES (?,?,?,?,?)",
                            ("revenue", cur.lastrowid, "revenue_anomaly", "high", f"{brand}/{cat}/{month}: rev anomaly"))

        # 生成客户
        tiers = ["S", "A", "B", "C"]
        for i, cust in enumerate(self.customers):
            tier = tiers[min(i // 3, 3)]
            region = self.rng.choice(self.regions)
            credit = self.rng.uniform(80, 99) if tier == "S" else self.rng.uniform(40, 95)
            cur.execute(
                "INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?,?)",
                (f"CUST-{i+1:03d}", cust, tier, region, round(credit, 1),
                 f"2024-{self.rng.randint(1,12):02d}-01", 1 if self.rng.random() > 0.1 else 0))

        # 生成客户订单
        for i, cust in enumerate(self.customers):
            cid = f"CUST-{i+1:03d}"
            scale = self.rng.uniform(3.0, 8.0) if self.config.concentration_risk and i < 3 else self.rng.uniform(0.3, 1.5)
            for month in months:
                if self.rng.random() < 0.15: continue
                brand = self.rng.choice(self.brands)
                cat = self.rng.choice(self.categories)
                qty = int(self.rng.uniform(500, 50000) * scale)
                val = qty * self.rng.uniform(50, 500)
                pd = self.rng.choice([15, 30, 45, 60, 90])
                qty_f, is_anom = self._inject_anomaly(float(qty))
                cur.execute(
                    "INSERT INTO customer_orders (customer_id,brand,category,month,order_quantity,order_value_usd,payment_days,is_anomaly) VALUES (?,?,?,?,?,?,?,?)",
                    (cid, brand, cat, month, int(qty_f), round(val, 2), pd, int(is_anom)))

        conn.commit()
        stats = {}
        for t in ["shipments", "revenue", "customers", "customer_orders", "anomaly_labels"]:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            stats[t] = cur.fetchone()[0]
        conn.close()

        return {
            "db_path": db_path, "brands": self.brands, "categories": self.categories,
            "regions": self.regions, "customers": self.customers, "months": months,
            "table_counts": stats, "total_anomalies": len(anomalies_meta),
            "anomalies": anomalies_meta[:20], "config": asdict(self.config),
        }


# ╔══════════════════════════════════════════╗
# ║  Part 3: 任务与验证器生成                 ║
# ╚══════════════════════════════════════════╝

@dataclass
class AWMTask:
    task_id: str
    question: str
    category: str
    difficulty: str
    expected_sql: str = ""
    verification_sql: str = ""
    verification_check: str = ""
    expected_keywords: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    context_hint: str = ""
    ground_truth: str = ""
    metadata: Dict = field(default_factory=dict)


class TaskGenerator:
    TEMPLATES = {
        "easy": [
            {"q": "{month}月{brand}的{category}出货量是多少？", "cat": "shipment",
             "sql": "SELECT SUM(quantity) FROM shipments WHERE brand='{brand}' AND category='{category}' AND month='{month}'",
             "v_sql": "SELECT SUM(quantity) FROM shipments WHERE brand='{brand}' AND category='{category}' AND month='{month}'",
             "v_chk": ">0", "kw": ["{brand}", "出货", "数量"]},
            {"q": "哪个品牌在{month}月出货量最高？", "cat": "shipment",
             "sql": "SELECT brand, SUM(quantity) as t FROM shipments WHERE month='{month}' GROUP BY brand ORDER BY t DESC LIMIT 1",
             "v_sql": "SELECT brand FROM shipments WHERE month='{month}' GROUP BY brand ORDER BY SUM(quantity) DESC LIMIT 1",
             "v_chk": "len>0", "kw": ["最高", "出货"]},
            {"q": "{brand}在所有品类中的总收入是多少？", "cat": "revenue",
             "sql": "SELECT SUM(revenue_usd) FROM revenue WHERE brand='{brand}'",
             "v_sql": "SELECT SUM(revenue_usd) FROM revenue WHERE brand='{brand}'",
             "v_chk": ">0", "kw": ["{brand}", "收入"]},
        ],
        "medium": [
            {"q": "{brand}的{category}从{month_start}到{month_end}的出货趋势如何？", "cat": "shipment",
             "v_sql": "SELECT COUNT(DISTINCT month) FROM shipments WHERE brand='{brand}' AND category='{category}'",
             "v_chk": ">3", "kw": ["趋势", "{brand}", "{category}"]},
            {"q": "各品牌{category}的平均ASP排名如何？", "cat": "revenue",
             "v_sql": "SELECT COUNT(DISTINCT brand) FROM revenue WHERE category='{category}'",
             "v_chk": ">1", "kw": ["ASP", "排名", "{category}"]},
            {"q": "前三大客户的订单集中度是多少？存在集中风险吗？", "cat": "customer",
             "v_sql": "SELECT COUNT(DISTINCT customer_id) FROM customer_orders",
             "v_chk": ">3", "kw": ["集中度", "风险", "客户"]},
        ],
        "hard": [
            {"q": "请识别{month}月所有品牌出货数据中的异常值，分析原因。", "cat": "anomaly",
             "v_sql": "SELECT COUNT(*) FROM shipments WHERE month='{month}' AND is_anomaly=1",
             "v_chk": ">=0", "kw": ["异常", "原因", "{month}"]},
            {"q": "哪些客户回款天数持续恶化？给出风险等级。", "cat": "customer",
             "v_sql": "SELECT COUNT(DISTINCT customer_id) FROM customer_orders WHERE payment_days > 60",
             "v_chk": ">=0", "kw": ["回款", "风险", "恶化"]},
            {"q": "对比{brand_a}和{brand_b}的出货量、收入和ASP变化，谁表现更好？", "cat": "revenue",
             "v_sql": "SELECT COUNT(*) FROM revenue WHERE brand IN ('{brand_a}', '{brand_b}')",
             "v_chk": ">0", "kw": ["{brand_a}", "{brand_b}", "对比"]},
        ],
        "extreme": [
            {"q": "基于过去数据，预测下季度各品牌出货量及置信区间。", "cat": "forecast",
             "kw": ["预测", "置信", "季度"]},
            {"q": "如果{brand}终止合作，对营收和产能影响多大？给出应急方案。", "cat": "risk",
             "kw": ["{brand}", "影响", "营收", "应急"]},
            {"q": "综合出货/收入/集中度/异常/回款，给业务健康评分(0-100)并列Top-3建议。", "cat": "health",
             "kw": ["健康", "评分", "建议", "综合"]},
        ],
    }

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed or int(time.time()))

    def generate_tasks(self, metadata: Dict,
                       num_per_difficulty: Dict[str, int] = None) -> List[AWMTask]:
        if num_per_difficulty is None:
            num_per_difficulty = {"easy": 5, "medium": 5, "hard": 3, "extreme": 2}
        brands = metadata.get("brands", ["Samsung"])
        categories = metadata.get("categories", ["智能手机_中端"])
        months = metadata.get("months", ["2025-06"])
        tasks = []
        tc = 0

        for diff, count in num_per_difficulty.items():
            templates = self.TEMPLATES.get(diff, [])
            if not templates: continue
            for _ in range(count):
                tpl = self.rng.choice(templates)
                params = {
                    "brand": self.rng.choice(brands),
                    "brand_a": brands[0] if len(brands) > 1 else "Samsung",
                    "brand_b": brands[1] if len(brands) > 1 else "Apple",
                    "category": self.rng.choice(categories),
                    "month": self.rng.choice(months),
                    "month_start": months[0], "month_end": months[-1],
                }
                q = tpl["q"].format(**params)
                vsql = tpl.get("v_sql", "").format(**params) if tpl.get("v_sql") else ""
                vchk = tpl.get("v_chk", "")
                kws = [k.format(**params) for k in tpl.get("kw", [])]
                tc += 1
                tasks.append(AWMTask(
                    task_id=f"AWM-{diff[0].upper()}{tc:03d}", question=q,
                    category=tpl.get("cat", "general"), difficulty=diff,
                    expected_sql=tpl.get("sql", "").format(**params) if tpl.get("sql") else "",
                    verification_sql=vsql, verification_check=vchk,
                    expected_keywords=kws, metadata={"params": params}))

        self.rng.shuffle(tasks)
        return tasks


# ╔══════════════════════════════════════════╗
# ║  Part 4: AWM 环境工厂 (主类)              ║
# ╚══════════════════════════════════════════╝

@dataclass
class AWMEnvironment:
    env_id: str
    scenario: ScenarioSeed
    db_path: str
    metadata: Dict
    tasks: List[AWMTask]
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def num_tasks(self) -> int: return len(self.tasks)

    @property
    def task_breakdown(self) -> Dict[str, int]:
        bd = defaultdict(int)
        for t in self.tasks: bd[t.difficulty] += 1
        return dict(bd)


class AWMEnvironmentFactory:
    """
    AWM 合成环境工厂 — V9.0 核心模块
    完整管线 (对齐 Snowflake AWM 论文):
      1. Scenario → 选择/自定义场景
      2. Database → SQLite + 合成数据
      3. Tasks → 参数化评估任务
      4. Verifier → SQL 验证 ground truth
      5. MCP API → 工具接口层
    """

    def __init__(self, output_dir: str = "./awm_envs", seed: int = None,
                 default_tasks_per_env: int = 15):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed or int(time.time())
        self.rng = random.Random(self.seed)
        self.default_tasks_per_env = default_tasks_per_env
        self.registry: Dict[str, AWMEnvironment] = {}
        self._load_registry()

    def _load_registry(self):
        rp = self.output_dir / "registry.json"
        if rp.exists():
            try:
                with open(rp) as f: json.load(f)
                logger.info("AWM: Registry loaded")
            except: pass

    def _save_registry(self):
        rp = self.output_dir / "registry.json"
        data = {}
        for eid, env in self.registry.items():
            data[eid] = {"env_id": eid, "scenario": env.scenario.name,
                         "domain": env.scenario.domain.value, "db_path": env.db_path,
                         "num_tasks": env.num_tasks, "breakdown": env.task_breakdown,
                         "created_at": env.created_at}
        with open(rp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def generate_one(self, domain: ScenarioDomain = None,
                     scenario: ScenarioSeed = None,
                     data_config: SyntheticDataConfig = None,
                     num_tasks: int = None) -> AWMEnvironment:
        if scenario is None:
            cands = [s for s in SCENARIO_TEMPLATES if s.domain == domain] if domain else SCENARIO_TEMPLATES
            scenario = self.rng.choice(cands)

        env_id = f"awm_{uuid.uuid4().hex[:8]}"
        logger.info(f"AWM: Generating {env_id} ({scenario.name})")

        if data_config is None:
            data_config = SyntheticDataConfig(
                num_brands=scenario.num_brands, num_categories=scenario.num_categories,
                num_regions=scenario.num_regions, num_months=scenario.time_span_months)
            if scenario.complexity == "easy":
                data_config.anomaly_rate = 0.02; data_config.num_brands = min(5, data_config.num_brands)
            elif scenario.complexity == "extreme":
                data_config.anomaly_rate = 0.08
                data_config.num_brands = max(12, data_config.num_brands)
                data_config.num_months = max(18, data_config.num_months)

        db_dir = self.output_dir / env_id
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(db_dir / "sales_data.db")

        gs = self.rng.randint(0, 2**31)
        gen = SyntheticDataGenerator(config=data_config, seed=gs)
        metadata = gen.generate_shipment_db(db_path)
        metadata["env_id"] = env_id
        metadata["scenario"] = scenario.name

        n = num_tasks or self.default_tasks_per_env
        td = {"easy": max(1, n//4), "medium": max(1, n//3), "hard": max(1, n//4),
              "extreme": max(1, n - n//4 - n//3 - n//4)}
        tg = TaskGenerator(seed=gs + 1)
        tasks = tg.generate_tasks(metadata, td)

        with open(db_dir / "tasks.json", "w") as f:
            json.dump([asdict(t) for t in tasks], f, ensure_ascii=False, indent=2)
        with open(db_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        env = AWMEnvironment(env_id=env_id, scenario=scenario, db_path=db_path,
                             metadata=metadata, tasks=tasks)
        self.registry[env_id] = env
        self._save_registry()
        logger.info(f"AWM: {env_id} ready — {metadata['table_counts'].get('shipments',0)} rows, {len(tasks)} tasks")
        return env

    def generate_batch(self, num_envs: int = 10, domain: ScenarioDomain = None,
                       diverse: bool = True) -> List[AWMEnvironment]:
        envs = []
        if diverse and domain is None:
            tpls = list(SCENARIO_TEMPLATES)
            for i in range(num_envs):
                envs.append(self.generate_one(scenario=tpls[i % len(tpls)]))
        else:
            for _ in range(num_envs):
                envs.append(self.generate_one(domain=domain))
        logger.info(f"AWM: Batch of {len(envs)} environments generated")
        return envs

    def load_env(self, env_id: str) -> Optional[AWMEnvironment]:
        if env_id in self.registry: return self.registry[env_id]
        ed = self.output_dir / env_id
        if not ed.exists(): return None
        mp = ed / "metadata.json"
        if not mp.exists(): return None
        with open(mp) as f: metadata = json.load(f)
        tasks = []
        tp = ed / "tasks.json"
        if tp.exists():
            with open(tp) as f:
                tasks = [AWMTask(**td) for td in json.load(f)]
        scenario = ScenarioSeed(domain=ScenarioDomain.ODM_SHIPMENT, name=metadata.get("scenario", "loaded"),
                                description="Loaded", entities=[], tools_required=[])
        env = AWMEnvironment(env_id=env_id, scenario=scenario, db_path=str(ed / "sales_data.db"),
                             metadata=metadata, tasks=tasks)
        self.registry[env_id] = env
        return env

    def get_stats(self) -> Dict:
        tt = sum(e.num_tasks for e in self.registry.values())
        doms = defaultdict(int)
        for e in self.registry.values(): doms[e.scenario.domain.value] += 1
        return {"total_environments": len(self.registry), "total_tasks": tt,
                "domains": dict(doms), "output_dir": str(self.output_dir)}


# ╔══════════════════════════════════════════╗
# ║  Part 5: AWM 评估集成                    ║
# ╚══════════════════════════════════════════╝

@dataclass
class AWMEvalResult:
    task_id: str; env_id: str; passed: bool; score: float
    answer: str = ""; verification_passed: bool = False
    keyword_score: float = 0.0; latency_ms: float = 0.0
    error: str = ""; details: Dict = field(default_factory=dict)

@dataclass
class AWMEvalReport:
    total_tasks: int = 0; passed: int = 0; failed: int = 0; avg_score: float = 0.0
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    by_category: Dict[str, Dict] = field(default_factory=dict)
    results: List[AWMEvalResult] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = ["═" * 55, "  AWM Eval Report", "═" * 55,
                 f"  Total: {self.total_tasks} | ✅ {self.passed} | ❌ {self.failed} | Score: {self.avg_score:.1%}",
                 f"  Time: {self.elapsed_sec:.1f}s", "", "  By Difficulty:"]
        for d, s in sorted(self.by_difficulty.items()):
            lines.append(f"    [{d}] {s['passed']}/{s['total']} ({s['score']:.0%})")
        lines.append("  By Category:")
        for c, s in sorted(self.by_category.items()):
            lines.append(f"    [{c}] {s['passed']}/{s['total']} ({s['score']:.0%})")
        lines.append("═" * 55)
        return "\n".join(lines)


class AWMEvalIntegration:
    def __init__(self, factory: AWMEnvironmentFactory):
        self.factory = factory

    def verify_with_sql(self, db_path: str, task: AWMTask, answer: str) -> Tuple[bool, str]:
        if not task.verification_sql: return True, "no_verification"
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(task.verification_sql)
            result = cur.fetchone()
            conn.close()
            if result is None: return False, "sql_returned_none"
            value = result[0]
            chk = task.verification_check
            if chk.startswith(">="): passed = float(value) >= float(chk[2:])
            elif chk.startswith(">"): passed = float(value) > float(chk[1:])
            elif chk.startswith("=="): passed = str(value) == chk[2:]
            elif chk.startswith("len>"): passed = value is not None and len(str(value)) > int(chk[4:])
            elif chk.startswith("contains:"): passed = chk[9:].lower() in str(answer).lower()
            else: passed = value is not None and value != 0
            return passed, f"value={value}, check={chk}"
        except Exception as e:
            return False, f"sql_error: {e}"

    def check_keywords(self, answer: str, keywords: List[str]) -> float:
        if not keywords: return 1.0
        return sum(1 for kw in keywords if kw.lower() in answer.lower()) / len(keywords)

    def eval_single_task(self, env: AWMEnvironment, task: AWMTask,
                         agent_fn: Callable[[str, str], str]) -> AWMEvalResult:
        t0 = time.time()
        try: answer = agent_fn(task.question, env.db_path)
        except Exception as e:
            return AWMEvalResult(task_id=task.task_id, env_id=env.env_id,
                                passed=False, score=0.0, error=str(e),
                                latency_ms=(time.time()-t0)*1000)
        lat = (time.time()-t0)*1000
        sql_ok, sql_det = self.verify_with_sql(env.db_path, task, answer)
        kw_sc = self.check_keywords(answer, task.expected_keywords)
        len_sc = min(1.0, len(answer) / 100)
        score = 0.5 * (1.0 if sql_ok else 0.0) + 0.3 * kw_sc + 0.2 * len_sc
        return AWMEvalResult(task_id=task.task_id, env_id=env.env_id,
                             passed=score >= 0.6, score=score, answer=answer[:500],
                             verification_passed=sql_ok, keyword_score=kw_sc,
                             latency_ms=lat, details={"sql": sql_det, "len": len(answer)})

    def run_eval_suite(self, agent_fn: Callable[[str, str], str],
                       envs: List[AWMEnvironment] = None,
                       max_tasks: int = None) -> AWMEvalReport:
        if envs is None: envs = list(self.factory.registry.values())
        if not envs: return AWMEvalReport()
        report = AWMEvalReport()
        t0 = time.time()
        tc = 0
        for env in envs:
            for task in env.tasks:
                if max_tasks and tc >= max_tasks: break
                r = self.eval_single_task(env, task, agent_fn)
                report.results.append(r)
                tc += 1
                for key, gv in [("difficulty", task.difficulty), ("category", task.category)]:
                    tgt = report.by_difficulty if key == "difficulty" else report.by_category
                    if gv not in tgt: tgt[gv] = {"total": 0, "passed": 0, "scores": [], "score": 0.0}
                    tgt[gv]["total"] += 1
                    tgt[gv]["scores"].append(r.score)
                    if r.passed: tgt[gv]["passed"] += 1
            if max_tasks and tc >= max_tasks: break

        report.total_tasks = len(report.results)
        report.passed = sum(1 for r in report.results if r.passed)
        report.failed = report.total_tasks - report.passed
        scores = [r.score for r in report.results]
        report.avg_score = sum(scores) / len(scores) if scores else 0
        report.elapsed_sec = time.time() - t0
        for st in list(report.by_difficulty.values()) + list(report.by_category.values()):
            sc = st.pop("scores", [])
            st["score"] = sum(sc) / len(sc) if sc else 0
        return report


# ╔══════════════════════════════════════════╗
# ║  Part 6: 工作流记忆 (Workflow Memory)     ║
# ╚══════════════════════════════════════════╝

@dataclass
class Workflow:
    workflow_id: str; name: str; description: str; goal: str
    steps: List[str]; required_tools: List[str]
    applicable_categories: List[str]
    complexity: int = 1; success_count: int = 0
    created_from: List[str] = field(default_factory=list)
    parent_workflows: List[str] = field(default_factory=list)
    created_at: str = ""
    def __post_init__(self):
        if not self.created_at: self.created_at = datetime.now().isoformat()

@dataclass
class TrajectoryRecord:
    trajectory_id: str; question: str; category: str
    tools_used: List[str]; steps_taken: List[Dict]
    success: bool; score: float; timestamp: str = ""


class AWMWorkflowMemory:
    """
    工作流记忆系统 — ICML 2025 AWM 论文核心
    1. 轨迹摄入 → 2. 工作流归纳 → 3. 工作流检索 → 4. 滚雪球效应 → 5. 在线学习
    """

    def __init__(self, storage_path: str = "./awm_workflows"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.workflows: Dict[str, Workflow] = {}
        self.trajectories: List[TrajectoryRecord] = []
        self._load()

    def _load(self):
        wfp = self.storage_path / "workflows.json"
        if wfp.exists():
            try:
                with open(wfp) as f:
                    for wd in json.load(f):
                        wf = Workflow(**wd)
                        self.workflows[wf.workflow_id] = wf
                logger.info(f"AWM Memory: {len(self.workflows)} workflows loaded")
            except Exception as e:
                logger.warning(f"AWM Memory: load failed: {e}")

    def _save(self):
        with open(self.storage_path / "workflows.json", "w") as f:
            json.dump([asdict(wf) for wf in self.workflows.values()], f, ensure_ascii=False, indent=2)

    def ingest_trajectory(self, traj: TrajectoryRecord):
        self.trajectories.append(traj)
        if traj.success and traj.score >= 0.7:
            self._try_induce(traj)

    def _try_induce(self, traj: TrajectoryRecord):
        steps = []
        tools = set()
        for s in traj.steps_taken:
            act = s.get("action", "")
            if act:
                abstract = act
                for b in BRAND_POOLS.get("tier1", []):
                    abstract = abstract.replace(b, "{BRAND}")
                for c in CATEGORY_POOLS:
                    abstract = abstract.replace(c, "{CATEGORY}")
                steps.append(abstract)
            t = s.get("tool", "")
            if t: tools.add(t)
        if len(steps) < 2: return

        sh = hashlib.md5("|".join(steps).encode()).hexdigest()[:8]
        for ex in self.workflows.values():
            eh = hashlib.md5("|".join(ex.steps).encode()).hexdigest()[:8]
            if sh == eh:
                ex.success_count += 1
                ex.created_from.append(traj.trajectory_id)
                self._save()
                return

        wid = f"WF-{len(self.workflows)+1:03d}"
        self.workflows[wid] = Workflow(
            workflow_id=wid, name=f"{traj.category}分析流程",
            description=f"从 {traj.question[:50]}... 归纳",
            goal=f"完成 {traj.category} 分析", steps=steps,
            required_tools=list(tools), applicable_categories=[traj.category],
            complexity=1 if len(steps) <= 3 else (2 if len(steps) <= 6 else 3),
            success_count=1, created_from=[traj.trajectory_id])
        self._save()
        logger.info(f"AWM Memory: Induced {wid} ({len(steps)} steps)")

    def get_workflows(self, query: str = "", category: str = "",
                      max_results: int = 5) -> List[Workflow]:
        cands = list(self.workflows.values())
        if category:
            cands = [w for w in cands if category in w.applicable_categories or not w.applicable_categories]
        if query:
            ql = query.lower()
            scored = []
            for wf in cands:
                sc = 0
                for w in ql.split():
                    if w in wf.name.lower(): sc += 3
                    if w in wf.description.lower(): sc += 1
                    if w in wf.goal.lower(): sc += 2
                sc += min(wf.success_count * 0.5, 5)
                scored.append((sc, wf))
            scored.sort(key=lambda x: -x[0])
            cands = [wf for _, wf in scored]
        else:
            cands.sort(key=lambda w: -w.success_count)
        return cands[:max_results]

    def compose_workflows(self, wf_ids: List[str], new_name: str = "") -> Optional[Workflow]:
        wfs = [self.workflows[w] for w in wf_ids if w in self.workflows]
        if len(wfs) < 2: return None
        steps, tools, cats = [], set(), set()
        for wf in wfs:
            steps.extend(wf.steps); tools.update(wf.required_tools)
            cats.update(wf.applicable_categories)
        nid = f"WF-C{len(self.workflows)+1:03d}"
        nwf = Workflow(workflow_id=nid,
            name=new_name or f"组合({'→'.join(w.name for w in wfs)})",
            description=f"组合自 {', '.join(wf_ids)}", goal=" + ".join(w.goal for w in wfs),
            steps=steps, required_tools=list(tools), applicable_categories=list(cats),
            complexity=max(w.complexity for w in wfs) + 1, parent_workflows=wf_ids)
        self.workflows[nid] = nwf
        self._save()
        return nwf

    def get_stats(self) -> Dict:
        return {
            "total_workflows": len(self.workflows),
            "total_trajectories": len(self.trajectories),
            "by_complexity": {i: sum(1 for w in self.workflows.values() if w.complexity == i) for i in (1,2,3)},
            "total_uses": sum(w.success_count for w in self.workflows.values()),
            "composed": sum(1 for w in self.workflows.values() if w.parent_workflows),
        }


# ╔══════════════════════════════════════════╗
# ║  Part 7: MCP 兼容层 + RLM 桥接           ║
# ╚══════════════════════════════════════════╝

class AWMToolInterface:
    """将合成 SQLite 暴露为 MCP 工具 (对齐 mcp_server.py)"""

    def __init__(self, env: AWMEnvironment):
        self.env = env; self.db_path = env.db_path

    def list_tools(self) -> List[Dict]:
        return [
            {"name": "awm_query_shipments", "description": "查询出货量", "readOnly": True,
             "inputSchema": {"type": "object", "properties": {
                 "brand": {"type": "string"}, "category": {"type": "string"},
                 "month": {"type": "string"}, "region": {"type": "string"}}}},
            {"name": "awm_query_revenue", "description": "查询收入", "readOnly": True,
             "inputSchema": {"type": "object", "properties": {
                 "brand": {"type": "string"}, "category": {"type": "string"}, "month": {"type": "string"}}}},
            {"name": "awm_query_customers", "description": "查询客户", "readOnly": True,
             "inputSchema": {"type": "object", "properties": {
                 "customer_id": {"type": "string"}, "tier": {"type": "string"}}}},
            {"name": "awm_run_sql", "description": "执行 SELECT", "readOnly": True,
             "inputSchema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
            {"name": "awm_detect_anomalies", "description": "查询异常标注", "readOnly": True,
             "inputSchema": {"type": "object", "properties": {
                 "table_name": {"type": "string"}, "month": {"type": "string"}}}},
        ]

    def call_tool(self, tool_name: str, args: Dict) -> Dict:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if tool_name == "awm_run_sql":
                sql = args.get("sql", "")
                if not sql.strip().upper().startswith("SELECT"):
                    return {"content": [{"type": "text", "text": "Only SELECT"}], "isError": True}
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchall()]
                result = json.dumps(rows, ensure_ascii=False, default=str)
            elif tool_name == "awm_query_shipments":
                conds, params = [], []
                for c in ["brand", "category", "month", "region"]:
                    if args.get(c): conds.append(f"{c}=?"); params.append(args[c])
                w = " AND ".join(conds) if conds else "1=1"
                cur.execute(f"SELECT brand,category,region,month,SUM(quantity) as qty FROM shipments WHERE {w} GROUP BY brand,category,region,month ORDER BY month LIMIT 200", params)
                result = json.dumps([dict(r) for r in cur.fetchall()], ensure_ascii=False)
            elif tool_name == "awm_query_revenue":
                conds, params = [], []
                for c in ["brand", "category", "month"]:
                    if args.get(c): conds.append(f"{c}=?"); params.append(args[c])
                w = " AND ".join(conds) if conds else "1=1"
                cur.execute(f"SELECT brand,category,month,revenue_usd,quantity,asp_usd,margin_pct FROM revenue WHERE {w} ORDER BY month LIMIT 200", params)
                result = json.dumps([dict(r) for r in cur.fetchall()], ensure_ascii=False)
            elif tool_name == "awm_query_customers":
                conds, params = [], []
                if args.get("customer_id"): conds.append("customer_id=?"); params.append(args["customer_id"])
                if args.get("tier"): conds.append("tier=?"); params.append(args["tier"])
                w = " AND ".join(conds) if conds else "1=1"
                cur.execute(f"SELECT * FROM customers WHERE {w} LIMIT 50", params)
                result = json.dumps([dict(r) for r in cur.fetchall()], ensure_ascii=False)
            elif tool_name == "awm_detect_anomalies":
                conds, params = [], []
                if args.get("table_name"): conds.append("table_name=?"); params.append(args["table_name"])
                if args.get("month"): conds.append("description LIKE ?"); params.append(f"%{args['month']}%")
                w = " AND ".join(conds) if conds else "1=1"
                cur.execute(f"SELECT * FROM anomaly_labels WHERE {w} LIMIT 100", params)
                result = json.dumps([dict(r) for r in cur.fetchall()], ensure_ascii=False)
            else:
                result = "Unknown tool"
            conn.close()
            return {"content": [{"type": "text", "text": result}], "isError": False}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}


class AWMForRLM:
    """AWM × RLM 桥接 — 让 RLM SecureREPL 加载合成环境数据"""

    def __init__(self, env: AWMEnvironment):
        self.env = env

    def load_as_dataframes(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.env.db_path)
        conn.row_factory = sqlite3.Row
        data = {}
        for t in ["shipments", "revenue", "customers", "customer_orders"]:
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {t}")
            data[t] = [dict(r) for r in cur.fetchall()]
        conn.close()
        return {**data, "metadata": self.env.metadata,
                "brands": self.env.metadata.get("brands", []),
                "categories": self.env.metadata.get("categories", []),
                "months": self.env.metadata.get("months", [])}

    def get_rlm_context(self) -> str:
        m = self.env.metadata
        tc = m["table_counts"]
        return (f"当前环境: {self.env.scenario.name}\n"
                f"品牌: {', '.join(m.get('brands', []))}\n"
                f"品类: {', '.join(m.get('categories', []))}\n"
                f"月份: {m.get('months',['?'])[0]} ~ {m.get('months',['?'])[-1]}\n"
                f"数据表: shipments({tc.get('shipments',0)}), revenue({tc.get('revenue',0)}), "
                f"customers({tc.get('customers',0)}), customer_orders({tc.get('customer_orders',0)})\n"
                f"已标注异常: {m.get('total_anomalies',0)}个\n"
                f"\n变量: shipments, revenue, customers, customer_orders, metadata")


# ╔══════════════════════════════════════════╗
# ║  Part 8: Demo & Self-Test                ║
# ╚══════════════════════════════════════════╝

def demo():
    print("=" * 60)
    print("  MRARFAI V9.0 — AWM 合成环境工厂 Demo")
    print("=" * 60)

    factory = AWMEnvironmentFactory(output_dir="/tmp/awm_demo", seed=42)

    print("\n📦 生成合成环境...")
    envs = factory.generate_batch(num_envs=3, diverse=True)

    for env in envs:
        m = env.metadata
        print(f"\n  🌍 {env.env_id} ({env.scenario.name})")
        print(f"     品牌: {len(m.get('brands',[]))} | 品类: {len(m.get('categories',[]))} | 月: {len(m.get('months',[]))}")
        print(f"     数据行: {m['table_counts']}")
        print(f"     异常: {m['total_anomalies']} | 任务: {env.num_tasks} ({env.task_breakdown})")

    print("\n📝 任务样本:")
    for t in envs[0].tasks[:5]:
        print(f"  [{t.task_id}] ({t.difficulty}) {t.question}")

    print("\n🔧 MCP 工具:")
    ti = AWMToolInterface(envs[0])
    for tool in ti.list_tools():
        print(f"  • {tool['name']}: {tool['description']}")

    print("\n📊 工具调用:")
    r = ti.call_tool("awm_run_sql", {"sql": "SELECT brand, SUM(quantity) as total FROM shipments GROUP BY brand ORDER BY total DESC LIMIT 5"})
    for row in json.loads(r["content"][0]["text"])[:5]:
        print(f"  {row['brand']}: {row['total']:,}")

    print("\n🔗 RLM 桥接:")
    bridge = AWMForRLM(envs[0])
    print(bridge.get_rlm_context())

    print("\n🧪 模拟评估:")
    def dummy_agent(q, db):
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT brand, SUM(quantity) as t FROM shipments GROUP BY brand ORDER BY t DESC LIMIT 5")
        rows = cur.fetchall()
        conn.close()
        ans = f"分析 {q[:30]}...\n"
        for b, qty in rows:
            ans += f"  {b}: {qty:,}\n"
        ans += "出货趋势整体增长，部分品牌存在异常波动。客户集中度较高需关注风险。"
        return ans

    evaluator = AWMEvalIntegration(factory)
    report = evaluator.run_eval_suite(agent_fn=dummy_agent, envs=envs[:1], max_tasks=10)
    print(report.summary())

    print("\n🧠 工作流记忆:")
    memory = AWMWorkflowMemory(storage_path="/tmp/awm_demo/workflows")
    memory.ingest_trajectory(TrajectoryRecord(
        trajectory_id="T-001", question="Samsung手机出货趋势", category="shipment",
        tools_used=["query_shipment", "calc_trend"],
        steps_taken=[{"action": "查询Samsung出货", "tool": "query_shipment"},
                     {"action": "按月汇总", "tool": "calc_trend"},
                     {"action": "检测异常", "tool": "detect_anomaly"},
                     {"action": "生成报告", "tool": "generate_report"}],
        success=True, score=0.85))
    memory.ingest_trajectory(TrajectoryRecord(
        trajectory_id="T-002", question="OPPO平板收入分析", category="revenue",
        tools_used=["query_revenue", "calc_asp"],
        steps_taken=[{"action": "查询OPPO收入", "tool": "query_revenue"},
                     {"action": "计算ASP", "tool": "calc_asp"},
                     {"action": "对比行业", "tool": "benchmark"}],
        success=True, score=0.9))
    print(f"  工作流数: {len(memory.workflows)}")
    for wf in memory.workflows.values():
        print(f"  📋 {wf.workflow_id}: {wf.name} ({len(wf.steps)} steps)")

    results = memory.get_workflows(query="品牌出货趋势")
    if results:
        print(f"  🔍 检索→ {results[0].name}")

    stats = factory.get_stats()
    print(f"\n📈 统计: {stats['total_environments']}环境, {stats['total_tasks']}任务")
    print("\n✅ AWM Demo 完成!")
    return factory, envs, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    demo()
