#!/usr/bin/env python3
"""
MRARFAI Knowledge Graph v1.0
==============================
参考 LinkedIn Text-to-SQL 论文 (arxiv:2507.14372) 的知识图谱设计。

LinkedIn 论文的三层架构：
  1. Knowledge Graph — 捕获数据语义（表结构、字段含义、关系）
  2. Text-to-SQL Agent — 从知识图谱检索上下文，生成查询
  3. Auto-Correction — 自动纠错幻觉和语法错误

本模块实现适配 MRARFAI 销售数据的版本：
  1. SalesOntology — 销售领域本体（概念、关系、指标定义）
  2. SynonymGraph — 同义词/别名图谱（中文商业术语映射）
  3. EntityCatalog — 实体目录（客户、区域、品类的标准化）
  4. QueryPatternLibrary — 查询模式库（常见问题→结构化查询）
  5. QueryValidator — 查询验证与自动纠错

集成方式：
  - 插入 SmartDataQuery._rule_based_plan() 之前
  - 大幅提升规则匹配精度，减少对LLM的依赖
  - 为LLM查询计划提供丰富的schema上下文
"""

import re
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field


# ============================================================
# 1. 销售领域本体 (Sales Ontology)
# ============================================================

@dataclass
class Metric:
    """指标定义"""
    name: str           # 标准名称
    dimension: str      # 所属维度
    unit: str = ""      # 单位
    description: str = ""
    aggregatable: bool = True  # 是否可聚合

@dataclass
class Relationship:
    """实体关系"""
    source: str
    target: str
    relation: str       # has_many / belongs_to / measured_by
    description: str = ""


class SalesOntology:
    """
    销售领域本体 — 定义所有概念、维度、指标及其关系
    
    LinkedIn 论文启示：
    "We construct a knowledge graph that captures up-to-date semantics
     by indexing database metadata, historical query logs, wikis, and code."
    
    对于 MRARFAI，我们的"数据库"是预计算的 results dict，
    所以本体直接映射到 results 的 key 结构。
    """

    # 维度定义（对应 SmartDataQuery 的 dimensions）
    DIMENSIONS = {
        'overview': {
            'label': '总览',
            'description': '全年营收汇总、同比增长、月度趋势、核心发现',
            'metrics': ['总营收', '同比增长率', '月度营收', '活跃客户数', '核心发现'],
            'data_keys': ['总营收', '总YoY', '月度总营收', '核心发现'],
        },
        'customers': {
            'label': '客户',
            'description': '客户分级(ABC)、金额排名、占比、集中度',
            'metrics': ['年度金额', '等级', 'H1', 'H2', '占比', '累计占比'],
            'data_keys': ['客户分级', '客户金额'],
            'filterable_by': ['customer_name', 'tier', 'top_n'],
        },
        'risks': {
            'label': '风险',
            'description': '流失预警、异常检测、风险等级评估',
            'metrics': ['风险等级', '原因', '年度金额', '月度异常'],
            'data_keys': ['流失预警', 'MoM异常'],
            'filterable_by': ['level', 'customer_name'],
        },
        'growth': {
            'label': '增长',
            'description': '增长机会识别、潜力评估',
            'metrics': ['机会', '潜力金额'],
            'data_keys': ['增长机会'],
        },
        'price_volume': {
            'label': '价量',
            'description': '价格与数量分解、单价变化、出货量变化',
            'metrics': ['单价变化', '数量变化', '金额变化'],
            'data_keys': ['价量分解'],
            'filterable_by': ['customer_name'],
        },
        'regions': {
            'label': '区域',
            'description': '区域分布、集中度(HHI)、Top区域',
            'metrics': ['区域金额', '占比', 'HHI指数', 'Top3集中度'],
            'data_keys': ['区域洞察'],
        },
        'categories': {
            'label': '品类',
            'description': '业务类别趋势、年度对比、增长率',
            'metrics': ['类别', '2024金额', '2025金额', '增长率'],
            'data_keys': ['类别趋势'],
        },
        'benchmark': {
            'label': '对标',
            'description': '行业对标、竞争分析(华勤/闻泰/龙旗)、市场定位',
            'metrics': ['行业排名', '市场份额', '竞争对标', '结构性风险', '战略机会'],
            'data_keys': ['行业对标'],
        },
        'forecast': {
            'label': '预测',
            'description': '营收预测、客户预测、品类预测、风险场景',
            'metrics': ['总营收预测', '客户预测', '品类预测', '风险场景'],
            'data_keys': ['预测'],
        },
    }

    # 实体关系
    RELATIONSHIPS = [
        Relationship('客户', '区域', 'belongs_to', '客户归属于某个区域'),
        Relationship('客户', '品类', 'has_many', '客户可能涉及多个业务品类'),
        Relationship('客户', '风险', 'measured_by', '客户有对应的风险评级'),
        Relationship('客户', '价量', 'measured_by', '客户有价量分解数据'),
        Relationship('客户', '增长', 'measured_by', '客户有增长机会评估'),
        Relationship('区域', '总览', 'contributes_to', '区域营收汇总为总营收'),
        Relationship('品类', '总览', 'contributes_to', '品类营收汇总为总营收'),
        Relationship('对标', '品类', 'references', '行业对标参考品类结构'),
        Relationship('预测', '客户', 'based_on', '预测基于客户历史数据'),
    ]

    @classmethod
    def get_dimension_info(cls, dim_name: str) -> dict:
        return cls.DIMENSIONS.get(dim_name, {})

    @classmethod
    def get_related_dimensions(cls, dim_name: str) -> List[str]:
        """获取与某维度相关的其他维度"""
        label = cls.DIMENSIONS.get(dim_name, {}).get('label', '')
        related = set()
        for rel in cls.RELATIONSHIPS:
            if rel.source == label:
                # 找到target对应的dimension key
                for k, v in cls.DIMENSIONS.items():
                    if v['label'] == rel.target:
                        related.add(k)
            elif rel.target == label:
                for k, v in cls.DIMENSIONS.items():
                    if v['label'] == rel.source:
                        related.add(k)
        return list(related)

    @classmethod
    def generate_schema_prompt(cls) -> str:
        """生成给LLM的schema描述（用于查询计划生成）"""
        lines = ["# 禾苗销售数据 Schema\n"]
        for dim_key, dim in cls.DIMENSIONS.items():
            filters = dim.get('filterable_by', [])
            filter_str = f"  可筛选: {', '.join(filters)}" if filters else ""
            lines.append(f"## {dim_key} ({dim['label']})")
            lines.append(f"  说明: {dim['description']}")
            lines.append(f"  指标: {', '.join(dim['metrics'])}")
            if filter_str:
                lines.append(filter_str)
            lines.append("")
        return "\n".join(lines)


# ============================================================
# 2. 同义词图谱 (Synonym Graph)
# ============================================================

class SynonymGraph:
    """
    中文商业术语同义词映射
    
    LinkedIn 论文启示：
    "historical query logs" → 从用户实际问法中学习同义词
    
    这里我们预定义常见的销售领域同义词，
    后续可以从 AgentMemory 的历史对话中自动扩展。
    """

    # 维度同义词 → 标准维度名
    DIMENSION_SYNONYMS = {
        # overview
        '总营收': 'overview', '营收': 'overview', '收入': 'overview', '业绩': 'overview',
        '销售额': 'overview', '全年': 'overview', '概览': 'overview', '总览': 'overview',
        '大盘': 'overview', '整体': 'overview', '汇总': 'overview', '月度': 'overview',
        '趋势': 'overview', '同比': 'overview', '环比': 'overview', 'yoy': 'overview',
        'mom': 'overview',
        # customers
        '客户': 'customers', '分级': 'customers', 'abc': 'customers', '排名': 'customers',
        'top': 'customers', '前几': 'customers', '大客户': 'customers', '小客户': 'customers',
        '占比': 'customers', '集中度': 'customers', '贡献': 'customers', '钱包份额': 'customers',
        # risks
        '风险': 'risks', '流失': 'risks', '预警': 'risks', '丢': 'risks', '掉': 'risks',
        '下降': 'risks', '下滑': 'risks', '暴跌': 'risks', '断崖': 'risks', '异常': 'risks',
        '危险': 'risks', '警告': 'risks', '问题': 'risks', '衰退': 'risks', '萎缩': 'risks',
        # growth
        '增长': 'growth', '机会': 'growth', '潜力': 'growth', '新客': 'growth',
        '拓展': 'growth', '提升': 'growth', '突破': 'growth',
        # price_volume
        '价量': 'price_volume', '单价': 'price_volume', '价格': 'price_volume',
        '出货量': 'price_volume', '数量': 'price_volume', '量价': 'price_volume',
        'asp': 'price_volume', '均价': 'price_volume',
        # regions
        '区域': 'regions', '地区': 'regions', '市场': 'regions', '华东': 'regions',
        '华南': 'regions', '华北': 'regions', '海外': 'regions', '国内': 'regions',
        # categories
        '品类': 'categories', '类别': 'categories', '产品': 'categories', '结构': 'categories',
        '智能手机': 'categories', '功能机': 'categories', 'iot': 'categories',
        '手机': 'categories',
        # benchmark
        '行业': 'benchmark', '竞争': 'benchmark', '对标': 'benchmark', '对手': 'benchmark',
        '华勤': 'benchmark', '闻泰': 'benchmark', '龙旗': 'benchmark',
        '份额': 'benchmark', '排名': 'benchmark',
        # forecast
        '预测': 'forecast', '2026': 'forecast', '未来': 'forecast', '前景': 'forecast',
        '下季': 'forecast', '下半年': 'forecast', '明年': 'forecast', '展望': 'forecast',
        '预估': 'forecast', '情景': 'forecast', '场景': 'forecast',
    }

    # 客户名别名 → 标准名（自动从数据中构建）
    CUSTOMER_ALIASES = {
        'hmd': 'HMD', 'nokia': 'Nokia', 'samsung': 'Samsung',
        'transsion': 'Transsion', '传音': 'Transsion',
        'motorola': 'Motorola', 'moto': 'Motorola', '摩托': 'Motorola', '摩托罗拉': 'Motorola',
        'xiaomi': 'Xiaomi', '小米': 'Xiaomi',
        'oppo': 'OPPO', 'vivo': 'Vivo',
        'realme': 'Realme', '真我': 'Realme',
        'tcl': 'TCL', 'zte': 'ZTE', '中兴': 'ZTE',
        'infinix': 'Infinix', 'tecno': 'TECNO', 'itel': 'itel',
    }

    # 指标同义词 → 标准指标名
    METRIC_SYNONYMS = {
        '金额': '年度金额', '营收': '年度金额', '收入': '年度金额', '销售': '年度金额',
        '等级': '等级', '级别': '等级', '评级': '等级',
        '增长': '增长率', '增速': '增长率', '涨幅': '增长率', '变化': '增长率',
        '下降': '增长率', '跌幅': '增长率',
        '占比': '占比', '比重': '占比', '比例': '占比',
        '单价': '单价变化', 'asp': '单价变化',
        '出货': '数量变化', '数量': '数量变化', '台数': '数量变化',
    }

    # 时间表达式 → 标准化
    TIME_EXPRESSIONS = {
        '上半年': 'H1', '下半年': 'H2', 'h1': 'H1', 'h2': 'H2',
        '一季度': 'Q1', '二季度': 'Q2', '三季度': 'Q3', '四季度': 'Q4',
        'q1': 'Q1', 'q2': 'Q2', 'q3': 'Q3', 'q4': 'Q4',
        '1月': 'M1', '2月': 'M2', '3月': 'M3', '4月': 'M4',
        '5月': 'M5', '6月': 'M6', '7月': 'M7', '8月': 'M8',
        '9月': 'M9', '10月': 'M10', '11月': 'M11', '12月': 'M12',
        '去年': '2024', '今年': '2025', '明年': '2026',
    }

    # 等级别名
    TIER_ALIASES = {
        'a级': 'A', 'a类': 'A', 'a档': 'A', '头部': 'A', '大客户': 'A', '核心': 'A',
        'b级': 'B', 'b类': 'B', 'b档': 'B', '中等': 'B', '腰部': 'B',
        'c级': 'C', 'c类': 'C', 'c档': 'C', '小客户': 'C', '尾部': 'C', '长尾': 'C',
    }

    # 风险等级别名
    RISK_LEVEL_ALIASES = {
        '高风险': 'high', '高危': 'high', '严重': 'high', '紧急': 'high',
        '中风险': 'medium', '中等风险': 'medium',
        '低风险': 'low', '轻微': 'low',
    }

    @classmethod
    def resolve_dimensions(cls, text: str) -> List[str]:
        """从文本中识别所有相关维度"""
        text_lower = text.lower()
        dims = set()
        for keyword, dim in cls.DIMENSION_SYNONYMS.items():
            if keyword.lower() in text_lower:
                dims.add(dim)
        return list(dims)

    @classmethod
    def resolve_customer(cls, text: str, known_customers: List[str] = None) -> Optional[str]:
        """从文本中识别客户名"""
        text_lower = text.lower()

        # 先查别名表
        for alias, standard in cls.CUSTOMER_ALIASES.items():
            if alias in text_lower:
                return standard

        # 再查已知客户列表（精确匹配）
        if known_customers:
            for cname in known_customers:
                if cname.lower() in text_lower:
                    return cname

        return None

    @classmethod
    def resolve_tier(cls, text: str) -> Optional[str]:
        """从文本中识别客户等级"""
        text_lower = text.lower()
        for alias, tier in cls.TIER_ALIASES.items():
            if alias in text_lower:
                return tier
        return None

    @classmethod
    def resolve_risk_level(cls, text: str) -> Optional[str]:
        """从文本中识别风险等级"""
        text_lower = text.lower()
        for alias, level in cls.RISK_LEVEL_ALIASES.items():
            if alias in text_lower:
                return level
        return None

    @classmethod
    def resolve_top_n(cls, text: str) -> Optional[int]:
        """从文本中识别 Top N"""
        # "top5", "前5", "top 10", "前10名"
        patterns = [
            r'top\s*(\d+)',
            r'前\s*(\d+)',
            r'最大的?\s*(\d+)',
            r'(\d+)\s*(?:个|家|名)(?:最大|最好|头部)',
        ]
        for pat in patterns:
            m = re.search(pat, text.lower())
            if m:
                return min(int(m.group(1)), 30)
        return None

    @classmethod
    def resolve_time(cls, text: str) -> Optional[str]:
        """从文本中识别时间表达式"""
        text_lower = text.lower()
        for alias, standard in cls.TIME_EXPRESSIONS.items():
            if alias in text_lower:
                return standard
        return None

    @classmethod
    def expand_aliases(cls, customer_data: List[dict]):
        """从实际数据中自动扩展客户别名"""
        for c in customer_data:
            name = c.get('客户', '')
            if name and name.lower() not in cls.CUSTOMER_ALIASES:
                cls.CUSTOMER_ALIASES[name.lower()] = name


# ============================================================
# 3. 实体目录 (Entity Catalog)
# ============================================================

class EntityCatalog:
    """
    实体目录 — 管理所有已知实体及其属性
    
    LinkedIn 论文启示：
    "We apply clustering to identify relevant tables for each team or product area."
    
    对应到 MRARFAI：根据客户、区域、品类进行聚类分组，
    帮助查询时快速定位相关数据。
    """

    def __init__(self):
        self.customers = {}    # name → {tier, amount, region, ...}
        self.regions = {}      # name → {amount, share, ...}
        self.categories = {}   # name → {amount_2024, amount_2025, growth, ...}
        self._built = False

    def build_from_data(self, data: dict, results: dict):
        """从原始数据构建实体目录"""
        # 客户
        for c in results.get('客户分级', []):
            self.customers[c['客户']] = {
                'tier': c.get('等级', 'C'),
                'amount': c.get('年度金额', 0),
                'h1': c.get('H1', 0),
                'h2': c.get('H2', 0),
                'share': c.get('占比', '0%'),
            }

        # 从价量分解补充
        for pv in results.get('价量分解', []):
            name = pv.get('客户', '')
            if name in self.customers:
                self.customers[name]['price_change'] = pv.get('单价变化', '')
                self.customers[name]['volume_change'] = pv.get('数量变化', '')

        # 从风险预警补充
        for r in results.get('流失预警', []):
            name = r.get('客户', '')
            if name in self.customers:
                self.customers[name]['risk'] = r.get('风险', '')
                self.customers[name]['risk_reason'] = r.get('原因', '')

        # 从增长机会补充
        for g in results.get('增长机会', []):
            name = g.get('客户', '')
            if name in self.customers:
                self.customers[name]['growth_opportunity'] = g.get('机会', '')
                self.customers[name]['growth_potential'] = g.get('潜力金额', 0)

        # 区域
        region_data = results.get('区域洞察', {})
        for r in region_data.get('详细', region_data.get('区域分布', [])):
            if isinstance(r, dict):
                self.regions[r.get('区域', '')] = {
                    'amount': r.get('金额', 0),
                    'share': r.get('占比', ''),
                }

        # 品类
        for cat in results.get('类别趋势', []):
            self.categories[cat.get('类别', '')] = {
                'amount_2024': cat.get('2024金额', 0),
                'amount_2025': cat.get('2025金额', 0),
                'growth': cat.get('增长率', ''),
            }

        # 扩展同义词表
        SynonymGraph.expand_aliases(data.get('客户金额', []))

        self._built = True

    def get_customer_profile(self, name: str) -> Optional[dict]:
        """获取客户完整画像"""
        # 精确匹配
        if name in self.customers:
            return {name: self.customers[name]}
        # 模糊匹配
        for cname, cdata in self.customers.items():
            if name.lower() in cname.lower():
                return {cname: cdata}
        return None

    def get_customers_by_tier(self, tier: str) -> dict:
        return {n: d for n, d in self.customers.items() if d.get('tier') == tier}

    def get_risky_customers(self, level: str = None) -> dict:
        result = {}
        for n, d in self.customers.items():
            risk = d.get('risk', '')
            if risk:
                if level is None or level in risk.lower():
                    result[n] = d
        return result

    def get_top_customers(self, n: int = 10) -> dict:
        sorted_c = sorted(self.customers.items(), key=lambda x: x[1].get('amount', 0), reverse=True)
        return dict(sorted_c[:n])

    def summarize(self) -> str:
        """生成实体目录摘要（给LLM的上下文）"""
        lines = [f"已索引: {len(self.customers)}客户, {len(self.regions)}区域, {len(self.categories)}品类"]
        # Top 5客户
        top5 = list(self.get_top_customers(5).items())
        if top5:
            lines.append("Top5客户: " + ", ".join(f"{n}({d['tier']}级¥{d['amount']}万)" for n, d in top5))
        # 风险客户
        risky = self.get_risky_customers()
        if risky:
            lines.append("风险客户: " + ", ".join(f"{n}({d.get('risk','')})" for n, d in risky.items()))
        return "\n".join(lines)


# ============================================================
# 4. 查询模式库 (Query Pattern Library)
# ============================================================

@dataclass
class QueryPattern:
    """查询模式"""
    pattern_name: str
    description: str
    triggers: List[str]        # 触发词
    dimensions: List[str]      # 需要的数据维度
    default_filters: dict = field(default_factory=dict)
    agent_hint: List[str] = field(default_factory=list)  # 推荐Agent
    examples: List[str] = field(default_factory=list)


class QueryPatternLibrary:
    """
    查询模式库 — 将常见问题模式映射到结构化查询
    
    LinkedIn 论文启示：
    "We build an interactive chatbot that supports various user intents,
     from data discovery to query writing to debugging."
    
    我们把 "user intents" 分为以下模式：
    """

    PATTERNS = [
        QueryPattern(
            pattern_name="total_revenue",
            description="查询总营收/整体业绩",
            triggers=["总营收", "营收多少", "业绩怎么样", "收入多少", "全年多少"],
            dimensions=["overview"],
            agent_hint=["analyst"],
            examples=["今年总营收多少？", "整体业绩怎么样？"],
        ),
        QueryPattern(
            pattern_name="customer_ranking",
            description="客户排名/分级查询",
            triggers=["排名", "最大", "top", "前几", "大客户", "abc分级"],
            dimensions=["customers"],
            agent_hint=["analyst"],
            examples=["Top5客户是谁？", "A级客户有哪些？"],
        ),
        QueryPattern(
            pattern_name="single_customer",
            description="单个客户详情（触发：提到具体客户名）",
            triggers=[],  # 由客户名匹配触发
            dimensions=["customers", "price_volume", "risks", "growth"],
            agent_hint=["analyst"],
            examples=["HMD今年表现怎么样？", "Nokia的价量分解"],
        ),
        QueryPattern(
            pattern_name="risk_alert",
            description="风险预警/流失分析",
            triggers=["风险", "流失", "预警", "可能丢", "下降严重", "异常"],
            dimensions=["risks", "customers"],
            default_filters={"level": "high"},
            agent_hint=["analyst", "risk"],
            examples=["哪些客户可能流失？", "高风险预警"],
        ),
        QueryPattern(
            pattern_name="growth_opportunity",
            description="增长机会分析",
            triggers=["增长机会", "哪里能增长", "潜力", "新增长点"],
            dimensions=["growth", "customers"],
            agent_hint=["analyst", "strategist"],
            examples=["有什么增长机会？", "哪些客户有潜力？"],
        ),
        QueryPattern(
            pattern_name="price_volume",
            description="价量分解分析",
            triggers=["价量", "单价", "出货量", "量价", "asp"],
            dimensions=["price_volume"],
            agent_hint=["analyst"],
            examples=["价量分解情况如何？", "哪些客户单价在涨？"],
        ),
        QueryPattern(
            pattern_name="regional_analysis",
            description="区域分布分析",
            triggers=["区域", "地区", "哪个市场", "集中度"],
            dimensions=["regions"],
            agent_hint=["analyst"],
            examples=["各区域分布如何？", "华东占比多少？"],
        ),
        QueryPattern(
            pattern_name="category_trend",
            description="品类趋势分析",
            triggers=["品类", "产品结构", "智能手机", "功能机", "iot"],
            dimensions=["categories"],
            agent_hint=["analyst"],
            examples=["各品类趋势如何？", "IoT增长多少？"],
        ),
        QueryPattern(
            pattern_name="competitive_benchmark",
            description="竞争对标分析",
            triggers=["行业对标", "竞争对手", "华勤", "闻泰", "龙旗", "市场份额", "行业地位"],
            dimensions=["benchmark"],
            agent_hint=["analyst", "strategist"],
            examples=["和华勤比怎么样？", "行业里排第几？"],
        ),
        QueryPattern(
            pattern_name="forecast",
            description="业绩预测",
            triggers=["预测", "2026", "未来", "前景", "展望", "下季度"],
            dimensions=["forecast"],
            agent_hint=["analyst", "strategist"],
            examples=["2026年预测如何？", "明年能增长多少？"],
        ),
        QueryPattern(
            pattern_name="ceo_report",
            description="CEO综合报告（全维度）",
            triggers=["ceo", "总结", "全面分析", "管理层", "报告", "汇报", "战略"],
            dimensions=["overview", "customers", "risks", "growth", "benchmark", "forecast"],
            agent_hint=["analyst", "risk", "strategist"],
            examples=["CEO该关注什么？", "给管理层做个全面分析"],
        ),
        QueryPattern(
            pattern_name="comparison",
            description="对比类查询（客户对比、时间对比）",
            triggers=["对比", "比较", "vs", "和.*比", "差多少", "变化"],
            dimensions=["customers", "price_volume"],
            agent_hint=["analyst"],
            examples=["HMD和Samsung比怎么样？", "上半年vs下半年"],
        ),
    ]

    @classmethod
    def match(cls, question: str, customer_name: str = None) -> Optional[QueryPattern]:
        """匹配最佳查询模式"""
        q_lower = question.lower()
        best_match = None
        best_score = 0

        for pattern in cls.PATTERNS:
            score = 0
            for trigger in pattern.triggers:
                if trigger.lower() in q_lower:
                    score += 1
            # 单客户模式：如果识别到客户名
            if pattern.pattern_name == "single_customer" and customer_name:
                score += 3  # 高优先级
            if score > best_score:
                best_score = score
                best_match = pattern

        return best_match if best_score > 0 else None

    @classmethod
    def get_all_examples(cls) -> List[str]:
        """获取所有示例问题（可用于前端推荐）"""
        examples = []
        for p in cls.PATTERNS:
            examples.extend(p.examples)
        return examples


# ============================================================
# 5. 查询验证器 (Query Validator)
# ============================================================

class QueryValidator:
    """
    查询结果验证与自动纠错
    
    LinkedIn 论文启示：
    "automatically corrects hallucinations and syntax errors"
    
    对应到 MRARFAI：
    - 验证查询计划的维度是否合法
    - 验证筛选条件是否有效
    - 检测明显的数据异常（如金额为负）
    - 建议补充查询（如问风险时自动补充客户数据）
    """

    VALID_DIMENSIONS = set(SalesOntology.DIMENSIONS.keys())
    VALID_TIERS = {'A', 'B', 'C'}
    VALID_RISK_LEVELS = {'high', 'medium', 'low'}

    @classmethod
    def validate_plan(cls, plan: dict) -> Tuple[dict, List[str]]:
        """
        验证并修正查询计划
        返回: (修正后的plan, 修正说明列表)
        """
        corrections = []
        dims = plan.get('dimensions', [])
        filters = plan.get('filters', {})

        # 1. 维度验证
        valid_dims = [d for d in dims if d in cls.VALID_DIMENSIONS]
        invalid_dims = [d for d in dims if d not in cls.VALID_DIMENSIONS]
        if invalid_dims:
            corrections.append(f"移除无效维度: {invalid_dims}")
        plan['dimensions'] = valid_dims

        # 2. 等级验证
        tier = filters.get('tier', '')
        if tier and tier.upper() not in cls.VALID_TIERS:
            corrections.append(f"无效等级'{tier}'，已移除")
            del filters['tier']
        elif tier:
            filters['tier'] = tier.upper()

        # 3. 风险等级验证
        level = filters.get('level', '')
        if level and level.lower() not in cls.VALID_RISK_LEVELS:
            corrections.append(f"无效风险等级'{level}'，默认high")
            filters['level'] = 'high'

        # 4. Top N 范围检查
        top_n = filters.get('top_n')
        if top_n is not None:
            top_n = max(1, min(int(top_n), 30))
            filters['top_n'] = top_n

        # 5. 智能补充：查风险时补充客户数据
        if 'risks' in valid_dims and 'customers' not in valid_dims:
            valid_dims.append('customers')
            corrections.append("自动补充客户数据（风险分析需要）")

        # 6. 智能补充：查单客户时补充价量
        if filters.get('customer_name') and 'price_volume' not in valid_dims:
            if len(valid_dims) <= 2:
                valid_dims.append('price_volume')
                corrections.append("自动补充价量分解（单客户分析需要）")

        plan['dimensions'] = valid_dims
        plan['filters'] = filters
        return plan, corrections

    @classmethod
    def validate_result(cls, result: dict) -> List[str]:
        """验证查询结果是否合理"""
        warnings = []

        # 检查空结果
        for key, value in result.items():
            if isinstance(value, list) and len(value) == 0:
                warnings.append(f"'{key}' 返回空结果")
            elif isinstance(value, dict) and not value:
                warnings.append(f"'{key}' 返回空结果")

        return warnings


# ============================================================
# 6. 知识图谱整合入口
# ============================================================

class SalesKnowledgeGraph:
    """
    销售知识图谱 — 整合所有组件的统一接口
    
    使用方式：
        kg = SalesKnowledgeGraph()
        kg.build(data, results)
        plan = kg.understand(question)  → 结构化查询计划
    """

    def __init__(self):
        self.ontology = SalesOntology
        self.synonyms = SynonymGraph
        self.catalog = EntityCatalog()
        self.patterns = QueryPatternLibrary
        self.validator = QueryValidator
        self._built = False

    def build(self, data: dict, results: dict):
        """从数据构建知识图谱"""
        self.catalog.build_from_data(data, results)
        self._built = True

    def understand(self, question: str) -> dict:
        """
        核心方法：将自然语言问题转为结构化查询计划
        
        流程：
        1. 同义词解析 → 识别维度、客户、等级、TopN
        2. 模式匹配 → 找到最佳查询模式
        3. 实体补充 → 从目录中补充上下文
        4. 验证纠错 → 确保计划合法
        
        返回:
        {
            "dimensions": ["customers", "price_volume"],
            "filters": {"customer_name": "HMD"},
            "limit": 10,
            "pattern": "single_customer",
            "agent_hint": ["analyst"],
            "corrections": [],
            "entity_context": "HMD: A级, ¥35000万, 单价+5%"
        }
        """
        # Step 1: 同义词解析
        dims = self.synonyms.resolve_dimensions(question)
        customer = self.synonyms.resolve_customer(
            question, list(self.catalog.customers.keys())
        )
        tier = self.synonyms.resolve_tier(question)
        risk_level = self.synonyms.resolve_risk_level(question)
        top_n = self.synonyms.resolve_top_n(question)
        time_expr = self.synonyms.resolve_time(question)

        # Step 2: 模式匹配
        pattern = self.patterns.match(question, customer)

        # 合并维度（同义词 + 模式）
        if pattern:
            dims = list(set(dims + pattern.dimensions))

        # Step 3: 构建filters
        filters = {}
        if customer:
            filters['customer_name'] = customer
        if tier:
            filters['tier'] = tier
        if risk_level:
            filters['level'] = risk_level
        if top_n:
            filters['top_n'] = top_n

        if pattern and pattern.default_filters:
            for k, v in pattern.default_filters.items():
                if k not in filters:
                    filters[k] = v

        # 无维度降级
        if not dims:
            dims = ['overview', 'customers']

        plan = {
            'dimensions': dims,
            'filters': filters,
            'limit': top_n or 15,
        }

        # Step 4: 验证纠错
        plan, corrections = self.validator.validate_plan(plan)

        # Step 5: 实体上下文（给LLM的额外背景）
        entity_context = ""
        if customer:
            profile = self.catalog.get_customer_profile(customer)
            if profile:
                for name, info in profile.items():
                    parts = [f"{name}: {info.get('tier','?')}级, ¥{info.get('amount',0)}万"]
                    if info.get('risk'):
                        parts.append(f"风险:{info['risk']}")
                    if info.get('growth_opportunity'):
                        parts.append(f"机会:{info['growth_opportunity']}")
                    if info.get('price_change'):
                        parts.append(f"单价{info['price_change']}")
                    entity_context = ", ".join(parts)

        return {
            'dimensions': plan['dimensions'],
            'filters': plan['filters'],
            'limit': plan.get('limit', 15),
            'pattern': pattern.pattern_name if pattern else 'unknown',
            'agent_hint': pattern.agent_hint if pattern else ['analyst'],
            'corrections': corrections,
            'entity_context': entity_context,
        }

    def get_schema_for_llm(self) -> str:
        """生成给LLM的完整知识图谱上下文"""
        parts = [
            self.ontology.generate_schema_prompt(),
            "",
            "# 实体目录摘要",
            self.catalog.summarize(),
        ]
        return "\n".join(parts)

    def get_suggested_questions(self) -> List[str]:
        """获取推荐问题列表"""
        return self.patterns.get_all_examples()
