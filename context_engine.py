#!/usr/bin/env python3
"""
MRARFAI V8.0 — Phase II: Context Engineering Layer (上下文工程层)
=================================================================
借鉴:
  - Stanford ACE: 可进化 Playbook（生成→反思→策展），+10.6% Agent 效果
  - Anthropic: 上下文工程最佳实践 — 最小高信号 Token 集
  - Google ADK: Session/WorkingContext 分离 + 自动压缩
  - Context Rot 研究: 300 token 精准 > 113K 散漫

核心理念:
  上下文不是「全部塞进去」,而是「精选最有信号的片段」。
  Session = 持久状态（ground truth）
  WorkingContext = 编译视图（给 LLM 的最优子集）

对比 V7.0:
  V7: 固定模板拼接上下文 + 简单截断
  V8: 锯齿压缩 + 语义缓存 + 上下文窗口管理 + 可进化 Playbook
"""

import json
import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict


# ============================================================
# 1. 上下文压缩器 (锯齿形压缩 — InftyThink+)
# ============================================================

class ContextCompressor:
    """
    Agent 间锯齿形压缩

    原理：每个 Agent 产出后立即压缩，再传给下一个 Agent。
    2000 tokens → 压缩 → 300 tokens → 下一个 Agent → 压缩 → 200 tokens

    效果：
    - 防止 context rot（长上下文性能衰减）
    - 节省 token 成本
    - 保留核心信息
    """

    # 压缩策略
    STRATEGIES = {
        "aggressive": 0.15,   # 保留 15%（Agent间传递）
        "moderate": 0.3,      # 保留 30%（报告摘要）
        "gentle": 0.5,        # 保留 50%（重要分析）
        "none": 1.0,          # 不压缩
    }

    @staticmethod
    def compress(text: str, strategy: str = "moderate",
                 preserve_numbers: bool = True,
                 preserve_keywords: List[str] = None) -> str:
        """
        压缩文本

        Args:
            text: 原始文本
            strategy: 压缩策略
            preserve_numbers: 是否保留数字/金额
            preserve_keywords: 必须保留的关键词

        Returns:
            压缩后的文本
        """
        if not text or strategy == "none":
            return text

        ratio = ContextCompressor.STRATEGIES.get(strategy, 0.3)
        target_len = max(int(len(text) * ratio), 100)

        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs:
            return text

        # 段落评分
        scored = []
        for p in paragraphs:
            score = ContextCompressor._score_paragraph(
                p, preserve_numbers, preserve_keywords
            )
            scored.append((score, p))

        # 按得分排序，保留最高分段落
        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        current_len = 0
        for score, p in scored:
            if current_len + len(p) > target_len:
                break
            result.append(p)
            current_len += len(p)

        # 如果结果为空，至少保留第一段
        if not result and paragraphs:
            result = [paragraphs[0][:target_len]]

        return "\n".join(result)

    @staticmethod
    def _score_paragraph(text: str, preserve_numbers: bool,
                         keywords: List[str] = None) -> float:
        """段落信号得分"""
        score = 0.0

        # 包含数字（金额/百分比/指标）
        if preserve_numbers:
            num_count = len(re.findall(r'\d+\.?\d*[%万亿元]?', text))
            score += min(num_count * 0.15, 0.5)

        # 包含关键信号词
        signal_words = [
            '核心', '关键', '重要', '结论', '发现', '建议',
            '风险', '异常', '警告', '机会', '增长', '下滑',
            '排名', 'Top', '最大', '最高', '突破',
        ]
        signal_count = sum(1 for w in signal_words if w in text)
        score += min(signal_count * 0.1, 0.3)

        # 包含自定义关键词
        if keywords:
            kw_count = sum(1 for kw in keywords if kw in text)
            score += min(kw_count * 0.2, 0.4)

        # 段落长度适中（太短没信息，太长有噪音）
        if 30 < len(text) < 300:
            score += 0.1

        return score


# ============================================================
# 2. 语义缓存 (FASA 缓存优化)
# ============================================================

class SemanticCache:
    """
    语义级缓存

    不是精确匹配问题字符串，而是语义相似的问题复用缓存结果。

    实现:
    - 关键词哈希 → 快速匹配
    - TTL 过期机制
    - 最大容量限制 (LRU 淘汰)
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(self, question: str) -> str:
        """生成语义缓存键"""
        # 提取关键词并排序（忽略语序差异）
        keywords = set()
        # 中文关键词
        for word in re.findall(r'[\u4e00-\u9fff]{2,}', question):
            keywords.add(word)
        # 英文/数字
        for word in re.findall(r'[a-zA-Z0-9]+', question.lower()):
            keywords.add(word)

        # 排序后哈希
        sorted_kw = sorted(keywords)
        key_str = "|".join(sorted_kw)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, question: str) -> Optional[Dict]:
        """查询缓存"""
        key = self._make_key(question)
        entry = self.cache.get(key)

        if entry is None:
            self.misses += 1
            return None

        # TTL 检查
        if time.time() - entry["timestamp"] > self.ttl:
            del self.cache[key]
            self.misses += 1
            return None

        # 命中 → 移到末尾 (LRU)
        self.cache.move_to_end(key)
        self.hits += 1
        return entry["data"]

    def put(self, question: str, data: Dict):
        """存入缓存"""
        key = self._make_key(question)

        # LRU 淘汰
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = {
            "data": data,
            "question": question[:100],
            "timestamp": time.time(),
        }

    def get_stats(self) -> Dict:
        """缓存统计"""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / max(total, 1):.1%}",
        }

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# ============================================================
# 3. 上下文构建器 (Working Context Compiler)
# ============================================================

@dataclass
class ContextSlot:
    """上下文槽位"""
    name: str
    content: str
    priority: int       # 1-10, 10 最高
    token_estimate: int
    source: str         # "system" / "data" / "memory" / "agent"


class WorkingContextBuilder:
    """
    Working Context 构建器

    借鉴 Google ADK:
    - Session = 持久状态（所有数据）
    - WorkingContext = 编译视图（给 LLM 的精选子集）

    原则:
    1. 分离存储与展示
    2. 显式变换（named processors）
    3. 默认最小作用域（每个 Agent 看到最小必要上下文）
    """

    # Token 预算 (按 ComplexityLevel)
    BUDGETS = {
        "skip": 500,       # 简单查询预算
        "light": 3000,     # 中等复杂度
        "full": 6000,      # 全流程
    }

    def __init__(self, budget_level: str = "full"):
        self.budget = self.BUDGETS.get(budget_level, 6000)
        self.slots: List[ContextSlot] = []

    def add_slot(self, name: str, content: str, priority: int = 5,
                 source: str = "system"):
        """添加上下文槽位"""
        if not content:
            return
        token_est = len(content) // 2  # 粗略估计中文 token
        self.slots.append(ContextSlot(
            name=name,
            content=content,
            priority=priority,
            token_estimate=token_est,
            source=source,
        ))

    def compile(self) -> str:
        """
        编译 Working Context

        按优先级排序，在 Token 预算内选择最高优先级槽位
        """
        # 按优先级降序排列
        sorted_slots = sorted(self.slots, key=lambda s: s.priority, reverse=True)

        selected = []
        used_tokens = 0

        for slot in sorted_slots:
            if used_tokens + slot.token_estimate > self.budget:
                # 尝试压缩后放入
                remaining_budget = self.budget - used_tokens
                if remaining_budget > 100:
                    compressed = ContextCompressor.compress(
                        slot.content,
                        strategy="aggressive"
                    )
                    compressed_tokens = len(compressed) // 2
                    if compressed_tokens <= remaining_budget:
                        selected.append(f"[{slot.name}]\n{compressed}")
                        used_tokens += compressed_tokens
                continue

            selected.append(f"[{slot.name}]\n{slot.content}")
            used_tokens += slot.token_estimate

        return "\n\n".join(selected)

    def get_stats(self) -> Dict:
        """构建统计"""
        return {
            "total_slots": len(self.slots),
            "budget": self.budget,
            "total_tokens_available": sum(s.token_estimate for s in self.slots),
        }


# ============================================================
# 4. 可进化 Playbook (借鉴 Stanford ACE)
# ============================================================

@dataclass
class PlaybookEntry:
    """Playbook 条目"""
    strategy: str         # 策略描述
    success_count: int    # 成功使用次数
    fail_count: int       # 失败次数
    last_used: float      # 最后使用时间
    source: str           # 来源 (initial / learned / refined)


class EvolvingPlaybook:
    """
    可进化的分析 Playbook

    借鉴 Stanford ACE:
    - 上下文 = 可进化的 Playbook
    - 通过 生成→反思→策展 三步不断改进
    - 防止 brevity bias（保留领域洞察）和 context collapse（防止细节丢失）

    MRARFAI 应用:
    - 每种查询类型都有最优分析策略
    - 策略随使用效果自动调优
    - 新策略通过反思自动生成
    """

    def __init__(self):
        self.entries: Dict[str, List[PlaybookEntry]] = {
            "customer_analysis": [
                PlaybookEntry(
                    strategy="先看 ABC 分级占比，再看同比趋势，最后聚焦异常客户",
                    success_count=10, fail_count=0,
                    last_used=time.time(), source="initial"
                ),
            ],
            "risk_assessment": [
                PlaybookEntry(
                    strategy="先扫描断崖下滑（>30%）客户，再看集中度 HHI，最后评估流失金额",
                    success_count=8, fail_count=1,
                    last_used=time.time(), source="initial"
                ),
            ],
            "growth_strategy": [
                PlaybookEntry(
                    strategy="先对标行业增长率，再识别低渗透客户，最后计算 TAM 提升空间",
                    success_count=6, fail_count=0,
                    last_used=time.time(), source="initial"
                ),
            ],
            "overview_report": [
                PlaybookEntry(
                    strategy="核心数字开头 → 客户分级 → 风险预警 → 增长机会 → 行动项",
                    success_count=12, fail_count=0,
                    last_used=time.time(), source="initial"
                ),
            ],
        }

    def get_strategy(self, query_type: str) -> str:
        """获取最优策略"""
        entries = self.entries.get(query_type, [])
        if not entries:
            return ""

        # 按成功率排序
        best = max(entries,
                   key=lambda e: e.success_count / max(e.success_count + e.fail_count, 1))
        best.last_used = time.time()
        return best.strategy

    def record_feedback(self, query_type: str, strategy: str, success: bool):
        """记录策略使用反馈"""
        entries = self.entries.get(query_type, [])
        for entry in entries:
            if entry.strategy == strategy:
                if success:
                    entry.success_count += 1
                else:
                    entry.fail_count += 1
                return

    def add_strategy(self, query_type: str, strategy: str, source: str = "learned"):
        """添加新策略"""
        if query_type not in self.entries:
            self.entries[query_type] = []

        self.entries[query_type].append(PlaybookEntry(
            strategy=strategy,
            success_count=0,
            fail_count=0,
            last_used=time.time(),
            source=source,
        ))

    def get_all_stats(self) -> Dict:
        """获取所有策略统计"""
        stats = {}
        for qtype, entries in self.entries.items():
            stats[qtype] = [
                {
                    "strategy": e.strategy[:60] + "...",
                    "success_rate": f"{e.success_count / max(e.success_count + e.fail_count, 1):.0%}",
                    "uses": e.success_count + e.fail_count,
                    "source": e.source,
                }
                for e in entries
            ]
        return stats


# ============================================================
# 5. YAML Schema 上下文 (McMillan 最佳实践)
# ============================================================

class YAMLContextSchema:
    """
    YAML-native 上下文 Schema

    借鉴 McMillan 研究:
    - YAML > JSON 对 LLM 的可读性
    - 结构化 Schema 帮助 LLM 理解数据维度
    - 文件式上下文 > 内联式上下文
    """

    @staticmethod
    def build_client_profile(company: str = "禾苗通讯",
                             analysis_focus: List[str] = None,
                             risk_tolerance: str = "medium") -> str:
        """构建客户档案 YAML"""
        focus = analysis_focus or ["季节性趋势", "客户健康度", "产品结构"]
        return f"""# 客户档案 (TELOS)
client_profile:
  company: "{company}"
  industry: "ODM/OEM 手机通讯"
  analysis_focus:
{chr(10).join(f'    - "{f}"' for f in focus)}
  risk_tolerance: "{risk_tolerance}"
  reporting_language: "zh-CN"
  key_metrics:
    - 年度营收及同比增长
    - 客户 ABC 分级变动
    - Top10 客户集中度 (HHI)
    - 异常检测 (5模型集成)
"""

    @staticmethod
    def build_data_schema(data: dict) -> str:
        """构建数据 Schema YAML"""
        total = data.get('总营收', 'N/A')
        yoy = data.get('总YoY', 'N/A')
        customers = data.get('客户金额', [])
        n_customers = len(customers) if isinstance(customers, list) else 0

        return f"""# 数据概览
data_overview:
  total_revenue: {total}
  yoy_growth: "{yoy}"
  customer_count: {n_customers}
  data_dimensions:
    - overview: 总营收, 同比, 月度列表
    - customers: 客户分级 (A/B/C), 金额, 占比
    - risks: 流失预警, 异常检测
    - growth: 增长机会, 潜力金额
    - price_volume: 价量分解
    - regions: 区域分布, HHI指数
    - categories: 品类趋势
    - benchmark: 行业对标
    - forecast: 预测场景
"""


# ============================================================
# 6. 全局实例
# ============================================================

_context_cache: Optional[SemanticCache] = None

def get_context_cache() -> SemanticCache:
    global _context_cache
    if _context_cache is None:
        _context_cache = SemanticCache(max_size=200, ttl_seconds=1800)
    return _context_cache

_playbook: Optional[EvolvingPlaybook] = None

def get_playbook() -> EvolvingPlaybook:
    global _playbook
    if _playbook is None:
        _playbook = EvolvingPlaybook()
    return _playbook


# ============================================================
# 7. 便捷接口
# ============================================================

def build_working_context(
    question: str,
    data: dict,
    results: dict,
    memory_context: str = "",
    level: str = "full",
    agent_id: str = "",
) -> str:
    """
    V8.0 上下文构建 — 一行调用

    根据查询类型和复杂度，构建最优 Working Context。
    """
    builder = WorkingContextBuilder(budget_level=level)

    # 系统指令 (最高优先级)
    client_yaml = YAMLContextSchema.build_client_profile()
    builder.add_slot("客户档案", client_yaml, priority=10, source="system")

    # 数据 Schema
    data_yaml = YAMLContextSchema.build_data_schema(data)
    builder.add_slot("数据概览", data_yaml, priority=9, source="system")

    # Playbook 策略
    playbook = get_playbook()
    query_type = _detect_query_type(question)
    strategy = playbook.get_strategy(query_type)
    if strategy:
        builder.add_slot("分析策略", f"推荐策略: {strategy}", priority=8, source="system")

    # 记忆上下文
    if memory_context:
        builder.add_slot("对话记忆", memory_context, priority=6, source="memory")

    # 数据上下文 (根据级别调整)
    data_context = json.dumps(results, ensure_ascii=False, default=str)
    if level == "skip":
        data_context = data_context[:500]
    elif level == "light":
        data_context = ContextCompressor.compress(data_context, "moderate")

    builder.add_slot("分析数据", data_context, priority=7, source="data")

    return builder.compile()


def compress_agent_output(output: str, strategy: str = "moderate") -> str:
    """压缩 Agent 输出（用于 Agent 间传递）"""
    return ContextCompressor.compress(output, strategy)


def _detect_query_type(question: str) -> str:
    """检测查询类型"""
    q = question.lower()
    if any(k in q for k in ['风险', '预警', '异常', '流失']):
        return "risk_assessment"
    if any(k in q for k in ['增长', '机会', '战略', '策略', '建议']):
        return "growth_strategy"
    if any(k in q for k in ['CEO', '总结', '全面', '概览', '报告']):
        return "overview_report"
    return "customer_analysis"
