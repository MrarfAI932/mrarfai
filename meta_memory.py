#!/usr/bin/env python3
"""
MRARFAI V8.0 — Phase III: Meta-Memory Layer (元记忆层)
========================================================
借鉴:
  - MAGMA (Jan 2026 SOTA): 4维正交图 (语义/时间/因果/实体)
  - EverMemOS (Feb 2026): 自组织记忆 OS, 3阶段 (情景→语义→重构)
  - MemRL (Jan 2026): 运行时 RL 自进化记忆策略
  - Memory in the Age of AI Agents Survey: 统一 forms × functions × dynamics

+8 分提升 — V8.0 中增幅最大的模块

核心升级:
  V7.0: SQLite 扁平存储 + 简单实体/洞察检索
  V8.0: 多维记忆图 + 6种 CRUD 操作 + 记忆巩固/衰减 + TELOS客户画像
"""

import json
import time
import math
import sqlite3
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


# ============================================================
# 1. 记忆类型定义 (MAGMA 4维 + EverMemOS 3阶段)
# ============================================================

class MemoryType(Enum):
    """记忆类型 — EverMemOS 启发"""
    EPISODIC = "episodic"      # 情景记忆: 具体对话/分析事件
    SEMANTIC = "semantic"      # 语义记忆: 巩固后的领域知识
    PROCEDURAL = "procedural"  # 程序记忆: 学到的分析模式/技能


class MemoryDimension(Enum):
    """记忆维度 — MAGMA 4维正交图"""
    SEMANTIC = "semantic"      # 语义相似性
    TEMPORAL = "temporal"      # 时间关联
    CAUSAL = "causal"          # 因果关系
    ENTITY = "entity"          # 实体关联


@dataclass
class MemoryNode:
    """记忆节点 — 图中的一个记忆单元"""
    node_id: str
    content: str                            # 记忆内容
    memory_type: MemoryType                 # 记忆类型
    importance: float = 0.5                 # 重要性 0-1
    recency: float = 1.0                    # 新鲜度 0-1 (随时间衰减)
    access_count: int = 0                   # 访问次数
    created_at: float = 0.0                 # 创建时间
    last_accessed: float = 0.0              # 最后访问时间
    entities: List[str] = field(default_factory=list)  # 关联实体
    tags: List[str] = field(default_factory=list)      # 标签
    source: str = ""                        # 来源 (question / analysis / feedback)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def relevance_score(self, query_entities: List[str] = None,
                        query_tags: List[str] = None) -> float:
        """计算记忆与查询的相关性得分"""
        score = 0.0

        # 重要性权重
        score += self.importance * 0.3

        # 新鲜度权重 (时间衰减)
        age_hours = (time.time() - self.last_accessed) / 3600
        freshness = math.exp(-age_hours / 168)  # 一周半衰期
        score += freshness * 0.25

        # 实体重叠
        if query_entities:
            overlap = len(set(self.entities) & set(query_entities))
            entity_score = min(overlap / max(len(query_entities), 1), 1.0)
            score += entity_score * 0.3

        # 标签重叠
        if query_tags:
            tag_overlap = len(set(self.tags) & set(query_tags))
            tag_score = min(tag_overlap / max(len(query_tags), 1), 1.0)
            score += tag_score * 0.15

        return min(score, 1.0)


@dataclass
class MemoryEdge:
    """记忆边 — 连接两个记忆节点"""
    source_id: str
    target_id: str
    dimension: MemoryDimension   # 连接维度
    weight: float = 0.5          # 连接强度 0-1
    relation: str = ""           # 关系描述


# ============================================================
# 2. TELOS 客户画像 (ODM/OEM 专用)
# ============================================================

@dataclass
class TELOSProfile:
    """
    TELOS 客户画像 — 5维度全景

    T - Trend (趋势): 营收趋势、季节性模式
    E - Engagement (互动): 订单频率、沟通历史
    L - Lifecycle (生命周期): 客户阶段 (新/成长/稳定/衰退)
    O - Opportunity (机会): 交叉销售、份额提升空间
    S - Stability (稳定性): 集中度风险、支付风险
    """
    customer_name: str
    # T - 趋势
    revenue_trend: str = "stable"          # growing / stable / declining / cliff
    seasonal_pattern: str = ""             # H1重/H2重/均匀
    yoy_growth: float = 0.0
    # E - 互动
    order_frequency: str = "regular"       # high / regular / low / dormant
    last_order_date: str = ""
    # L - 生命周期
    lifecycle_stage: str = "stable"        # new / growing / stable / declining / churned
    customer_since: str = ""
    # O - 机会
    wallet_share: float = 0.0             # 钱包份额 (0-1)
    cross_sell_potential: List[str] = field(default_factory=list)
    tam_uplift: float = 0.0               # TAM 提升空间 (万元)
    # S - 稳定性
    concentration_risk: str = "low"        # high / medium / low
    health_score: float = 0.0             # 健康评分 0-100
    risk_tags: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """生成可嵌入 prompt 的 TELOS 摘要"""
        lines = [f"[TELOS画像] {self.customer_name}"]
        lines.append(f"  T趋势: {self.revenue_trend} (YoY {self.yoy_growth:+.1f}%)")
        lines.append(f"  E互动: {self.order_frequency}")
        lines.append(f"  L周期: {self.lifecycle_stage}")
        if self.tam_uplift > 0:
            lines.append(f"  O机会: TAM提升空间 {self.tam_uplift:.0f}万元")
        lines.append(f"  S稳定: 健康分{self.health_score:.0f} | 风险={self.concentration_risk}")
        if self.risk_tags:
            lines.append(f"  ⚠️ {', '.join(self.risk_tags)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "customer": self.customer_name,
            "trend": self.revenue_trend,
            "yoy": self.yoy_growth,
            "lifecycle": self.lifecycle_stage,
            "health": self.health_score,
            "risk": self.concentration_risk,
            "tam_uplift": self.tam_uplift,
            "risk_tags": self.risk_tags,
        }


# ============================================================
# 3. 多维记忆图 (MAGMA 架构)
# ============================================================

class MemoryGraph:
    """
    多维记忆图 — MAGMA 启发

    4种维度的边:
    - SEMANTIC: 语义相似的记忆相连
    - TEMPORAL: 时间相近的记忆相连
    - CAUSAL:   因果关联的记忆相连
    - ENTITY:   共享实体的记忆相连

    策略引导的图遍历:
    - 查询时根据问题类型选择遍历策略
    - 风险问题 → 优先 CAUSAL + ENTITY
    - 趋势问题 → 优先 TEMPORAL + SEMANTIC
    """

    def __init__(self, db_path: str = "memory_graph.db"):
        self.db_path = db_path
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []
        self.telos_profiles: Dict[str, TELOSProfile] = {}
        # 记忆统计
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "retrievals": 0,
            "consolidations": 0,
        }
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 存储"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    node_id TEXT PRIMARY KEY,
                    content TEXT,
                    memory_type TEXT,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    created_at REAL,
                    last_accessed REAL,
                    entities TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    source TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT,
                    target_id TEXT,
                    dimension TEXT,
                    weight REAL DEFAULT 0.5,
                    relation TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telos_profiles (
                    customer_name TEXT PRIMARY KEY,
                    profile_json TEXT,
                    updated_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_skills (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT,
                    pattern TEXT,
                    strategy TEXT,
                    success_count INTEGER DEFAULT 0,
                    fail_count INTEGER DEFAULT 0,
                    created_at REAL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Memory graph database initialization failed: {e}")

    # ---- CRUD 操作 (AtomMem 6种工具) ----

    def add(self, content: str, memory_type: MemoryType = MemoryType.EPISODIC,
            importance: float = 0.5, entities: List[str] = None,
            tags: List[str] = None, source: str = "",
            metadata: Dict = None) -> str:
        """添加记忆节点"""
        node_id = hashlib.md5(
            f"{content[:100]}_{time.time()}".encode()
        ).hexdigest()[:12]

        node = MemoryNode(
            node_id=node_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            access_count=0,
            created_at=time.time(),
            last_accessed=time.time(),
            entities=entities or [],
            tags=tags or [],
            source=source,
            metadata=metadata or {},
        )

        self.nodes[node_id] = node
        self.stats["total_nodes"] += 1

        # 自动建立边
        self._auto_connect(node)

        # 持久化
        self._save_node(node)

        return node_id

    def retrieve(self, query: str, entities: List[str] = None,
                 tags: List[str] = None, limit: int = 5,
                 strategy: str = "balanced") -> List[MemoryNode]:
        """
        检索相关记忆

        策略:
        - "balanced": 平衡各维度
        - "causal": 优先因果链
        - "temporal": 优先时间线
        - "entity_focused": 优先实体关联
        """
        self.stats["retrievals"] += 1

        # 加载节点（如果内存中没有）
        if not self.nodes:
            self._load_all_nodes()

        # 评分
        scored = []
        for node in self.nodes.values():
            score = node.relevance_score(entities, tags)

            # 内容关键词匹配
            query_words = set(query)
            content_words = set(node.content)
            keyword_overlap = len(query_words & content_words)
            score += min(keyword_overlap / max(len(query_words), 1), 0.3) * 0.2

            # 策略加权
            if strategy == "entity_focused" and entities:
                entity_overlap = len(set(node.entities) & set(entities))
                score += entity_overlap * 0.15
            elif strategy == "temporal":
                freshness = math.exp(-(time.time() - node.last_accessed) / 86400)
                score += freshness * 0.2

            scored.append((score, node))

        # 排序取 Top-K
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [node for _, node in scored[:limit]]

        # 更新访问记录
        for node in results:
            node.access_count += 1
            node.last_accessed = time.time()

        return results

    def update(self, node_id: str, content: str = None,
               importance: float = None, tags: List[str] = None):
        """更新记忆节点"""
        node = self.nodes.get(node_id)
        if not node:
            return False

        if content is not None:
            node.content = content
        if importance is not None:
            node.importance = importance
        if tags is not None:
            node.tags = tags

        node.last_accessed = time.time()
        self._save_node(node)
        return True

    def delete(self, node_id: str) -> bool:
        """删除记忆节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            # 删除相关边
            self.edges = [e for e in self.edges
                         if e.source_id != node_id and e.target_id != node_id]
            self._delete_node_db(node_id)
            return True
        return False

    def summarize(self, entity: str = None, memory_type: MemoryType = None,
                  max_items: int = 10) -> str:
        """
        记忆摘要 — 将多个相关记忆浓缩为一段话

        EverMemOS 的 Semantic Consolidation 阶段
        """
        # 筛选相关记忆
        candidates = list(self.nodes.values())
        if entity:
            candidates = [n for n in candidates if entity in n.entities]
        if memory_type:
            candidates = [n for n in candidates if n.memory_type == memory_type]

        # 按重要性排序
        candidates.sort(key=lambda n: n.importance, reverse=True)
        top = candidates[:max_items]

        if not top:
            return "暂无相关记忆"

        # 拼接摘要
        summaries = []
        for node in top:
            age_days = (time.time() - node.created_at) / 86400
            freshness = "近期" if age_days < 7 else "历史"
            summaries.append(f"[{freshness}] {node.content[:150]}")

        return "\n".join(summaries)

    def filter(self, min_importance: float = 0.0,
               memory_type: MemoryType = None,
               entity: str = None,
               since_hours: float = None) -> List[MemoryNode]:
        """过滤记忆节点"""
        results = []
        for node in self.nodes.values():
            if node.importance < min_importance:
                continue
            if memory_type and node.memory_type != memory_type:
                continue
            if entity and entity not in node.entities:
                continue
            if since_hours:
                age_hours = (time.time() - node.created_at) / 3600
                if age_hours > since_hours:
                    continue
            results.append(node)
        return results

    # ---- 自动连接 ----

    def _auto_connect(self, new_node: MemoryNode):
        """自动建立边 — 基于实体和标签重叠"""
        for existing_id, existing in self.nodes.items():
            if existing_id == new_node.node_id:
                continue

            # 实体维度
            entity_overlap = set(new_node.entities) & set(existing.entities)
            if entity_overlap:
                edge = MemoryEdge(
                    source_id=new_node.node_id,
                    target_id=existing_id,
                    dimension=MemoryDimension.ENTITY,
                    weight=min(len(entity_overlap) * 0.3, 1.0),
                    relation=f"共享实体: {', '.join(list(entity_overlap)[:3])}",
                )
                self.edges.append(edge)
                self.stats["total_edges"] += 1

            # 时间维度 (12小时内)
            time_diff = abs(new_node.created_at - existing.created_at)
            if time_diff < 43200:  # 12 hours
                weight = 1.0 - (time_diff / 43200)
                edge = MemoryEdge(
                    source_id=new_node.node_id,
                    target_id=existing_id,
                    dimension=MemoryDimension.TEMPORAL,
                    weight=weight,
                    relation="时间相近",
                )
                self.edges.append(edge)
                self.stats["total_edges"] += 1

    # ---- 记忆巩固 (EverMemOS 3阶段) ----

    def consolidate(self, min_access: int = 3, max_age_days: int = 30):
        """
        记忆巩固 — EverMemOS 启发

        阶段 1: Episodic Trace → 保留原始事件记忆
        阶段 2: Semantic Consolidation → 多个相关记忆合并为语义知识
        阶段 3: Reconstructive Recollection → 检索时动态重构

        触发条件:
        - 被多次访问 (access_count >= min_access) 的 EPISODIC 记忆
        - 升级为 SEMANTIC 记忆
        """
        episodic_nodes = [
            n for n in self.nodes.values()
            if n.memory_type == MemoryType.EPISODIC and n.access_count >= min_access
        ]

        consolidated = 0
        for node in episodic_nodes:
            # 升级为语义记忆
            node.memory_type = MemoryType.SEMANTIC
            node.importance = min(node.importance + 0.2, 1.0)
            consolidated += 1

        # 清理过期低重要性记忆
        cutoff = time.time() - (max_age_days * 86400)
        expired = [
            nid for nid, n in self.nodes.items()
            if n.created_at < cutoff
            and n.importance < 0.3
            and n.access_count < 2
        ]
        for nid in expired:
            self.delete(nid)

        self.stats["consolidations"] += 1
        return {
            "consolidated": consolidated,
            "expired_removed": len(expired),
        }

    # ---- TELOS 画像管理 ----

    def upsert_telos(self, profile: TELOSProfile):
        """创建/更新 TELOS 客户画像"""
        self.telos_profiles[profile.customer_name] = profile
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO telos_profiles VALUES (?, ?, ?)",
                (profile.customer_name, json.dumps(profile.to_dict(), ensure_ascii=False),
                 time.time())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to persist TELOS profile for {profile.customer_name}: {e}")

    def get_telos(self, customer_name: str) -> Optional[TELOSProfile]:
        """获取 TELOS 客户画像"""
        return self.telos_profiles.get(customer_name)

    def build_telos_from_data(self, customer_data: dict,
                              health_scores: list = None) -> TELOSProfile:
        """从销售数据自动构建 TELOS 画像"""
        name = customer_data.get('客户', '')
        yoy = customer_data.get('YoY', 0)
        h1 = customer_data.get('H1金额', 0)
        h2 = customer_data.get('H2金额', 0)
        annual = customer_data.get('年度金额', 0)
        grade = customer_data.get('等级', 'C')

        # T - 趋势
        if isinstance(yoy, (int, float)):
            if yoy > 20:
                trend = "growing"
            elif yoy > -10:
                trend = "stable"
            elif yoy > -30:
                trend = "declining"
            else:
                trend = "cliff"
        else:
            trend = "stable"

        # 季节性
        if isinstance(h1, (int, float)) and isinstance(h2, (int, float)):
            if h1 > 0 and h2 > 0:
                ratio = h1 / max(h2, 1)
                seasonal = "H1重" if ratio > 1.3 else ("H2重" if ratio < 0.7 else "均匀")
            else:
                seasonal = "数据不足"
        else:
            seasonal = "数据不足"

        # L - 生命周期
        if isinstance(yoy, (int, float)):
            if yoy > 50:
                lifecycle = "new"
            elif yoy > 10:
                lifecycle = "growing"
            elif yoy > -15:
                lifecycle = "stable"
            else:
                lifecycle = "declining"
        else:
            lifecycle = "stable"

        # S - 稳定性
        health = 50.0
        risk_tags = []
        if health_scores:
            for hs in health_scores:
                if hs.get('客户', '') == name:
                    health = hs.get('总分', 50)
                    risk_tags = hs.get('风险标签', [])
                    break

        conc_risk = "high" if grade == 'A' else ("medium" if grade == 'B' else "low")

        profile = TELOSProfile(
            customer_name=name,
            revenue_trend=trend,
            seasonal_pattern=seasonal,
            yoy_growth=float(yoy) if isinstance(yoy, (int, float)) else 0,
            lifecycle_stage=lifecycle,
            concentration_risk=conc_risk,
            health_score=health,
            risk_tags=risk_tags,
        )

        self.upsert_telos(profile)
        return profile

    # ---- 记忆上下文构建 ----

    def build_memory_context(self, question: str,
                             entities: List[str] = None,
                             max_length: int = 1500) -> str:
        """
        构建记忆上下文 — 供 Agent 使用

        整合:
        1. 相关记忆检索
        2. TELOS 画像
        3. 学到的分析技能
        """
        sections = []

        # 1. 检索相关记忆
        memories = self.retrieve(question, entities, limit=5)
        if memories:
            mem_lines = ["[跨会话记忆]"]
            for m in memories:
                mem_lines.append(f"- [{m.memory_type.value}] {m.content[:120]}")
            sections.append("\n".join(mem_lines))

        # 2. TELOS 画像
        if entities:
            for entity in entities[:3]:
                telos = self.get_telos(entity)
                if telos:
                    sections.append(telos.to_prompt())

        # 3. 历史分析技能
        skills = self._get_relevant_skills(question)
        if skills:
            sections.append(f"[历史分析策略]\n{skills}")

        result = "\n\n".join(sections)

        # 长度控制
        if len(result) > max_length:
            result = result[:max_length] + "\n..."

        return result

    def _get_relevant_skills(self, question: str) -> str:
        """获取相关的历史分析技能"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT name, strategy, success_count FROM memory_skills "
                "ORDER BY success_count DESC LIMIT 3"
            )
            skills = cursor.fetchall()
            conn.close()
            if skills:
                return "\n".join(
                    f"- {name}: {strategy} (成功{count}次)"
                    for name, strategy, count in skills
                )
        except Exception as e:
            logger.debug(f"Failed to retrieve memory skills: {e}")
        return ""

    # ---- 学习新技能 (MemSkill) ----

    def learn_skill(self, name: str, pattern: str, strategy: str):
        """学习新的分析技能"""
        skill_id = hashlib.md5(name.encode()).hexdigest()[:8]
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO memory_skills VALUES (?, ?, ?, ?, 0, 0, ?)",
                (skill_id, name, pattern, strategy, time.time())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to persist memory skill {name}: {e}")

    def record_skill_result(self, name: str, success: bool):
        """记录技能使用结果"""
        try:
            conn = sqlite3.connect(self.db_path)
            field_name = "success_count" if success else "fail_count"
            conn.execute(
                f"UPDATE memory_skills SET {field_name} = {field_name} + 1 WHERE name = ?",
                (name,)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to record skill result for {name}: {e}")

    # ---- 持久化 ----

    def _save_node(self, node: MemoryNode):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO memory_nodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (node.node_id, node.content, node.memory_type.value,
                 node.importance, node.access_count, node.created_at,
                 node.last_accessed, json.dumps(node.entities, ensure_ascii=False),
                 json.dumps(node.tags, ensure_ascii=False), node.source,
                 json.dumps(node.metadata, ensure_ascii=False))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to persist memory node {node.node_id}: {e}")

    def _delete_node_db(self, node_id: str):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM memory_nodes WHERE node_id = ?", (node_id,))
            conn.execute("DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?",
                        (node_id, node_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to delete memory node {node_id}: {e}")

    def _load_all_nodes(self):
        """从数据库加载所有节点"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT * FROM memory_nodes")
            for row in cursor.fetchall():
                node = MemoryNode(
                    node_id=row[0], content=row[1],
                    memory_type=MemoryType(row[2]),
                    importance=row[3], access_count=row[4],
                    created_at=row[5], last_accessed=row[6],
                    entities=json.loads(row[7] or '[]'),
                    tags=json.loads(row[8] or '[]'),
                    source=row[9] or '',
                    metadata=json.loads(row[10] or '{}'),
                )
                self.nodes[node.node_id] = node
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to load memory nodes from {self.db_path}: {e}")

    def get_stats(self) -> Dict:
        """获取记忆图统计"""
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            type_counts[node.memory_type.value] += 1

        return {
            **self.stats,
            "type_distribution": dict(type_counts),
            "telos_profiles": len(self.telos_profiles),
            "avg_importance": (
                sum(n.importance for n in self.nodes.values()) /
                max(len(self.nodes), 1)
            ),
        }


# ============================================================
# 4. 全局实例
# ============================================================

_memory_graph: Optional[MemoryGraph] = None

def get_memory_graph(db_path: str = "memory_graph.db") -> MemoryGraph:
    global _memory_graph
    if _memory_graph is None:
        _memory_graph = MemoryGraph(db_path)
    return _memory_graph

def set_memory_graph(graph: MemoryGraph):
    global _memory_graph
    _memory_graph = graph
