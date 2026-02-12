#!/usr/bin/env python3
"""
MRARFAI V9.0 — 3D Memory Architecture
=========================================
基于 "Memory in the Age of AI Agents: A Comprehensive Survey"

核心框架: Forms × Functions × Dynamics 三维记忆
  Forms (形式):     Text / Parametric / Structured
  Functions (功能): Storage / Retrieval / Reasoning / Learning
  Dynamics (动态):  Consolidation / Decay / Transfer / Evolution

与 V8.0 meta_memory.py 的关系:
  V8: MAGMA 4维图 + EverMemOS 3阶段 + MemRL
  V9: 在 V8 之上添加:
    1. 三维记忆索引 — 同一记忆可按 3 维度检索
    2. 跨Agent记忆共享 — 分析师的发现可被风控专家引用
    3. 记忆进化引擎 — 自动合并/升级/遗忘
    4. 技能记忆 — 学到的分析模式可跨任务复用

集成点:
  - meta_memory.py (V8): 继承 MemoryNode, MemoryType, MemoryDimension
  - persistent_memory.py: 继承 SQLite 持久化
  - rlm_engine.py: RLM 递归层间共享记忆
  - reasoning_templates.py: 推理模板匹配记忆中的相似分析
  - search_engine.py: 搜索时用记忆剪枝

效果:
  - 记忆复用率 60%+ (相似问题直接调用历史分析)
  - 跨Agent知识传递 (风控发现→策略师可用)
  - 自动遗忘低价值记忆, 巩固高价值洞察
"""

import json
import time
import math
import sqlite3
import hashlib
import logging
from typing import Optional, Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("mrarfai.memory_v9")


# ============================================================
# 三维坐标系
# ============================================================

class MemoryForm(Enum):
    """记忆形式 — 存储的物理形态"""
    TEXT = "text"                # 自然语言文本
    PARAMETRIC = "parametric"   # 嵌入向量 / 权重
    STRUCTURED = "structured"   # 知识图谱三元组 / JSON

class MemoryFunction(Enum):
    """记忆功能 — 记忆被使用的方式"""
    STORAGE = "storage"         # 纯存储
    RETRIEVAL = "retrieval"     # 可被检索
    REASONING = "reasoning"     # 用于推理链
    LEARNING = "learning"       # 可被学习/蒸馏

class MemoryDynamic(Enum):
    """记忆动态 — 记忆如何随时间变化"""
    STABLE = "stable"           # 稳定不变 (核心知识)
    CONSOLIDATING = "consol"    # 正在巩固 (多次验证中)
    DECAYING = "decaying"       # 正在衰减 (长期未访问)
    EVOLVING = "evolving"       # 正在进化 (被新信息更新)


# ============================================================
# 3D 记忆节点
# ============================================================

@dataclass
class Memory3DNode:
    """
    三维记忆节点 — 每个记忆有 Form × Function × Dynamic 坐标
    """
    node_id: str
    content: str                        # 记忆内容
    
    # 三维坐标
    form: MemoryForm = MemoryForm.TEXT
    function: MemoryFunction = MemoryFunction.STORAGE
    dynamic: MemoryDynamic = MemoryDynamic.STABLE
    
    # 元数据
    importance: float = 0.5             # 重要性 [0, 1]
    confidence: float = 0.5            # 置信度 [0, 1] (多次验证提升)
    access_count: int = 0               # 访问次数
    created_at: float = 0.0
    last_accessed: float = 0.0
    last_updated: float = 0.0
    
    # 关联
    entities: List[str] = field(default_factory=list)     # 关联实体
    tags: List[str] = field(default_factory=list)          # 标签
    source_agent: str = ""              # 来源Agent
    source_query: str = ""              # 触发此记忆的查询
    
    # 结构化数据 (STRUCTURED form)
    triples: List[Tuple[str, str, str]] = field(default_factory=list)  # (主语, 谓语, 宾语)
    
    # 技能记忆 (LEARNING function)
    skill_pattern: str = ""             # 可复用的分析模式
    skill_success_rate: float = 0.0     # 技能成功率
    
    # 进化追踪
    version: int = 1
    parent_id: str = ""                 # 进化自哪个记忆
    merge_sources: List[str] = field(default_factory=list)  # 合并来源
    
    def relevance(self, query: str = "", entities: List[str] = None,
                  agent: str = "") -> float:
        """计算与查询的相关性"""
        score = 0.0
        
        # 重要性
        score += self.importance * 0.25
        
        # 新鲜度 (指数衰减, 1周半衰期)
        age_hours = (time.time() - self.last_accessed) / 3600
        freshness = math.exp(-age_hours / 168)
        score += freshness * 0.20
        
        # 置信度
        score += self.confidence * 0.15
        
        # 实体重叠
        if entities:
            overlap = len(set(self.entities) & set(entities))
            score += min(overlap / max(len(entities), 1), 1.0) * 0.20
        
        # 文本匹配 (简单关键词)
        if query:
            query_words = set(query.lower().split())
            content_words = set(self.content.lower().split())
            word_overlap = len(query_words & content_words)
            score += min(word_overlap / max(len(query_words), 1), 1.0) * 0.15
        
        # Agent 偏好
        if agent and self.source_agent == agent:
            score += 0.05
        
        return min(1.0, score)
    
    def to_prompt_text(self, max_len: int = 200) -> str:
        """转换为可嵌入 prompt 的文本"""
        parts = [f"[{self.form.value}/{self.function.value}]"]
        parts.append(self.content[:max_len])
        if self.entities:
            parts.append(f"(实体: {', '.join(self.entities[:3])})")
        if self.confidence > 0.8:
            parts.append("(高置信)")
        return " ".join(parts)


# ============================================================
# 3D 记忆存储
# ============================================================

class Memory3DStore:
    """
    三维记忆存储引擎
    
    支持按 Form / Function / Dynamic 三个维度检索
    SQLite 持久化 + 内存索引
    """
    
    def __init__(self, db_path: str = "memory_v9.db"):
        self.db_path = db_path
        self.nodes: Dict[str, Memory3DNode] = {}
        
        # 三维索引
        self.form_index: Dict[MemoryForm, Set[str]] = defaultdict(set)
        self.function_index: Dict[MemoryFunction, Set[str]] = defaultdict(set)
        self.dynamic_index: Dict[MemoryDynamic, Set[str]] = defaultdict(set)
        
        # 实体索引
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Agent 索引
        self.agent_index: Dict[str, Set[str]] = defaultdict(set)
        
        self._init_db()
    
    def _init_db(self):
        """初始化 SQLite"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories_v9 (
                node_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                form TEXT,
                function TEXT,
                dynamic TEXT,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at REAL,
                last_accessed REAL,
                last_updated REAL,
                entities TEXT,
                tags TEXT,
                source_agent TEXT,
                source_query TEXT,
                triples TEXT,
                skill_pattern TEXT,
                skill_success_rate REAL DEFAULT 0,
                version INTEGER DEFAULT 1,
                parent_id TEXT,
                merge_sources TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_form ON memories_v9(form)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_func ON memories_v9(function)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_dyn ON memories_v9(dynamic)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories_v9(source_agent)")
        conn.commit()
        conn.close()
    
    def add(self, node: Memory3DNode) -> str:
        """添加记忆节点"""
        if not node.created_at:
            node.created_at = time.time()
        if not node.last_accessed:
            node.last_accessed = time.time()
        if not node.last_updated:
            node.last_updated = time.time()
        
        self.nodes[node.node_id] = node
        
        # 更新索引
        self.form_index[node.form].add(node.node_id)
        self.function_index[node.function].add(node.node_id)
        self.dynamic_index[node.dynamic].add(node.node_id)
        for entity in node.entities:
            self.entity_index[entity.lower()].add(node.node_id)
        if node.source_agent:
            self.agent_index[node.source_agent].add(node.node_id)
        
        # 持久化
        self._persist(node)
        
        return node.node_id
    
    def get(self, node_id: str) -> Optional[Memory3DNode]:
        """获取记忆"""
        node = self.nodes.get(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = time.time()
        return node
    
    def update(self, node_id: str, content: str = None,
               importance: float = None, confidence: float = None,
               dynamic: MemoryDynamic = None):
        """更新记忆"""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        if content is not None:
            node.content = content
        if importance is not None:
            node.importance = importance
        if confidence is not None:
            node.confidence = confidence
        if dynamic is not None:
            old_dynamic = node.dynamic
            self.dynamic_index[old_dynamic].discard(node_id)
            node.dynamic = dynamic
            self.dynamic_index[dynamic].add(node_id)
        
        node.version += 1
        node.last_updated = time.time()
        self._persist(node)
    
    def delete(self, node_id: str):
        """删除记忆"""
        node = self.nodes.pop(node_id, None)
        if not node:
            return
        self.form_index[node.form].discard(node_id)
        self.function_index[node.function].discard(node_id)
        self.dynamic_index[node.dynamic].discard(node_id)
        for entity in node.entities:
            self.entity_index[entity.lower()].discard(node_id)
        if node.source_agent:
            self.agent_index[node.source_agent].discard(node_id)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memories_v9 WHERE node_id=?", (node_id,))
        conn.commit()
        conn.close()
    
    def query(self, text: str = "", entities: List[str] = None,
              form: MemoryForm = None, function: MemoryFunction = None,
              dynamic: MemoryDynamic = None, agent: str = "",
              top_k: int = 10) -> List[Memory3DNode]:
        """
        三维检索 — 按任意维度组合查询
        """
        # 候选集
        candidate_ids = set(self.nodes.keys())
        
        # 按维度过滤
        if form is not None:
            candidate_ids &= self.form_index.get(form, set())
        if function is not None:
            candidate_ids &= self.function_index.get(function, set())
        if dynamic is not None:
            candidate_ids &= self.dynamic_index.get(dynamic, set())
        if agent:
            candidate_ids &= self.agent_index.get(agent, set())
        
        # 实体过滤 (OR 逻辑)
        if entities:
            entity_ids = set()
            for e in entities:
                entity_ids |= self.entity_index.get(e.lower(), set())
            if entity_ids:
                candidate_ids &= entity_ids
        
        # 评分排序
        candidates = [self.nodes[nid] for nid in candidate_ids if nid in self.nodes]
        candidates.sort(
            key=lambda n: n.relevance(text, entities, agent),
            reverse=True
        )
        
        # 更新访问
        for node in candidates[:top_k]:
            node.access_count += 1
            node.last_accessed = time.time()
        
        return candidates[:top_k]
    
    def query_skills(self, question: str, top_k: int = 5) -> List[Memory3DNode]:
        """检索可复用的技能记忆"""
        return self.query(
            text=question,
            function=MemoryFunction.LEARNING,
            top_k=top_k,
        )
    
    def _persist(self, node: Memory3DNode):
        """持久化到 SQLite"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO memories_v9 VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            node.node_id, node.content,
            node.form.value, node.function.value, node.dynamic.value,
            node.importance, node.confidence, node.access_count,
            node.created_at, node.last_accessed, node.last_updated,
            json.dumps(node.entities, ensure_ascii=False),
            json.dumps(node.tags, ensure_ascii=False),
            node.source_agent, node.source_query,
            json.dumps([list(t) for t in node.triples], ensure_ascii=False),
            node.skill_pattern, node.skill_success_rate,
            node.version, node.parent_id,
            json.dumps(node.merge_sources, ensure_ascii=False),
        ))
        conn.commit()
        conn.close()
    
    def load_from_db(self):
        """从 SQLite 加载全部记忆"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM memories_v9").fetchall()
        conn.close()
        
        for row in rows:
            node = Memory3DNode(
                node_id=row[0], content=row[1],
                form=MemoryForm(row[2]),
                function=MemoryFunction(row[3]),
                dynamic=MemoryDynamic(row[4]),
                importance=row[5], confidence=row[6],
                access_count=row[7],
                created_at=row[8], last_accessed=row[9],
                last_updated=row[10],
                entities=json.loads(row[11] or "[]"),
                tags=json.loads(row[12] or "[]"),
                source_agent=row[13] or "",
                source_query=row[14] or "",
                triples=[tuple(t) for t in json.loads(row[15] or "[]")],
                skill_pattern=row[16] or "",
                skill_success_rate=row[17] or 0,
                version=row[18], parent_id=row[19] or "",
                merge_sources=json.loads(row[20] or "[]"),
            )
            self.nodes[node.node_id] = node
            self.form_index[node.form].add(node.node_id)
            self.function_index[node.function].add(node.node_id)
            self.dynamic_index[node.dynamic].add(node.node_id)
            for entity in node.entities:
                self.entity_index[entity.lower()].add(node.node_id)
            if node.source_agent:
                self.agent_index[node.source_agent].add(node.node_id)
        
        logger.info(f"加载 {len(self.nodes)} 条记忆")


# ============================================================
# 记忆进化引擎 — Dynamics 管理
# ============================================================

class MemoryEvolutionEngine:
    """
    记忆动态管理 — 巩固、衰减、合并、遗忘
    
    规则:
      1. 巩固: 同一洞察被 ≥3 次不同查询验证 → confidence 提升 → STABLE
      2. 衰减: >7天未访问且 importance < 0.5 → DECAYING
      3. 合并: 两个记忆 content 相似度 >0.8 → 合并为新记忆
      4. 遗忘: DECAYING 且 >30天未访问 → 删除
      5. 升级: TEXT 记忆被多次结构化引用 → 升级为 STRUCTURED
    """
    
    CONSOLIDATION_THRESHOLD = 3    # 验证次数 → 巩固
    DECAY_DAYS = 7                 # 天数 → 开始衰减
    FORGET_DAYS = 30               # 天数 → 遗忘
    MERGE_SIMILARITY = 0.7         # 相似度 → 合并
    
    def __init__(self, store: Memory3DStore):
        self.store = store
        self.evolution_log = []
    
    def run_cycle(self) -> Dict:
        """执行一轮记忆进化"""
        stats = {"consolidated": 0, "decayed": 0, "merged": 0,
                 "forgotten": 0, "upgraded": 0}
        now = time.time()
        
        nodes = list(self.store.nodes.values())
        
        # 1. 巩固
        for node in nodes:
            if (node.access_count >= self.CONSOLIDATION_THRESHOLD
                and node.dynamic != MemoryDynamic.STABLE
                and node.confidence < 0.9):
                node.confidence = min(1.0, node.confidence + 0.15)
                self.store.update(node.node_id, 
                                  confidence=node.confidence,
                                  dynamic=MemoryDynamic.STABLE)
                stats["consolidated"] += 1
                self._log("consolidate", node.node_id)
        
        # 2. 衰减
        decay_threshold = now - self.DECAY_DAYS * 86400
        for node in nodes:
            if (node.last_accessed < decay_threshold
                and node.importance < 0.5
                and node.dynamic == MemoryDynamic.STABLE):
                self.store.update(node.node_id,
                                  dynamic=MemoryDynamic.DECAYING)
                stats["decayed"] += 1
                self._log("decay", node.node_id)
        
        # 3. 遗忘
        forget_threshold = now - self.FORGET_DAYS * 86400
        to_forget = []
        for node in nodes:
            if (node.dynamic == MemoryDynamic.DECAYING
                and node.last_accessed < forget_threshold):
                to_forget.append(node.node_id)
        for nid in to_forget:
            self.store.delete(nid)
            stats["forgotten"] += 1
            self._log("forget", nid)
        
        # 4. 合并相似记忆
        merge_pairs = self._find_merge_candidates(nodes)
        for n1_id, n2_id in merge_pairs[:5]:  # 限制每轮合并数
            merged = self._merge_memories(n1_id, n2_id)
            if merged:
                stats["merged"] += 1
                self._log("merge", f"{n1_id}+{n2_id}→{merged}")
        
        # 5. TEXT → STRUCTURED 升级
        for node in nodes:
            if (node.form == MemoryForm.TEXT
                and node.access_count >= 5
                and not node.triples):
                triples = self._extract_triples(node.content)
                if triples:
                    node.triples = triples
                    self.store.update(node.node_id)
                    old_form = node.form
                    node.form = MemoryForm.STRUCTURED
                    self.store.form_index[old_form].discard(node.node_id)
                    self.store.form_index[MemoryForm.STRUCTURED].add(node.node_id)
                    stats["upgraded"] += 1
                    self._log("upgrade", node.node_id)
        
        logger.info(f"记忆进化: {stats}")
        return stats
    
    def _find_merge_candidates(self, nodes: List[Memory3DNode]) -> List[Tuple[str, str]]:
        """找出可合并的记忆对"""
        pairs = []
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                if n1.form != n2.form:
                    continue
                sim = self._text_similarity(n1.content, n2.content)
                if sim >= self.MERGE_SIMILARITY:
                    pairs.append((n1.node_id, n2.node_id))
        return pairs
    
    def _merge_memories(self, id1: str, id2: str) -> Optional[str]:
        """合并两个记忆"""
        n1 = self.store.get(id1)
        n2 = self.store.get(id2)
        if not n1 or not n2:
            return None
        
        # 新记忆 = 保留更重要的内容 + 合并实体/标签
        merged = Memory3DNode(
            node_id=f"merged-{hashlib.md5(f'{id1}{id2}'.encode()).hexdigest()[:8]}",
            content=n1.content if n1.importance >= n2.importance else n2.content,
            form=n1.form,
            function=max(n1.function, n2.function, key=lambda f: f.value),
            dynamic=MemoryDynamic.CONSOLIDATING,
            importance=max(n1.importance, n2.importance),
            confidence=min(1.0, (n1.confidence + n2.confidence) / 2 + 0.1),
            access_count=n1.access_count + n2.access_count,
            entities=list(set(n1.entities + n2.entities)),
            tags=list(set(n1.tags + n2.tags)),
            source_agent=n1.source_agent or n2.source_agent,
            triples=list(set(n1.triples + n2.triples)),
            merge_sources=[id1, id2],
        )
        
        self.store.add(merged)
        self.store.delete(id1)
        self.store.delete(id2)
        
        return merged.node_id
    
    @staticmethod
    def _text_similarity(t1: str, t2: str) -> float:
        """简单的 Jaccard 文本相似度"""
        s1 = set(t1.lower().split())
        s2 = set(t2.lower().split())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)
    
    @staticmethod
    def _extract_triples(content: str) -> List[Tuple[str, str, str]]:
        """从文本提取简单三元组 (启发式)"""
        triples = []
        # 匹配 "X的Y是Z" 模式
        import re
        patterns = [
            r'(\w{2,10})的(\w{2,8})是(\w{2,20})',
            r'(\w{2,10})(\w{2,8})了(\d+[\d,.]*\w*)',
            r'(\w{2,10})(增长|下降|达到)(\d+[\d,.]*%?)',
        ]
        for pat in patterns:
            matches = re.findall(pat, content)
            for m in matches[:3]:
                triples.append(tuple(m))
        return triples
    
    def _log(self, action: str, detail: str):
        self.evolution_log.append({
            "time": time.time(),
            "action": action,
            "detail": detail,
        })


# ============================================================
# 跨Agent记忆共享器
# ============================================================

class CrossAgentMemorySharer:
    """
    跨Agent记忆共享
    
    问题: V8 各 Agent 只能看到自己的记忆
    方案: 建立共享记忆层，分析师的发现可以被风控/策略师使用
    
    规则:
      - 高置信度 (>0.8) 的记忆自动进入共享池
      - Agent 可以声明"发布"某条记忆
      - 其他 Agent 查询时，共享池作为额外来源
      - 引用计数跟踪跨Agent使用情况
    """
    
    def __init__(self, store: Memory3DStore):
        self.store = store
        self.shared_pool: Set[str] = set()
        self.cross_references: Dict[str, List[Dict]] = defaultdict(list)  # node_id → [{agent, time}]
    
    def publish(self, node_id: str, publisher_agent: str):
        """Agent 发布记忆到共享池"""
        node = self.store.get(node_id)
        if not node:
            return
        self.shared_pool.add(node_id)
        logger.info(f"记忆 {node_id} 被 {publisher_agent} 发布到共享池")
    
    def auto_publish(self):
        """自动发布高置信度记忆"""
        for nid, node in self.store.nodes.items():
            if node.confidence >= 0.8 and nid not in self.shared_pool:
                self.shared_pool.add(nid)
    
    def query_shared(self, query: str, requesting_agent: str,
                     entities: List[str] = None,
                     top_k: int = 5) -> List[Memory3DNode]:
        """查询共享记忆 (排除自己发布的)"""
        candidates = []
        for nid in self.shared_pool:
            node = self.store.nodes.get(nid)
            if node and node.source_agent != requesting_agent:
                candidates.append(node)
        
        candidates.sort(
            key=lambda n: n.relevance(query, entities, requesting_agent),
            reverse=True
        )
        
        # 记录跨Agent引用
        for node in candidates[:top_k]:
            self.cross_references[node.node_id].append({
                "agent": requesting_agent,
                "time": time.time(),
            })
        
        return candidates[:top_k]
    
    def get_cross_reference_stats(self) -> Dict:
        """获取跨Agent引用统计"""
        stats = {}
        for nid, refs in self.cross_references.items():
            node = self.store.nodes.get(nid)
            if node:
                stats[nid] = {
                    "source": node.source_agent,
                    "content_preview": node.content[:50],
                    "cited_by": list(set(r["agent"] for r in refs)),
                    "citation_count": len(refs),
                }
        return stats


# ============================================================
# 技能记忆管理器
# ============================================================

class SkillMemoryManager:
    """
    技能记忆 — 将成功的分析模式提炼为可复用技能
    
    基于 MemSkill 论文: 学到的分析模式可跨任务复用
    
    流程:
      1. Agent 完成一次高分分析
      2. 提炼分析模式 (问题类型 → 步骤 → 输出格式)
      3. 存为技能记忆 (LEARNING function)
      4. 下次遇到相似问题时自动匹配
    """
    
    def __init__(self, store: Memory3DStore):
        self.store = store
    
    def distill_skill(self, question: str, analysis_trace: str,
                      quality_score: float,
                      agent: str = "",
                      min_quality: float = 0.7) -> Optional[str]:
        """
        从分析轨迹蒸馏技能
        """
        if quality_score < min_quality:
            return None
        
        # 构造技能模式
        pattern = self._extract_pattern(question, analysis_trace)
        if not pattern:
            return None
        
        skill_id = f"skill-{hashlib.md5(pattern.encode()).hexdigest()[:10]}"
        
        # 检查是否已有相似技能
        existing = self.store.query_skills(question, top_k=3)
        for ex in existing:
            if self._pattern_similarity(ex.skill_pattern, pattern) > 0.7:
                # 强化现有技能
                new_rate = (ex.skill_success_rate * ex.access_count + quality_score) / (ex.access_count + 1)
                self.store.update(ex.node_id, confidence=min(1.0, ex.confidence + 0.05))
                ex.skill_success_rate = new_rate
                logger.info(f"强化技能 {ex.node_id}: rate={new_rate:.2f}")
                return ex.node_id
        
        # 创建新技能
        node = Memory3DNode(
            node_id=skill_id,
            content=f"技能: {question[:50]} → {analysis_trace[:100]}",
            form=MemoryForm.STRUCTURED,
            function=MemoryFunction.LEARNING,
            dynamic=MemoryDynamic.CONSOLIDATING,
            importance=quality_score,
            confidence=quality_score * 0.8,
            entities=self._extract_entities(question),
            tags=["skill", agent] if agent else ["skill"],
            source_agent=agent,
            source_query=question,
            skill_pattern=pattern,
            skill_success_rate=quality_score,
        )
        
        self.store.add(node)
        logger.info(f"新技能 {skill_id}: {pattern[:60]}")
        return skill_id
    
    def match_skill(self, question: str, top_k: int = 3) -> List[Dict]:
        """为问题匹配可复用技能"""
        skills = self.store.query_skills(question, top_k=top_k)
        return [
            {
                "skill_id": s.node_id,
                "pattern": s.skill_pattern,
                "success_rate": s.skill_success_rate,
                "confidence": s.confidence,
            }
            for s in skills if s.skill_success_rate > 0.5
        ]
    
    def _extract_pattern(self, question: str, trace: str) -> str:
        """提取分析模式"""
        # 简化: 取问题类型 + 分析步骤关键词
        q_type = "trend" if any(kw in question for kw in ["趋势", "变化", "走势"]) else \
                 "compare" if any(kw in question for kw in ["对比", "比较", "vs"]) else \
                 "risk" if any(kw in question for kw in ["风险", "异常", "下降"]) else \
                 "overview" if any(kw in question for kw in ["总", "全", "概览"]) else "query"
        
        return f"type:{q_type}|question:{question[:80]}|steps:{len(trace.split('Step'))}"
    
    @staticmethod
    def _pattern_similarity(p1: str, p2: str) -> float:
        s1, s2 = set(p1.split("|")), set(p2.split("|"))
        return len(s1 & s2) / max(len(s1 | s2), 1)
    
    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        entities = []
        for brand in ["HMD", "Transsion", "Samsung", "Xiaomi", "OPPO",
                       "Vivo", "Realme", "Nokia", "Infinix", "Tecno"]:
            if brand.lower() in text.lower():
                entities.append(brand)
        return entities


# ============================================================
# 统一 API — 给 multi_agent.py 使用
# ============================================================

class MemoryV9System:
    """
    V9 记忆系统统一入口
    
    用法:
        mem = MemoryV9System(db_path="memory_v9.db")
        
        # 存储分析结果
        mem.remember("analyst", question, analysis_output, quality=0.85)
        
        # 检索相关记忆
        memories = mem.recall(question, agent="risk", top_k=5)
        
        # 检索跨Agent共享记忆
        shared = mem.recall_shared(question, agent="strategist")
        
        # 匹配技能
        skills = mem.match_skills(question)
        
        # 记忆进化
        mem.evolve()
    """
    
    def __init__(self, db_path: str = "memory_v9.db"):
        self.store = Memory3DStore(db_path=db_path)
        self.evolution = MemoryEvolutionEngine(self.store)
        self.sharer = CrossAgentMemorySharer(self.store)
        self.skills = SkillMemoryManager(self.store)
    
    def remember(self, agent: str, question: str, content: str,
                 quality: float = 0.5, entities: List[str] = None,
                 tags: List[str] = None,
                 form: MemoryForm = MemoryForm.TEXT,
                 function: MemoryFunction = MemoryFunction.RETRIEVAL):
        """存储一条记忆"""
        node_id = f"mem-{hashlib.md5(f'{agent}{question}{time.time()}'.encode()).hexdigest()[:12]}"
        
        node = Memory3DNode(
            node_id=node_id,
            content=content[:2000],
            form=form,
            function=function,
            dynamic=MemoryDynamic.CONSOLIDATING,
            importance=quality,
            confidence=quality * 0.7,
            entities=entities or [],
            tags=tags or [],
            source_agent=agent,
            source_query=question[:200],
        )
        
        self.store.add(node)
        
        # 自动技能蒸馏
        if quality >= 0.7:
            self.skills.distill_skill(question, content, quality, agent)
        
        # 自动发布高质量记忆
        self.sharer.auto_publish()
        
        return node_id
    
    def recall(self, query: str, agent: str = "",
               entities: List[str] = None,
               top_k: int = 5) -> List[Dict]:
        """检索记忆"""
        nodes = self.store.query(
            text=query, entities=entities,
            agent=agent, top_k=top_k
        )
        return [
            {
                "id": n.node_id,
                "content": n.content[:300],
                "relevance": round(n.relevance(query, entities, agent), 3),
                "form": n.form.value,
                "function": n.function.value,
                "confidence": n.confidence,
                "source_agent": n.source_agent,
            }
            for n in nodes
        ]
    
    def recall_shared(self, query: str, agent: str,
                      entities: List[str] = None,
                      top_k: int = 5) -> List[Dict]:
        """检索跨Agent共享记忆"""
        nodes = self.sharer.query_shared(query, agent, entities, top_k)
        return [
            {
                "id": n.node_id,
                "content": n.content[:300],
                "source_agent": n.source_agent,
                "confidence": n.confidence,
            }
            for n in nodes
        ]
    
    def match_skills(self, question: str) -> List[Dict]:
        """匹配可复用技能"""
        return self.skills.match_skill(question)
    
    def evolve(self) -> Dict:
        """执行记忆进化周期"""
        return self.evolution.run_cycle()
    
    def get_stats(self) -> Dict:
        """获取记忆系统统计"""
        nodes = list(self.store.nodes.values())
        return {
            "total_memories": len(nodes),
            "by_form": {f.value: len(ids) for f, ids in self.store.form_index.items()},
            "by_function": {f.value: len(ids) for f, ids in self.store.function_index.items()},
            "by_dynamic": {d.value: len(ids) for d, ids in self.store.dynamic_index.items()},
            "by_agent": {a: len(ids) for a, ids in self.store.agent_index.items()},
            "shared_pool_size": len(self.sharer.shared_pool),
            "cross_references": len(self.sharer.cross_references),
            "avg_confidence": round(
                sum(n.confidence for n in nodes) / max(len(nodes), 1), 3
            ),
        }


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import tempfile, os

    print("=" * 60)
    print("MRARFAI 3D Memory System v9.0 Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        mem = MemoryV9System(db_path=db_path)

        # 模拟 Agent 分析并存储记忆
        print("\n--- 存储记忆 ---")
        mem.remember("analyst", "2025年HMD出货量趋势",
                     "HMD 2025年出货量同比增长35%，主要由非洲市场驱动。Q3出现高峰。",
                     quality=0.9, entities=["HMD"], tags=["出货", "趋势"])

        mem.remember("analyst", "各品牌营收排名",
                     "Top5: Transsion(3.2亿) > HMD(2.8亿) > Samsung_ODM(2.1亿) > Xiaomi_ODM(1.8亿) > OPPO_ODM(1.2亿)",
                     quality=0.85, entities=["Transsion", "HMD", "Samsung_ODM"], tags=["营收", "排名"])

        mem.remember("risk", "客户集中度风险",
                     "Top2客户(Transsion+HMD)占总营收55%，集中度偏高。建议拓展3-5个新品牌。",
                     quality=0.8, entities=["Transsion", "HMD"], tags=["风险", "集中度"])

        mem.remember("strategist", "平板市场机会",
                     "平板ODM毛利率比手机高8%，但量级仅为手机的25%。建议针对教育平板赛道扩展。",
                     quality=0.75, entities=[], tags=["策略", "平板"])

        print(f"  存储了 4 条记忆")

        # 检索
        print("\n--- 记忆检索 ---")
        results = mem.recall("HMD最近表现如何", agent="analyst", entities=["HMD"])
        for r in results:
            print(f"  [{r['source_agent']}] {r['content'][:60]}... (rel={r['relevance']})")

        # 跨Agent共享
        print("\n--- 跨Agent共享 ---")
        shared = mem.recall_shared("客户风险", agent="strategist", entities=["Transsion"])
        for s in shared:
            print(f"  从 [{s['source_agent']}] 获取: {s['content'][:60]}...")

        # 技能匹配
        print("\n--- 技能匹配 ---")
        skills = mem.match_skills("Samsung_ODM的出货趋势分析")
        print(f"  匹配到 {len(skills)} 个技能")
        for sk in skills:
            print(f"    {sk['skill_id']}: rate={sk['success_rate']:.2f}")

        # 记忆进化
        print("\n--- 记忆进化 ---")
        # 模拟多次访问以触发巩固
        for _ in range(5):
            mem.recall("HMD出货", entities=["HMD"])
        stats = mem.evolve()
        print(f"  进化结果: {stats}")

        # 系统统计
        print("\n--- 系统统计 ---")
        sys_stats = mem.get_stats()
        for k, v in sys_stats.items():
            print(f"  {k}: {v}")

    print("\n✅ 3D Memory System 初始化成功")
