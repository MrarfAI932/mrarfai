#!/usr/bin/env python3
"""
MRARFAI v3.3 — Persistent Agent Memory
========================================
跨会话持久化记忆系统

三层记忆架构:
  1. Session Memory (会话记忆): 当前对话的短期记忆（已有AgentMemory）
  2. Entity Memory (实体记忆): 客户/产品/区域的历史画像，跨会话积累
  3. Insight Memory (洞察记忆): 过去分析的核心结论、用户偏好

存储: SQLite WAL模式，与observability.db共享连接策略
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================
# 数据模型
# ============================================================

@dataclass
class EntityProfile:
    """实体画像 — 某个客户/产品/区域的累计知识"""
    entity_type: str         # customer / product / region
    entity_name: str
    mentions: int = 0
    last_mentioned: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)  # 动态属性
    history: List[dict] = field(default_factory=list)  # [{date, context, insight}]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.entity_type,
            "name": self.entity_name,
            "mentions": self.mentions,
            "last_mentioned": self.last_mentioned,
            "attributes": self.attributes,
            "history": self.history[-5:],  # 只返回最近5条
            "tags": self.tags,
        }

    def to_prompt(self) -> str:
        """生成可嵌入prompt的实体上下文"""
        lines = [f"[实体记忆] {self.entity_name} ({self.entity_type})"]
        if self.attributes:
            for k, v in list(self.attributes.items())[:5]:
                lines.append(f"  - {k}: {v}")
        if self.tags:
            lines.append(f"  标签: {', '.join(self.tags)}")
        if self.history:
            latest = self.history[-1]
            lines.append(f"  上次分析({latest.get('date', '?')}): {latest.get('insight', '')[:80]}")
        return "\n".join(lines)


@dataclass
class InsightRecord:
    """洞察记录 — 过去分析的核心结论"""
    insight_id: str
    question: str
    answer_summary: str
    agents_used: List[str]
    key_findings: List[str]
    entities_involved: List[str]
    timestamp: str
    quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.insight_id,
            "question": self.question,
            "summary": self.answer_summary[:200],
            "agents": self.agents_used,
            "findings": self.key_findings,
            "entities": self.entities_involved,
            "time": self.timestamp,
            "quality": self.quality_score,
        }


@dataclass
class UserPreferences:
    """用户偏好 — 从历史交互中学习"""
    preferred_metrics: List[str] = field(default_factory=list)  # 常关注的指标
    preferred_agents: List[str] = field(default_factory=list)   # 常用的Agent
    focus_entities: List[str] = field(default_factory=list)     # 重点关注的实体
    language_style: str = "formal"  # formal / casual
    detail_level: str = "medium"    # brief / medium / detailed
    interaction_count: int = 0

    def to_dict(self) -> dict:
        return {
            "metrics": self.preferred_metrics,
            "agents": self.preferred_agents,
            "focus_entities": self.focus_entities,
            "style": self.language_style,
            "detail": self.detail_level,
            "interactions": self.interaction_count,
        }

    def to_prompt(self) -> str:
        parts = []
        if self.focus_entities:
            parts.append(f"用户重点关注: {', '.join(self.focus_entities[:5])}")
        if self.preferred_metrics:
            parts.append(f"常看指标: {', '.join(self.preferred_metrics[:5])}")
        if self.detail_level == "brief":
            parts.append("用户偏好简洁回答")
        elif self.detail_level == "detailed":
            parts.append("用户偏好详细分析")
        return "; ".join(parts) if parts else ""


# ============================================================
# PersistentMemoryStore — SQLite持久化
# ============================================================

class PersistentMemoryStore:
    """
    跨会话持久化记忆存储
    
    表结构:
    - entities: 实体画像
    - insights: 历史洞察
    - preferences: 用户偏好
    - entity_history: 实体事件日志
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_type TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    mentions INTEGER DEFAULT 0,
                    last_mentioned TEXT,
                    attributes TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '[]',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (entity_type, entity_name)
                );

                CREATE TABLE IF NOT EXISTS entity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    context TEXT,
                    insight TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer_summary TEXT,
                    agents_used TEXT DEFAULT '[]',
                    key_findings TEXT DEFAULT '[]',
                    entities_involved TEXT DEFAULT '[]',
                    quality_score REAL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_entity_history_entity
                    ON entity_history(entity_type, entity_name);
                CREATE INDEX IF NOT EXISTS idx_insights_time
                    ON insights(created_at DESC);
            """)
            conn.commit()
        finally:
            conn.close()

    # ---- Entity Memory ----

    def upsert_entity(self, entity_type: str, entity_name: str,
                      attributes: Dict = None, tags: List[str] = None) -> EntityProfile:
        """创建或更新实体画像"""
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            row = conn.execute(
                "SELECT * FROM entities WHERE entity_type=? AND entity_name=?",
                (entity_type, entity_name)
            ).fetchone()

            if row:
                # 合并attributes
                existing_attrs = json.loads(row["attributes"] or "{}")
                if attributes:
                    existing_attrs.update(attributes)
                existing_tags = json.loads(row["tags"] or "[]")
                if tags:
                    existing_tags = list(set(existing_tags + tags))

                conn.execute("""
                    UPDATE entities SET 
                        mentions = mentions + 1,
                        last_mentioned = ?,
                        attributes = ?,
                        tags = ?,
                        updated_at = ?
                    WHERE entity_type = ? AND entity_name = ?
                """, (now, json.dumps(existing_attrs, ensure_ascii=False),
                      json.dumps(existing_tags, ensure_ascii=False),
                      now, entity_type, entity_name))
                conn.commit()

                return EntityProfile(
                    entity_type=entity_type,
                    entity_name=entity_name,
                    mentions=row["mentions"] + 1,
                    last_mentioned=now,
                    attributes=existing_attrs,
                    tags=existing_tags,
                )
            else:
                attrs = attributes or {}
                tgs = tags or []
                conn.execute("""
                    INSERT INTO entities (entity_type, entity_name, mentions,
                        last_mentioned, attributes, tags)
                    VALUES (?, ?, 1, ?, ?, ?)
                """, (entity_type, entity_name, now,
                      json.dumps(attrs, ensure_ascii=False),
                      json.dumps(tgs, ensure_ascii=False)))
                conn.commit()

                return EntityProfile(
                    entity_type=entity_type,
                    entity_name=entity_name,
                    mentions=1,
                    last_mentioned=now,
                    attributes=attrs,
                    tags=tgs,
                )
        finally:
            conn.close()

    def get_entity(self, entity_type: str, entity_name: str) -> Optional[EntityProfile]:
        """获取实体画像"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM entities WHERE entity_type=? AND entity_name=?",
                (entity_type, entity_name)
            ).fetchone()

            if not row:
                return None

            # 加载历史
            history = conn.execute(
                "SELECT event_date, context, insight FROM entity_history "
                "WHERE entity_type=? AND entity_name=? ORDER BY created_at DESC LIMIT 10",
                (entity_type, entity_name)
            ).fetchall()

            return EntityProfile(
                entity_type=entity_type,
                entity_name=entity_name,
                mentions=row["mentions"],
                last_mentioned=row["last_mentioned"] or "",
                attributes=json.loads(row["attributes"] or "{}"),
                history=[dict(h) for h in history],
                tags=json.loads(row["tags"] or "[]"),
            )
        finally:
            conn.close()

    def add_entity_event(self, entity_type: str, entity_name: str,
                         context: str, insight: str):
        """记录实体事件"""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO entity_history (entity_type, entity_name, event_date, context, insight)
                VALUES (?, ?, ?, ?, ?)
            """, (entity_type, entity_name, datetime.now().strftime("%Y-%m-%d"),
                  context[:200], insight[:500]))
            conn.commit()
        finally:
            conn.close()

    def get_relevant_entities(self, text: str, limit: int = 5) -> List[EntityProfile]:
        """根据文本查找相关实体"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM entities ORDER BY mentions DESC, last_mentioned DESC LIMIT 50"
            ).fetchall()

            matched = []
            for row in rows:
                if row["entity_name"] in text:
                    profile = EntityProfile(
                        entity_type=row["entity_type"],
                        entity_name=row["entity_name"],
                        mentions=row["mentions"],
                        last_mentioned=row["last_mentioned"] or "",
                        attributes=json.loads(row["attributes"] or "{}"),
                        tags=json.loads(row["tags"] or "[]"),
                    )
                    matched.append(profile)
                    if len(matched) >= limit:
                        break

            return matched
        finally:
            conn.close()

    def get_top_entities(self, entity_type: str = None, limit: int = 10) -> List[EntityProfile]:
        """获取最常提及的实体"""
        conn = self._get_conn()
        try:
            if entity_type:
                rows = conn.execute(
                    "SELECT * FROM entities WHERE entity_type=? "
                    "ORDER BY mentions DESC LIMIT ?",
                    (entity_type, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entities ORDER BY mentions DESC LIMIT ?",
                    (limit,)
                ).fetchall()

            return [
                EntityProfile(
                    entity_type=r["entity_type"],
                    entity_name=r["entity_name"],
                    mentions=r["mentions"],
                    last_mentioned=r["last_mentioned"] or "",
                    attributes=json.loads(r["attributes"] or "{}"),
                    tags=json.loads(r["tags"] or "[]"),
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ---- Insight Memory ----

    def save_insight(self, insight: InsightRecord):
        """保存分析洞察"""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO insights
                    (insight_id, question, answer_summary, agents_used,
                     key_findings, entities_involved, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                insight.insight_id,
                insight.question,
                insight.answer_summary,
                json.dumps(insight.agents_used, ensure_ascii=False),
                json.dumps(insight.key_findings, ensure_ascii=False),
                json.dumps(insight.entities_involved, ensure_ascii=False),
                insight.quality_score,
            ))
            conn.commit()
        finally:
            conn.close()

    def get_related_insights(self, question: str, limit: int = 3) -> List[InsightRecord]:
        """获取与问题相关的历史洞察"""
        conn = self._get_conn()
        try:
            # 简单的关键词匹配（生产环境可用向量检索）
            rows = conn.execute(
                "SELECT * FROM insights ORDER BY created_at DESC LIMIT 50"
            ).fetchall()

            scored = []
            q_tokens = set(question)
            for row in rows:
                q_old = row["question"]
                overlap = len(set(q_old) & q_tokens) / max(len(q_tokens), 1)
                scored.append((overlap, row))

            scored.sort(key=lambda x: -x[0])
            
            results = []
            for score, row in scored[:limit]:
                if score < 0.1:
                    continue
                results.append(InsightRecord(
                    insight_id=row["insight_id"],
                    question=row["question"],
                    answer_summary=row["answer_summary"] or "",
                    agents_used=json.loads(row["agents_used"] or "[]"),
                    key_findings=json.loads(row["key_findings"] or "[]"),
                    entities_involved=json.loads(row["entities_involved"] or "[]"),
                    timestamp=row["created_at"],
                    quality_score=row["quality_score"] or 0,
                ))

            return results
        finally:
            conn.close()

    def get_recent_insights(self, limit: int = 10) -> List[InsightRecord]:
        """获取最近N条洞察"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM insights ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [
                InsightRecord(
                    insight_id=r["insight_id"],
                    question=r["question"],
                    answer_summary=r["answer_summary"] or "",
                    agents_used=json.loads(r["agents_used"] or "[]"),
                    key_findings=json.loads(r["key_findings"] or "[]"),
                    entities_involved=json.loads(r["entities_involved"] or "[]"),
                    timestamp=r["created_at"],
                    quality_score=r["quality_score"] or 0,
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ---- User Preferences ----

    def save_preference(self, key: str, value: Any):
        """保存用户偏好"""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO preferences (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
            """, (key, json.dumps(value, ensure_ascii=False)))
            conn.commit()
        finally:
            conn.close()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """获取用户偏好"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key=?", (key,)
            ).fetchone()
            if row:
                return json.loads(row["value"])
            return default
        finally:
            conn.close()

    def load_preferences(self) -> UserPreferences:
        """加载完整用户偏好"""
        return UserPreferences(
            preferred_metrics=self.get_preference("preferred_metrics", []),
            preferred_agents=self.get_preference("preferred_agents", []),
            focus_entities=self.get_preference("focus_entities", []),
            language_style=self.get_preference("language_style", "formal"),
            detail_level=self.get_preference("detail_level", "medium"),
            interaction_count=self.get_preference("interaction_count", 0),
        )

    def update_preferences_from_interaction(self, question: str,
                                             agents_used: List[str],
                                             entities: List[str]):
        """从交互中自动学习用户偏好"""
        # 更新交互计数
        count = self.get_preference("interaction_count", 0)
        self.save_preference("interaction_count", count + 1)

        # 更新常用Agent
        agent_freq = defaultdict(int, self.get_preference("agent_freq", {}))
        for a in agents_used:
            agent_freq[a] += 1
        self.save_preference("agent_freq", dict(agent_freq))
        top_agents = sorted(agent_freq.keys(), key=lambda x: -agent_freq[x])[:5]
        self.save_preference("preferred_agents", top_agents)

        # 更新关注实体
        entity_freq = defaultdict(int, self.get_preference("entity_freq", {}))
        for e in entities:
            entity_freq[e] += 1
        self.save_preference("entity_freq", dict(entity_freq))
        top_entities = sorted(entity_freq.keys(), key=lambda x: -entity_freq[x])[:10]
        self.save_preference("focus_entities", top_entities)

        # 更新关注指标（从问题中提取）
        metric_keywords = {
            "营收": "营收", "收入": "营收", "金额": "营收",
            "出货": "出货量", "数量": "出货量",
            "利润": "利润", "毛利": "利润",
            "增长": "增长率", "同比": "增长率", "环比": "增长率",
            "客户": "客户分布", "集中": "客户集中度",
            "产品": "产品结构", "结构": "产品结构",
            "风险": "风险", "流失": "流失风险",
        }
        metric_freq = defaultdict(int, self.get_preference("metric_freq", {}))
        for kw, metric in metric_keywords.items():
            if kw in question:
                metric_freq[metric] += 1
        self.save_preference("metric_freq", dict(metric_freq))
        top_metrics = sorted(metric_freq.keys(), key=lambda x: -metric_freq[x])[:5]
        self.save_preference("preferred_metrics", top_metrics)

    # ---- 上下文生成 ----

    def build_memory_context(self, question: str) -> str:
        """
        根据当前问题，构建持久化记忆上下文（嵌入prompt）
        
        输出格式:
        [持久化记忆]
        - 用户偏好: ...
        - 相关实体: ...
        - 历史洞察: ...
        """
        parts = []

        # 用户偏好
        prefs = self.load_preferences()
        pref_text = prefs.to_prompt()
        if pref_text:
            parts.append(pref_text)

        # 相关实体画像
        entities = self.get_relevant_entities(question, limit=3)
        for e in entities:
            parts.append(e.to_prompt())

        # 相关历史洞察
        insights = self.get_related_insights(question, limit=2)
        for ins in insights:
            parts.append(
                f"[历史分析] Q: {ins.question[:50]}\n"
                f"  发现: {'; '.join(ins.key_findings[:2]) if ins.key_findings else ins.answer_summary[:80]}"
            )

        if not parts:
            return ""

        return "[持久化记忆]\n" + "\n".join(parts)

    # ---- 清理 ----

    def cleanup(self, days: int = 180):
        """清理过期数据"""
        conn = self._get_conn()
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            conn.execute("DELETE FROM entity_history WHERE created_at < ?", (cutoff,))
            conn.execute("DELETE FROM insights WHERE created_at < ?", (cutoff,))
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """获取记忆系统统计"""
        conn = self._get_conn()
        try:
            entities = conn.execute("SELECT COUNT(*) as c FROM entities").fetchone()["c"]
            insights = conn.execute("SELECT COUNT(*) as c FROM insights").fetchone()["c"]
            events = conn.execute("SELECT COUNT(*) as c FROM entity_history").fetchone()["c"]
            prefs = conn.execute("SELECT COUNT(*) as c FROM preferences").fetchone()["c"]
            return {
                "entities": entities,
                "insights": insights,
                "entity_events": events,
                "preferences": prefs,
            }
        finally:
            conn.close()


# ============================================================
# 全局实例
# ============================================================

_persistent_memory: Optional[PersistentMemoryStore] = None


def get_persistent_memory(db_path: str = "memory.db") -> PersistentMemoryStore:
    """获取全局持久化记忆实例"""
    global _persistent_memory
    if _persistent_memory is None:
        _persistent_memory = PersistentMemoryStore(db_path)
    return _persistent_memory


def set_persistent_memory(store: PersistentMemoryStore):
    """设置全局持久化记忆实例"""
    global _persistent_memory
    _persistent_memory = store
