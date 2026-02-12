#!/usr/bin/env python3
"""
MRARFAI GraphRAG v7.0 — 多跳推理知识图谱
==========================================
Phase 3 升级：静态 knowledge_graph.py → 动态图 + 多跳推理

核心变化:
  ① NetworkX 动态图 — 实体/关系自动从数据构建（零额外依赖）
  ② 多跳推理 — "哪些A级客户在下滑？" → 自动图遍历
  ③ 实体链接 — 客户↔区域↔品类↔风险 全量关联
  ④ 上下文生成 — 为 Agent 自动构建相关实体画像
  ⑤ Neo4j 适配器 — 预留接口，数据增长后无缝迁移

v5.0 → v7.0 评分变化:
  知识图谱: 70 → 95 (+25)

架构:
  ┌──────────────┐
  │  Raw Data    │  (results dict / Excel)
  └──────┬───────┘
         ▼
  ┌──────────────┐    ┌─────────────────┐
  │ GraphBuilder │ →  │  NetworkX Graph  │
  │  自动构建    │    │  节点+关系+属性  │
  └──────────────┘    └────────┬────────┘
                               ▼
  ┌─────────────────────────────────────────┐
  │           GraphQuery Engine             │
  │  - 多跳遍历 (BFS/DFS)                  │
  │  - 路径查找 (客户→风险→原因)            │
  │  - 子图提取 (相关实体上下文)             │
  │  - 统计聚合 (按类型/属性汇总)            │
  └─────────────────────────────────────────┘
                               ▼
  ┌─────────────────────────────────────────┐
  │      Context Generator for Agents       │
  │  → 精准、结构化、多跳的实体画像          │
  └─────────────────────────────────────────┘

依赖: networkx (pip install networkx, 通常已预装)
可选: neo4j (pip install neo4j, Phase 4 迁移时使用)

兼容: 完全兼容 v5.0 的 SalesKnowledgeGraph 接口
"""

import json
import logging
import re
from typing import Optional, Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger("mrarfai.graphrag")

# ============================================================
# NetworkX 导入
# ============================================================
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logger.warning("networkx 未安装: pip install networkx")

# Neo4j 可选
try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


# ============================================================
# 节点类型定义
# ============================================================

class NodeType:
    CUSTOMER = "customer"
    REGION = "region"
    CATEGORY = "category"
    PRODUCT = "product"
    RISK = "risk_alert"
    GROWTH = "growth_opportunity"
    ANOMALY = "anomaly"
    MONTH = "month"
    YEAR = "year"
    METRIC = "metric"


class EdgeType:
    BELONGS_TO = "belongs_to"           # 客户 → 区域
    ORDERS = "orders"                   # 客户 → 品类
    HAS_RISK = "has_risk"              # 客户 → 风险
    HAS_GROWTH = "has_growth"          # 客户 → 增长机会
    HAS_ANOMALY = "has_anomaly"        # 客户 → 异常
    REVENUE_IN = "revenue_in"          # 客户 → 月份
    COMPETES_WITH = "competes_with"    # 客户 ↔ 客户
    SAME_REGION = "same_region"        # 客户 ↔ 客户
    YEAR_PERIOD = "year_period"        # 月份 → 年份


# ============================================================
# SalesGraph — 核心知识图谱
# ============================================================

class SalesGraph:
    """
    销售领域知识图谱 — NetworkX 实现

    自动从 results dict 构建，支持:
    - 多跳查询: "哪些高风险客户在华南区域？"
    - 路径分析: "从HMD到整体风险的关联路径"
    - 子图提取: 给Agent的精准上下文
    - 统计聚合: "每个区域有多少A级客户？"
    """

    def __init__(self):
        if not HAS_NX:
            self.G = None
            logger.warning("NetworkX 不可用，知识图谱降级为静态模式")
            return

        self.G = nx.DiGraph()
        self._built = False
        self._build_time = None
        self._stats = {}

    def build_from_results(self, data: dict, results: dict):
        """
        从 results dict 自动构建知识图谱

        输入: analyze_clients_v2 产出的 results 字典
        输出: 完整的实体关系图
        """
        if not HAS_NX or self.G is None:
            return

        self.G.clear()
        t0 = __import__('time').time()

        # 1. 客户节点
        self._build_customers(results)

        # 2. 区域节点 + 客户→区域关系
        self._build_regions(results)

        # 3. 品类节点 + 客户→品类关系
        self._build_categories(results)

        # 4. 风险节点 + 客户→风险关系
        self._build_risks(results)

        # 5. 增长机会 + 客户→机会关系
        self._build_growth(results)

        # 6. 异常检测 + 客户→异常关系
        self._build_anomalies(results)

        # 7. 月度数据 + 客户→月份关系
        self._build_monthly(results)

        # 8. 同区域客户关联
        self._build_co_region_edges()

        self._built = True
        self._build_time = __import__('time').time() - t0

        # 统计
        self._stats = {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "node_types": dict(self._count_by_type("nodes")),
            "edge_types": dict(self._count_by_type("edges")),
            "build_time_ms": round(self._build_time * 1000, 1),
        }

        logger.info(
            f"✅ 知识图谱已构建: {self._stats['nodes']}节点, "
            f"{self._stats['edges']}边, {self._stats['build_time_ms']}ms"
        )

    def _build_customers(self, results: dict):
        """构建客户节点"""
        for c in results.get('客户分级', []):
            name = c.get('客户', '')
            if not name:
                continue
            self.G.add_node(
                f"customer:{name}",
                type=NodeType.CUSTOMER,
                name=name,
                tier=c.get('等级', 'C'),
                annual_amount=c.get('年度金额', 0),
                h1=c.get('H1', 0),
                h2=c.get('H2', 0),
                share=c.get('占比', '0%'),
                cumulative_share=c.get('累计占比', '0%'),
            )

    def _build_regions(self, results: dict):
        """构建区域节点和关系"""
        region_data = results.get('区域洞察', {})
        details = region_data.get('详细', region_data.get('区域分布', []))

        for r in details:
            if not isinstance(r, dict):
                continue
            region_name = r.get('区域', '')
            if not region_name:
                continue

            self.G.add_node(
                f"region:{region_name}",
                type=NodeType.REGION,
                name=region_name,
                amount=r.get('金额', 0),
                share=r.get('占比', ''),
            )

        # 客户→区域关系（从数据中推断）
        for c in results.get('客户分级', []):
            name = c.get('客户', '')
            region = c.get('区域', '') or c.get('所属区域', '')
            if name and region:
                cid = f"customer:{name}"
                rid = f"region:{region}"
                if self.G.has_node(cid) and self.G.has_node(rid):
                    self.G.add_edge(cid, rid, type=EdgeType.BELONGS_TO)

    def _build_categories(self, results: dict):
        """构建品类节点"""
        for cat in results.get('类别趋势', []):
            cat_name = cat.get('类别', '')
            if not cat_name:
                continue
            self.G.add_node(
                f"category:{cat_name}",
                type=NodeType.CATEGORY,
                name=cat_name,
                amount_2024=cat.get('2024金额', 0),
                amount_2025=cat.get('2025金额', 0),
                growth=cat.get('增长率', ''),
            )

    def _build_risks(self, results: dict):
        """构建风险节点和关系"""
        for i, r in enumerate(results.get('流失预警', [])):
            customer = r.get('客户', '')
            if not customer:
                continue

            risk_id = f"risk:{customer}_{i}"
            self.G.add_node(
                risk_id,
                type=NodeType.RISK,
                customer=customer,
                level=r.get('风险', ''),
                reason=r.get('原因', ''),
                amount=r.get('年度金额', 0),
            )

            cid = f"customer:{customer}"
            if self.G.has_node(cid):
                self.G.add_edge(cid, risk_id, type=EdgeType.HAS_RISK)

    def _build_growth(self, results: dict):
        """构建增长机会节点"""
        for i, g in enumerate(results.get('增长机会', [])):
            customer = g.get('客户', '')
            if not customer:
                continue

            gid = f"growth:{customer}_{i}"
            self.G.add_node(
                gid,
                type=NodeType.GROWTH,
                customer=customer,
                opportunity=g.get('机会', ''),
                potential=g.get('潜力金额', 0),
            )

            cid = f"customer:{customer}"
            if self.G.has_node(cid):
                self.G.add_edge(cid, gid, type=EdgeType.HAS_GROWTH)

    def _build_anomalies(self, results: dict):
        """构建异常检测节点"""
        anomaly_data = results.get('异常检测', results.get('MoM异常', []))
        if isinstance(anomaly_data, list):
            for i, a in enumerate(anomaly_data):
                customer = a.get('客户', '')
                if not customer:
                    continue
                aid = f"anomaly:{customer}_{i}"
                self.G.add_node(
                    aid,
                    type=NodeType.ANOMALY,
                    customer=customer,
                    month=a.get('月份', ''),
                    severity=a.get('严重度', ''),
                    detail=a.get('描述', str(a)),
                )
                cid = f"customer:{customer}"
                if self.G.has_node(cid):
                    self.G.add_edge(cid, aid, type=EdgeType.HAS_ANOMALY)

    def _build_monthly(self, results: dict):
        """构建月度数据节点"""
        for c in results.get('客户分级', []):
            name = c.get('客户', '')
            cid = f"customer:{name}"
            if not self.G.has_node(cid):
                continue

            # 查找月度数据
            for m_key in [f'{i}月' for i in range(1, 13)]:
                val = c.get(m_key, 0)
                if val and val > 0:
                    mid = f"month:{m_key}"
                    if not self.G.has_node(mid):
                        self.G.add_node(mid, type=NodeType.MONTH, name=m_key)
                    self.G.add_edge(cid, mid, type=EdgeType.REVENUE_IN, amount=val)

    def _build_co_region_edges(self):
        """构建同区域客户关联"""
        region_customers = defaultdict(list)
        for node, data in self.G.nodes(data=True):
            if data.get('type') == NodeType.CUSTOMER:
                for _, target, edata in self.G.out_edges(node, data=True):
                    if edata.get('type') == EdgeType.BELONGS_TO:
                        region_customers[target].append(node)

        for region, customers in region_customers.items():
            for i, c1 in enumerate(customers):
                for c2 in customers[i+1:]:
                    self.G.add_edge(c1, c2, type=EdgeType.SAME_REGION, region=region)

    def _count_by_type(self, what: str) -> List[Tuple[str, int]]:
        """按类型统计"""
        counts = defaultdict(int)
        if what == "nodes":
            for _, data in self.G.nodes(data=True):
                counts[data.get('type', 'unknown')] += 1
        else:
            for _, _, data in self.G.edges(data=True):
                counts[data.get('type', 'unknown')] += 1
        return sorted(counts.items(), key=lambda x: -x[1])

    # ============================================================
    # 多跳查询 API
    # ============================================================

    def query_neighbors(self, node_id: str, edge_type: str = None,
                        max_hops: int = 1) -> List[dict]:
        """
        多跳邻居查询

        示例:
          query_neighbors("customer:HMD", edge_type="has_risk")
          → HMD 的所有风险节点

          query_neighbors("customer:HMD", max_hops=2)
          → HMD 的所有 1-2 跳邻居（包含区域、风险、品类等）
        """
        if not self._built or not self.G.has_node(node_id):
            return []

        results = []
        visited = {node_id}

        # BFS 多跳遍历
        queue = [(node_id, 0)]
        while queue:
            current, hop = queue.pop(0)
            if hop >= max_hops:
                continue

            for _, neighbor, edata in self.G.out_edges(current, data=True):
                if neighbor in visited:
                    continue
                if edge_type and edata.get('type') != edge_type:
                    continue

                visited.add(neighbor)
                node_data = dict(self.G.nodes[neighbor])
                node_data['_id'] = neighbor
                node_data['_hop'] = hop + 1
                node_data['_edge_type'] = edata.get('type', '')
                results.append(node_data)
                queue.append((neighbor, hop + 1))

            # 双向 — 也查入边
            for source, _, edata in self.G.in_edges(current, data=True):
                if source in visited:
                    continue
                if edge_type and edata.get('type') != edge_type:
                    continue

                visited.add(source)
                node_data = dict(self.G.nodes[source])
                node_data['_id'] = source
                node_data['_hop'] = hop + 1
                node_data['_edge_type'] = edata.get('type', '')
                results.append(node_data)
                queue.append((source, hop + 1))

        return results

    def query_by_type(self, node_type: str, filters: Dict = None) -> List[dict]:
        """
        按类型查询节点

        示例:
          query_by_type("customer", {"tier": "A"})
          → 所有 A 级客户

          query_by_type("risk_alert", {"level": "高"})
          → 所有高风险警报
        """
        results = []
        for node, data in self.G.nodes(data=True):
            if data.get('type') != node_type:
                continue

            if filters:
                match = all(
                    str(data.get(k, '')).lower() == str(v).lower() or
                    str(v).lower() in str(data.get(k, '')).lower()
                    for k, v in filters.items()
                )
                if not match:
                    continue

            node_data = dict(data)
            node_data['_id'] = node
            results.append(node_data)

        return results

    def find_path(self, source: str, target: str) -> List[str]:
        """
        查找两个实体间的最短路径

        示例:
          find_path("customer:HMD", "region:华南")
          → ["customer:HMD", "region:华南"]
        """
        if not self._built:
            return []

        try:
            path = nx.shortest_path(self.G, source, target)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # 尝试无向图
            try:
                path = nx.shortest_path(self.G.to_undirected(), source, target)
                return path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []

    def get_subgraph(self, center_node: str, radius: int = 2) -> Dict:
        """
        提取子图 — 以某节点为中心，半径内的所有实体

        用途: 为 Agent 构建精准上下文
        """
        if not self._built or not self.G.has_node(center_node):
            return {"nodes": [], "edges": []}

        # BFS 收集节点
        visited = {center_node}
        queue = [(center_node, 0)]
        while queue:
            current, hop = queue.pop(0)
            if hop >= radius:
                continue
            for _, neighbor in self.G.out_edges(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))
            for source, _ in self.G.in_edges(current):
                if source not in visited:
                    visited.add(source)
                    queue.append((source, hop + 1))

        # 构建子图
        sub = self.G.subgraph(visited)
        nodes = [{"id": n, **dict(d)} for n, d in sub.nodes(data=True)]
        edges = [{"source": u, "target": v, **dict(d)} for u, v, d in sub.edges(data=True)]

        return {"nodes": nodes, "edges": edges}

    def aggregate(self, group_type: str, target_type: str,
                  metric: str = "count") -> Dict[str, Any]:
        """
        统计聚合

        示例:
          aggregate("region", "customer", "count")
          → {"华南": 5, "华东": 3, ...}

          aggregate("region", "customer", "sum:annual_amount")
          → {"华南": 12000, "华东": 8000, ...}
        """
        if not self._built:
            return {}

        result = defaultdict(lambda: 0 if 'sum' in metric else [])

        # 找到所有 group_type 节点
        groups = self.query_by_type(group_type)

        for group in groups:
            gid = group['_id']
            gname = group.get('name', gid)

            # 查找关联的 target_type 节点
            neighbors = self.query_neighbors(gid, max_hops=1)
            targets = [n for n in neighbors if n.get('type') == target_type]

            if metric == "count":
                result[gname] = len(targets)
            elif metric.startswith("sum:"):
                field = metric.split(":")[1]
                result[gname] = sum(t.get(field, 0) for t in targets)
            elif metric == "list":
                result[gname] = [t.get('name', t.get('_id')) for t in targets]

        return dict(result)

    # ============================================================
    # Agent 上下文生成 — 核心价值
    # ============================================================

    def generate_entity_context(self, question: str, max_tokens: int = 1500) -> str:
        """
        从问题中提取实体，自动生成多跳上下文

        这是 GraphRAG 的核心价值：
        传统 RAG 只能检索文本片段
        GraphRAG 能跟踪实体关系，提供结构化的多跳上下文

        示例:
          问: "HMD 的风险是什么？"
          → 自动查找 HMD → 风险节点 → 同区域客户 → 区域整体情况
        """
        if not self._built:
            return ""

        # 实体识别
        entities = self._extract_entities(question)
        if not entities:
            return ""

        context_parts = []

        for entity_id in entities[:3]:  # 最多3个实体
            entity = self.G.nodes.get(entity_id, {})
            entity_name = entity.get('name', entity_id)

            # 基本属性
            attrs = {k: v for k, v in entity.items()
                     if k not in ('type', 'name') and v}
            if attrs:
                context_parts.append(
                    f"【{entity_name}】{json.dumps(attrs, ensure_ascii=False, default=str)}"
                )

            # 1跳关联
            neighbors = self.query_neighbors(entity_id, max_hops=1)
            for n in neighbors[:10]:
                ntype = n.get('type', '')
                nname = n.get('name', n.get('customer', ''))
                edge = n.get('_edge_type', '')

                if ntype == NodeType.RISK:
                    context_parts.append(
                        f"  ⚠️ 风险: {n.get('level','')} | {n.get('reason','')}"
                    )
                elif ntype == NodeType.GROWTH:
                    context_parts.append(
                        f"  📈 机会: {n.get('opportunity','')} (潜力¥{n.get('potential',0)}万)"
                    )
                elif ntype == NodeType.ANOMALY:
                    context_parts.append(
                        f"  🔍 异常: {n.get('month','')} {n.get('detail','')[:80]}"
                    )
                elif ntype == NodeType.REGION:
                    context_parts.append(f"  📍 区域: {nname}")
                elif ntype == NodeType.CUSTOMER and edge == EdgeType.SAME_REGION:
                    context_parts.append(
                        f"  👥 同区域: {nname} ({n.get('tier','')}级 ¥{n.get('annual_amount',0)}万)"
                    )

            # 2跳 — 同区域风险
            if entity.get('type') == NodeType.CUSTOMER:
                two_hop = self.query_neighbors(entity_id, max_hops=2)
                co_risks = [n for n in two_hop
                           if n.get('type') == NodeType.RISK
                           and n.get('_hop') == 2]
                if co_risks:
                    context_parts.append(
                        f"  ⚠️ 关联风险(2跳): {len(co_risks)}个同区域客户也有风险预警"
                    )

        result = "\n".join(context_parts)

        # Token 控制
        if len(result) > max_tokens * 2:
            result = result[:max_tokens * 2] + "\n...[更多实体关系已截断]"

        return result

    def _extract_entities(self, question: str) -> List[str]:
        """从问题中识别图中的实体"""
        found = []
        q_lower = question.lower()

        # 精确匹配客户名
        for node, data in self.G.nodes(data=True):
            if data.get('type') != NodeType.CUSTOMER:
                continue
            name = data.get('name', '')
            if name.lower() in q_lower:
                found.append(node)

        # 匹配区域
        for node, data in self.G.nodes(data=True):
            if data.get('type') != NodeType.REGION:
                continue
            name = data.get('name', '')
            if name in question:
                found.append(node)

        # 匹配品类
        for node, data in self.G.nodes(data=True):
            if data.get('type') != NodeType.CATEGORY:
                continue
            name = data.get('name', '')
            if name in question:
                found.append(node)

        return found

    def get_agent_hint(self, question: str) -> List[str]:
        """
        为 multi_agent_v7 的路由提供 Agent 推荐

        基于图中实体关系判断需要哪些 Agent:
        - 有风险节点 → risk agent
        - 有增长节点 → strategist agent
        - 默认 → analyst agent
        """
        entities = self._extract_entities(question)
        agents = {"analyst"}

        for eid in entities:
            neighbors = self.query_neighbors(eid, max_hops=1)
            for n in neighbors:
                ntype = n.get('type', '')
                if ntype == NodeType.RISK:
                    agents.add("risk")
                elif ntype == NodeType.GROWTH:
                    agents.add("strategist")
                elif ntype == NodeType.ANOMALY:
                    agents.add("risk")

        return list(agents)

    # ============================================================
    # 统计和导出
    # ============================================================

    def get_stats(self) -> Dict:
        """获取图谱统计"""
        return self._stats if self._built else {"built": False}

    def to_json(self) -> str:
        """导出为 JSON（用于前端可视化）"""
        if not self._built:
            return "{}"

        nodes = [{"id": n, **{k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}}
                 for n, d in self.G.nodes(data=True)]
        edges = [{"source": u, "target": v, "type": d.get('type', '')}
                 for u, v, d in self.G.edges(data=True)]

        return json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, default=str)

    def summary(self) -> str:
        """图谱摘要（人类可读）"""
        if not self._built:
            return "知识图谱未构建"

        s = self._stats
        lines = [
            f"📊 知识图谱: {s.get('nodes',0)}节点, {s.get('edges',0)}边",
        ]

        nt = s.get('node_types', {})
        if nt:
            lines.append("  节点: " + ", ".join(f"{t}={c}" for t, c in nt.items()))

        # A级客户概要
        a_customers = self.query_by_type(NodeType.CUSTOMER, {"tier": "A"})
        if a_customers:
            names = [c.get('name', '') for c in a_customers[:5]]
            lines.append(f"  A级客户({len(a_customers)}): {', '.join(names)}")

        # 风险概要
        risks = self.query_by_type(NodeType.RISK)
        if risks:
            lines.append(f"  风险预警: {len(risks)}条")

        return "\n".join(lines)


# ============================================================
# Neo4j 适配器 — Phase 4 迁移用
# ============================================================

class Neo4jAdapter:
    """
    Neo4j 适配器 — 当数据量超过 NetworkX 能力时迁移

    使用方法:
        adapter = Neo4jAdapter("bolt://localhost:7687", "neo4j", "password")
        adapter.sync_from_sales_graph(sales_graph)

    迁移时机 (任一满足):
        - 客户数 > 500
        - 需要持久化图数据
        - 需要并发查询
        - 需要 Cypher 复杂查询
    """

    def __init__(self, uri: str, user: str, password: str):
        if not HAS_NEO4J:
            raise ImportError("neo4j 未安装: pip install neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def sync_from_sales_graph(self, graph: SalesGraph):
        """将 NetworkX 图同步到 Neo4j"""
        if not graph._built or not graph.G:
            return

        with self.driver.session() as session:
            # 清空
            session.run("MATCH (n) DETACH DELETE n")

            # 写入节点
            for node, data in graph.G.nodes(data=True):
                props = {k: v for k, v in data.items()
                        if isinstance(v, (str, int, float, bool))}
                node_type = data.get('type', 'Entity')
                session.run(
                    f"CREATE (n:{node_type} $props)",
                    props={**props, "graph_id": node},
                )

            # 写入边
            for u, v, data in graph.G.edges(data=True):
                edge_type = data.get('type', 'RELATES_TO').upper()
                session.run(
                    f"""
                    MATCH (a {{graph_id: $source}})
                    MATCH (b {{graph_id: $target}})
                    CREATE (a)-[:{edge_type}]->(b)
                    """,
                    source=u, target=v,
                )

    def close(self):
        self.driver.close()


# ============================================================
# 全局实例 + v5.0 兼容
# ============================================================

_sales_graph: Optional[SalesGraph] = None


def get_sales_graph() -> SalesGraph:
    """获取全局知识图谱实例"""
    global _sales_graph
    if _sales_graph is None:
        _sales_graph = SalesGraph()
    return _sales_graph


def build_knowledge_graph(data: dict, results: dict) -> SalesGraph:
    """构建并返回知识图谱"""
    graph = get_sales_graph()
    graph.build_from_results(data, results)
    return graph


# 兼容 v5.0 的 SalesKnowledgeGraph 接口
class SalesKnowledgeGraph:
    """v5.0 兼容层"""

    def __init__(self, data=None, results=None):
        self._graph = get_sales_graph()
        if data and results:
            self._graph.build_from_results(data, results)

    def get_entity_context(self, question: str) -> str:
        return self._graph.generate_entity_context(question)

    def get_agent_hint(self, question: str) -> List[str]:
        return self._graph.get_agent_hint(question)

    def summary(self) -> str:
        return self._graph.summary()


# ============================================================
# 模块信息
# ============================================================

__version__ = "7.0.0"
__all__ = [
    "SalesGraph",
    "SalesKnowledgeGraph",
    "Neo4jAdapter",
    "build_knowledge_graph",
    "get_sales_graph",
    "NodeType",
    "EdgeType",
    "HAS_NX",
    "HAS_NEO4J",
]

if __name__ == "__main__":
    print(f"MRARFAI GraphRAG v{__version__}")
    print(f"NetworkX: {'✅' if HAS_NX else '❌'}")
    print(f"Neo4j:    {'✅' if HAS_NEO4J else '❌ (可选)'}")

    if HAS_NX:
        # 模拟测试
        g = SalesGraph()
        mock_results = {
            '客户分级': [
                {'客户': 'HMD', '等级': 'A', '年度金额': 4200, 'H1': 2000, 'H2': 2200, '占比': '15%'},
                {'客户': 'Samsung', '等级': 'A', '年度金额': 3800, 'H1': 1800, 'H2': 2000, '占比': '13%'},
                {'客户': 'Xiaomi', '等级': 'B', '年度金额': 1500, 'H1': 700, 'H2': 800, '占比': '5%'},
            ],
            '流失预警': [
                {'客户': 'HMD', '风险': '中高', '原因': '连续3月下滑', '年度金额': 4200},
            ],
            '增长机会': [
                {'客户': 'Samsung', '机会': '新品类渗透', '潜力金额': 800},
            ],
            '类别趋势': [
                {'类别': '手机', '2024金额': 20000, '2025金额': 28000, '增长率': '40%'},
            ],
            '区域洞察': {'详细': [
                {'区域': '华南', '金额': 15000, '占比': '50%'},
                {'区域': '华东', '金额': 8000, '占比': '27%'},
            ]},
        }

        g.build_from_results({}, mock_results)
        print(f"\n{g.summary()}")

        # 多跳测试
        ctx = g.generate_entity_context("HMD的风险是什么？")
        print(f"\n多跳上下文:\n{ctx}")

        # Agent hint
        hint = g.get_agent_hint("HMD有什么风险？")
        print(f"\nAgent推荐: {hint}")
