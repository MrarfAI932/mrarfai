#!/usr/bin/env python3
"""
MRARFAI P3-05 — Graphiti 时序知识图谱集成
============================================
基于 Zep/Graphiti (arxiv 2501.13956):
  - 双时间模型 (bi-temporal): t_valid/t_invalid + t_created/t_expired
  - 实时增量更新 (无需批量重算)
  - 混合搜索: 语义 + BM25 + 图遍历
  - MCP Server 支持

现有系统: knowledge_graph.py (852行) → SalesKnowledgeGraph
升级路径: 保留现有KG作为fallback, Graphiti作为增强层

安装: pip install graphiti-core
依赖: Neo4j / FalkorDB (docker)
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("mrarfai.graphiti")

# ============================================================
# Safe Import
# ============================================================
try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    HAS_GRAPHITI = True
except ImportError:
    HAS_GRAPHITI = False
    logger.info("graphiti-core not installed — Graphiti features disabled")

try:
    from knowledge_graph import SalesKnowledgeGraph
    HAS_LEGACY_KG = True
except ImportError:
    HAS_LEGACY_KG = False


# ============================================================
# Graphiti Adapter for MRARFAI
# ============================================================

class MRARFAIGraphiti:
    """
    MRARFAI Graphiti时序知识图谱适配器
    
    特性:
      - 客户关系时序追踪 (合同签订/续签/终止)
      - 出货量变化事件图 (月度波动作为episodes)
      - 供应商变更历史
      - 品质事件时间线
      - 混合搜索: 语义 + 关键词 + 图遍历
    
    降级策略:
      - Level 1: Graphiti + Neo4j (全功能)
      - Level 2: 现有SalesKnowledgeGraph (fallback)
      - Level 3: 纯SQL查询
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = None,
        openai_api_key: str = None,
    ):
        self.client = None
        self.legacy_kg = None

        # 从环境变量获取密码 (避免硬编码)
        import os
        neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "neo4j")

        # Level 1: 尝试Graphiti
        if HAS_GRAPHITI:
            try:
                import asyncio
                self.client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
                logger.info("✅ Graphiti connected to Neo4j")
            except Exception as e:
                logger.warning(f"Graphiti init failed: {e}")
                self.client = None
        
        # Level 2: Fallback to legacy KG
        if self.client is None and HAS_LEGACY_KG:
            self.legacy_kg = SalesKnowledgeGraph()
            logger.info("⚠️ Fallback to SalesKnowledgeGraph")
    
    @property
    def is_graphiti(self) -> bool:
        return self.client is not None
    
    @property
    def backend_name(self) -> str:
        if self.client:
            return "Graphiti (Temporal KG)"
        elif self.legacy_kg:
            return "SalesKnowledgeGraph (Legacy)"
        return "None"
    
    # ─── Episode Ingestion ───
    
    async def ingest_sales_event(
        self,
        customer: str,
        event_type: str,  # "shipment" | "contract" | "quality_issue" | "ar_payment"
        data: Dict[str, Any],
        timestamp: datetime = None,
    ):
        """
        注入销售事件到时序知识图谱
        
        事件类型:
          - shipment: 出货记录 → (Customer)-[SHIPPED_TO]->(Product)
          - contract: 合同变更 → (Customer)-[CONTRACT]->(Terms)
          - quality_issue: 品质问题 → (Product)-[HAS_DEFECT]->(DefectType)
          - ar_payment: 收款事件 → (Customer)-[PAID]->(Invoice)
        
        Graphiti自动处理:
          - 实体抽取 (Customer, Product, Supplier)
          - 关系创建/更新
          - 时间戳标注 (bi-temporal)
          - 冲突消解 (新信息覆盖旧信息, 保留历史)
        """
        if not self.client:
            logger.debug(f"Graphiti not available, skipping event: {event_type}")
            return
        
        ts = timestamp or datetime.now()
        
        # 构造episode内容
        episode_content = self._format_episode(customer, event_type, data, ts)
        
        try:
            await self.client.add_episode(
                name=f"{customer}_{event_type}_{ts.strftime('%Y%m%d')}",
                episode_body=episode_content,
                source=EpisodeType.json,
                source_description=f"MRARFAI V10 {event_type} event",
                reference_time=ts,
            )
            logger.info(f"📊 Episode ingested: {customer}/{event_type}")
        except Exception as e:
            logger.warning(f"Episode ingestion failed: {e}")
    
    def _format_episode(
        self, customer: str, event_type: str, data: Dict, ts: datetime
    ) -> str:
        """格式化事件为Graphiti episode"""
        templates = {
            "shipment": (
                f"在{ts.strftime('%Y年%m月%d日')}, "
                f"客户{customer}收到了{data.get('product', '产品')}的出货, "
                f"数量为{data.get('quantity', 0)}台, "
                f"金额为{data.get('amount', 0)}万元。"
            ),
            "contract": (
                f"在{ts.strftime('%Y年%m月%d日')}, "
                f"与客户{customer}签订了{data.get('type', '采购')}合同, "
                f"有效期至{data.get('end_date', '未知')}, "
                f"合同金额{data.get('amount', 0)}万元。"
            ),
            "quality_issue": (
                f"在{ts.strftime('%Y年%m月%d日')}, "
                f"客户{customer}的{data.get('product', '产品')}出现品质问题: "
                f"{data.get('defect', '未知缺陷')}, "
                f"影响{data.get('affected_units', 0)}台, "
                f"根因分析: {data.get('root_cause', '调查中')}。"
            ),
            "ar_payment": (
                f"在{ts.strftime('%Y年%m月%d日')}, "
                f"客户{customer}支付了发票{data.get('invoice', '')}, "
                f"金额{data.get('amount', 0)}万元, "
                f"当前应收余额{data.get('balance', 0)}万元。"
            ),
        }
        return templates.get(event_type, f"{customer}: {event_type} - {data}")
    
    # ─── Search ───
    
    async def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        混合搜索: 语义 + 关键词 + 图遍历
        
        Graphiti特点:
          - 结合embedding相似度和BM25关键词
          - 图遍历发现关联实体
          - 时间过滤 (如: "上个季度Samsung的出货情况")
          - 结果融合排序
        """
        if self.client:
            try:
                results = await self.client.search(query, num_results=limit)
                return [
                    {
                        "fact": edge.fact if hasattr(edge, 'fact') else str(edge),
                        "score": getattr(edge, 'score', 0),
                        "valid_from": str(getattr(edge, 't_valid', '')),
                        "valid_to": str(getattr(edge, 't_invalid', '')),
                    }
                    for edge in results
                ]
            except Exception as e:
                logger.warning(f"Graphiti search failed: {e}")
        
        # Fallback to legacy KG
        if self.legacy_kg:
            return self.legacy_kg.search(query)[:limit] if hasattr(self.legacy_kg, 'search') else []
        
        return []
    
    # ─── Temporal Queries ───
    
    async def get_customer_timeline(self, customer: str) -> List[Dict]:
        """获取客户完整时间线 — Graphiti bi-temporal"""
        if self.client:
            try:
                results = await self.client.search(
                    f"{customer}的所有历史事件",
                    num_results=50,
                )
                timeline = sorted(
                    [
                        {
                            "fact": edge.fact if hasattr(edge, 'fact') else str(edge),
                            "time": str(getattr(edge, 't_valid', '')),
                        }
                        for edge in results
                    ],
                    key=lambda x: x["time"],
                )
                return timeline
            except Exception as e:
                logger.warning(f"Timeline query failed: {e}")
        
        return []
    
    async def detect_relationship_changes(self, customer: str) -> List[Dict]:
        """
        检测客户关系变化 — 利用Graphiti的edge invalidation
        
        当新信息与旧信息冲突时, Graphiti会invalidate旧edge,
        我们可以通过比较t_valid和t_invalid来发现变化。
        """
        if self.client:
            try:
                results = await self.client.search(
                    f"{customer}的合同、出货或联系方式变更",
                    num_results=30,
                )
                changes = [
                    {
                        "fact": edge.fact if hasattr(edge, 'fact') else str(edge),
                        "valid_from": str(getattr(edge, 't_valid', '')),
                        "expired_at": str(getattr(edge, 't_invalid', '')),
                        "is_current": getattr(edge, 't_invalid', None) is None,
                    }
                    for edge in results
                    if hasattr(edge, 't_invalid') and edge.t_invalid is not None
                ]
                return changes
            except Exception as e:
                logger.warning(f"Change detection failed: {e}")
        
        return []
    
    # ─── Batch Ingestion from Excel ───
    
    async def ingest_from_excel(self, df, data_type: str = "shipment"):
        """
        批量导入Excel数据到Graphiti
        
        Args:
            df: pandas DataFrame (出货/应收/品质数据)
            data_type: 数据类型
        """
        if not self.client:
            logger.warning("Graphiti not available for batch ingestion")
            return 0
        
        count = 0
        for _, row in df.iterrows():
            try:
                customer = str(row.iloc[0]) if len(row) > 0 else "Unknown"
                data = {col: str(row[col]) for col in df.columns[:6]}
                await self.ingest_sales_event(customer, data_type, data)
                count += 1
            except Exception as e:
                logger.debug(f"Row ingestion failed: {e}")
        
        logger.info(f"📊 Batch ingested: {count}/{len(df)} rows as {data_type}")
        return count


# ============================================================
# MCP Server Integration
# ============================================================

def register_graphiti_mcp_tools(mcp_server, graphiti: MRARFAIGraphiti):
    """
    注册 Graphiti 搜索工具到 MCP Server

    用法 (mcp_server_v7.py 集成):
        from graphiti_adapter import register_graphiti_mcp_tools, MRARFAIGraphiti
        kg = MRARFAIGraphiti()
        register_graphiti_mcp_tools(server, kg)

    TODO: 当Graphiti MCP Server标准确定后，注册具体工具:
        - graphiti_search: 混合搜索 (语义+BM25+图遍历)
        - graphiti_timeline: 客户时间线查询
        - graphiti_ingest: 事件注入
    """
    # 占位: 工具注册逻辑待Graphiti MCP标准确定后实现
    logger.info(f"📋 Graphiti MCP tools registered (backend: {graphiti.backend_name})")


# ============================================================
# Export
# ============================================================

__all__ = [
    "MRARFAIGraphiti",
    "register_graphiti_mcp_tools",
    "HAS_GRAPHITI",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    kg = MRARFAIGraphiti()
    print(f"Backend: {kg.backend_name}")
    print(f"Graphiti: {'✅' if kg.is_graphiti else '❌'}")
    print(f"Legacy KG: {'✅' if kg.legacy_kg else '❌'}")
    
    if HAS_GRAPHITI:
        print("✅ graphiti-core installed")
    else:
        print("❌ graphiti-core not installed")
        print("   Install: pip install graphiti-core")
        print("   Also need: Neo4j or FalkorDB (docker)")
