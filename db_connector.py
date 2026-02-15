#!/usr/bin/env python3
"""
MRARFAI — 数据库连接器
==========================
支持多种数据源接入:
  - SQLite (Demo/本地)
  - MySQL (ERP: 用友/金蝶/SAP)
  - PostgreSQL (MES/自研系统)
  - REST API (第三方平台)

用法:
    from db_connector import get_connector, DatabaseConfig

    config = DatabaseConfig(type="sqlite", path="data.db")
    db = get_connector(config)
    suppliers = db.query_suppliers()
"""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("mrarfai.db")


# ============================================================
# 数据库配置
# ============================================================
@dataclass
class DatabaseConfig:
    """数据库连接配置"""
    type: str = "none"          # none / sqlite / mysql / postgresql / api
    host: str = ""
    port: int = 0
    database: str = ""
    username: str = ""
    password: str = ""
    path: str = ""              # SQLite 文件路径
    api_url: str = ""           # REST API base URL
    api_key: str = ""
    extra: Dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量加载配置"""
        return cls(
            type=os.environ.get("DB_TYPE", "none"),
            host=os.environ.get("DB_HOST", ""),
            port=int(os.environ.get("DB_PORT", "0") or "0"),
            database=os.environ.get("DB_NAME", ""),
            username=os.environ.get("DB_USER", ""),
            password=os.environ.get("DB_PASSWORD", ""),
            path=os.environ.get("DB_PATH", ""),
            api_url=os.environ.get("DB_API_URL", ""),
            api_key=os.environ.get("DB_API_KEY", ""),
        )

    @classmethod
    def from_dict(cls, d: Dict) -> "DatabaseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# 抽象连接器接口
# ============================================================
class BaseConnector:
    """数据库连接器基类"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connected = False

    def connect(self) -> bool:
        """建立连接"""
        raise NotImplementedError

    def disconnect(self):
        """断开连接"""
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def test_connection(self) -> Dict:
        """测试连接"""
        try:
            ok = self.connect()
            return {"status": "ok" if ok else "failed", "type": self.config.type}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── Agent 数据查询接口 ──
    def query_suppliers(self) -> List[Dict]:
        """采购Agent: 查询供应商列表"""
        return []

    def query_orders(self) -> List[Dict]:
        """采购Agent: 查询采购订单"""
        return []

    def query_accounts_receivable(self) -> List[Dict]:
        """财务Agent: 查询应收账款"""
        return []

    def query_margins(self) -> List[Dict]:
        """财务Agent: 查询毛利数据"""
        return []

    def query_yields(self) -> List[Dict]:
        """品质Agent: 查询良率数据"""
        return []

    def query_returns(self) -> List[Dict]:
        """品质Agent: 查询退货数据"""
        return []

    def query_competitors(self) -> List[Dict]:
        """市场Agent: 查询竞品数据"""
        return []

    def query_risk_clients(self) -> List[Dict]:
        """风控Agent: 查询客户风险数据"""
        return []

    def query_anomalies(self) -> List[Dict]:
        """风控Agent: 查询异常检测结果"""
        return []

    def query_health_scores(self) -> List[Dict]:
        """风控Agent: 查询客户健康评分"""
        return []

    def query_industry_benchmark(self) -> List[Dict]:
        """策略Agent: 查询行业对标数据"""
        return []

    def query_forecast(self) -> List[Dict]:
        """策略Agent: 查询预测数据"""
        return []

    def query_sales(self, start_date: str = "", end_date: str = "") -> List[Dict]:
        """销售Agent: 查询销售数据"""
        return []

    def execute_raw(self, sql: str, params: tuple = ()) -> List[Dict]:
        """执行原始SQL (仅支持SQL类数据库)"""
        return []


# ============================================================
# 无数据库模式 — 使用 Agent 内置样本数据
# ============================================================
class NoDBConnector(BaseConnector):
    """无数据库连接 — 回退到 Agent 内置样本数据"""

    def connect(self) -> bool:
        self._connected = True
        return True

    def test_connection(self) -> Dict:
        return {"status": "ok", "type": "none", "message": "使用内置样本数据"}


# ============================================================
# SQLite 连接器 — Demo/本地数据
# ============================================================
class SQLiteConnector(BaseConnector):
    """SQLite 数据库连接器"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._conn = None

    def connect(self) -> bool:
        try:
            import sqlite3
            self._conn = sqlite3.connect(self.config.path)
            self._conn.row_factory = sqlite3.Row
            self._connected = True
            logger.info(f"SQLite 已连接: {self.config.path}")
            return True
        except Exception as e:
            logger.error(f"SQLite 连接失败: {e}")
            return False

    def disconnect(self):
        if self._conn:
            self._conn.close()
        super().disconnect()

    def execute_raw(self, sql: str, params: tuple = ()) -> List[Dict]:
        if not self._connected or not self._conn:
            return []
        try:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"SQL执行失败: {e}")
            return []

    def query_suppliers(self) -> List[Dict]:
        return self.execute_raw("SELECT * FROM suppliers ORDER BY name")

    def query_orders(self) -> List[Dict]:
        return self.execute_raw("SELECT * FROM orders ORDER BY created_date DESC LIMIT 100")

    def query_accounts_receivable(self) -> List[Dict]:
        return self.execute_raw("SELECT * FROM accounts_receivable ORDER BY due_date")

    def query_margins(self) -> List[Dict]:
        return self.execute_raw("SELECT * FROM margins ORDER BY product")

    def query_yields(self) -> List[Dict]:
        return self.execute_raw("SELECT * FROM yields ORDER BY month DESC")

    def query_returns(self) -> List[Dict]:
        return self.execute_raw("SELECT * FROM returns ORDER BY date DESC LIMIT 100")

    def query_competitors(self) -> List[Dict]:
        return self.execute_raw("SELECT * FROM competitors ORDER BY revenue_billion DESC")

    def query_sales(self, start_date: str = "", end_date: str = "") -> List[Dict]:
        sql = "SELECT * FROM sales"
        params = []
        if start_date:
            sql += " WHERE date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?" if start_date else " WHERE date <= ?"
            params.append(end_date)
        sql += " ORDER BY date DESC LIMIT 500"
        return self.execute_raw(sql, tuple(params))


# ============================================================
# MySQL 连接器 — ERP 系统
# ============================================================
class MySQLConnector(BaseConnector):
    """MySQL 数据库连接器 (用友/金蝶/SAP 等 ERP 系统)"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._conn = None

    def connect(self) -> bool:
        try:
            import pymysql
            self._conn = pymysql.connect(
                host=self.config.host,
                port=self.config.port or 3306,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
            )
            self._connected = True
            logger.info(f"MySQL 已连接: {self.config.host}:{self.config.port}/{self.config.database}")
            return True
        except ImportError:
            logger.error("MySQL 连接需要 pymysql: pip install pymysql")
            return False
        except Exception as e:
            logger.error(f"MySQL 连接失败: {e}")
            return False

    def disconnect(self):
        if self._conn:
            self._conn.close()
        super().disconnect()

    def execute_raw(self, sql: str, params: tuple = ()) -> List[Dict]:
        if not self._connected or not self._conn:
            return []
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except Exception as e:
            logger.error(f"MySQL查询失败: {e}")
            return []

    def query_suppliers(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_suppliers",
                "SELECT * FROM suppliers ORDER BY name LIMIT 200"))

    def query_orders(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_orders",
                "SELECT * FROM purchase_orders ORDER BY created_date DESC LIMIT 200"))

    def query_accounts_receivable(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_ar",
                "SELECT * FROM accounts_receivable ORDER BY due_date LIMIT 200"))

    def query_margins(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_margins",
                "SELECT * FROM product_margins ORDER BY product LIMIT 200"))


# ============================================================
# REST API 连接器 — 第三方平台
# ============================================================
class APIConnector(BaseConnector):
    """REST API 数据连接器"""

    def connect(self) -> bool:
        try:
            import requests
            resp = requests.get(
                f"{self.config.api_url}/health",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=5,
            )
            self._connected = resp.status_code == 200
            return self._connected
        except ImportError:
            logger.error("API 连接需要 requests: pip install requests")
            return False
        except Exception as e:
            logger.error(f"API 连接失败: {e}")
            return False

    def _get(self, endpoint: str, params: Dict = None) -> List[Dict]:
        if not self._connected:
            return []
        try:
            import requests
            resp = requests.get(
                f"{self.config.api_url}{endpoint}",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                params=params,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("data", [])
            return []
        except Exception as e:
            logger.error(f"API请求失败: {e}")
            return []

    def query_suppliers(self) -> List[Dict]:
        return self._get("/api/suppliers")

    def query_orders(self) -> List[Dict]:
        return self._get("/api/orders")

    def query_accounts_receivable(self) -> List[Dict]:
        return self._get("/api/ar")

    def query_margins(self) -> List[Dict]:
        return self._get("/api/margins")


# ============================================================
# PostgreSQL 连接器 — 生产环境 (V10.1)
# ============================================================
class PostgresConnector(BaseConnector):
    """
    PostgreSQL 数据库连接器 — 生产级

    特性:
      - psycopg2 连接池 (SimpleConnectionPool)
      - 自动 rollback (防止事务泄露)
      - 与 init_postgres.sql schema 对齐
    """

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._pool = None

    def connect(self) -> bool:
        try:
            import psycopg2
            from psycopg2 import pool as pg_pool
            self._pool = pg_pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=int(self.config.extra.get("maxconn", 10)) if self.config.extra else 10,
                host=self.config.host or "localhost",
                port=self.config.port or 5432,
                dbname=self.config.database or "mrarfai_sales",
                user=self.config.username or "mrarfai",
                password=self.config.password,
            )
            # 验证连接 (try/finally 防止连接泄露)
            conn = self._pool.getconn()
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            finally:
                self._pool.putconn(conn)
            self._connected = True
            logger.info(f"PostgreSQL 已连接: {self.config.host}:{self.config.port or 5432}/{self.config.database}")
            return True
        except ImportError:
            logger.error("PostgreSQL 连接需要 psycopg2: pip install psycopg2-binary")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL 连接失败: {e}")
            return False

    def disconnect(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None
        super().disconnect()

    def _get_conn(self):
        """从连接池获取连接"""
        if self._pool:
            conn = self._pool.getconn()
            conn.autocommit = False
            return conn
        return None

    def _put_conn(self, conn):
        """归还连接到池 (含 rollback 防止事务泄露)"""
        if conn and self._pool:
            try:
                conn.rollback()
            except Exception:
                pass
            self._pool.putconn(conn)

    def execute_raw(self, sql: str, params: tuple = ()) -> List[Dict]:
        if not self._connected or not self._pool:
            return []
        conn = None
        try:
            conn = self._get_conn()
            import psycopg2.extras
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                conn.commit()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"PostgreSQL 查询失败: {e}")
            # rollback 由 _put_conn 统一处理
            return []
        finally:
            if conn:
                self._put_conn(conn)

    def test_connection(self) -> Dict:
        if not self._connected:
            return {"connected": False, "error": "Not connected"}
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                ver = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")
                tables = cur.fetchone()[0]
            conn.commit()
            return {"connected": True, "version": ver, "tables": tables, "pool_size": self._pool.maxconn}
        except Exception as e:
            return {"connected": False, "error": str(e)}
        finally:
            if conn:
                self._put_conn(conn)

    # --- Agent 查询方法 (对齐 init_postgres.sql schema) ---

    def query_suppliers(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_suppliers",
                "SELECT * FROM dim_supplier ORDER BY supplier_name LIMIT 200"))

    def query_orders(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_orders",
                "SELECT * FROM fact_purchase_orders ORDER BY order_date DESC LIMIT 200"))

    def query_accounts_receivable(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_ar",
                "SELECT * FROM fact_accounts_receivable ORDER BY due_date LIMIT 200"))

    def query_margins(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_margins",
                "SELECT * FROM fact_product_margins ORDER BY product_name LIMIT 200"))

    def query_yields(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_yields",
                "SELECT * FROM fact_quality_yields ORDER BY record_date DESC LIMIT 200"))

    def query_returns(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_returns",
                "SELECT * FROM fact_quality_returns ORDER BY return_date DESC LIMIT 200"))

    def query_competitors(self) -> List[Dict]:
        return self.execute_raw(
            self.config.extra.get("sql_competitors",
                "SELECT * FROM dim_competitor ORDER BY competitor_name LIMIT 200"))

    def query_sales(self, start_date: str = "", end_date: str = "") -> List[Dict]:
        sql = "SELECT * FROM fact_sales"
        params = []
        conditions = []
        if start_date:
            conditions.append("sale_date >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("sale_date <= %s")
            params.append(end_date)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY sale_date DESC LIMIT 500"
        return self.execute_raw(sql, tuple(params))


# ============================================================
# 工厂函数
# ============================================================
_CONNECTORS = {
    "none": NoDBConnector,
    "sqlite": SQLiteConnector,
    "mysql": MySQLConnector,
    "postgresql": PostgresConnector,
    "api": APIConnector,
}


def get_connector(config: DatabaseConfig = None) -> BaseConnector:
    """获取数据库连接器实例"""
    if config is None:
        config = DatabaseConfig.from_env()
    cls = _CONNECTORS.get(config.type, NoDBConnector)
    connector = cls(config)
    connector.connect()
    return connector


def get_db_status(config: DatabaseConfig = None) -> Dict:
    """获取数据库连接状态"""
    if config is None:
        config = DatabaseConfig.from_env()
    if config.type == "none":
        return {
            "type": "none",
            "status": "ok",
            "message": "使用内置样本数据（可在设置中配置数据库连接）",
        }
    connector = _CONNECTORS.get(config.type, NoDBConnector)(config)
    result = connector.test_connection()
    connector.disconnect()
    return result


# ============================================================
# DB → Agent Engine 桥接
# ============================================================
def create_engines_from_db(config: DatabaseConfig = None) -> Dict:
    """
    从数据库加载数据并创建 Agent Engine 实例。

    返回 dict: {agent_name: engine_instance}
    如果 DB_TYPE=none 或连接失败，返回空 dict（Agent 将使用内置样本数据）。
    """
    if config is None:
        config = DatabaseConfig.from_env()
    if config.type == "none":
        return {}

    try:
        connector = get_connector(config)
        if not connector.is_connected():
            logger.warning("数据库未连接，使用内置样本数据")
            return {}
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return {}

    engines = {}

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas 未安装，无法从 DB 创建 Engine")
        connector.disconnect()
        return {}

    # ── 采购 Agent ──
    try:
        suppliers_data = connector.query_suppliers()
        orders_data = connector.query_orders()
        if suppliers_data or orders_data:
            from agent_procurement import ProcurementEngine
            sup_df = pd.DataFrame(suppliers_data) if suppliers_data else None
            ord_df = pd.DataFrame(orders_data) if orders_data else None
            engines["procurement"] = ProcurementEngine.from_dataframes(sup_df, ord_df)
            logger.info(f"DB → 采购Agent: {len(suppliers_data)} suppliers, {len(orders_data)} orders")
    except Exception as e:
        logger.error(f"DB→采购Engine失败: {e}")

    # ── 财务 Agent ──
    try:
        ar_data = connector.query_accounts_receivable()
        margin_data = connector.query_margins()
        if ar_data or margin_data:
            from agent_finance import FinanceEngine
            ar_df = pd.DataFrame(ar_data) if ar_data else None
            mg_df = pd.DataFrame(margin_data) if margin_data else None
            engines["finance"] = FinanceEngine.from_dataframes(ar_df, mg_df)
            logger.info(f"DB → 财务Agent: {len(ar_data)} AR, {len(margin_data)} margins")
    except Exception as e:
        logger.error(f"DB→财务Engine失败: {e}")

    # ── 品质 Agent ──
    try:
        yields_data = connector.query_yields()
        returns_data = connector.query_returns()
        if yields_data or returns_data:
            from agent_quality import QualityEngine
            yd_df = pd.DataFrame(yields_data) if yields_data else None
            rt_df = pd.DataFrame(returns_data) if returns_data else None
            engines["quality"] = QualityEngine.from_dataframes(yd_df, rt_df)
            logger.info(f"DB → 品质Agent: {len(yields_data)} yields, {len(returns_data)} returns")
    except Exception as e:
        logger.error(f"DB→品质Engine失败: {e}")

    # ── 市场 Agent ──
    try:
        competitors_data = connector.query_competitors()
        if competitors_data:
            from agent_market import MarketEngine
            comp_df = pd.DataFrame(competitors_data)
            engines["market"] = MarketEngine.from_dataframes(comp_df)
            logger.info(f"DB → 市场Agent: {len(competitors_data)} competitors")
    except Exception as e:
        logger.error(f"DB→市场Engine失败: {e}")

    # ── 风控 Agent ──
    try:
        risk_clients = connector.query_risk_clients()
        anomalies = connector.query_anomalies()
        health_scores = connector.query_health_scores()
        if risk_clients or anomalies or health_scores:
            from agent_risk import RiskEngine
            engines["risk"] = RiskEngine(
                risk_clients=risk_clients if risk_clients else None,
                anomalies=anomalies if anomalies else None,
                health_scores=health_scores if health_scores else None,
            )
            logger.info(f"DB → 风控Agent: {len(risk_clients)} clients, {len(anomalies)} anomalies")
    except Exception as e:
        logger.error(f"DB→风控Engine失败: {e}")

    # ── 策略 Agent ──
    try:
        benchmark = connector.query_industry_benchmark()
        forecast = connector.query_forecast()
        if benchmark or forecast:
            from agent_strategist import StrategistEngine
            positioning = benchmark[0] if benchmark else None
            competitive = benchmark[1] if len(benchmark) > 1 else None
            engines["strategist"] = StrategistEngine(
                positioning=positioning,
                competitive=competitive,
                forecast=forecast[0] if forecast else None,
            )
            logger.info(f"DB → 策略Agent: benchmark={len(benchmark)}, forecast={len(forecast)}")
    except Exception as e:
        logger.error(f"DB→策略Engine失败: {e}")

    connector.disconnect()
    return engines
