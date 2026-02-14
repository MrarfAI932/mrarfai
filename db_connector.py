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
        return self.execute_raw("SELECT * FROM competitors ORDER BY revenue DESC")

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
# 工厂函数
# ============================================================
_CONNECTORS = {
    "none": NoDBConnector,
    "sqlite": SQLiteConnector,
    "mysql": MySQLConnector,
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
