#!/usr/bin/env python3
"""
MRARFAI SQL Data Layer v1.0
=============================
生产级数据库对接 — 替代 Excel 上传

支持:
  - SQLite (本地/测试)
  - MySQL (禾苗生产环境)
  - PostgreSQL (云端部署)
  - Excel 兼容 (平滑过渡)

核心能力:
  1. 统一 DataAdapter 接口 — 无论数据源如何切换
  2. 预置查询模板 — 常用销售分析SQL
  3. 连接池管理 — 生产级连接复用
  4. SmartDataQuery 集成 — 替代 DataFrame 查询
"""

import os
import json
import time
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger("mrarfai.sql")


# ============================================================
# Config
# ============================================================

@dataclass
class DBConfig:
    """数据库连接配置"""
    engine: str = "sqlite"          # sqlite / mysql / postgresql
    host: str = "localhost"
    port: int = 3306
    database: str = "mrarfai_sales"
    user: str = ""
    password: str = ""
    sqlite_path: str = "sales_data.db"
    pool_size: int = 5
    pool_timeout: int = 30
    charset: str = "utf8mb4"

    @classmethod
    def from_env(cls):
        """从环境变量读取配置"""
        return cls(
            engine=os.getenv("MRARFAI_DB_ENGINE", "sqlite"),
            host=os.getenv("MRARFAI_DB_HOST", "localhost"),
            port=int(os.getenv("MRARFAI_DB_PORT", "3306")),
            database=os.getenv("MRARFAI_DB_NAME", "mrarfai_sales"),
            user=os.getenv("MRARFAI_DB_USER", ""),
            password=os.getenv("MRARFAI_DB_PASSWORD", ""),
            sqlite_path=os.getenv("MRARFAI_DB_SQLITE", "sales_data.db"),
        )

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# Connection Manager
# ============================================================

class ConnectionManager:
    """数据库连接管理（含简易连接池）"""

    def __init__(self, config: DBConfig):
        self.config = config
        self._pool: list = []
        self._active = 0

    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = self._acquire()
        try:
            yield conn
        finally:
            self._release(conn)

    def _acquire(self):
        if self._pool:
            return self._pool.pop()

        cfg = self.config

        if cfg.engine == "sqlite":
            import sqlite3
            conn = sqlite3.connect(cfg.sqlite_path)
            conn.row_factory = sqlite3.Row
            return conn

        elif cfg.engine == "mysql":
            try:
                import pymysql
                return pymysql.connect(
                    host=cfg.host, port=cfg.port,
                    user=cfg.user, password=cfg.password,
                    database=cfg.database, charset=cfg.charset,
                    cursorclass=pymysql.cursors.DictCursor,
                    connect_timeout=cfg.pool_timeout,
                )
            except ImportError:
                raise ImportError("pip install pymysql")

        elif cfg.engine == "postgresql":
            try:
                import psycopg2
                import psycopg2.extras
                return psycopg2.connect(
                    host=cfg.host, port=cfg.port,
                    user=cfg.user, password=cfg.password,
                    dbname=cfg.database,
                    cursor_factory=psycopg2.extras.RealDictCursor,
                )
            except ImportError:
                raise ImportError("pip install psycopg2-binary")

        raise ValueError(f"Unsupported engine: {cfg.engine}")

    def _release(self, conn):
        if len(self._pool) < self.config.pool_size:
            self._pool.append(conn)
        else:
            conn.close()

    def close_all(self):
        for conn in self._pool:
            try:
                conn.close()
            except Exception:
                pass
        self._pool.clear()


# ============================================================
# Schema Definition — 禾苗销售数据表结构
# ============================================================

SALES_SCHEMA = """
-- 客户维度表
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name   TEXT NOT NULL,
    region          TEXT DEFAULT '',
    industry        TEXT DEFAULT 'ODM',
    tier            TEXT DEFAULT '',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 产品维度表
CREATE TABLE IF NOT EXISTS dim_product (
    product_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name    TEXT NOT NULL,
    category        TEXT DEFAULT '',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 销售事实表 (月度粒度)
CREATE TABLE IF NOT EXISTS fact_sales (
    sale_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL,
    product_id      INTEGER,
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    revenue         REAL DEFAULT 0,
    units           INTEGER DEFAULT 0,
    cost            REAL DEFAULT 0,
    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
    FOREIGN KEY (product_id) REFERENCES dim_product(product_id)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_sales_year_month ON fact_sales(year, month);
CREATE INDEX IF NOT EXISTS idx_sales_customer ON fact_sales(customer_id);
"""


# ============================================================
# Pre-built Query Templates — 常用分析SQL
# ============================================================

QUERY_TEMPLATES = {
    "total_revenue": """
        SELECT year, SUM(revenue) as total_revenue
        FROM fact_sales
        WHERE year = ?
        GROUP BY year
    """,

    "yoy_comparison": """
        SELECT
            curr.year as current_year,
            curr.total as current_revenue,
            prev.total as previous_revenue,
            ROUND((curr.total - prev.total) / prev.total * 100, 2) as yoy_pct
        FROM
            (SELECT year, SUM(revenue) as total FROM fact_sales WHERE year = ? GROUP BY year) curr,
            (SELECT year, SUM(revenue) as total FROM fact_sales WHERE year = ? GROUP BY year) prev
    """,

    "monthly_revenue": """
        SELECT month, SUM(revenue) as revenue
        FROM fact_sales
        WHERE year = ?
        GROUP BY month
        ORDER BY month
    """,

    "customer_ranking": """
        SELECT c.customer_name, SUM(f.revenue) as total_revenue,
               COUNT(DISTINCT f.month) as active_months
        FROM fact_sales f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        WHERE f.year = ?
        GROUP BY c.customer_name
        ORDER BY total_revenue DESC
        LIMIT ?
    """,

    "customer_monthly": """
        SELECT c.customer_name, f.month, SUM(f.revenue) as revenue
        FROM fact_sales f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        WHERE f.year = ?
        GROUP BY c.customer_name, f.month
        ORDER BY c.customer_name, f.month
    """,

    "product_breakdown": """
        SELECT p.product_name, p.category,
               SUM(CASE WHEN f.year = ? THEN f.revenue ELSE 0 END) as current_revenue,
               SUM(CASE WHEN f.year = ? THEN f.revenue ELSE 0 END) as previous_revenue
        FROM fact_sales f
        JOIN dim_product p ON f.product_id = p.product_id
        WHERE f.year IN (?, ?)
        GROUP BY p.product_name, p.category
        ORDER BY current_revenue DESC
    """,

    "customer_churn_data": """
        SELECT c.customer_name,
               GROUP_CONCAT(f.revenue ORDER BY f.month) as monthly_revenues
        FROM fact_sales f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        WHERE f.year = ?
        GROUP BY c.customer_name
    """,

    "concentration": """
        SELECT c.customer_name, SUM(f.revenue) as revenue
        FROM fact_sales f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        WHERE f.year = ?
        GROUP BY c.customer_name
        ORDER BY revenue DESC
    """,
}


# ============================================================
# DataAdapter — 统一数据接口
# ============================================================

class DataAdapter:
    """
    统一数据适配器 — 无论 SQL/Excel 都通过此接口

    Usage:
        adapter = DataAdapter(DBConfig.from_env())
        adapter.init_schema()  # 首次使用

        # 查询
        revenue = adapter.query("total_revenue", [2024])
        ranking = adapter.query("customer_ranking", [2024, 10])

        # 自定义SQL
        result = adapter.raw_query("SELECT * FROM fact_sales WHERE year=? LIMIT 10", [2024])

        # Excel 导入
        adapter.import_from_dataframe(df, year=2024)

        # 生成 context_data 给 Agent
        context = adapter.build_context(year=2024, question="营收分析")
    """

    def __init__(self, config: DBConfig = None):
        self.config = config or DBConfig.from_env()
        self.conn_mgr = ConnectionManager(self.config)
        self._query_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 300  # 5 min

    def init_schema(self):
        """初始化数据库表结构"""
        with self.conn_mgr.get_connection() as conn:
            if self.config.engine == "sqlite":
                conn.executescript(SALES_SCHEMA)
            else:
                cursor = conn.cursor()
                # MySQL/PG: 逐个执行
                for stmt in SALES_SCHEMA.split(";"):
                    stmt = stmt.strip()
                    if stmt and not stmt.startswith("--"):
                        cursor.execute(stmt)
                conn.commit()
        logger.info("Schema initialized")

    def query(self, template_name: str, params: list = None) -> List[dict]:
        """执行预置查询模板"""
        sql = QUERY_TEMPLATES.get(template_name)
        if not sql:
            raise ValueError(f"Unknown template: {template_name}")
        return self.raw_query(sql, params)

    def raw_query(self, sql: str, params: list = None) -> List[dict]:
        """执行原始 SQL"""
        # 简单缓存
        cache_key = hashlib.md5(f"{sql}:{params}".encode()).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached:
            result, ts = cached
            if time.time() - ts < self._cache_ttl:
                return result

        with self.conn_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params or [])
            rows = cursor.fetchall()
            # 转为 dict list
            if rows and hasattr(rows[0], 'keys'):
                result = [dict(r) for r in rows]
            else:
                cols = [d[0] for d in cursor.description] if cursor.description else []
                result = [dict(zip(cols, r)) for r in rows]

        self._query_cache[cache_key] = (result, time.time())
        return result

    def import_from_dataframe(self, df, year: int = 2024):
        """
        从 pandas DataFrame 导入数据到 SQL（Excel → SQL 平滑过渡）

        期望 DataFrame 列:
        - 客户名称 (customer_name)
        - 产品 (product, optional)
        - 1月-12月 或 月份列
        """
        with self.conn_mgr.get_connection() as conn:
            cursor = conn.cursor()

            # 探测列名
            cols = list(df.columns)
            name_col = None
            for c in cols:
                if "客户" in str(c) or "名称" in str(c) or "customer" in str(c).lower():
                    name_col = c
                    break
            if not name_col:
                name_col = cols[0]

            # 月份列
            month_cols = {}
            for c in cols:
                for m in range(1, 13):
                    if f"{m}月" in str(c) or str(c).strip() == str(m):
                        month_cols[m] = c
                        break

            for _, row in df.iterrows():
                customer_name = str(row[name_col]).strip()
                if not customer_name:
                    continue

                # Upsert customer
                cursor.execute(
                    "INSERT OR IGNORE INTO dim_customer (customer_name) VALUES (?)",
                    (customer_name,)
                )
                cursor.execute(
                    "SELECT customer_id FROM dim_customer WHERE customer_name = ?",
                    (customer_name,)
                )
                cid = cursor.fetchone()
                customer_id = cid[0] if isinstance(cid, (tuple, list)) else cid["customer_id"]

                # Insert monthly data
                for month, col_name in month_cols.items():
                    revenue = float(row.get(col_name, 0) or 0)
                    cursor.execute(
                        "INSERT INTO fact_sales (customer_id, year, month, revenue) VALUES (?, ?, ?, ?)",
                        (customer_id, year, month, revenue),
                    )

            conn.commit()
        logger.info(f"Imported {len(df)} rows for year {year}")

    def build_context(self, year: int = 2024, question: str = "") -> str:
        """
        构建 Agent 所需的 context_data 字符串

        替代原来 SmartDataQuery 从 DataFrame 提取数据的流程
        """
        sections = []

        # 总营收
        try:
            rev = self.query("total_revenue", [year])
            if rev:
                sections.append(f"【{year}年总营收】{rev[0].get('total_revenue', 'N/A')} 万元")
        except Exception:
            pass

        # 同比
        try:
            yoy = self.query("yoy_comparison", [year, year - 1])
            if yoy:
                sections.append(
                    f"【同比】{year}年 vs {year-1}年: "
                    f"当年{yoy[0].get('current_revenue', 'N/A')} / "
                    f"去年{yoy[0].get('previous_revenue', 'N/A')} / "
                    f"增长{yoy[0].get('yoy_pct', 'N/A')}%"
                )
        except Exception:
            pass

        # 月度趋势
        try:
            monthly = self.query("monthly_revenue", [year])
            if monthly:
                vals = [f"{r['month']}月:{r['revenue']}" for r in monthly]
                sections.append(f"【月度营收】" + ", ".join(vals))
        except Exception:
            pass

        # Top客户
        try:
            top = self.query("customer_ranking", [year, 10])
            if top:
                lines = [f"{r['customer_name']}: {r['total_revenue']}万 (活跃{r['active_months']}月)"
                         for r in top]
                sections.append(f"【Top10客户】\n" + "\n".join(lines))
        except Exception:
            pass

        # 关键词敏感查询
        if any(k in question for k in ["风险", "流失", "预警"]):
            try:
                churn = self.query("customer_churn_data", [year])
                if churn:
                    lines = [f"{r['customer_name']}: [{r['monthly_revenues']}]"
                             for r in churn[:20]]
                    sections.append(f"【客户月度数据(流失分析用)】\n" + "\n".join(lines))
            except Exception:
                pass

        if any(k in question for k in ["集中", "依赖"]):
            try:
                conc = self.query("concentration", [year])
                if conc:
                    lines = [f"{r['customer_name']}: {r['revenue']}万" for r in conc[:10]]
                    sections.append(f"【客户营收排名(集中度分析用)】\n" + "\n".join(lines))
            except Exception:
                pass

        return "\n\n".join(sections) if sections else f"[{year}年数据查询无结果]"

    def get_stats(self) -> dict:
        """数据库统计信息"""
        stats = {"engine": self.config.engine}
        try:
            stats["customers"] = len(self.raw_query("SELECT COUNT(*) as c FROM dim_customer"))
            stats["products"] = len(self.raw_query("SELECT COUNT(*) as c FROM dim_product"))
            years = self.raw_query("SELECT DISTINCT year FROM fact_sales ORDER BY year")
            stats["years"] = [r["year"] for r in years]
            total = self.raw_query("SELECT COUNT(*) as c FROM fact_sales")
            stats["total_records"] = total[0]["c"] if total else 0
        except Exception as e:
            stats["error"] = str(e)
        return stats

    def close(self):
        self.conn_mgr.close_all()


# ============================================================
# Integration: multi_agent.py SmartDataQuery 替换
# ============================================================

def create_adapter_from_session(session_state: dict = None) -> Optional[DataAdapter]:
    """
    从 Streamlit session_state 创建 DataAdapter

    优先级: SQL配置 > Excel数据(自动导入SQLite)
    """
    ss = session_state or {}

    # 如果有 SQL 配置
    if ss.get("db_config"):
        config = DBConfig.from_dict(ss["db_config"])
        adapter = DataAdapter(config)
        return adapter

    # 如果有 DataFrame（Excel上传），自动导入 SQLite
    df = ss.get("data")
    if df is not None and hasattr(df, 'shape') and df.shape[0] > 0:
        config = DBConfig(engine="sqlite", sqlite_path=":memory:")
        adapter = DataAdapter(config)
        adapter.init_schema()
        adapter.import_from_dataframe(df)
        return adapter

    return None


# ============================================================
# CLI Test
# ============================================================

def main():
    """测试 SQL 数据层"""
    import tempfile

    print("🧪 Testing SQL Data Layer...\n")

    # 创建临时 SQLite
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    config = DBConfig(engine="sqlite", sqlite_path=db_path)
    adapter = DataAdapter(config)
    adapter.init_schema()
    print("✅ Schema created")

    # 模拟数据插入
    with adapter.conn_mgr.get_connection() as conn:
        cursor = conn.cursor()
        # 客户
        customers = ["华为", "小米", "OPPO", "vivo", "传音"]
        for name in customers:
            cursor.execute("INSERT INTO dim_customer (customer_name, region) VALUES (?, ?)",
                          (name, "华南" if name in ("华为", "OPPO") else "华东"))

        # 月度数据
        import random
        random.seed(42)
        for cid in range(1, 6):
            base = random.randint(500, 3000)
            for month in range(1, 13):
                rev = base + random.randint(-200, 300)
                cursor.execute("INSERT INTO fact_sales (customer_id, year, month, revenue) VALUES (?,?,?,?)",
                              (cid, 2024, month, rev))
                # 去年数据 (偏低)
                cursor.execute("INSERT INTO fact_sales (customer_id, year, month, revenue) VALUES (?,?,?,?)",
                              (cid, 2023, month, rev * 0.7))
        conn.commit()
    print("✅ Test data inserted")

    # 测试查询
    rev = adapter.query("total_revenue", [2024])
    print(f"✅ Total revenue 2024: {rev}")

    yoy = adapter.query("yoy_comparison", [2024, 2023])
    print(f"✅ YoY: {yoy}")

    ranking = adapter.query("customer_ranking", [2024, 5])
    print(f"✅ Top 5 customers: {[r['customer_name'] for r in ranking]}")

    # 测试 context 构建
    ctx = adapter.build_context(2024, "今年营收怎么样？有没有流失风险？")
    print(f"✅ Context built ({len(ctx)} chars):")
    print(ctx[:500])

    stats = adapter.get_stats()
    print(f"✅ Stats: {stats}")

    adapter.close()
    os.unlink(db_path)
    print("\n✅ All SQL tests passed!")


if __name__ == "__main__":
    main()
