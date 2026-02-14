-- ============================================================
-- MRARFAI V10.0 — PostgreSQL 初始化脚本
-- ============================================================
-- 用于 docker-compose 自动初始化
-- 挂载到: /docker-entrypoint-initdb.d/01_schema.sql

-- 客户维度表
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id     SERIAL PRIMARY KEY,
    customer_name   TEXT NOT NULL,
    region          TEXT DEFAULT '',
    industry        TEXT DEFAULT 'ODM',
    tier            TEXT DEFAULT '',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 产品维度表
CREATE TABLE IF NOT EXISTS dim_product (
    product_id      SERIAL PRIMARY KEY,
    product_name    TEXT NOT NULL,
    category        TEXT DEFAULT '',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 销售事实表 (月度粒度)
CREATE TABLE IF NOT EXISTS fact_sales (
    sale_id         SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES dim_customer(customer_id),
    product_id      INTEGER REFERENCES dim_product(product_id),
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    revenue         DOUBLE PRECISION DEFAULT 0,
    units           INTEGER DEFAULT 0,
    cost            DOUBLE PRECISION DEFAULT 0
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_sales_year_month ON fact_sales(year, month);
CREATE INDEX IF NOT EXISTS idx_sales_customer ON fact_sales(customer_id);

-- ============================================================
-- 观测性数据表 (原 observability.db)
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_traces (
    trace_id        TEXT PRIMARY KEY,
    session_id      TEXT DEFAULT '',
    agent_id        TEXT NOT NULL,
    question        TEXT DEFAULT '',
    answer          TEXT DEFAULT '',
    elapsed_ms      DOUBLE PRECISION DEFAULT 0,
    status          TEXT DEFAULT 'completed',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_metrics (
    id              SERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    DOUBLE PRECISION DEFAULT 0,
    tags            JSONB DEFAULT '{}',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_traces_agent ON agent_traces(agent_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON agent_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_agent ON agent_metrics(agent_id);

-- ============================================================
-- 记忆持久化表 (原 memory.db)
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_memory (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    agent_id        TEXT DEFAULT '',
    memory_type     TEXT DEFAULT 'conversation',
    content         TEXT DEFAULT '',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memory_session ON agent_memory(session_id);

-- ============================================================
-- 审计日志表
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT DEFAULT '',
    action          TEXT NOT NULL,
    detail          TEXT DEFAULT '',
    ip_address      TEXT DEFAULT '',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);
