#!/usr/bin/env python3
"""
MRARFAI — 创建 Demo SQLite 数据库
=====================================
生成一个包含样本数据的 SQLite 数据库，用于测试 DB 连接功能。

用法:
    python create_demo_db.py
    # 生成 demo_data.db
    # 然后在 .env 中设置:
    #   DB_TYPE=sqlite
    #   DB_PATH=demo_data.db
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data.db")


def create_demo_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── 供应商表 ──
    c.execute("""CREATE TABLE suppliers (
        name TEXT, category TEXT, lead_time_days INTEGER,
        quality_score REAL, price_index REAL, on_time_rate REAL,
        defect_rate REAL, credit_rating TEXT
    )""")
    suppliers = [
        ("华星光电", "屏幕", 7, 9.5, 0.95, 0.98, 0.008, "A"),
        ("京东方", "屏幕", 10, 9.2, 0.92, 0.95, 0.012, "A"),
        ("汇顶科技", "芯片", 14, 8.8, 1.05, 0.90, 0.015, "B+"),
        ("紫光展锐", "芯片", 21, 8.5, 0.88, 0.85, 0.020, "B"),
        ("欣旺达", "电池", 5, 9.0, 0.98, 0.96, 0.010, "A-"),
        ("比亚迪电子", "结构件", 10, 9.3, 0.90, 0.93, 0.008, "A"),
        ("蓝思科技", "盖板", 8, 8.9, 1.02, 0.91, 0.013, "B+"),
        ("立讯精密", "连接器", 6, 9.1, 0.97, 0.97, 0.006, "A"),
    ]
    c.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?,?,?,?)", suppliers)

    # ── 采购订单表 ──
    c.execute("""CREATE TABLE orders (
        po_id TEXT, supplier TEXT, material TEXT, quantity INTEGER,
        total_amount REAL, status TEXT, created_date TEXT, expected_date TEXT
    )""")
    orders = [
        ("PO-2026-001", "华星光电", "6.5寸LCD", 50000, 850, "delivered", "2026-01-05", "2026-01-12"),
        ("PO-2026-002", "紫光展锐", "T618芯片", 30000, 420, "shipped", "2026-01-08", "2026-01-22"),
        ("PO-2026-003", "欣旺达", "5000mAh电池", 40000, 320, "pending", "2026-01-15", "2026-01-20"),
        ("PO-2026-004", "比亚迪电子", "中框", 25000, 275, "delivered", "2026-01-03", "2026-01-13"),
        ("PO-2026-005", "蓝思科技", "玻璃盖板", 35000, 490, "delayed", "2026-01-10", "2026-01-18"),
        ("PO-2026-006", "汇顶科技", "指纹模组", 20000, 180, "pending", "2026-01-20", "2026-02-03"),
        ("PO-2026-007", "京东方", "6.7寸OLED", 15000, 675, "shipped", "2026-01-12", "2026-01-22"),
        ("PO-2026-008", "立讯精密", "Type-C接口", 60000, 96, "delivered", "2026-01-02", "2026-01-08"),
    ]
    c.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", orders)

    # ── 应收账款表 ──
    c.execute("""CREATE TABLE accounts_receivable (
        customer TEXT, invoice_no TEXT, amount REAL, currency TEXT,
        due_date TEXT, status TEXT, overdue_days INTEGER
    )""")
    ar = [
        ("三星印度", "INV-2026-001", 520, "USD", "2026-02-15", "outstanding", 0),
        ("Transsion", "INV-2026-002", 380, "USD", "2026-01-20", "overdue", 24),
        ("小米", "INV-2026-003", 890, "RMB", "2026-02-28", "outstanding", 0),
        ("OPPO", "INV-2026-004", 650, "RMB", "2026-01-10", "overdue", 34),
        ("Nokia", "INV-2026-005", 210, "EUR", "2026-03-01", "outstanding", 0),
        ("中兴", "INV-2026-006", 430, "RMB", "2026-01-25", "overdue", 19),
    ]
    c.executemany("INSERT INTO accounts_receivable VALUES (?,?,?,?,?,?,?)", ar)

    # ── 毛利表 ──
    c.execute("""CREATE TABLE margins (
        product TEXT, customer TEXT, revenue REAL, cost REAL
    )""")
    margins = [
        ("A68 Pro", "三星印度", 2800, 2380),
        ("A68 Pro", "Transsion", 2650, 2380),
        ("S92 Lite", "小米", 1580, 1420),
        ("S92 Lite", "OPPO", 1520, 1420),
        ("X15 Max", "Nokia", 3200, 2720),
        ("X15 Max", "中兴", 3050, 2720),
    ]
    c.executemany("INSERT INTO margins VALUES (?,?,?,?)", margins)

    # ── 良率表 ──
    c.execute("""CREATE TABLE yields (
        line TEXT, product TEXT, month TEXT, total INTEGER,
        passed INTEGER, defect_cosmetic INTEGER, defect_functional INTEGER
    )""")
    yields = [
        ("L1", "A68 Pro", "2026-01", 50000, 48500, 800, 700),
        ("L1", "A68 Pro", "2025-12", 48000, 46200, 1000, 800),
        ("L2", "S92 Lite", "2026-01", 35000, 34300, 400, 300),
        ("L2", "S92 Lite", "2025-12", 33000, 32100, 500, 400),
        ("L3", "X15 Max", "2026-01", 20000, 19400, 350, 250),
        ("L3", "X15 Max", "2025-12", 18000, 17280, 420, 300),
    ]
    c.executemany("INSERT INTO yields VALUES (?,?,?,?,?,?,?)", yields)

    # ── 退货表 ──
    c.execute("""CREATE TABLE returns (
        customer TEXT, product TEXT, quantity INTEGER,
        reason TEXT, date TEXT, severity TEXT
    )""")
    returns = [
        ("三星印度", "A68 Pro", 120, "屏幕漏光", "2026-01-18", "高"),
        ("Transsion", "A68 Pro", 85, "电池鼓包", "2026-01-22", "严重"),
        ("小米", "S92 Lite", 45, "WiFi断连", "2026-01-25", "中"),
        ("OPPO", "S92 Lite", 30, "外观划痕", "2026-01-20", "低"),
        ("Nokia", "X15 Max", 15, "按键失灵", "2026-01-28", "中"),
    ]
    c.executemany("INSERT INTO returns VALUES (?,?,?,?,?,?)", returns)

    # ── 竞品表 ──
    c.execute("""CREATE TABLE competitors (
        company TEXT, ticker TEXT, revenue_billion REAL,
        growth_pct REAL, main_clients TEXT, strengths TEXT,
        weaknesses TEXT, market_share_pct REAL
    )""")
    competitors = [
        ("华勤技术", "603296.SH", 580, 12.5, "三星,小米,OPPO", "规模大,研发强", "成本高", 18.5),
        ("闻泰科技", "600745.SH", 520, 8.2, "三星,联想,小米", "半导体协同,产能大", "利润薄", 16.8),
        ("龙旗科技", "300726.SZ", 280, 15.0, "小米,荣耀,realme", "性价比高,效率高", "品牌少", 9.2),
        ("禾苗通讯", "未上市", 45, 22.0, "三星,Nokia,Transsion", "灵活,非洲市场", "规模小", 1.5),
    ]
    c.executemany("INSERT INTO competitors VALUES (?,?,?,?,?,?,?,?)", competitors)

    conn.commit()
    conn.close()
    print(f"✅ Demo 数据库已创建: {DB_PATH}")
    print("   供应商: 8 条 | 订单: 8 条 | 应收: 6 条 | 毛利: 6 条")
    print("   良率: 6 条 | 退货: 5 条 | 竞品: 4 条")
    print(f"\n📝 配置 .env:")
    print(f"   DB_TYPE=sqlite")
    print(f"   DB_PATH={DB_PATH}")


if __name__ == "__main__":
    create_demo_db()
