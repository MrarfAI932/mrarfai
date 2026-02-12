#!/usr/bin/env python3
"""
MRARFAI Golden Dataset v5.0
===============================
Phase 1 升级：建立质量回归测试基准

功能：
  ① 管理「问题 + 标准答案 + 标签」测试集
  ② 对接 Langfuse Dataset API（可选）
  ③ 运行回归测试 + LLM-as-Judge 自动评分
  ④ 版本对比（v4.3 vs v5.0 质量差异）

存储：SQLite 本地 + Langfuse 云端（可选双写）
"""

import json
import sqlite3
import time
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any

logger = logging.getLogger("mrarfai.golden")


# ============================================================
# 数据模型
# ============================================================

@dataclass
class GoldenCase:
    """一条黄金测试用例"""
    case_id: str                      # 唯一ID: "GC-001"
    question: str                     # 用户问题
    expected_keywords: List[str]      # 答案中必须包含的关键词
    expected_pattern: str = ""        # 答案的结构描述
    category: str = "general"         # 分类: general/risk/product/trend/region
    difficulty: str = "medium"        # easy/medium/hard
    context_hint: str = ""            # 提示：这个问题需要什么数据
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class RegressionResult:
    """回归测试结果"""
    case_id: str
    question: str
    agent_output: str
    scores: Dict                      # LLM-as-Judge 评分
    keyword_hits: int                 # 关键词命中数
    keyword_total: int                # 关键词总数
    keyword_score: float              # 关键词命中率
    overall_score: float              # 综合分数
    elapsed_ms: float = 0
    error: str = ""


@dataclass
class RegressionReport:
    """一次回归测试的完整报告"""
    version: str                      # 代码版本标签
    timestamp: str
    total_cases: int
    results: List[RegressionResult]
    avg_scores: Dict[str, float]      # 各维度平均分
    overall_avg: float
    keyword_avg: float
    elapsed_total_ms: float


# ============================================================
# 内置黄金测试集 — 覆盖核心业务场景
# ============================================================

DEFAULT_GOLDEN_CASES = [
    # --- 区域分析 ---
    GoldenCase(
        case_id="GC-001",
        question="华东区今年整体表现如何？",
        expected_keywords=["营收", "同比", "增长", "客户"],
        expected_pattern="包含总营收、增长率、主要客户、趋势判断",
        category="region",
        difficulty="easy",
    ),
    GoldenCase(
        case_id="GC-002",
        question="对比各区域的销售表现，哪个区域增长最快？",
        expected_keywords=["区域", "增长", "对比", "最快"],
        expected_pattern="包含各区域数据对比、增长率排名",
        category="region",
        difficulty="medium",
    ),
    
    # --- 客户风险 ---
    GoldenCase(
        case_id="GC-010",
        question="哪些客户有流失风险？给出具体数据支撑",
        expected_keywords=["流失", "风险", "下降", "客户"],
        expected_pattern="包含具体客户名、下降幅度、月度趋势、风险等级",
        category="risk",
        difficulty="medium",
    ),
    GoldenCase(
        case_id="GC-011",
        question="客户集中度怎么样？有什么风险？",
        expected_keywords=["集中度", "Top", "占比", "风险"],
        expected_pattern="包含Top3/Top5占比、HHI指数或类似指标、建议",
        category="risk",
        difficulty="medium",
    ),
    GoldenCase(
        case_id="GC-012",
        question="最近三个月零出货的客户有哪些？",
        expected_keywords=["零出货", "客户", "月"],
        expected_pattern="包含具体客户列表、之前的出货量对比",
        category="risk",
        difficulty="easy",
    ),
    
    # --- 产品分析 ---
    GoldenCase(
        case_id="GC-020",
        question="各产品线的营收占比和增长趋势是什么？",
        expected_keywords=["产品", "占比", "增长", "趋势"],
        expected_pattern="包含各产品线金额、占比、同比增长率",
        category="product",
        difficulty="medium",
    ),
    GoldenCase(
        case_id="GC-021",
        question="哪些产品是明星产品？哪些在萎缩？",
        expected_keywords=["明星", "增长", "萎缩", "产品"],
        expected_pattern="包含BCG分类或类似分析、具体产品线名称",
        category="product",
        difficulty="hard",
    ),
    
    # --- 趋势预测 ---
    GoldenCase(
        case_id="GC-030",
        question="今年的月度出货趋势如何？有什么规律？",
        expected_keywords=["月度", "趋势", "峰", "谷"],
        expected_pattern="包含月度数据走势、峰值月份、季节性规律",
        category="trend",
        difficulty="medium",
    ),
    GoldenCase(
        case_id="GC-031",
        question="按目前趋势，下个季度预计营收多少？",
        expected_keywords=["预测", "季度", "营收", "趋势"],
        expected_pattern="包含预测数字、预测方法说明、置信度",
        category="trend",
        difficulty="hard",
    ),
    
    # --- 综合策略 ---
    GoldenCase(
        case_id="GC-040",
        question="给我一份年度销售总结，重点是风险和机会",
        expected_keywords=["总结", "风险", "机会", "建议"],
        expected_pattern="结构化报告：业绩概览、关键风险、增长机会、行动建议",
        category="general",
        difficulty="hard",
    ),
    GoldenCase(
        case_id="GC-041",
        question="如果要提升明年营收20%，你有什么建议？",
        expected_keywords=["建议", "增长", "策略", "行动"],
        expected_pattern="包含具体策略、数据支撑、优先级排序",
        category="general",
        difficulty="hard",
    ),
]


# ============================================================
# 数据集管理器（SQLite）
# ============================================================

class GoldenDatasetManager:
    """管理黄金测试集的增删改查和持久化"""
    
    def __init__(self, db_path: str = "golden_dataset.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _get_conn(self):
        return self._conn
    
    def _init_db(self):
        conn = self._get_conn()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS golden_cases (
                    case_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    expected_keywords TEXT,
                    expected_pattern TEXT,
                    category TEXT DEFAULT 'general',
                    difficulty TEXT DEFAULT 'medium',
                    context_hint TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regression_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT,
                    timestamp TEXT,
                    total_cases INTEGER,
                    overall_avg REAL,
                    keyword_avg REAL,
                    scores_json TEXT,
                    elapsed_ms REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regression_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    case_id TEXT,
                    agent_output TEXT,
                    scores_json TEXT,
                    keyword_score REAL,
                    overall_score REAL,
                    error TEXT,
                    FOREIGN KEY (run_id) REFERENCES regression_runs(run_id)
                )
            """)
    
    def load_defaults(self):
        """加载内置测试用例（不覆盖已存在的）"""
        count = 0
        for case in DEFAULT_GOLDEN_CASES:
            if not self.get_case(case.case_id):
                self.add_case(case)
                count += 1
        return count
    
    def add_case(self, case: GoldenCase):
        """添加测试用例"""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO golden_cases 
                (case_id, question, expected_keywords, expected_pattern,
                 category, difficulty, context_hint, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case.case_id, case.question,
                json.dumps(case.expected_keywords, ensure_ascii=False),
                case.expected_pattern, case.category,
                case.difficulty, case.context_hint, case.created_at,
            ))
    
    def get_case(self, case_id: str) -> Optional[GoldenCase]:
        conn = self._get_conn()
        with conn:
            row = conn.execute(
                "SELECT * FROM golden_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row:
                return self._row_to_case(row)
        return None
    
    def list_cases(self, category: str = None) -> List[GoldenCase]:
        conn = self._get_conn()
        with conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM golden_cases WHERE category = ? ORDER BY case_id",
                    (category,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM golden_cases ORDER BY case_id"
                ).fetchall()
        return [self._row_to_case(r) for r in rows]
    
    def count(self) -> int:
        conn = self._get_conn()
        with conn:
            return conn.execute("SELECT COUNT(*) FROM golden_cases").fetchone()[0]
    
    def delete_case(self, case_id: str):
        conn = self._get_conn()
        with conn:
            conn.execute("DELETE FROM golden_cases WHERE case_id = ?", (case_id,))
    
    def _row_to_case(self, row) -> GoldenCase:
        return GoldenCase(
            case_id=row[0],
            question=row[1],
            expected_keywords=json.loads(row[2]) if row[2] else [],
            expected_pattern=row[3] or "",
            category=row[4] or "general",
            difficulty=row[5] or "medium",
            context_hint=row[6] or "",
            created_at=row[7] or "",
        )
    
    # ---- 回归测试记录 ----
    
    def save_regression_run(self, report: RegressionReport) -> int:
        """保存一次回归测试结果"""
        conn = self._get_conn()
        with conn:
            cursor = conn.execute("""
                INSERT INTO regression_runs 
                (version, timestamp, total_cases, overall_avg, keyword_avg, scores_json, elapsed_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                report.version, report.timestamp, report.total_cases,
                report.overall_avg, report.keyword_avg,
                json.dumps(report.avg_scores, ensure_ascii=False),
                report.elapsed_total_ms,
            ))
            run_id = cursor.lastrowid
            
            for r in report.results:
                conn.execute("""
                    INSERT INTO regression_details 
                    (run_id, case_id, agent_output, scores_json, keyword_score, overall_score, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, r.case_id, r.agent_output[:500],
                    json.dumps(r.scores, ensure_ascii=False),
                    r.keyword_score, r.overall_score, r.error,
                ))
            
            return run_id
    
    def get_regression_history(self, limit: int = 10) -> List[Dict]:
        """获取历史回归测试记录（用于版本对比）"""
        conn = self._get_conn()
        with conn:
            rows = conn.execute("""
                SELECT version, timestamp, total_cases, overall_avg, keyword_avg, scores_json, elapsed_ms
                FROM regression_runs
                ORDER BY run_id DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        return [{
            "version": r[0],
            "timestamp": r[1],
            "total_cases": r[2],
            "overall_avg": r[3],
            "keyword_avg": r[4],
            "avg_scores": json.loads(r[5]) if r[5] else {},
            "elapsed_ms": r[6],
        } for r in rows]


# ============================================================
# 回归测试执行器
# ============================================================

def run_regression(
    agent_fn: Callable,
    version_tag: str = "dev",
    categories: List[str] = None,
    provider: str = "claude",
    api_key: str = "",
    db_path: str = "golden_dataset.db",
    use_judge: bool = True,
) -> RegressionReport:
    """
    运行完整的回归测试
    
    参数:
        agent_fn: Agent 调用函数，签名 fn(question) -> str
        version_tag: 版本标签，如 "v4.3", "v5.0-langfuse"
        categories: 只测试某些分类，None=全部
        provider: LLM-as-Judge 用的 provider
        api_key: API Key
        db_path: 数据库路径
        use_judge: 是否使用 LLM-as-Judge（False 时只做关键词匹配）
    
    返回:
        RegressionReport
    """
    t0 = time.time()
    
    # 加载测试集
    mgr = GoldenDatasetManager(db_path)
    if mgr.count() == 0:
        mgr.load_defaults()
    
    cases = mgr.list_cases()
    if categories:
        cases = [c for c in cases if c.category in categories]
    
    if not cases:
        return RegressionReport(
            version=version_tag,
            timestamp=datetime.now().isoformat(),
            total_cases=0, results=[], avg_scores={},
            overall_avg=0, keyword_avg=0, elapsed_total_ms=0,
        )
    
    # 导入 judge（如果需要）
    judge_fn = None
    if use_judge:
        try:
            from llm_judge import judge_output
            judge_fn = judge_output
        except ImportError:
            logger.warning("llm_judge.py 未找到，仅使用关键词匹配")
    
    # 执行测试
    results = []
    dim_totals = {}
    kw_totals = []
    
    for i, case in enumerate(cases):
        logger.info(f"[{i+1}/{len(cases)}] {case.case_id}: {case.question[:40]}...")
        case_t0 = time.time()
        
        # 调用 Agent
        try:
            output = agent_fn(case.question)
            if not isinstance(output, str):
                output = str(output)
        except Exception as e:
            results.append(RegressionResult(
                case_id=case.case_id,
                question=case.question,
                agent_output="",
                scores={},
                keyword_hits=0,
                keyword_total=len(case.expected_keywords),
                keyword_score=0,
                overall_score=0,
                error=str(e),
            ))
            continue
        
        # 关键词匹配
        hits = sum(1 for kw in case.expected_keywords if kw in output)
        kw_score = hits / len(case.expected_keywords) if case.expected_keywords else 1.0
        kw_totals.append(kw_score)
        
        # LLM-as-Judge 评分
        scores = {}
        if judge_fn and api_key:
            scores = judge_fn(
                question=case.question,
                output=output,
                context="",  # 回归测试不提供 context，靠 Agent 自己找数据
                provider=provider,
                api_key=api_key,
            )
            
            for dim in ["correctness", "relevance", "hallucination"]:
                if dim in scores and scores[dim].get("score", -1) >= 0:
                    if dim not in dim_totals:
                        dim_totals[dim] = []
                    dim_totals[dim].append(scores[dim]["score"])
        
        # 综合分数 = LLM 评分 * 0.7 + 关键词匹配 * 0.3
        judge_avg = scores.get("overall", kw_score)
        if judge_avg < 0:
            judge_avg = kw_score
        overall = round(judge_avg * 0.7 + kw_score * 0.3, 3)
        
        elapsed = (time.time() - case_t0) * 1000
        
        results.append(RegressionResult(
            case_id=case.case_id,
            question=case.question,
            agent_output=output[:500],
            scores=scores,
            keyword_hits=hits,
            keyword_total=len(case.expected_keywords),
            keyword_score=round(kw_score, 3),
            overall_score=overall,
            elapsed_ms=round(elapsed, 1),
        ))
    
    # 汇总
    total_elapsed = (time.time() - t0) * 1000
    avg_scores = {
        dim: round(sum(vals) / len(vals), 3)
        for dim, vals in dim_totals.items()
    }
    overall_avg = round(
        sum(r.overall_score for r in results if not r.error) / max(len([r for r in results if not r.error]), 1),
        3
    )
    keyword_avg = round(sum(kw_totals) / max(len(kw_totals), 1), 3)
    
    report = RegressionReport(
        version=version_tag,
        timestamp=datetime.now().isoformat(),
        total_cases=len(cases),
        results=results,
        avg_scores=avg_scores,
        overall_avg=overall_avg,
        keyword_avg=keyword_avg,
        elapsed_total_ms=round(total_elapsed, 1),
    )
    
    # 保存到数据库
    run_id = mgr.save_regression_run(report)
    logger.info(f"✅ 回归测试完成 (run #{run_id}): 版本={version_tag}, "
                f"总分={overall_avg:.2f}, 关键词={keyword_avg:.2f}")
    
    return report


def compare_versions(db_path: str = "golden_dataset.db", limit: int = 5) -> str:
    """对比最近几个版本的回归测试结果"""
    mgr = GoldenDatasetManager(db_path)
    history = mgr.get_regression_history(limit)
    
    if not history:
        return "暂无回归测试记录。运行 run_regression() 生成第一条。"
    
    lines = ["版本对比（最近 {} 次）:".format(len(history))]
    lines.append("-" * 70)
    lines.append(f"{'版本':<15} {'综合分':>8} {'关键词':>8} {'用例数':>6} {'耗时':>10}")
    lines.append("-" * 70)
    
    for h in history:
        lines.append(
            f"{h['version']:<15} {h['overall_avg']:>8.3f} "
            f"{h['keyword_avg']:>8.3f} {h['total_cases']:>6} "
            f"{h['elapsed_ms']:>8.0f}ms"
        )
    
    # 检测退步
    if len(history) >= 2:
        latest = history[0]
        prev = history[1]
        diff = latest["overall_avg"] - prev["overall_avg"]
        if diff < -0.05:
            lines.append(f"\n⚠️  质量下降警告: {latest['version']} 比 {prev['version']} "
                        f"降低了 {abs(diff):.3f}")
        elif diff > 0.05:
            lines.append(f"\n✅ 质量提升: {latest['version']} 比 {prev['version']} "
                        f"提升了 {diff:.3f}")
    
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("MRARFAI Golden Dataset Manager")
    print("=" * 60)
    
    mgr = GoldenDatasetManager()
    loaded = mgr.load_defaults()
    total = mgr.count()
    
    print(f"✅ 数据库: golden_dataset.db")
    print(f"   本次新加载: {loaded} 条")
    print(f"   总测试用例: {total} 条")
    
    # 按分类统计
    categories = {}
    for case in mgr.list_cases():
        categories[case.category] = categories.get(case.category, 0) + 1
    
    print(f"\n📊 分类统计:")
    for cat, count in sorted(categories.items()):
        print(f"   {cat}: {count} 条")
    
    # 显示历史对比
    print(f"\n{compare_versions()}")
    
    if "--list" in sys.argv:
        print(f"\n📋 全部测试用例:")
        for case in mgr.list_cases():
            print(f"  [{case.case_id}] ({case.category}/{case.difficulty}) {case.question}")
            print(f"         关键词: {', '.join(case.expected_keywords)}")
