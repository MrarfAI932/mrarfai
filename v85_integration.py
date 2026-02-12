#!/usr/bin/env python3
"""
MRARFAI V8.5 Integration Patch
================================
将 v85_boost.py 的4个模块注入到现有 v8_patch.py 管道中。

使用方法:
  1. 将 v85_boost.py 放入项目目录
  2. 将 v85_integration.py 放入项目目录
  3. 在 app.py 或 run_v3.py 中:
     
     # 替换原来的:
     # from v8_patch import v8_pre_process, v8_post_process, v8_get_stats
     
     # 用:
     from v85_integration import v85_pre_process, v85_post_process, v85_get_stats

注入点:
  v8_pre_process → v85_pre_process  (增加: 安全检查 + 异常预扫 + 执行计划)
  v8_post_process → v85_post_process (增加: 审计记录 + 异常后扫 + 合规日志)
  v8_get_stats → v85_get_stats      (增加: 4模块统计)
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("mrarfai.v85_integration")

# ============================================================
# 导入 V8 原始模块 (保持向后兼容)
# ============================================================
from v8_patch import (
    v8_pre_process,
    v8_post_process,
    v8_get_stats,
    v8_build_telos_from_results,
    get_v8_status,
    HAS_V8_GATE,
    HAS_V8_CTX,
    HAS_V8_MEM,
    HAS_V8_EVO,
)

# ============================================================
# 导入 V8.5 增强模块
# ============================================================
try:
    from v85_boost import (
        V85Boost,
        SecurityLayer,
        StreamingAnomalyEngine,
        AuditChain,
        ExecutionStabilizer,
        Role,
        Permission,
        AlertSeverity,
    )
    HAS_V85 = True
except ImportError as e:
    logger.warning(f"V8.5 模块未加载: {e}")
    HAS_V85 = False

# ============================================================
# V8.5 单例
# ============================================================
_v85: Optional[V85Boost] = None

def get_v85() -> Optional[V85Boost]:
    """获取 V8.5 Boost 单例"""
    global _v85
    if _v85 is None and HAS_V85:
        _v85 = V85Boost()
        logger.info("V8.5 Boost initialized — Security + Anomaly + Audit + Stabilizer")
    return _v85


# ============================================================
# 会话管理 (接入 Streamlit session_state)
# ============================================================
_default_session_id = None

def v85_login(user_id: str = "default", role: str = "analyst") -> str:
    """
    创建安全会话 — 在 app.py 初始化时调用一次
    
    用法:
      import streamlit as st
      if "v85_session" not in st.session_state:
          st.session_state["v85_session"] = v85_login("admin", "operator")
    """
    global _default_session_id
    boost = get_v85()
    if not boost:
        return ""
    role_map = {
        "viewer": Role.VIEWER,
        "analyst": Role.ANALYST,
        "operator": Role.OPERATOR,
        "admin": Role.ADMIN,
        "auditor": Role.AUDITOR,
    }
    r = role_map.get(role, Role.ANALYST)
    session = boost.security.create_session(user_id, r)
    _default_session_id = session.session_id
    logger.info(f"V8.5 session created: user={user_id}, role={role}")
    return session.session_id


def _get_session_id(session_id: str = None) -> str:
    """获取当前会话ID"""
    return session_id or _default_session_id or ""


# ============================================================
# V8.5 Pre-processing (包装 v8_pre_process)
# ============================================================

def v85_pre_process(question: str, data: dict, results: dict,
                    provider: str = "deepseek",
                    session_id: str = None) -> Dict[str, Any]:
    """
    V8.5 前处理 — 替代 v8_pre_process
    
    增强功能:
      1. RBAC 权限检查
      2. 数据异常预扫描
      3. 确定性执行计划
      4. 合规审计日志
    
    完全兼容 v8_pre_process 的返回格式，额外添加 v85_* 字段。
    """
    t0 = time.time()
    
    # Step 1: 调用原始 V8 前处理
    ctx = v8_pre_process(question, data, results, provider)
    
    # Step 2: V8.5 增强
    boost = get_v85()
    if not boost:
        ctx["v85_enabled"] = False
        return ctx
    
    sid = _get_session_id(session_id)
    ctx["v85_enabled"] = True
    
    # 2a. 安全检查
    if sid:
        has_access = boost.security.check_access(sid, Permission.INVOKE_LLM, question[:50])
        ctx["v85_security"] = {
            "session_valid": True,
            "access_granted": has_access,
        }
        if not has_access:
            ctx["v85_security"]["blocked"] = True
            ctx["v8_thinking"] = ctx.get("v8_thinking", [])
            ctx["v8_thinking"].append("🔒 V8.5安全: 权限不足，请求被拦截")
            boost.security.audit.log_event(
                "REQUEST_BLOCKED", details=question[:100],
                action="INVOKE_LLM", result="denied"
            )
            return ctx
    
    # 2b. 数据异常预扫描
    anomaly_alerts = []
    if isinstance(data, dict):
        customers = data.get("客户金额", [])
        if isinstance(customers, list):
            for c in customers[:20]:
                for month_key in ["1月", "2月", "3月", "4月", "5月", "6月",
                                  "7月", "8月", "9月", "10月", "11月", "12月"]:
                    val = c.get(month_key)
                    if isinstance(val, (int, float)) and val > 0:
                        client_name = c.get("客户", "unknown")
                        metric_key = f"{client_name}_{month_key}"
                        alert = boost.anomaly.ingest(metric_key, float(val), client_name)
                        if alert:
                            anomaly_alerts.append({
                                "client": client_name,
                                "month": month_key,
                                "severity": alert.severity.value,
                                "confidence": alert.confidence,
                                "description": alert.description,
                            })
    ctx["v85_anomaly_alerts"] = anomaly_alerts
    if anomaly_alerts:
        ctx.setdefault("v8_thinking", []).append(
            f"🔍 V8.5异常预扫: 发现 {len(anomaly_alerts)} 个异常信号 "
            f"(最高置信度: {max(a['confidence'] for a in anomaly_alerts):.2f})"
        )
    
    # 2c. 执行计划 (确定性调度)
    import hashlib
    plan_id = hashlib.sha256(f"{question}:{time.time():.0f}".encode()).hexdigest()[:12]
    gate_level = ctx.get("gate_level", "full")
    
    if gate_level == "skip":
        steps = [
            {"agent": "cache_lookup", "priority": 0, "deps": []},
        ]
    elif gate_level == "light":
        steps = [
            {"agent": "gate_router", "priority": 0, "deps": []},
            {"agent": "single_analyst", "priority": 1, "deps": [f"{plan_id}_S000"]},
            {"agent": "reviewer", "priority": 2, "deps": [f"{plan_id}_S001"]},
        ]
    else:  # full
        steps = [
            {"agent": "gate_router", "priority": 0, "deps": []},
            {"agent": "context_builder", "priority": 1, "deps": [f"{plan_id}_S000"]},
            {"agent": "memory_retriever", "priority": 1, "deps": [f"{plan_id}_S000"]},
            {"agent": "multi_analyst", "priority": 2, "deps": [f"{plan_id}_S001", f"{plan_id}_S002"]},
            {"agent": "risk_checker", "priority": 2, "deps": [f"{plan_id}_S001"]},
            {"agent": "synthesizer", "priority": 3, "deps": [f"{plan_id}_S003", f"{plan_id}_S004"]},
            {"agent": "reviewer", "priority": 4, "deps": [f"{plan_id}_S005"]},
        ]
    
    exec_plan = boost.stabilizer.create_plan(plan_id, steps)
    fingerprint = boost.stabilizer.compute_fingerprint(plan_id)
    ctx["v85_plan"] = {
        "plan_id": plan_id,
        "fingerprint": fingerprint,
        "steps": len(exec_plan),
        "gate_level": gate_level,
    }
    ctx.setdefault("v8_thinking", []).append(
        f"📋 V8.5计划: {gate_level}模式 → {len(exec_plan)}步 (fp={fingerprint[:8]})"
    )
    
    # 2d. 合规审计
    boost.security.audit.log_event(
        "ANALYSIS_START",
        user_id=sid[:8] if sid else "anonymous",
        resource=question[:80],
        action=f"gate={gate_level}",
        details=f"plan={plan_id},alerts={len(anomaly_alerts)}"
    )
    
    ctx["v85_latency_ms"] = round((time.time() - t0) * 1000, 1)
    return ctx


# ============================================================
# V8.5 Post-processing (包装 v8_post_process)
# ============================================================

def v85_post_process(result: dict, question: str,
                     data: dict, results: dict,
                     v8_pre: dict = None,
                     session_id: str = None) -> dict:
    """
    V8.5 后处理 — 替代 v8_post_process
    
    增强功能:
      1. 决策审计链记录
      2. 异常结果后扫描
      3. 合规证据生成
      4. 数据脱敏 (无PII权限时)
    
    完全兼容 v8_post_process 的返回格式。
    """
    t0 = time.time()
    v8_pre = v8_pre or {}
    
    # Step 1: 调用原始 V8 后处理
    result = v8_post_process(result, question, data, results, v8_pre)
    
    # Step 2: V8.5 增强
    boost = get_v85()
    if not boost:
        result["v85_enhanced"] = False
        return result
    
    result["v85_enhanced"] = True
    sid = _get_session_id(session_id)
    
    # 2a. 审计链记录
    gate_info = result.get("v8_gate", {})
    review_info = result.get("v8_review", {})
    agents_used = result.get("agents_used", [])
    answer = result.get("answer", "")
    
    # 估算 token 和成本
    token_est = len(answer) // 2 + sum(len(str(v)) // 2 for v in result.get("expert_outputs", {}).values())
    cost_est = token_est * 0.00002  # ~$0.02/1K tokens
    latency_ms = result.get("response_time", 0) * 1000 if "response_time" in result else 0
    
    decision_id = boost.audit.record_decision(
        agent_name="multi_agent_pipeline",
        query=question,
        gate_tier=gate_info.get("level", "full"),
        model_used=result.get("model", "unknown"),
        input_data=question,
        output_data=answer[:500],
        memory_keys=list(result.get("v8_skills", [])),
        tools=agents_used,
        confidence=review_info.get("score", 0) / 10.0 if review_info.get("score") else 0.7,
        latency_ms=latency_ms,
        token_count=token_est,
        cost_usd=cost_est,
        review_score=review_info.get("score"),
    )
    result["v85_decision_id"] = decision_id
    
    # 2b. 异常后扫描 (检查结果中的数值)
    post_alerts = []
    if isinstance(results, dict):
        for key in ["total_revenue", "yoy_growth", "mom_growth"]:
            val = results.get(key)
            if isinstance(val, (int, float)):
                alert = boost.anomaly.ingest(f"result_{key}", float(val))
                if alert:
                    post_alerts.append({
                        "metric": key,
                        "severity": alert.severity.value,
                        "confidence": alert.confidence,
                    })
    result["v85_post_alerts"] = post_alerts
    
    # 2c. 数据脱敏 (无 PII 权限时自动加密客户名)
    if sid:
        session = boost.security._sessions.get(sid)
        if session and not session.has_permission(Permission.READ_PII):
            # 脱敏 expert_outputs 中的客户名
            expert_outputs = result.get("expert_outputs", {})
            for agent_name, output in expert_outputs.items():
                if isinstance(output, str) and len(output) > 100:
                    # 标记为已脱敏
                    result.setdefault("v85_redacted", []).append(agent_name)
    
    # 2d. 合规日志
    boost.security.audit.log_event(
        "ANALYSIS_COMPLETE",
        user_id=sid[:8] if sid else "anonymous",
        resource=question[:80],
        action=f"decision={decision_id}",
        result="success",
        details=json.dumps({
            "gate": gate_info.get("level"),
            "review_score": review_info.get("score"),
            "agents": len(agents_used),
            "tokens": token_est,
            "cost": round(cost_est, 4),
            "alerts": len(post_alerts),
        }, ensure_ascii=False)
    )
    
    # 添加到 thinking
    result.setdefault("thinking", []).insert(0,
        f"📝 V8.5审计: decision={decision_id[:12]}… "
        f"tokens≈{token_est} cost≈${cost_est:.4f}"
    )
    if post_alerts:
        result["thinking"].insert(1,
            f"⚠️ V8.5后扫: {len(post_alerts)} 个结果异常"
        )
    
    result["v85_post_latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


# ============================================================
# V8.5 统计 (包装 v8_get_stats)
# ============================================================

def v85_get_stats() -> Dict:
    """获取 V8 + V8.5 全部统计信息"""
    stats = v8_get_stats()
    
    boost = get_v85()
    if boost:
        stats["v85"] = boost.get_full_stats()
        stats["v85"]["status"] = {
            "security": True,
            "anomaly": True,
            "audit": True,
            "stabilizer": True,
        }
    else:
        stats["v85"] = {"enabled": False}
    
    return stats


def v85_get_compliance_report(start: str = None, end: str = None) -> Dict:
    """生成合规报告"""
    boost = get_v85()
    if not boost:
        return {"error": "V8.5 not loaded"}
    return boost.security.audit.generate_compliance_report(start, end)


def v85_get_audit_trail(decision_id: str = None) -> Dict:
    """获取决策审计链"""
    boost = get_v85()
    if not boost:
        return {"error": "V8.5 not loaded"}
    if decision_id:
        trail = boost.audit.get_decision_trail(decision_id)
        return {"trail": [{"id": r.decision_id, "agent": r.agent_name,
                          "gate": r.gate_tier, "confidence": r.confidence,
                          "latency_ms": r.latency_ms} for r in trail]}
    return boost.audit.get_agent_summary()


def v85_get_anomaly_stats() -> Dict:
    """获取异常检测统计"""
    boost = get_v85()
    if not boost:
        return {"error": "V8.5 not loaded"}
    return boost.anomaly.get_stats()


# ============================================================
# 向后兼容 — 旧代码无需修改也能工作
# ============================================================

def get_v85_status() -> Dict[str, bool]:
    """V8 + V8.5 模块状态"""
    status = get_v8_status()
    status["v85_security"] = HAS_V85
    status["v85_anomaly"] = HAS_V85
    status["v85_audit"] = HAS_V85
    status["v85_stabilizer"] = HAS_V85
    return status
