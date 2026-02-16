#!/usr/bin/env python3
"""
MRARFAI — 认证模块
====================
简单的用户名+密码登录，支持多用户和角色。
密码使用 SHA-256 哈希存储，不明文保存。

用法:
    from auth import require_login, get_current_user, logout

    # 在 app.py 最前面调用
    require_login()   # 未登录会显示登录页，阻止后续代码
    user = get_current_user()
    print(user["role"])  # "admin" / "viewer"
"""

import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict

# ============================================================
# 用户数据库（生产环境应换成真数据库）
# ============================================================

def _hash_pw(password: str) -> str:
    """SHA-256 哈希密码"""
    return hashlib.sha256(password.encode()).hexdigest()

# ============================================================
# 角色 → Agent 权限映射
# ============================================================
ROLE_PERMISSIONS = {
    "admin": {
        "agents": ["sales", "procurement", "quality", "finance", "market", "risk", "strategist"],
        "collab": True,
        "upload": True,
        "export": True,
        "label": "管理员",
    },
    "sales_manager": {
        "agents": ["sales", "risk", "market"],
        "collab": True,
        "upload": True,
        "export": True,
        "label": "销售经理",
    },
    "procurement_manager": {
        "agents": ["procurement", "quality", "finance"],
        "collab": True,
        "upload": True,
        "export": True,
        "label": "采购经理",
    },
    "quality_manager": {
        "agents": ["quality"],
        "collab": False,
        "upload": True,
        "export": True,
        "label": "品质经理",
    },
    "finance_manager": {
        "agents": ["finance"],
        "collab": False,
        "upload": True,
        "export": True,
        "label": "财务经理",
    },
    "viewer": {
        "agents": ["sales", "market"],
        "collab": False,
        "upload": False,
        "export": False,
        "label": "只读访客",
    },
}

def get_role_permissions(role: str) -> dict:
    """获取角色的权限配置"""
    return ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["viewer"])

def get_allowed_agents(role: str) -> list:
    """获取角色可访问的 Agent 列表"""
    return get_role_permissions(role).get("agents", [])

def can_access_agent(role: str, agent_name: str) -> bool:
    """检查角色是否可访问指定 Agent"""
    return agent_name in get_allowed_agents(role)

def can_use_collab(role: str) -> bool:
    """检查角色是否可使用跨 Agent 协作"""
    return get_role_permissions(role).get("collab", False)

def can_upload(role: str) -> bool:
    """检查角色是否可上传数据"""
    return get_role_permissions(role).get("upload", False)

def can_export(role: str) -> bool:
    """检查角色是否可导出报告"""
    return get_role_permissions(role).get("export", False)


# 默认用户 — 可通过 users.json 覆盖
DEFAULT_USERS = {
    "admin": {
        "password_hash": _hash_pw("mrarfai2025"),
        "role": "admin",
        "display_name": "管理员",
        "company": "MRARFAI",
    },
    "sprocomm": {
        "password_hash": _hash_pw("sprocomm888"),
        "role": "admin",
        "display_name": "禾苗通讯",
        "company": "Sprocomm",
    },
    "sales": {
        "password_hash": _hash_pw("sales123"),
        "role": "sales_manager",
        "display_name": "销售部",
        "company": "Sprocomm",
    },
    "procurement": {
        "password_hash": _hash_pw("proc123"),
        "role": "procurement_manager",
        "display_name": "采购部",
        "company": "Sprocomm",
    },
    "quality": {
        "password_hash": _hash_pw("quality123"),
        "role": "quality_manager",
        "display_name": "品质部",
        "company": "Sprocomm",
    },
    "finance": {
        "password_hash": _hash_pw("finance123"),
        "role": "finance_manager",
        "display_name": "财务部",
        "company": "Sprocomm",
    },
    "viewer": {
        "password_hash": _hash_pw("view123"),
        "role": "viewer",
        "display_name": "访客",
        "company": "Guest",
    },
}

def _load_users() -> dict:
    """加载用户数据库 — 优先从 users.json 读取"""
    users_file = os.path.join(os.path.dirname(__file__), "users.json")
    if os.path.exists(users_file):
        try:
            with open(users_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_USERS


def _save_users(users: dict):
    """保存用户数据库"""
    users_file = os.path.join(os.path.dirname(__file__), "users.json")
    with open(users_file, "w") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


# ============================================================
# 认证逻辑
# ============================================================

def authenticate(username: str, password: str) -> Optional[Dict]:
    """验证用户名+密码，返回用户信息或 None"""
    users = _load_users()
    user = users.get(username.lower().strip())
    if not user:
        return None
    if user["password_hash"] == _hash_pw(password):
        return {
            "username": username.lower().strip(),
            "role": user["role"],
            "display_name": user["display_name"],
            "company": user.get("company", ""),
            "login_time": datetime.now().isoformat(),
        }
    return None


# Session 超时配置 (默认 4 小时)
SESSION_TIMEOUT_HOURS = int(os.environ.get("SESSION_TIMEOUT_HOURS", "4") or "4")


def is_logged_in() -> bool:
    """是否已登录"""
    return st.session_state.get("auth_user") is not None


def check_session_timeout() -> bool:
    """
    检查 session 是否已超时。
    返回 True 表示已超时（需要重新登录），False 表示仍有效。
    """
    if not is_logged_in():
        return False

    login_time = st.session_state.get("auth_login_time")
    last_activity = st.session_state.get("auth_last_activity")

    now = datetime.now()

    # 基于最后活跃时间检测（如果有的话），否则用登录时间
    ref_time = last_activity or login_time

    if ref_time:
        try:
            if isinstance(ref_time, str):
                ref_time = datetime.fromisoformat(ref_time)
            elapsed = (now - ref_time).total_seconds()
            if elapsed > SESSION_TIMEOUT_HOURS * 3600:
                return True
        except Exception:
            pass

    # 更新最后活跃时间
    st.session_state["auth_last_activity"] = now.isoformat()
    return False


def get_current_user() -> Optional[Dict]:
    """获取当前用户信息"""
    return st.session_state.get("auth_user")


def logout():
    """登出"""
    st.session_state.pop("auth_user", None)
    st.session_state.pop("auth_login_time", None)
    st.session_state.pop("auth_last_activity", None)


def is_admin() -> bool:
    """当前用户是否是管理员"""
    user = get_current_user()
    return user and user.get("role") == "admin"


# ============================================================
# 登录页面 UI
# ============================================================

def _render_login_page():
    """渲染登录页面 — 白色卡片 + OLED深色背景
    CSS 由 ui_theme.inject_login_theme() 提供。
    """
    from ui_theme import inject_login_theme

    # ── 注入登录页 CSS（来自设计系统 Single Source of Truth）──
    inject_login_theme()

    # ── 读取 logo base64 ──
    _login_logo_b64 = ""
    try:
        import os as _os
        _logo_path = _os.path.join(_os.path.dirname(__file__), "logo_b64.txt")
        with open(_logo_path, "r") as _lf:
            _login_logo_b64 = _lf.read().strip()
    except Exception:
        pass

    _horse_logo_html = (
        f'<img class="horse-logo" src="data:image/png;base64,{_login_logo_b64}" />'
        if _login_logo_b64
        else '<div class="horse-icon-fallback">&#x1F40E;</div>'
    )

    # ── System status indicator ──
    st.markdown("""
    <div style="position:fixed; top:20px; right:24px; z-index:100;
         display:flex; align-items:center; gap:8px;">
        <div class="pulse-dot" style="width:6px; height:6px; background:#FFFFFF;"></div>
        <span style="font-family:'Inter',sans-serif; font-size:10px;
              color:#505050; letter-spacing:0.08em; text-transform:uppercase;">
            SYSTEM ONLINE
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Logo + 品牌名 + 分隔线 ──
    st.markdown(f"""
    <div class="login-logo-area">
        {_horse_logo_html}
        <div class="brand-name">MRARFAI</div>
        <div class="brand-sub">ENTERPRISE AGENT PLATFORM</div>
    </div>
    <div class="login-divider"></div>
    """, unsafe_allow_html=True)

    # ── 表单字段 ──
    st.markdown('<div class="login-label">USERNAME</div>', unsafe_allow_html=True)
    username = st.text_input(
        "用户名", label_visibility="collapsed", key="login_user",
        placeholder="Enter username",
    )

    st.markdown('<div class="login-label">PASSWORD</div>', unsafe_allow_html=True)
    password = st.text_input(
        "密码", type="password", label_visibility="collapsed",
        key="login_pass", placeholder="Enter password",
    )

    # ── SIGN IN 按钮 ──
    login_clicked = st.button("SIGN IN", key="login_btn", use_container_width=True)

    # ── 登录逻辑 ──
    if login_clicked:
        if not username or not password:
            st.markdown(
                '<div class="login-error">Please enter username and password</div>',
                unsafe_allow_html=True,
            )
        else:
            user = authenticate(username, password)
            if user:
                st.session_state["auth_user"] = user
                st.session_state["auth_login_time"] = datetime.now().isoformat()
                st.session_state["auth_last_activity"] = datetime.now().isoformat()
                st.rerun()
            else:
                st.markdown(
                    '<div class="login-error">Invalid username or password</div>',
                    unsafe_allow_html=True,
                )

    # ── 底部链接 + 页脚 ──
    st.markdown("""
    <div style="position:fixed; bottom:48px; left:0; right:0; z-index:10; pointer-events:auto;">
        <div class="login-links-outer">
            <a href="#">Forgot Password?</a>
            <span class="sep">&middot;</span>
            <a href="#">Contact Support</a>
        </div>
        <div class="login-page-footer-outer">
            &copy; 2025 MRARFAI &middot; Powered by Multi-Agent Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)


def require_login():
    """
    在 app.py 最前面调用。未登录则显示登录页并 st.stop()。
    Session 超时自动登出。
    """
    # 检查 session 超时
    if check_session_timeout():
        logout()
        st.warning(f"Session expired ({SESSION_TIMEOUT_HOURS}h). Please login again.")

    if not is_logged_in():
        _render_login_page()
        st.stop()
