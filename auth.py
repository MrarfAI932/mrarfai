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

SP_GREEN = "#00FF88"

def _render_login_page():
    """渲染登录页面"""

    # 全屏深色背景 + Command Center 动效
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── Keyframes ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 0 1px rgba(255,255,255,0.06); }
        50%      { box-shadow: 0 0 18px rgba(255,255,255,0.10), 0 0 0 1px rgba(255,255,255,0.18); }
    }
    @keyframes scanLine {
        0%   { top: -2px; opacity: 0; }
        10%  { opacity: 0.6; }
        90%  { opacity: 0.6; }
        100% { top: 100%; opacity: 0; }
    }
    @keyframes titleShimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes dotPulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(255,255,255,0.5); }
        50%      { opacity: 0.7; box-shadow: 0 0 6px 3px rgba(255,255,255,0.2); }
    }

    /* ── Base ── */
    [data-testid="stApp"] {
        background: #080808;
    }
    /* Grid background overlay */
    [data-testid="stApp"]::before {
        content: "";
        position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 60px 60px;
    }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stHeader"] { display: none; }
    #MainMenu, footer, .stDeployButton,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] { display: none !important; }

    /* ── Login Container ── */
    .login-container {
        max-width: 380px; margin: 20vh auto 0 auto; padding: 40px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(12,12,12,0.95);
        animation: fadeInUp 0.6s ease-out, glowPulse 4s ease-in-out 0.6s infinite;
        position: relative; overflow: hidden;
    }
    /* Scan line effect */
    .login-container::after {
        content: "";
        position: absolute; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
        animation: scanLine 5s ease-in-out infinite;
        pointer-events: none;
    }

    /* ── Logo ── */
    .login-logo {
        display: flex; align-items: center; gap: 12px; margin-bottom: 32px;
        position: relative; z-index: 1;
    }
    .login-logo-box {
        width: 40px; height: 40px;
        display: flex; align-items: center; justify-content: center;
    }
    .login-logo-box img {
        width: 40px; height: auto;
        filter: brightness(0) invert(1);
    }
    .login-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 1.2rem; letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #FFFFFF;
        display: flex; align-items: center; gap: 8px;
    }
    .login-title img.title-horse {
        width: 36px; height: auto;
        filter: brightness(0) invert(1);
    }
    .login-subtitle {
        font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
        color: #6a6a6a; letter-spacing: 0.1em; text-transform: uppercase;
    }

    /* ── Secure Badge ── */
    .login-badge {
        display: flex; align-items: center; gap: 6px;
        padding: 6px 10px; margin-bottom: 20px;
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12);
        font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
        color: #6a6a6a; letter-spacing: 0.1em; text-transform: uppercase;
        position: relative; z-index: 1;
    }
    .login-badge .badge-dot {
        width: 5px; height: 5px; border-radius: 50%; background: #FFFFFF;
        animation: dotPulse 2s ease-in-out infinite;
    }

    /* ── Labels & Errors ── */
    .login-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
        color: #6a6a6a; letter-spacing: 0.1em; text-transform: uppercase;
        margin-bottom: 4px; position: relative; z-index: 1;
    }
    .login-error {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        color: #D94040; padding: 8px 12px; margin-top: 12px;
        border: 1px solid rgba(217,64,64,0.2);
        background: rgba(217,64,64,0.06);
    }
    .login-footer {
        font-family: 'JetBrains Mono', monospace; font-size: 0.5rem;
        color: #4a4a4a; text-align: center; margin-top: 32px;
        letter-spacing: 0.05em; position: relative; z-index: 1;
    }

    /* ── Streamlit Input Overrides ── */
    .login-container .stTextInput input {
        background: #111111 !important; border: 1px solid #2f2f2f !important;
        color: #FFFFFF !important; font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important; border-radius: 0 !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .login-container .stTextInput input:focus {
        border-color: #FFFFFF !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.25), 0 0 12px rgba(255,255,255,0.08) !important;
    }
    .login-container .stButton button {
        width: 100%; background: #FFFFFF !important; color: #0C0C0C !important;
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
        font-size: 0.75rem !important; letter-spacing: 0.1em !important;
        text-transform: uppercase !important; border: none !important;
        border-radius: 0 !important; padding: 10px !important; margin-top: 16px;
        transition: transform 0.15s, box-shadow 0.15s, background 0.15s;
        position: relative; z-index: 1;
    }
    .login-container .stButton button:hover {
        background: #e0e0e0 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(255,255,255,0.25);
    }
    .login-container .stButton button:active {
        transform: translateY(0px);
        box-shadow: 0 1px 4px rgba(255,255,255,0.15);
    }
    /* ── 移动端适配 ── */
    @media (max-width: 768px) {
        .login-container { max-width: 90vw; padding: 24px 20px; margin-top: 10vh; }
        .login-title { font-size: 1rem; }
        .login-subtitle { font-size: 0.5rem; }
    }
    @media (max-width: 480px) {
        .login-container { max-width: 95vw; padding: 20px 16px; margin-top: 6vh; }
        .login-title { font-size: 0.9rem; }
    }
    </style>""", unsafe_allow_html=True)

    # 读取 logo base64
    _login_logo_b64 = ""
    try:
        import os as _os
        _logo_path = _os.path.join(_os.path.dirname(__file__), "logo_b64.txt")
        with open(_logo_path, "r") as _lf:
            _login_logo_b64 = _lf.read().strip()
    except Exception:
        pass

    _horse_img = f'<img class="title-horse" src="data:image/png;base64,{_login_logo_b64}" />' if _login_logo_b64 else ''

    # Login form — 居中容器（只有一个马logo，在MRARFAI旁边）
    st.markdown(f"""
    <div class="login-container">
        <div class="login-logo">
            <div>
                <div class="login-title">{_horse_img}MRARFAI</div>
                <div class="login-subtitle">V10.0 · Enterprise Agent Platform</div>
            </div>
        </div>
        <div class="login-badge">
            <span class="badge-dot"></span>
            SECURE ACCESS · V10.0
        </div>
    """, unsafe_allow_html=True)

    # 用 columns 居中 input 区域
    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        st.markdown('<div class="login-label">USERNAME</div>', unsafe_allow_html=True)
        username = st.text_input("用户名", label_visibility="collapsed", key="login_user",
                                  placeholder="username")

        st.markdown('<div class="login-label">PASSWORD</div>', unsafe_allow_html=True)
        password = st.text_input("密码", type="password", label_visibility="collapsed",
                                  key="login_pass", placeholder="••••••••")

        login_clicked = st.button("SIGN IN", key="login_btn", use_container_width=True)

        if login_clicked:
            if not username or not password:
                st.markdown('<div class="login-error">⚠ 请输入用户名和密码</div>',
                           unsafe_allow_html=True)
            else:
                user = authenticate(username, password)
                if user:
                    st.session_state["auth_user"] = user
                    st.session_state["auth_login_time"] = datetime.now().isoformat()
                    st.session_state["auth_last_activity"] = datetime.now().isoformat()
                    st.rerun()
                else:
                    st.markdown('<div class="login-error">⚠ 用户名或密码错误</div>',
                               unsafe_allow_html=True)

    st.markdown("""
        <div class="login-footer">
            © 2026 MRARFAI · Powered by Multi-Agent Intelligence
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
        st.warning(f"⏰ 会话已超时（{SESSION_TIMEOUT_HOURS}小时），请重新登录。")

    if not is_logged_in():
        _render_login_page()
        st.stop()
