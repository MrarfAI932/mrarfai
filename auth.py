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
    """渲染登录页面 — 白色卡片 + 深色背景"""

    # 全屏深色背景 + 白色卡片居中
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── Keyframes ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Base ── */
    [data-testid="stApp"] {
        background: #0a0a0a !important;
    }
    /* 细网格背景 */
    [data-testid="stApp"]::before {
        content: "";
        position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stHeader"] { display: none; }
    #MainMenu, footer, .stDeployButton,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] { display: none !important; }

    /* ── 白色卡片容器 ── */
    .login-card {
        max-width: 420px; margin: 12vh auto 0 auto;
        padding: 48px 40px 36px;
        background: #FFFFFF;
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        animation: fadeInUp 0.5s ease-out;
        position: relative; z-index: 1;
    }

    /* ── Logo 区域 (居中) ── */
    .login-logo-area {
        text-align: center; margin-bottom: 28px;
    }
    .login-logo-area img.horse-logo {
        width: 72px; height: auto; margin-bottom: 16px;
        filter: brightness(0);  /* 纯黑色 */
    }
    /* 纯文字马头 fallback */
    .login-logo-area .horse-icon-fallback {
        font-size: 56px; margin-bottom: 8px; line-height: 1;
    }
    .login-logo-area .brand-name {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 900; font-size: 28px;
        letter-spacing: 0.12em; color: #0a0a0a;
        text-transform: uppercase;
    }
    .login-logo-area .brand-sub {
        font-family: 'Inter', sans-serif;
        font-weight: 600; font-size: 11px;
        letter-spacing: 0.18em; color: #888;
        text-transform: uppercase; margin-top: 4px;
    }

    /* ── 分隔线 ── */
    .login-divider {
        height: 1px; background: #e8e8e8; margin: 24px 0;
    }

    /* ── 标签 ── */
    .login-label {
        font-family: 'Inter', sans-serif;
        font-weight: 700; font-size: 12px;
        color: #1a1a1a; letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* ── 错误提示 ── */
    .login-error {
        font-family: 'Inter', sans-serif; font-size: 13px;
        color: #dc3545; padding: 10px 14px; margin-top: 12px;
        border: 1px solid rgba(220,53,69,0.2);
        background: rgba(220,53,69,0.06);
        border-radius: 8px;
    }

    /* ── 底部链接 ── */
    .login-links {
        text-align: center; margin-top: 24px;
        font-family: 'Inter', sans-serif; font-size: 13px;
    }
    .login-links a {
        color: #666; text-decoration: none;
        transition: color 0.2s;
    }
    .login-links a:hover { color: #0a0a0a; }
    .login-links .sep {
        color: #ccc; margin: 0 10px;
    }

    /* ── 页脚 ── */
    .login-footer {
        font-family: 'Inter', sans-serif; font-size: 12px;
        color: #555; text-align: center; margin-top: 24px;
        position: relative; z-index: 1;
    }

    /* ── Streamlit Input 覆盖 (白色卡片内) ── */
    .login-card .stTextInput input {
        background: #FFFFFF !important;
        border: 1.5px solid #d0d0d0 !important;
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .login-card .stTextInput input::placeholder {
        color: #aaa !important;
    }
    .login-card .stTextInput input:focus {
        border-color: #0a0a0a !important;
        box-shadow: 0 0 0 3px rgba(10,10,10,0.08) !important;
    }
    /* SIGN IN 按钮 — 黑色 */
    .login-card .stButton button {
        width: 100%;
        background: #0a0a0a !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px !important;
        margin-top: 8px;
        transition: background 0.2s, transform 0.15s, box-shadow 0.15s;
        position: relative; z-index: 1;
    }
    .login-card .stButton button:hover {
        background: #222 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .login-card .stButton button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    /* 隐藏 Streamlit 默认 label */
    .login-card .stTextInput label,
    .login-card .stButton label { display: none !important; }

    /* ── 移动端适配 ── */
    @media (max-width: 768px) {
        .login-card { max-width: 90vw; padding: 36px 24px 28px; margin-top: 8vh; }
        .login-logo-area .brand-name { font-size: 24px; }
    }
    @media (max-width: 480px) {
        .login-card { max-width: 95vw; padding: 28px 20px 24px; margin-top: 5vh; }
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

    _horse_logo = (
        f'<img class="horse-logo" src="data:image/png;base64,{_login_logo_b64}" />'
        if _login_logo_b64
        else '<div class="horse-icon-fallback">\U0001F40E</div>'
    )

    # ── 白色卡片 HTML ──
    st.markdown(f"""
    <div class="login-card">
        <div class="login-logo-area">
            {_horse_logo}
            <div class="brand-name">MRARFAI</div>
            <div class="brand-sub">Enterprise Agent Platform</div>
        </div>
        <div class="login-divider"></div>
    """, unsafe_allow_html=True)

    # 用 columns 让输入框在卡片内居中
    col1, col2, col3 = st.columns([0.6, 2, 0.6])
    with col2:
        st.markdown('<div class="login-label">USERNAME</div>', unsafe_allow_html=True)
        username = st.text_input("用户名", label_visibility="collapsed", key="login_user",
                                  placeholder="Enter username")

        st.markdown('<div class="login-label">PASSWORD</div>', unsafe_allow_html=True)
        password = st.text_input("密码", type="password", label_visibility="collapsed",
                                  key="login_pass", placeholder="Enter password")

        login_clicked = st.button("SIGN IN", key="login_btn", use_container_width=True)

        if login_clicked:
            if not username or not password:
                st.markdown('<div class="login-error">\u26a0 Please enter username and password</div>',
                           unsafe_allow_html=True)
            else:
                user = authenticate(username, password)
                if user:
                    st.session_state["auth_user"] = user
                    st.session_state["auth_login_time"] = datetime.now().isoformat()
                    st.session_state["auth_last_activity"] = datetime.now().isoformat()
                    st.rerun()
                else:
                    st.markdown('<div class="login-error">\u26a0 Invalid username or password</div>',
                               unsafe_allow_html=True)

    st.markdown("""
        <div class="login-links">
            <a href="#">Forgot Password?</a>
            <span class="sep">|</span>
            <a href="#">Contact Support</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 卡片外页脚
    st.markdown("""
    <div class="login-footer">
        &copy; 2024 MRARFAI &middot; Powered by Multi-Agent Intelligence
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
