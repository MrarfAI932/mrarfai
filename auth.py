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
    """渲染登录页面 — 白色卡片 + 深色背景 (Streamlit 兼容方案)

    核心思路:
    把 [data-testid="stMainBlockContainer"] 当作白色卡片，同时确保其
    所有内部子容器(stVerticalBlock, stVerticalBlockBorderWrapper,
    block-container, stElementContainer 等)都是透明背景、零额外 padding，
    使白色背景从最外层一路穿透到所有 widget。
    """

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

    # ================================================================
    # 一次性注入所有 CSS — 这是唯一的 st.markdown(style) 调用
    # ================================================================
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    @keyframes fadeInUp {
        from { opacity:0; transform:translateY(24px); }
        to   { opacity:1; transform:translateY(0); }
    }

    /* ── 1. 全局深色背景 ── */
    html, body,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    .stApp {
        background: #0a0a0a !important;
    }
    [data-testid="stApp"]::before {
        content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    [data-testid="stMain"],
    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    /* ── 2. 彻底隐藏 Streamlit 所有 chrome ── */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stBottom"],
    #MainMenu, footer, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    /* 顶部留白也消除 */
    [data-testid="stAppViewContainer"] > section > div {
        padding-top: 0 !important;
    }

    /* ── 3. 白色卡片 = stMainBlockContainer ── */
    [data-testid="stMainBlockContainer"] {
        max-width: 420px !important;
        margin: 8vh auto 0 auto !important;
        background: #FFFFFF !important;
        border-radius: 16px !important;
        padding: 48px 40px 36px 40px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5) !important;
        animation: fadeInUp 0.5s ease-out;
        position: relative;
        z-index: 1;
        overflow: hidden !important;
    }

    /* ── 3a. 穿透所有内部 Streamlit 包装器 ── */
    [data-testid="stMainBlockContainer"] .block-container,
    [data-testid="stMainBlockContainer"] .stMainBlockContainer {
        max-width: 100% !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        gap: 4px !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"] {
        background: transparent !important;
        margin: 0 !important;
        width: 100% !important;
    }
    /* login-label 所在的 stElementContainer — 确保标签和下方输入框之间有间距 */
    [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(.login-label) {
        margin-bottom: 2px !important;
    }

    /* ── 4. Logo 区域 ── */
    .login-logo-area {
        text-align: center; margin-bottom: 10px;
    }
    .login-logo-area img.horse-logo {
        width: 64px; height: auto; margin-bottom: 14px;
        filter: brightness(0);
    }
    .login-logo-area .horse-icon-fallback {
        font-size: 52px; margin-bottom: 8px; line-height: 1;
    }
    .login-logo-area .brand-name {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 900; font-size: 26px;
        letter-spacing: 0.12em; color: #0a0a0a;
    }
    .login-logo-area .brand-sub {
        font-family: 'Inter', sans-serif;
        font-weight: 600; font-size: 11px;
        letter-spacing: 0.15em; color: #999;
        margin-top: 4px;
    }

    /* ── 5. 分隔线 ── */
    .login-divider {
        height: 1px; background: #e8e8e8;
        margin: 18px auto 20px auto; width: 75%;
    }

    /* ── 6. USERNAME / PASSWORD 标签 ── */
    .login-label {
        font-family: 'Inter', sans-serif;
        font-weight: 700; font-size: 11px;
        color: #333; letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 12px 0 6px 2px;
        padding: 0;
        line-height: 1;
        display: block;
        position: relative;
        z-index: 2;
    }

    /* ── 7. 错误提示 ── */
    .login-error {
        font-family: 'Inter', sans-serif; font-size: 13px;
        color: #dc3545; padding: 10px 14px; margin-top: 10px;
        border: 1px solid rgba(220,53,69,0.2);
        background: rgba(220,53,69,0.06);
        border-radius: 8px; text-align: center;
    }

    /* ── 8. 底部链接 ── */
    .login-links {
        text-align: center; margin-top: 22px;
        font-family: 'Inter', sans-serif; font-size: 13px;
    }
    .login-links a { color: #888; text-decoration: none; }
    .login-links a:hover { color: #0a0a0a; }
    .login-links .sep { color: #ddd; margin: 0 10px; }

    /* ── 9. 页脚 ── */
    .login-page-footer {
        font-family: 'Inter', sans-serif; font-size: 11px;
        color: #aaa; text-align: center; margin-top: 16px;
    }

    /* ── 10. Input 样式覆盖 (BaseUI + Dark Theme 强制白色) ── */
    /* 隐藏 Streamlit 原生 label */
    [data-testid="stMainBlockContainer"] .stTextInput label,
    [data-testid="stMainBlockContainer"] .stTextInput > label {
        display: none !important; height: 0 !important;
        margin: 0 !important; padding: 0 !important;
        min-height: 0 !important; overflow: hidden !important;
    }
    /* TextInput 整体 wrapper */
    [data-testid="stMainBlockContainer"] [data-testid="stTextInput"],
    [data-testid="stMainBlockContainer"] .stTextInput {
        width: 100% !important;
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stMainBlockContainer"] .stTextInput > div {
        width: 100% !important;
    }
    /* ★ 核弹覆盖: stTextInput 内所有 div 强制白色 (Dark theme) */
    [data-testid="stMainBlockContainer"] .stTextInput div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
    }
    /* ★ 整个 stTextInput 组件加上可见边框 (这才是目标设计里的圆角框) */
    [data-testid="stMainBlockContainer"] .stTextInput [data-testid="stTextInputRootElement"],
    [data-testid="stMainBlockContainer"] .stTextInput [data-baseweb="input"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #d0d0d0 !important;
        border-radius: 8px !important;
        padding: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        position: relative !important;
        overflow: hidden !important;
        transition: border-color 0.2s;
    }
    /* 聚焦时边框变深 */
    [data-testid="stMainBlockContainer"] .stTextInput [data-testid="stTextInputRootElement"]:focus-within,
    [data-testid="stMainBlockContainer"] .stTextInput [data-baseweb="input"]:focus-within {
        border-color: #999 !important;
    }
    /* Input 本体 — 无自身边框 */
    [data-testid="stMainBlockContainer"] .stTextInput input {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: #1a1a1a !important;
        caret-color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        padding: 10px 14px !important;
        height: auto !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    [data-testid="stMainBlockContainer"] .stTextInput input::placeholder {
        color: #bbb !important;
        -webkit-text-fill-color: #bbb !important;
    }
    [data-testid="stMainBlockContainer"] .stTextInput input:focus {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    /* 密码眼睛 */
    [data-testid="stMainBlockContainer"] .stTextInput button {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        color: #aaa !important;
        padding: 4px 8px !important;
        min-height: 0 !important;
        height: auto !important;
    }
    [data-testid="stMainBlockContainer"] .stTextInput button svg {
        fill: #aaa !important;
        stroke: #aaa !important;
        width: 18px !important;
        height: 18px !important;
    }
    [data-testid="stMainBlockContainer"] .stTextInput button:hover {
        background: #f5f5f5 !important;
        color: #666 !important;
    }
    [data-testid="stMainBlockContainer"] .stTextInput button:hover svg {
        fill: #666 !important;
        stroke: #666 !important;
    }
    /* 自动填充白底黑字 */
    [data-testid="stMainBlockContainer"] .stTextInput input:-webkit-autofill,
    [data-testid="stMainBlockContainer"] .stTextInput input:-webkit-autofill:focus {
        -webkit-box-shadow: 0 0 0 1000px #FFFFFF inset !important;
        -webkit-text-fill-color: #1a1a1a !important;
        background-color: #FFFFFF !important;
    }

    /* ── 11. SIGN IN 按钮 ── */
    [data-testid="stMainBlockContainer"] .stButton > button,
    [data-testid="stMainBlockContainer"] [data-testid="stBaseButton-secondary"] {
        width: 100% !important;
        background: #0a0a0a !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 0 !important;
        margin-top: 10px;
        cursor: pointer;
        line-height: 1.2 !important;
        min-height: 0 !important;
        height: 40px !important;
        transition: background 0.2s, transform 0.1s;
    }
    [data-testid="stMainBlockContainer"] .stButton > button:hover {
        background: #222 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    [data-testid="stMainBlockContainer"] .stButton > button:active {
        transform: translateY(0);
    }

    /* ── 12. 移动端 ── */
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            max-width: 90vw !important;
            padding: 36px 28px 28px 28px !important;
            margin-top: 5vh !important;
        }
    }
    @media (max-width: 480px) {
        [data-testid="stMainBlockContainer"] {
            max-width: 96vw !important;
            padding: 28px 20px 22px 20px !important;
            margin-top: 3vh !important;
            border-radius: 12px !important;
        }
    }
    </style>""", unsafe_allow_html=True)

    # ================================================================
    # Logo + 品牌名 + 分隔线
    # ================================================================
    st.markdown(f"""
    <div class="login-logo-area">
        {_horse_logo_html}
        <div class="brand-name">MRARFAI</div>
        <div class="brand-sub">\u4f01\u4e1aAgent\u5e73\u53f0</div>
    </div>
    <div class="login-divider"></div>
    """, unsafe_allow_html=True)

    # ================================================================
    # 表单字段 — Streamlit 原生 widget，CSS 会自动样式化
    # ================================================================
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
                '<div class="login-error">\u26a0 Please enter username and password</div>',
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
                    '<div class="login-error">\u26a0 Invalid username or password</div>',
                    unsafe_allow_html=True,
                )

    # ================================================================
    # 底部链接 + 页脚 (仍然在卡片内)
    # ================================================================
    st.markdown("""
    <div class="login-links">
        <a href="#">Forgot Password?</a>
        <span class="sep">|</span>
        <a href="#">Contact Support</a>
    </div>
    <div class="login-page-footer">
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
