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


def is_logged_in() -> bool:
    """是否已登录"""
    return st.session_state.get("auth_user") is not None


def get_current_user() -> Optional[Dict]:
    """获取当前用户信息"""
    return st.session_state.get("auth_user")


def logout():
    """登出"""
    st.session_state.pop("auth_user", None)
    st.session_state.pop("auth_login_time", None)


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

    # 全屏深色背景
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    [data-testid="stApp"] { background: #080808; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stHeader"] { display: none; }
    
    .login-container {
        max-width: 380px; margin: 10vh auto; padding: 40px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(12,12,12,0.95);
    }
    .login-logo {
        display: flex; align-items: center; gap: 12px; margin-bottom: 32px;
    }
    .login-logo-box {
        width: 40px; height: 40px; background: #00FF88;
        display: flex; align-items: center; justify-content: center;
    }
    .login-logo-box span {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 1.1rem; color: #0C0C0C;
    }
    .login-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 1.2rem; color: #FFFFFF; letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .login-subtitle {
        font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
        color: #6a6a6a; letter-spacing: 0.1em; text-transform: uppercase;
    }
    .login-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
        color: #6a6a6a; letter-spacing: 0.1em; text-transform: uppercase;
        margin-bottom: 4px;
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
        letter-spacing: 0.05em;
    }

    /* Style Streamlit inputs */
    .login-container .stTextInput input {
        background: #111111 !important; border: 1px solid #2f2f2f !important;
        color: #FFFFFF !important; font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important; border-radius: 0 !important;
    }
    .login-container .stTextInput input:focus {
        border-color: #00FF88 !important; box-shadow: none !important;
    }
    .login-container .stButton button {
        width: 100%; background: #00FF88 !important; color: #0C0C0C !important;
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
        font-size: 0.75rem !important; letter-spacing: 0.1em !important;
        text-transform: uppercase !important; border: none !important;
        border-radius: 0 !important; padding: 10px !important; margin-top: 16px;
    }
    .login-container .stButton button:hover {
        background: #00cc6e !important;
    }
    </style>""", unsafe_allow_html=True)

    # Login form
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-logo">
                <div class="login-logo-box"><span>S</span></div>
                <div>
                    <div class="login-title">SPROCOMM AI</div>
                    <div class="login-subtitle">MRARFAI v9.0 · Sales Intelligence</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

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
                    st.rerun()
                else:
                    st.markdown('<div class="login-error">⚠ 用户名或密码错误</div>',
                               unsafe_allow_html=True)

        st.markdown("""
            <div class="login-footer">
                © 2025 MRARFAI · Powered by Multi-Agent Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)


def require_login():
    """
    在 app.py 最前面调用。未登录则显示登录页并 st.stop()。
    """
    if not is_logged_in():
        _render_login_page()
        st.stop()
