"""
MRARFAI API — 认证路由
POST /api/auth/login  → 登录
GET  /api/auth/me     → 当前用户信息
"""

import sys
import os

from fastapi import APIRouter, HTTPException

# 添加项目根目录
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from api.schemas import LoginRequest, LoginResponse, UserInfo
from api.deps import create_access_token, get_current_user
from fastapi import Depends

# 导入原有认证模块 (只使用 authenticate 函数，避免导入 streamlit)
from auth import authenticate

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """
    登录认证。
    前端发送 email + password，后端用 username 部分匹配。
    例如: admin@sprocomm.com → username="admin"
    """
    # 从 email 提取 username (取 @ 前面的部分)
    username = req.email.split("@")[0].lower().strip() if "@" in req.email else req.email.lower().strip()

    result = authenticate(username, req.password)
    if result is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 生成 JWT
    token = create_access_token({
        "username": result["username"],
        "role": result["role"],
        "display_name": result["display_name"],
        "company": result.get("company", ""),
    })

    return LoginResponse(
        token=token,
        user=UserInfo(
            username=result["username"],
            displayName=result["display_name"],
            role=result["role"],
            company=result.get("company", ""),
        ),
    )


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    """返回当前用户信息"""
    return {
        "username": user.get("username", "anonymous"),
        "displayName": user.get("display_name", "匿名"),
        "role": user.get("role", "viewer"),
        "company": user.get("company", ""),
    }
