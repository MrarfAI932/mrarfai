"""
MRARFAI API v10.0
FastAPI 后端入口 — 包装现有 Python 业务逻辑
"""

import sys
import os

# 确保项目根目录在 Python path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# 加载 .env
from dotenv import load_dotenv
load_dotenv(os.path.join(_root, ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import auth, agents, data, audit

app = FastAPI(
    title="MRARFAI API",
    description="多智能体销售情报平台 API",
    version="10.0",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路由 ──
app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
app.include_router(agents.router, prefix="/api/agents", tags=["智能体"])
app.include_router(data.router, prefix="/api/data", tags=["数据"])
app.include_router(audit.router, prefix="/api/audit", tags=["审计"])


@app.get("/")
async def root():
    return {
        "service": "MRARFAI API",
        "version": "10.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
