#!/bin/bash
# MRARFAI v10 — 启动脚本
# 同时启动 FastAPI 后端 (8000) + Next.js 前端 (3000)

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "============================================"
echo "  MRARFAI v10 — 多智能体销售情报平台"
echo "============================================"
echo ""

# 检查 Python 依赖
python3 -c "import fastapi" 2>/dev/null || {
  echo "[!] 安装 Python 依赖..."
  pip3 install fastapi uvicorn python-jose[cryptography] python-multipart python-dotenv
}

# 检查 Node 依赖
if [ ! -d "frontend/node_modules" ]; then
  echo "[!] 安装前端依赖..."
  cd frontend && pnpm install && cd ..
fi

echo ""
echo "[1/2] 启动 FastAPI 后端 → http://localhost:8000"
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "[2/2] 启动 Next.js 前端 → http://localhost:3000"
cd frontend && pnpm dev &
FRONTEND_PID=$!

echo ""
echo "============================================"
echo "  前端: http://localhost:3000"
echo "  后端: http://localhost:8000"
echo "  API 文档: http://localhost:8000/docs"
echo "============================================"
echo "  按 Ctrl+C 停止所有服务"
echo ""

# 等待两个进程
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
