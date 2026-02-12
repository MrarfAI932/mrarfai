# ============================================================
# MRARFAI V9.0 Dockerfile — 生产级多阶段构建
# ============================================================
# 用法:
#   docker build -t mrarfai:v9.0 .
#   docker run -p 8501:8501 --env-file .env mrarfai:v9.0
#
# 特性:
#   - 多阶段构建（Builder → Runtime），镜像更小更安全
#   - 非 root 用户运行
#   - Healthcheck 自动监控
#   - Streamlit 配置优化
#   - 环境变量注入（API Keys 不打入镜像）
#   - 支持 Railway / Render / 自托管部署
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime

LABEL maintainer="mrarf"
LABEL version="9.0"
LABEL description="MRARFAI V9.0 — Multi-Agent Sales Intelligence Platform"

WORKDIR /app
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从 Builder 阶段复制编译好的 wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# 创建非 root 用户
RUN groupadd -r mrarfai && useradd -r -g mrarfai -d /app -s /sbin/nologin mrarfai

# Streamlit 配置
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml

# 应用代码 — 全部 42 个源代码文件
COPY *.py /app/

# 数据目录（运行时挂载或上传）
RUN mkdir -p /app/uploads /app/data /app/logs /app/db

# 修改权限
RUN chown -R mrarfai:mrarfai /app

# 切换到非 root 用户
USER mrarfai

# 环境变量（敏感信息运行时通过 --env-file 或平台 Secrets 注入）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Railway/Render 使用 $PORT 环境变量
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动 — 使用 shell 形式确保 $PORT 被正确展开
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --server.enableCORS=false --server.enableXsrfProtection=false --server.maxUploadSize=50"]
