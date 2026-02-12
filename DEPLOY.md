# MRARFAI V9.0 — 中美双线部署指南

> 24 AI 模块 · 27,679 行代码 · 7 篇前沿论文 · 双线全球部署

---

## 部署架构

| 线路 | 平台 | 节点 | 用途 |
|------|------|------|------|
| **主线** | Railway | 美国 (US-West) | 全球生产环境，自动 HTTPS |
| **备线** | Streamlit Cloud | 全球 CDN | 中国可访问的演示/备用 |
| **备选** | Render | 美国/欧洲 | 免费备用，支持 Docker |

---

## 方法一：Railway 部署（推荐，主力生产）

### 前置条件
- GitHub 仓库已推送最新代码
- Railway 账号 (https://railway.com)
- DeepSeek / Anthropic API Key

### 步骤

```bash
# 1. 确保代码已推送到 GitHub
cd ~/Desktop/mrarfai
git add -A
git commit -m "V9.0 deploy: Railway + Streamlit Cloud dual deployment"
git push origin main
```

**2. 在 Railway 创建项目**
1. 打开 https://railway.com/dashboard
2. 点击 **New Project** → **Deploy from GitHub repo**
3. 选择 `MrarfAI932/mrarfai` 仓库
4. Railway 会自动检测 `Dockerfile` 并开始构建

**3. 配置环境变量**

在 Railway Dashboard → **Variables** 中添加：

| 变量名 | 值 | 必填 |
|--------|-----|------|
| `DEEPSEEK_API_KEY` | `sk-xxx` | 是 |
| `ANTHROPIC_API_KEY` | `sk-ant-xxx` | 是 |
| `MRARFAI_ENV` | `production` | 是 |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-xxx` | 可选 |
| `LANGFUSE_SECRET_KEY` | `sk-lf-xxx` | 可选 |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | 可选 |

**4. 生成域名**

Railway → Settings → **Networking** → Generate Domain

得到类似：`mrarfai-production.up.railway.app`

> 可绑定自定义域名（需 Developer Plan，$5/月起）

**5. 验证部署**
```bash
# 健康检查
curl https://your-app.up.railway.app/_stcore/health

# 应返回 "ok"
```

---

## 方法二：Streamlit Cloud 部署（免费，中国线路）

### 步骤

**1. 打开 Streamlit Cloud**

访问 https://share.streamlit.io

**2. 创建新应用**
1. 用 GitHub 账号登录
2. 点击 **New app**
3. 配置：
   - Repository: `MrarfAI932/mrarfai`
   - Branch: `main`
   - Main file: `app.py`
4. 点击 **Deploy**

**3. 配置 Secrets**

部署后在 **Settings** → **Secrets** 中添加：

```toml
DEEPSEEK_API_KEY = "sk-xxx"
ANTHROPIC_API_KEY = "sk-ant-xxx"
MRARFAI_ENV = "production"
```

**4. 访问地址**

部署成功后得到：`https://mrarfai-xxx.streamlit.app`

> 中国大陆直接访问，无需翻墙

---

## 方法三：Render 部署（免费备用）

### 步骤

**1. 打开 Render**

访问 https://render.com

**2. 创建新服务**
1. 点击 **New** → **Web Service**
2. 连接 GitHub 仓库 `MrarfAI932/mrarfai`
3. 选择 **Docker** 部署方式
4. Plan: **Starter** (免费)

**3. 配置环境变量**

与 Railway 相同，在 Dashboard → **Environment** 中添加。

**4. 访问地址**

`https://mrarfai.onrender.com`

> 注意：免费套餐有冷启动（~30 秒），15 分钟无访问会休眠

---

## 方法四：本地 Docker 部署

```bash
# 构建
docker build -t mrarfai:v9.0 .

# 运行（需要先创建 .env 文件）
docker run -p 8501:8501 --env-file .env mrarfai:v9.0

# 或使用 Docker Compose
docker compose up -d

# 访问
open http://localhost:8501
```

---

## 登录账号

| 用户名 | 密码 | 角色 | 用途 |
|--------|------|------|------|
| `admin` | `mrarfai2025` | 管理员 | 系统管理 |
| `sprocomm` | `sprocomm888` | 管理员 | 禾苗客户 |
| `viewer` | `view123` | 只读 | 演示访客 |

---

## 故障排查

### 构建失败
```bash
# 本地测试 Docker 构建
docker build -t mrarfai:v9.0 . 2>&1 | tail -20
```

### 健康检查失败
- 确认 `PORT` 环境变量已设置
- 检查 `/_stcore/health` 返回 200

### 中国访问慢
- 优先使用 Streamlit Cloud（有 CDN）
- Railway 可选 Singapore 节点

### API 调用失败
- 确认环境变量中 API Key 正确
- DeepSeek: https://platform.deepseek.com
- Claude: https://console.anthropic.com

---

## 更新部署

```bash
# 推送新代码后，Railway / Streamlit Cloud 会自动重新部署
git add -A
git commit -m "update: description"
git push origin main

# Railway: 自动触发，约 2-3 分钟
# Streamlit Cloud: 自动触发，约 1-2 分钟
# Render: 自动触发，约 3-5 分钟
```
