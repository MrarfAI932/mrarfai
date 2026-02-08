# MRARFAI 销售分析 Agent v3.2

> 禾苗通讯 Sprocomm (01401.HK) 智能销售分析平台

## 🚀 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Dashboard
streamlit run app.py

# 3. 浏览器自动打开 http://localhost:8501
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `app.py` | Streamlit Dashboard（12个分析Tab + 反馈收集） |
| `analyze_clients_v2.py` | 核心分析引擎（12维度） |
| `industry_benchmark.py` | 行业对标（IDC/Counterpoint数据） |
| `forecast_engine.py` | 预测引擎（季节性+趋势+情景） |
| `ai_narrator.py` | AI叙事（Claude/DeepSeek API） |
| `feedback_collector.py` | 用户反馈收集与统计 |
| `run_v3.py` | 命令行版本（生成Markdown报告） |
| `DEPLOY.md` | 部署指南（Streamlit Cloud/Railway/Docker） |

## 📊 Dashboard 功能

| Tab | 内容 |
|-----|------|
| 总览 | 营收/出货/客户KPI + 月度趋势 |
| 客户分析 | ABC分级 + 集中度 + 单客户趋势 |
| 价量分解 | 单价趋势 → 增长质量判断 |
| 预警中心 | 流失风险评分 + 异常检测 |
| 增长机会 | 高增长/新兴/换道客户识别 |
| 产品结构 | FP/SP/PAD × CKD/SKD/CBU |
| 区域分析 | HHI集中度 + 区域分布 |
| 行业对标 | 华勤/闻泰/龙旗竞争对比 |
| 预测 | Q1 2026预测 + 4种情景分析 |
| CEO备忘录 | 管理层战略摘要 + 行动建议 |
| 导出 | 完整报告下载（MD/JSON/Prompt） |
| 反馈 | 用户满意度 + 建议收集（v3.2新增） |

## 🚀 部署

详见 [DEPLOY.md](DEPLOY.md)

```bash
# Streamlit Cloud（推荐）
# 1. 推送到GitHub → 2. share.streamlit.io 选择仓库 → 3. Deploy

# Docker（国内云服务器）
docker build -t mrarfai .
docker run -p 8501:8501 mrarfai

# Railway
# 自动检测 Procfile 部署
```

## 🤖 AI 功能

在侧边栏启用「AI深度叙事」，填入API Key：
- **Claude**: `sk-ant-api03-...`
- **DeepSeek**: `sk-...`

AI会基于全部分析数据生成管理层级的战略备忘录。

## 🔧 命令行用法

```bash
# 基础分析
python run_v3.py

# 指定文件
python run_v3.py --revenue data/金额.xlsx --quantity data/数量.xlsx

# 启用AI
python run_v3.py --ai-key sk-ant-xxx --provider claude
```

## 📈 v3.2 更新日志（Week 5-6）

- ✅ 错误处理：数据加载失败不再白屏，给出清晰错误提示
- ✅ 反馈收集：内置用户满意度调查（KPI目标>80%）
- ✅ 使用日志：匿名记录分析行为，用于优化产品
- ✅ 部署就绪：Streamlit Cloud / Railway / Docker 三套配置
- ✅ 使用指南：欢迎页增加操作说明和FAQ
- ✅ 数据安全：会话结束自动清除，不保存用户数据

---
© 2025-2026 MRARFAI | Built for Sprocomm
