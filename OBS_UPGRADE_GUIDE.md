# MRARFAI v3.2 可观测性升级指南

## 📦 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `observability.py` | ~650 | 核心可观测性引擎（OpenTelemetry标准） |
| `multi_agent.py` | v3.2 | 全链路追踪集成 |
| `obs_tab.py` | ~280 | Streamlit可观测性仪表盘 |
| `test_obs.py` | ~400 | 40项测试（全部通过） |

## 🏗️ 架构设计

```
用户问题
  ↓
ask_multi_agent() ─── [Trace 开始]
  │
  ├── SmartDataQuery ──── [data_query span]
  │     └── KG查询（零API）
  │
  ├── SmartRouter ────── [routing span]
  │     └── LLM调用 ──── [llm_call span: routing_llm]
  │
  ├── ParallelExecutor ── [agent spans]
  │     ├── 分析师 ───── [llm_call span: agent_analyst]
  │     ├── 风控 ─────── [llm_call span: agent_risk]
  │     └── 策略师 ───── [llm_call span: agent_strategist]
  │
  └── Reporter ────────── [reporter span]
        └── LLM调用 ──── [llm_call span: reporter_llm]
  │
  [Trace 结束] → SQLite持久化
```

## 🔧 集成步骤

### 1. 放置文件
```
your_project/
  ├── observability.py     # 新增
  ├── obs_tab.py           # 新增
  ├── test_obs.py          # 新增
  ├── multi_agent.py       # 替换（v3.2）
  ├── knowledge_graph.py   # 不变
  ├── chat_tab.py          # 不变
  └── app.py               # 需小改
```

### 2. 在 app.py 中添加仪表盘Tab
```python
# 在现有tab列表中添加：
from obs_tab import render_obs_tab

# 在tab渲染区域添加：
with tab_obs:
    render_obs_tab()
```

### 3. 在 chat_tab.py 中展示trace信息（可选）
```python
# 在ask_multi_agent返回后，可以获取trace信息：
result = ask_multi_agent(question, data, results, ...)
trace_id = result.get("trace_id", "")
obs = result.get("obs_summary", {})

# 展示成本信息（可选）
if obs:
    st.caption(f"Trace: {trace_id[:8]} | "
               f"Tokens: {obs['total_tokens']} | "
               f"Cost: ${obs['total_cost_usd']:.4f}")
```

## 📊 可观测性能力清单

### 实时指标
- ✅ 每次请求的完整trace（trace_id唯一标识）
- ✅ 各阶段延迟分解（data_query / routing / agents / reporter）
- ✅ 每次LLM调用的token/cost追踪
- ✅ 知识图谱路由 vs LLM路由的分布

### 历史分析
- ✅ 延迟分位数（P50/P90/P95/P99）
- ✅ 每日趋势（查询量、延迟、成本、错误率）
- ✅ 成本分解（按Provider、按阶段）
- ✅ 路由统计（来源分布、模式分布、Agent使用频率）

### 质量信号
- ✅ 用户反馈（1-5星评分 + 文本）
- ✅ KG纠正率（自动纠正占比）
- ✅ 错误率追踪

### 运维能力
- ✅ SQLite WAL模式（并发安全）
- ✅ 自动清理（默认90天）
- ✅ JSON/CSV导出
- ✅ 数据库统计（traces/spans/大小）

## 💰 成本追踪

内置价格表（可动态更新）：

| Provider | Model | Input $/1M | Output $/1M |
|----------|-------|-----------|-------------|
| DeepSeek | deepseek-chat | $0.14 | $0.28 |
| Claude | claude-sonnet-4 | $3.00 | $15.00 |
| OpenAI | gpt-4o | $2.50 | $10.00 |

更新价格：
```python
from observability import CostCalculator
CostCalculator.update_pricing("deepseek", "deepseek-v3", 0.20, 0.40)
```

## 🔄 向后兼容

- `ask_multi_agent` 返回的dict新增 `trace_id` 和 `obs_summary` 字段
- 原有字段（answer/agents_used/thinking/expert_outputs/hitl_triggers）完全不变
- chat_tab.py 无需改动即可运行
- 可观测性模块 import 失败时自动降级（HAS_OBS=False）

## 🧪 测试

```bash
python -m unittest test_obs -v
# 40 tests, all passing
```

测试覆盖：
- 数据模型（Span/Trace/LLMUsage）
- 成本计算（DeepSeek/Claude/未知Provider）
- SQLite持久化（CRUD/反馈/清理/并发）
- AgentTracer（生命周期/嵌套/LLM追踪/错误/禁用）
- 指标聚合（概览/延迟/成本/路由/质量）
- 导出（JSON/CSV）
- 集成模式（完整pipeline模拟/并发trace隔离）
