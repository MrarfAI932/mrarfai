import { useState, useMemo } from "react";

// ================================================================
// MRARFAI V10.0 — 2026.2.14 R5终极审计修正版
//
// 基于 frontier_audit_r5_final 五轮累计修正:
//   ★ LangGraph >=0.3 → 1.0 GA (Oct 29 2025)
//   ★ LangChain-core >=0.3 → 1.0 (Nov 17 2025)
//   ★ CrewAI 幽灵依赖清理 (1.9.3 原生A2A)
//   ★ A2A agent.json → agent-card.json (v0.3.0)
//   ★ MCP spec 2025-11-25 (Tasks+Extensions+CIMD)
//   ★ V10 Agent 质量门 (HITL+CriticAgent+Langfuse)
//   ★ OpenAI Agents SDK 0.8.4 竞品评估
//   ★ ADK 0.5.0 四语言 · Deep Agents 0.4.1
//   ★ 2026三大框架格局: LangGraph+CrewAI+AutoGen
// ================================================================

// ── R5 五轮审计发现 ──────────────────────────────────────────
const AUDIT_ROUNDS = [
  { id: 5, label: "R5", color: "#f472b6", items: [
    { severity: "critical", title: "CrewAI幽灵依赖", detail: "multi_agent.py:121 from crewai import — 但requirements已注释crewai>=0.80。运行时HAS_CREWAI=False。CrewAI现已1.9.3原生A2A+Flows", src: "multi_agent.py:121-123" },
    { severity: "critical", title: "V10 Agent零质量门", detail: "grep确认: 4个V10 Agent(procurement/quality/finance/market)零HITL调用、零CriticAgent调用。纯关键词路由→直接返回", src: "platform_gateway.py:407+" },
    { severity: "warn", title: "OpenAI Agents SDK未评估", detail: "0.8.4 (Feb 11) Handoffs+Guardrails+Sessions+Tracing。GPT-5.3-Codex-Spark(Feb 12)。竞品生态需了解", src: "N/A" },
    { severity: "warn", title: "2026三大框架格局", detail: "Iterathon: 86%企业用agent框架。LangGraph(图控制)+CrewAI(角色协作)+AutoGen(对话)三足鼎立。你只用LangGraph", src: "N/A" },
    { severity: "ok", title: "ADK 0.5.0确认", detail: "Python+TS+Java+Go 4语言。MCP原生+A2A原生。可选整合路径", src: "N/A" },
  ]},
  { id: 4, label: "R4", color: "#f59e0b", items: [
    { severity: "critical", title: "langgraph>=0.3 → 1.0 GA (落后两代)", detail: "requirements_v7.txt:17 — LangGraph 1.0 GA Oct 29 2025, Durable State, HITL一等API, Node Caching", src: "requirements_v7.txt:17" },
    { severity: "critical", title: "langchain-core>=0.3 → 1.0", detail: "requirements_v7.txt:18 — LangChain 1.0 Nov 17, create_agent+Middleware架构", src: "requirements_v7.txt:18" },
    { severity: "critical", title: "Deep Agents 0.4.1 未使用", detail: "Feb 11发布。Planning+Sub-agents+Virtual FS。已集成Langfuse", src: "N/A" },
  ]},
  { id: 3, label: "R3", color: "#38bdf8", items: [
    { severity: "critical", title: "MCP spec 2025-11-25 (非06-18)", detail: "Tasks异步+Extensions+CIMD。OAuth 2.1。Registry 2000+。Python SDK v2 Q1'26", src: "mcp_server_v7.py:41,356" },
    { severity: "critical", title: "A2A agent.json→agent-card.json", detail: "a2a_server_v7.py:629 5处旧路径。A2A v0.3.0 (Jul 30 2025)", src: "a2a_server_v7.py:629" },
    { severity: "warn", title: "A2UI v0.8 缺失", detail: "Google Agent→UI声明式协议。AG-UI (CopilotKit)。已集成Deep Agents", src: "N/A" },
  ]},
  { id: 2, label: "R2", color: "#64748b", items: [
    { severity: "ok", title: "MCP已用官方SDK (之前误判)", detail: "mcp_server_v7.py:41 确认使用官方Python SDK", src: "mcp_server_v7.py:41" },
    { severity: "critical", title: "LangGraph仅覆盖3/7 Agent", detail: "multi_agent.py:906 — 4个V10 Agent未接入LangGraph统一编排", src: "multi_agent.py:906" },
  ]},
];

// ── 全量审计条目 (对标 frontier_audit_r5) ─────────────────────
const AUDIT_ITEMS = [
  // PROTOCOLS
  { section: "protocols", name: "MCP", delta: -2,
    frontier: "spec 2025-11-25 (latest)\nTasks异步 · Extensions · CIMD\nOAuth 2.1 · Registry 2000+\nPython SDK v2 Q1'26\n捐Linux Foundation AAIF",
    current: "✅ mcp SDK (py:41)\n✅ stdio+streamable_http\n❌ 仅5 Sales工具\n❌ 无Tasks/Extensions\n❌ mcp>=1.0 (v2即将)",
    fix: "升级到2025-11-25 spec · Tasks · 扩展工具到全Agent",
    src: "mcp_server_v7.py:41,356" },
  { section: "protocols", name: "A2A", delta: -2,
    frontier: "v0.3.0 (Jul 30 2025)\nagent.json→agent-card.json\ngRPC+REST+JSON-RPC三传输\nmTLS · Extensions · 5 SDK\n捐Linux Foundation",
    current: "✅ 自建JSON-RPC完整\n✅ 7Agent全注册\n❌ 5处旧路径agent.json!\n❌ 无gRPC · 无mTLS",
    fix: "agent.json→agent-card.json (1hr) · 迁移官方SDK",
    src: "a2a_server_v7.py:629" },
  { section: "protocols", name: "A2UI + AG-UI", delta: -1,
    frontier: "A2UI v0.8 (Google Dec'25)\nAG-UI (CopilotKit)\n声明式UI · 已集成Deep Agents",
    current: "❌ 无 · Streamlit SSR",
    fix: "P3: 前端迁移后考虑",
    src: "N/A" },
  // FRAMEWORKS
  { section: "frameworks", name: "LangGraph 版本", delta: -3, critical: true,
    frontier: "1.0 GA (Oct 29 2025)\nDurable State · 内建持久化\nHITL一等API · Node Caching\nDeferred Nodes · Model Hooks\nUber/LinkedIn/Klarna投产\nFeb 10 2026: sandbox集成",
    current: "❌ require langgraph>=0.3\n落后两个大版本!\nStateGraph API兼容但缺1.0能力",
    fix: "★ 立即改 langgraph>=1.0",
    src: "requirements_v7.txt:17" },
  { section: "frameworks", name: "LangChain 版本", delta: -3, critical: true,
    frontier: "1.0 (Nov 17 2025)\ncreate_agent() · Middleware架构\nModel Profiles · 统一docs\nlangchain.agents替代prebuilt",
    current: "❌ require langchain-core>=0.3\n❌ 零create_agent · 零Middleware\n手写Agent+StateGraph循环",
    fix: "★ 升级 langchain>=1.0 · create_agent重构",
    src: "requirements_v7.txt:18" },
  { section: "frameworks", name: "编排覆盖范围", delta: -3, critical: true,
    frontier: "所有Agent统一编排\n质量门(Critic+HITL)不可绕过",
    current: "❌ LangGraph仅3Agent\n❌ V10四个走keyword路由\n❌ V10 Agent: 零HITL调用(grep)\n❌ V10 Agent: 零CriticAgent调用(grep)",
    fix: "4个V10 Agent接入LangGraph+质量门",
    src: "multi_agent.py:906" },
  { section: "frameworks", name: "CrewAI 幽灵依赖", delta: -1,
    frontier: "CrewAI 1.9.3 (Jan 30 2026)\n独立于LangChain · 原生A2A\nFlows事件驱动 · Pydantic v2\n100K+认证开发者",
    current: "⚠ multi_agent.py:121 import crewai\n⚠ requirements注释: crewai>=0.80\n⚠ HAS_CREWAI=False (未安装)\n结论: 幽灵代码不影响运行但不干净",
    fix: "清理: 移除或激活。若激活→升级到1.9.3",
    src: "multi_agent.py:121-123" },
  // PATTERNS
  { section: "patterns", name: "设计模式 5/7", delta: 0,
    frontier: "ReAct·Reflection·ToolUse\nPlanning·Multi-Agent·Seq·HITL",
    current: "✅ 5/7 (critic/tool/multi/seq/hitl)\n❌ ReAct · ❌ Planning",
    fix: "可选加ReAct",
    src: "critic_agent.py · hitl_engine.py" },
  { section: "patterns", name: "记忆架构", delta: 0,
    frontier: "三层 + Graph Memory\nGraphiti · Letta · A-MEM",
    current: "✅ 3D Memory · 4维MAGMA\n✅ knowledge_graph 852行",
    fix: "可选加Graphiti",
    src: "memory_v9.py:77" },
  { section: "patterns", name: "结构化合约", delta: -2,
    frontier: "Pydantic/JSON Schema\nLangChain 1.0: TypedDict必须\nCrewAI: Pydantic v2",
    current: "❌ 零Pydantic\n✅ TypedDict (AgentState)\n⚠ Agent间dict通信",
    fix: "Pydantic输出模型",
    src: "multi_agent.py:1967" },
  // INFRA
  { section: "infra", name: "可观测性", delta: -1,
    frontier: "OTEL · Langfuse · Datadog\nLangfuse已集成Deep Agents",
    current: "✅ Langfuse v3 OTEL (V9)\n❌ V10四Agent无trace",
    fix: "V10接入Langfuse",
    src: "langfuse_v3_otel.py" },
  { section: "infra", name: "安全护栏", delta: 0,
    frontier: "CoSAI 12类威胁\nCisco AgenticOps (Feb 10)\nMCP安全白皮书 (Adversa Feb'26)",
    current: "✅ CircuitBreaker·RetryConfig\n✅ AuditLog·RateLimiter\n✅ input_guardrails.py",
    fix: "可选整合",
    src: "guardrails.py" },
  { section: "infra", name: "多模型路由", delta: 0,
    frontier: "59%企业3+LLM\nGPT-5.3-Codex-Spark (Feb 12)\nQwen3-Coder-Next (Feb 3)",
    current: "✅ adaptive_gate.py 3档\n✅ DeepSeek/GPT-4o/多模型",
    fix: "可选加reasoning model",
    src: "adaptive_gate.py:496" },
  { section: "infra", name: "部署", delta: -3,
    frontier: "Container→Cloud Run/K8s\nADK Agent Engine一键部署\nCrewAI AMP Suite",
    current: "⚠ docker-compose写V9\n❌ 未实际部署 · SQLite",
    fix: "更新V10·实际部署·PostgreSQL",
    src: "docker-compose.yml:4" },
  // DATA
  { section: "data", name: "真实数据接入", delta: -4,
    frontier: "生产底线: 真实数据\nMeridian: $17M融资做金融Agent\n数据=产品的灵魂",
    current: "✅ Sales真实Excel\n❌ 4个V10 Agent全SAMPLE_\n(Procurement 8条/Quality 7条/\nFinance 9条/Market 4条)",
    fix: "DataBridge: Excel→反推→灌入 (3-5天)",
    src: "agent_*py SAMPLE_*" },
];

// ── 2026 Agent框架全景 ───────────────────────────────────────
const FRAMEWORK_LANDSCAPE = [
  { name: "LangGraph", version: "1.0 GA", release: "Oct 29 '25", status: "using", statusNote: "✅ 用(但>=0.3)", desc: "图控制·Durable State·HITL一等API", category: "core" },
  { name: "CrewAI", version: "1.9.3", release: "Jan 30 '26", status: "ghost", statusNote: "⚠ 幽灵代码", desc: "角色协作·原生A2A·Flows·Pydantic v2", category: "core" },
  { name: "OpenAI Agents", version: "0.8.4", release: "Feb 11 '26", status: "missing", statusNote: "❌ 未评估", desc: "Handoffs·Guardrails·Sessions·Tracing", category: "competitor" },
  { name: "Google ADK", version: "0.5.0", release: "Feb '26", status: "optional", statusNote: "❌ 可选", desc: "MCP+A2A原生·4语言(Py/TS/Java/Go)", category: "optional" },
  { name: "Deep Agents", version: "0.4.1", release: "Feb 11 '26", status: "missing", statusNote: "❌ 未使用", desc: "Planning+Sub-agents+Virtual FS", category: "competitor" },
  { name: "AutoGen", version: "0.5+", release: "Jan '26", status: "missing", statusNote: "❌ 无", desc: "对话协作·多Agent·Microsoft", category: "core" },
];

// ── Agent矩阵 (R5修正版) ────────────────────────────────────
const PLATFORM_AGENTS = [
  {
    id: "sales",
    icon: "📊",
    name: "Sales Intelligence",
    nameCn: "销售分析Agent",
    status: "live",
    statusLabel: "V9 已有",
    description: "客户分析、出货趋势、异常检测、营收预测、价量分解",
    mcpTools: ["query_shipment", "detect_anomaly", "forecast_revenue", "health_score"],
    a2aSkills: ["revenue_analysis", "churn_alert", "price_volume_decomposition"],
    existingFiles: ["analyze_clients_v2.py", "anomaly_detector.py", "forecast_engine.py", "health_score.py"],
    effort: "已完成",
    color: "#00FF88",
    auditStatus: { langgraph: true, hitl: true, critic: true, langfuse: true, realData: true },
    qualityNote: "✅ LangGraph编排 · ✅ HITL · ✅ CriticAgent · ✅ 真实Excel数据",
  },
  {
    id: "procurement",
    icon: "🛒",
    name: "Procurement Agent",
    nameCn: "采购管理Agent",
    status: "new",
    statusLabel: "V10 新建",
    description: "供应商报价比较、采购订单跟踪、交期预警、成本分析、自动催单",
    mcpTools: ["compare_quotes", "track_po", "alert_delay", "analyze_cost"],
    a2aSkills: ["supplier_evaluation", "delivery_prediction", "cost_optimization"],
    existingFiles: [],
    effort: "2-3 周",
    color: "#FF6B35",
    keyTech: "MCP连接ERP → PO数据 → A2A委派给Sales Agent做成本关联分析",
    auditStatus: { langgraph: false, hitl: false, critic: false, langfuse: false, realData: false },
    qualityNote: "❌ 零LangGraph · ❌ 零HITL · ❌ 零CriticAgent · ❌ SAMPLE_数据(8条)",
    auditDelta: -3,
  },
  {
    id: "quality",
    icon: "🔍",
    name: "Quality Control Agent",
    nameCn: "品质管控Agent",
    status: "new",
    statusLabel: "V10 新建",
    description: "良品率监控、退货分析、客诉分类、质量趋势预警、根因追溯",
    mcpTools: ["monitor_yield", "analyze_returns", "classify_complaints", "trace_root_cause"],
    a2aSkills: ["quality_alert", "defect_pattern", "supplier_quality_score"],
    existingFiles: [],
    effort: "2-3 周",
    color: "#A855F7",
    keyTech: "复用anomaly_detector异常检测引擎 → 应用于良品率数据",
    auditStatus: { langgraph: false, hitl: false, critic: false, langfuse: false, realData: false },
    qualityNote: "❌ 零LangGraph · ❌ 零HITL · ❌ 零CriticAgent · ❌ SAMPLE_数据(7条)",
    auditDelta: -3,
  },
  {
    id: "finance",
    icon: "💰",
    name: "Finance Agent",
    nameCn: "财务分析Agent",
    status: "new",
    statusLabel: "V10 新建",
    description: "应收账款追踪、毛利分析、现金流预测、发票自动匹配、账期管理",
    mcpTools: ["track_ar", "analyze_margin", "forecast_cashflow", "match_invoice"],
    a2aSkills: ["margin_alert", "payment_prediction", "cost_variance"],
    existingFiles: [],
    effort: "3-4 周",
    color: "#06B6D4",
    keyTech: "与Sales Agent联动 → 客户出货×单价=应收 → 自动对账",
    auditStatus: { langgraph: false, hitl: false, critic: false, langfuse: false, realData: false },
    qualityNote: "❌ 零LangGraph · ❌ 零HITL · ❌ 零CriticAgent · ❌ SAMPLE_数据(9条)",
    auditDelta: -3,
  },
  {
    id: "market",
    icon: "🌍",
    name: "Market Intelligence Agent",
    nameCn: "市场情报Agent",
    status: "new",
    statusLabel: "V10 新建",
    description: "竞品价格监控、行业报告摘要、展会信息聚合、客户舆情监控",
    mcpTools: ["monitor_competitor", "summarize_report", "aggregate_events", "track_sentiment"],
    a2aSkills: ["market_brief", "competitor_alert", "trend_insight"],
    existingFiles: ["industry_benchmark.py"],
    effort: "2-3 周",
    color: "#F59E0B",
    keyTech: "Browser Agent自动采集 + RAG知识库(已有) + 定期WeChat推送(已有)",
    auditStatus: { langgraph: false, hitl: false, critic: false, langfuse: false, realData: false },
    qualityNote: "❌ 零LangGraph · ❌ 零HITL · ❌ 零CriticAgent · ❌ SAMPLE_数据(4条)",
    auditDelta: -3,
  },
  {
    id: "hr",
    icon: "👥",
    name: "HR & Operations Agent",
    nameCn: "人事运营Agent",
    status: "future",
    statusLabel: "规划中",
    description: "考勤统计、产能规划、排班优化、绩效追踪、招聘筛选",
    mcpTools: ["analyze_attendance", "plan_capacity", "optimize_schedule"],
    a2aSkills: ["capacity_forecast", "performance_summary"],
    existingFiles: [],
    effort: "Phase 2",
    color: "#EC4899",
    auditStatus: { langgraph: false, hitl: false, critic: false, langfuse: false, realData: false },
    qualityNote: "规划中 — 需从零设计含质量门",
  },
  {
    id: "doc",
    icon: "📄",
    name: "Document Agent",
    nameCn: "文档处理Agent",
    status: "future",
    statusLabel: "规划中",
    description: "合同审查、报关单据生成、产品规格书提取、邮件自动回复",
    mcpTools: ["review_contract", "generate_customs_doc", "extract_spec", "draft_email"],
    a2aSkills: ["document_qa", "compliance_check"],
    existingFiles: ["pdf_report.py"],
    effort: "Phase 2",
    color: "#8B5CF6",
    auditStatus: { langgraph: false, hitl: false, critic: false, langfuse: false, realData: false },
    qualityNote: "规划中 — 需从零设计含质量门",
  },
  {
    id: "customer",
    icon: "💬",
    name: "Customer Communication Agent",
    nameCn: "客户沟通Agent",
    status: "future",
    statusLabel: "规划中",
    description: "多语言客户沟通、询价自动回复、交期确认、售后跟进",
    mcpTools: ["auto_reply_inquiry", "confirm_delivery", "followup_aftersales"],
    a2aSkills: ["inquiry_handler", "delivery_communicator"],
    existingFiles: ["wechat_notify.py"],
    effort: "Phase 3 (Voice)",
    color: "#14B8A6",
    auditStatus: { langgraph: false, hitl: false, critic: false, langfuse: false, realData: false },
    qualityNote: "规划中 — 需从零设计含质量门",
  },
];

// ── 架构层 (R5修正: CrewAI→LangGraph, agent.json→agent-card.json) ─
const ARCH_LAYERS = [
  { name: "Frontend", nameCn: "前端层", items: [
    "Streamlit (现有) → Next.js (升级)",
    "A2UI v0.8 + AG-UI (规划中)",
    "WeChat Mini Program",
  ], color: "#58a6ff" },
  { name: "Gateway", nameCn: "网关层", items: [
    "MCP Gateway (OAuth 2.1, spec 2025-11-25)",
    "A2A Discovery (/.well-known/agent-card.json)",
    "Rate Limiting + Audit Log + mTLS",
  ], color: "#bc8cff" },
  { name: "Orchestrator", nameCn: "编排层", items: [
    "LangGraph 1.0 GA 统一编排 (7/7 Agent)",
    "LangChain 1.0 create_agent + Middleware",
    "Adaptive Gate (V8已有) + CriticAgent质量门",
    "HITL 一等API + Durable State",
  ], color: "#f778ba" },
  { name: "Agents", nameCn: "Agent层", items: [
    "Sales ✅ (LangGraph+HITL+Critic)",
    "Procurement ⚠ (需接入质量门)",
    "Quality ⚠ (需接入质量门)",
    "Finance ⚠ (需接入质量门)",
    "Market ⚠ (需接入质量门)",
    "HR / Doc / Customer ⏳",
  ], color: "#00FF88" },
  { name: "Infra", nameCn: "基础设施", items: [
    "Langfuse v3 OTEL (全Agent覆盖)",
    "SQLite → PostgreSQL (生产)",
    "Memory V9 3D + MAGMA (已有)",
    "Docker V10 + Cloud Run/K8s",
    "Pydantic v2 结构化合约",
  ], color: "#6e7681" },
];

// ── 实施路线 (P0紧急修正 + 原Phase) ────────────────────────
const PHASES = [
  {
    phase: "P0",
    title: "紧急修正 — 版本+路径+幽灵依赖",
    duration: "30min - 1hr",
    color: "#ef4444",
    priority: "CRITICAL",
    tasks: [
      "requirements: langgraph>=1.0 (从>=0.3)",
      "requirements: langchain>=1.0, langchain-core>=1.0 (从>=0.3)",
      "清理CrewAI幽灵import (multi_agent.py:121) 或升级到1.9.3",
      "a2a_server_v7.py: agent.json→agent-card.json (5处)",
      "MCP spec对齐2025-11-25 (Tasks+Extensions)",
    ],
    outcome: "版本、路径、依赖全部对齐2026最前沿标准"
  },
  {
    phase: "P0",
    title: "V10 Agent 质量门补齐",
    duration: "2-3 天",
    color: "#ef4444",
    priority: "CRITICAL",
    tasks: [
      "4个V10 Agent接入LangGraph统一编排",
      "每个V10 Agent添加CriticAgent调用",
      "每个V10 Agent添加HITL检查点",
      "每个V10 Agent接入Langfuse trace",
      "DataBridge: Excel真实数据灌入V10 Agent (3-5天)",
    ],
    outcome: "7/7 Agent统一编排 · 质量门不可绕过 · 全链路可观测"
  },
  {
    phase: "P1",
    title: "框架升级 + 协议对齐",
    duration: "1 周",
    color: "#f59e0b",
    priority: "HIGH",
    tasks: [
      "create_agent() + Middleware重构 (LangChain 1.0)",
      "MCP升级2025-11-25 Tasks异步",
      "Pydantic v2输出模型 (替代dict通信)",
      "A2A迁移官方SDK + gRPC传输",
      "docker-compose更新V10 + PostgreSQL",
    ],
    outcome: "可写入BP: 对标LangGraph 1.0 + MCP 2025-11-25 + A2A v0.3"
  },
  {
    phase: "P2",
    title: "扩展Agent + 企业级",
    duration: "Week 4-8",
    color: "#38bdf8",
    priority: "MEDIUM",
    tasks: [
      "Quality Control Agent (品质管控)",
      "Finance Agent (财务分析)",
      "Market Intelligence Agent (市场情报)",
      "跨Agent协作: 销售异常→品质关联→财务影响",
      "评估Deep Agents 0.4.1 + MCP Registry",
      "Cloud Run/K8s实际部署",
    ],
    outcome: "5个Agent覆盖核心业务 · 真实部署"
  },
  {
    phase: "P3",
    title: "企业级 + Voice + 开放",
    duration: "Week 9-14",
    color: "#22c55e",
    priority: "FUTURE",
    tasks: [
      "Customer Voice Agent (语音客服)",
      "Document Agent (文档处理)",
      "HR Agent (人事运营)",
      "A2UI/AG-UI前端升级",
      "ADK封装 · Node Caching · Durable State · reasoning models",
      "白标系统: 可卖给其他ODM公司",
    ],
    outcome: "完整ODM/OEM企业Agent平台 · 断档式领先"
  },
];

// ── 协作拓扑 (不变) ─────────────────────────────────────────
const COLLAB_SCENARIOS = [
  {
    scenario: "客户出货异常 → 全链路追踪",
    flow: ["📊 Sales", "🔍 Quality", "💰 Finance", "🌍 Market"],
    desc: "Sales检测到客户A出货量骤降 → 委派Quality查良品率 → Finance查应收账款 → Market查是否竞品抢单",
    color: "#FF4136",
  },
  {
    scenario: "新客户询价 → 智能报价",
    flow: ["💬 Customer", "📊 Sales", "🛒 Procurement", "💰 Finance"],
    desc: "Customer Agent收到询价 → Sales查历史定价 → Procurement查物料成本 → Finance算毛利空间 → 自动生成报价单",
    color: "#0074D9",
  },
  {
    scenario: "月度经营复盘 → 自动生成报告",
    flow: ["📊 Sales", "🛒 Procurement", "🔍 Quality", "💰 Finance", "📄 Doc"],
    desc: "每月1号自动触发 → 各Agent提交本月数据 → Doc Agent汇总生成PDF报告 → WeChat推送管理层",
    color: "#2ECC40",
  },
  {
    scenario: "供应商交期延迟 → 主动应对",
    flow: ["🛒 Procurement", "📊 Sales", "💬 Customer"],
    desc: "Procurement检测到供应商延迟 → 查Sales受影响订单 → Customer Agent主动通知受影响客户并协商新交期",
    color: "#F59E0B",
  },
];

// ── delta映射 ────────────────────────────────────────────────
const DELTA_MAP = {
  "0":  { label: "✅", color: "#22c55e" },
  "-1": { label: "微", color: "#38bdf8" },
  "-2": { label: "改", color: "#f59e0b" },
  "-3": { label: "差", color: "#ef4444" },
  "-4": { label: "🚨", color: "#dc2626" },
};

// ── section映射 ──────────────────────────────────────────────
const SECTION_MAP = {
  protocols: { icon: "🔗", label: "协议层" },
  frameworks: { icon: "🏗️", label: "框架层 (核心)" },
  patterns: { icon: "🧩", label: "设计模式" },
  infra: { icon: "⚙️", label: "基础设施" },
  data: { icon: "📊", label: "数据层" },
};

// ════════════════════════════════════════════════════════════
// 子组件
// ════════════════════════════════════════════════════════════

function AuditRoundTabs() {
  const [activeRound, setActiveRound] = useState(5);
  const totalFindings = AUDIT_ROUNDS.reduce((sum, r) => sum + r.items.length, 0);
  const criticalCount = AUDIT_ROUNDS.reduce((sum, r) => sum + r.items.filter(i => i.severity === "critical").length, 0);
  const round = AUDIT_ROUNDS.find(r => r.id === activeRound);

  const severityStyle = {
    critical: { bg: "rgba(239,68,68,0.12)", color: "#ef4444", label: "🔴" },
    warn: { bg: "rgba(245,158,11,0.12)", color: "#f59e0b", label: "🟡" },
    ok: { bg: "rgba(34,197,94,0.12)", color: "#22c55e", label: "🟢" },
  };

  return (
    <div style={{ marginBottom: 16, padding: "14px 16px", borderRadius: 10, background: "rgba(244,114,182,0.04)", border: "1px solid rgba(244,114,182,0.12)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
        <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "#f472b6" }}>五轮审计修正</span>
        <div style={{ display: "flex", gap: 3, marginLeft: 8 }}>
          {AUDIT_ROUNDS.map(r => (
            <button key={r.id} onClick={() => setActiveRound(r.id)} style={{
              padding: "3px 10px", borderRadius: 4, border: "none", cursor: "pointer",
              background: activeRound === r.id ? r.color + "30" : "rgba(255,255,255,0.04)",
              color: activeRound === r.id ? r.color : "#64748b",
              fontSize: "0.65rem", fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
            }}>
              {r.label} ({r.items.length})
            </button>
          ))}
        </div>
        <span style={{ marginLeft: "auto", fontSize: "0.6rem", color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
          {totalFindings} 处修正 · {criticalCount} 严重
        </span>
      </div>
      {round?.items.map((item, i) => {
        const sev = severityStyle[item.severity];
        return (
          <div key={i} style={{
            marginBottom: 4, padding: "6px 10px", borderRadius: 5,
            background: sev.bg, fontSize: "0.7rem", lineHeight: 1.6,
            border: `1px solid ${sev.color}15`,
          }}>
            <span style={{ fontWeight: 700, marginRight: 6 }}>{sev.label}</span>
            <span style={{ color: "#f1f5f9", fontWeight: 600 }}>{item.title}</span>
            <span style={{ color: "#94a3b8", marginLeft: 8 }}>— {item.detail}</span>
            {item.src !== "N/A" && (
              <code style={{ marginLeft: 8, fontSize: "0.58rem", color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
                {item.src}
              </code>
            )}
          </div>
        );
      })}
    </div>
  );
}

function FrameworkLandscape() {
  const statusColors = {
    using: "#22c55e",
    ghost: "#f59e0b",
    missing: "#ef4444",
    optional: "#38bdf8",
  };

  return (
    <div style={{ marginBottom: 16, padding: "14px 16px", borderRadius: 10, background: "rgba(167,139,250,0.03)", border: "1px solid rgba(167,139,250,0.1)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "#a78bfa" }}>2026.2 Agent框架全景</span>
        <span style={{ fontSize: "0.58rem", color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
          Iterathon: 86%企业用agent框架 · 三足鼎立
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6 }}>
        {FRAMEWORK_LANDSCAPE.map(fw => (
          <div key={fw.name} style={{
            padding: "8px 10px", borderRadius: 6,
            background: "rgba(255,255,255,0.025)",
            border: `1px solid ${statusColors[fw.status]}18`,
            borderLeft: `3px solid ${statusColors[fw.status]}`,
            fontSize: "0.7rem", lineHeight: 1.6,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontWeight: 700, color: "#f1f5f9" }}>{fw.name}</span>
              <span style={{ fontSize: "0.55rem", color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>{fw.version}</span>
            </div>
            <div style={{ color: statusColors[fw.status], fontWeight: 600, fontSize: "0.62rem" }}>{fw.statusNote}</div>
            <div style={{ color: "#475569", fontSize: "0.58rem" }}>{fw.desc}</div>
            <div style={{ color: "#334155", fontSize: "0.52rem", fontFamily: "'JetBrains Mono', monospace" }}>{fw.release}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function QualityGateBar({ agent }) {
  const gates = [
    { key: "langgraph", label: "LangGraph", icon: "🔗" },
    { key: "hitl", label: "HITL", icon: "🧑" },
    { key: "critic", label: "Critic", icon: "🔍" },
    { key: "langfuse", label: "Langfuse", icon: "📊" },
    { key: "realData", label: "真实数据", icon: "💾" },
  ];
  const passCount = gates.filter(g => agent.auditStatus[g.key]).length;
  const total = gates.length;

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 4 }}>
        <span style={{ fontSize: "0.55rem", color: "#64748b", fontWeight: 600, letterSpacing: "0.1em" }}>质量门</span>
        <span style={{
          fontSize: "0.55rem", fontFamily: "'JetBrains Mono', monospace", fontWeight: 700,
          color: passCount === total ? "#22c55e" : passCount >= 3 ? "#f59e0b" : "#ef4444",
        }}>
          {passCount}/{total}
        </span>
      </div>
      <div style={{ display: "flex", gap: 3 }}>
        {gates.map(gate => {
          const pass = agent.auditStatus[gate.key];
          return (
            <span key={gate.key} style={{
              fontSize: "0.55rem", padding: "2px 6px", borderRadius: 3,
              background: pass ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.08)",
              color: pass ? "#22c55e" : "#ef4444",
              border: `1px solid ${pass ? "#22c55e" : "#ef4444"}18`,
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {gate.icon} {gate.label}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function AuditDetailRow({ item, isOpen, onToggle }) {
  const d = DELTA_MAP[String(item.delta)] || DELTA_MAP["0"];
  return (
    <div style={{ marginBottom: 4 }}>
      <div onClick={onToggle} style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "8px 12px", cursor: "pointer",
        background: isOpen ? "rgba(255,255,255,0.04)" : item.critical ? "#ef444406" : "rgba(255,255,255,0.015)",
        border: `1px solid ${isOpen ? d.color + "28" : "rgba(255,255,255,0.04)"}`,
        borderRadius: isOpen ? "6px 6px 0 0" : 6,
        transition: "all 0.2s",
      }}>
        <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "#e2e8f0" }}>
          {item.critical && <span style={{ color: "#ef4444", marginRight: 4 }}>!!</span>}
          {item.name}
        </span>
        <span style={{
          fontSize: "0.6rem", fontWeight: 700, padding: "2px 7px", borderRadius: 4,
          background: d.color + "18", color: d.color,
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          {d.label} {item.delta}
        </span>
      </div>
      {isOpen && (
        <div style={{
          padding: "10px 12px", fontSize: "0.7rem", lineHeight: 1.6,
          background: "rgba(255,255,255,0.02)",
          border: `1px solid ${d.color}15`, borderTop: "none",
          borderRadius: "0 0 6px 6px",
        }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 8 }}>
            <div>
              <div style={{ color: "#64748b", fontWeight: 700, fontSize: "0.58rem", marginBottom: 2, letterSpacing: "0.05em" }}>最前沿 2026.2.14</div>
              <div style={{ color: "#cbd5e1", whiteSpace: "pre-line" }}>{item.frontier}</div>
            </div>
            <div>
              <div style={{ color: "#64748b", fontWeight: 700, fontSize: "0.58rem", marginBottom: 2, letterSpacing: "0.05em" }}>你的代码</div>
              <div style={{ color: "#cbd5e1", whiteSpace: "pre-line" }}>{item.current}</div>
            </div>
          </div>
          <div style={{ padding: "5px 10px", borderRadius: 4, background: d.color + "08", marginBottom: 4 }}>
            <span style={{ color: "#38bdf8", fontWeight: 700 }}>→ </span>
            <span style={{ color: "#e2e8f0" }}>{item.fix}</span>
          </div>
          <div style={{ fontSize: "0.55rem", color: "#475569", fontFamily: "'JetBrains Mono', monospace" }}>
            {item.src}
          </div>
        </div>
      )}
    </div>
  );
}

function CoreFindings() {
  return (
    <div style={{
      padding: "12px 16px", borderRadius: 8, marginBottom: 16,
      background: "linear-gradient(135deg, #ef444406, #f59e0b04)",
      border: "1px solid #ef444418",
    }}>
      <div style={{ fontSize: "0.72rem", fontWeight: 700, color: "#ef4444", marginBottom: 6 }}>
        五轮验证后的核心结论
      </div>
      <div style={{ fontSize: "0.7rem", lineHeight: 1.8, color: "#cbd5e1" }}>
        <code style={{ background: "#1e293b", padding: "1px 4px", borderRadius: 2, fontSize: "0.62rem", fontFamily: "'JetBrains Mono', monospace" }}>langgraph&gt;=0.3</code>
        {" + "}
        <code style={{ background: "#1e293b", padding: "1px 4px", borderRadius: 2, fontSize: "0.62rem", fontFamily: "'JetBrains Mono', monospace" }}>langchain-core&gt;=0.3</code>
        {" = "}
        <strong style={{ color: "#f59e0b" }}>落后整整两代</strong>
        <br />
        4个V10 Agent <strong style={{ color: "#f59e0b" }}>零HITL、零CriticAgent、零Langfuse</strong> (grep确认)
        <br />
        CrewAI在代码中import但requirements注释 → <strong style={{ color: "#ef4444" }}>幽灵依赖</strong>
        <br />
        A2A 5处 <code style={{ background: "#1e293b", padding: "1px 4px", borderRadius: 2, fontSize: "0.62rem", fontFamily: "'JetBrains Mono', monospace" }}>agent.json</code> → 应为 <code style={{ background: "#1e293b", padding: "1px 4px", borderRadius: 2, fontSize: "0.62rem", fontFamily: "'JetBrains Mono', monospace" }}>agent-card.json</code> (A2A v0.3.0)
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// 主组件
// ════════════════════════════════════════════════════════════

export default function PlatformArchitecture() {
  const [view, setView] = useState("agents");
  const [expandedAgent, setExpandedAgent] = useState(null);
  const [expandedPhase, setExpandedPhase] = useState(0);
  const [auditOpen, setAuditOpen] = useState({});

  const toggleAudit = (idx) => setAuditOpen(prev => ({ ...prev, [idx]: !prev[idx] }));

  const statusColors = { live: "#00FF88", new: "#FF6B35", future: "#6e7681" };
  const statusBg = { live: "rgba(0,255,136,0.08)", new: "rgba(255,107,53,0.08)", future: "rgba(110,118,129,0.05)" };

  // 审计统计
  const auditStats = useMemo(() => {
    const counts = {};
    AUDIT_ITEMS.forEach(item => {
      const k = String(item.delta);
      counts[k] = (counts[k] || 0) + 1;
    });
    return counts;
  }, []);

  const sections = useMemo(() => [...new Set(AUDIT_ITEMS.map(i => i.section))], []);

  // Agent质量统计
  const agentQualityStats = useMemo(() => {
    const v10 = PLATFORM_AGENTS.filter(a => a.status === "new");
    const withGates = v10.filter(a => a.auditStatus.hitl && a.auditStatus.critic);
    return { total: v10.length, withGates: withGates.length };
  }, []);

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(165deg, #040710, #0b1020 40%, #090e18)",
      color: "#e6edf3",
      fontFamily: "'DM Sans', 'Noto Sans SC', -apple-system, sans-serif",
      padding: "1.5rem 1rem",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;600&display=swap');
        * { box-sizing: border-box; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
        @keyframes pulse2 { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
      `}</style>

      {/* ── Header ── */}
      <div style={{ maxWidth: 960, margin: "0 auto 1.5rem", textAlign: "center" }}>
        <div style={{
          display: "inline-flex", gap: 6, marginBottom: 8,
          fontSize: "0.6rem", fontFamily: "'JetBrains Mono', monospace",
        }}>
          <span style={{ padding: "3px 10px", borderRadius: 20, background: "rgba(0,255,136,0.1)", color: "#00FF88" }}>V10.0 PLATFORM</span>
          <span style={{ padding: "3px 10px", borderRadius: 20, background: "rgba(188,140,255,0.1)", color: "#bc8cff" }}>MCP 2025-11-25 + A2A v0.3.0</span>
          <span style={{ padding: "3px 10px", borderRadius: 20, background: "rgba(239,68,68,0.1)", color: "#ef4444" }}>R5 审计修正</span>
          <span style={{ padding: "3px 10px", borderRadius: 20, background: "rgba(247,120,186,0.1)", color: "#f778ba" }}>SPROCOMM</span>
        </div>
        <h1 style={{
          fontSize: "clamp(1.6rem, 3.5vw, 2.4rem)", fontWeight: 900, margin: "0.5rem 0 0.3rem",
          background: "linear-gradient(135deg, #00FF88, #58a6ff, #bc8cff, #f472b6)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>
          MRARFAI 企业Agent平台
        </h1>
        <p style={{ color: "#6e7681", fontSize: "0.78rem", margin: "0 0 4px" }}>
          从「销售分析工具」→「多Agent企业智能平台」· 禾苗通讯全业务覆盖
        </p>
        <p style={{ color: "#475569", fontSize: "0.6rem", margin: 0, fontFamily: "'JetBrains Mono', monospace" }}>
          LangGraph 1.0 GA · LangChain 1.0 · MCP 2025-11-25 · A2A v0.3.0 · agent-card.json · Pydantic v2
        </p>
      </div>

      {/* ── Nav Tabs ── */}
      <div style={{ maxWidth: 960, margin: "0 auto 1.2rem", display: "flex", gap: 5, justifyContent: "center", flexWrap: "wrap" }}>
        {[
          { key: "agents", label: "Agent矩阵", count: PLATFORM_AGENTS.length, icon: "🤖" },
          { key: "audit", label: "R5审计", count: AUDIT_ITEMS.length, icon: "🔬" },
          { key: "arch", label: "架构层级", icon: "🏗️" },
          { key: "phases", label: "实施路线", count: PHASES.length, icon: "📅" },
          { key: "collab", label: "协作拓扑", icon: "🔗" },
        ].map(t => (
          <button key={t.key} onClick={() => setView(t.key)} style={{
            padding: "7px 14px", borderRadius: 8, fontSize: "0.75rem",
            border: view === t.key ? "1px solid #58a6ff" : "1px solid #1c2128",
            background: view === t.key ? "rgba(88,166,255,0.1)" : "transparent",
            color: view === t.key ? "#58a6ff" : "#6e7681",
            cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
          }}>
            {t.icon} {t.label} {t.count ? `(${t.count})` : ""}
          </button>
        ))}
      </div>

      <div style={{ maxWidth: 960, margin: "0 auto" }}>

        {/* ══════════ AGENTS VIEW ══════════ */}
        {view === "agents" && (
          <div>
            {/* Stats bar */}
            <div style={{
              display: "flex", gap: 16, marginBottom: 12, justifyContent: "center", flexWrap: "wrap",
              fontSize: "0.7rem", fontFamily: "'JetBrains Mono', monospace",
            }}>
              <span><span style={{ color: "#00FF88" }}>●</span> 已有 {PLATFORM_AGENTS.filter(a => a.status === "live").length}</span>
              <span><span style={{ color: "#FF6B35" }}>●</span> V10新建 {PLATFORM_AGENTS.filter(a => a.status === "new").length}</span>
              <span><span style={{ color: "#6e7681" }}>●</span> 规划 {PLATFORM_AGENTS.filter(a => a.status === "future").length}</span>
              <span style={{ color: "#484f58" }}>MCP: {PLATFORM_AGENTS.reduce((s, a) => s + a.mcpTools.length, 0)}</span>
              <span style={{ color: "#484f58" }}>A2A: {PLATFORM_AGENTS.reduce((s, a) => s + a.a2aSkills.length, 0)}</span>
              <span style={{ color: agentQualityStats.withGates === agentQualityStats.total ? "#22c55e" : "#ef4444" }}>
                质量门: {agentQualityStats.withGates}/{agentQualityStats.total} V10
              </span>
            </div>

            {/* V10 Alert Banner */}
            {agentQualityStats.withGates < agentQualityStats.total && (
              <div style={{
                marginBottom: 12, padding: "10px 14px", borderRadius: 8,
                background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)",
                fontSize: "0.7rem", color: "#fca5a5",
              }}>
                <strong style={{ color: "#ef4444" }}>R5审计发现:</strong>{" "}
                {agentQualityStats.total - agentQualityStats.withGates}个V10 Agent 零HITL · 零CriticAgent · 零Langfuse trace · 纯关键词路由→直接返回。
                需接入LangGraph 1.0统一编排+质量门。
              </div>
            )}

            {/* Agent Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 10 }}>
              {PLATFORM_AGENTS.map((agent, i) => (
                <div key={agent.id}
                  onClick={() => setExpandedAgent(expandedAgent === agent.id ? null : agent.id)}
                  style={{
                    animation: `fadeUp 0.35s ease ${i * 0.05}s both`,
                    background: statusBg[agent.status],
                    border: expandedAgent === agent.id
                      ? `1px solid ${agent.color}55`
                      : `1px solid ${agent.color}15`,
                    borderRadius: 12, padding: "1.2rem", cursor: "pointer",
                    transition: "all 0.25s",
                    borderLeft: `3px solid ${agent.color}`,
                  }}
                >
                  {/* Header */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: "1.5rem" }}>{agent.icon}</span>
                      <div>
                        <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "#f0f6fc" }}>{agent.nameCn}</div>
                        <div style={{ fontSize: "0.62rem", fontFamily: "'JetBrains Mono', monospace", color: "#6e7681" }}>{agent.name}</div>
                      </div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3 }}>
                      <span style={{
                        fontSize: "0.58rem", padding: "2px 8px", borderRadius: 999,
                        background: `${statusColors[agent.status]}18`,
                        color: statusColors[agent.status],
                        border: `1px solid ${statusColors[agent.status]}30`,
                        fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
                      }}>
                        {agent.statusLabel}
                      </span>
                      {agent.auditDelta !== undefined && (
                        <span style={{
                          fontSize: "0.5rem", padding: "1px 5px", borderRadius: 3,
                          background: "#ef444418", color: "#ef4444",
                          fontFamily: "'JetBrains Mono', monospace", fontWeight: 700,
                        }}>
                          审计 {DELTA_MAP[String(agent.auditDelta)]?.label} {agent.auditDelta}
                        </span>
                      )}
                    </div>
                  </div>

                  <p style={{ fontSize: "0.75rem", color: "#8b949e", lineHeight: 1.6, margin: "0 0 8px" }}>
                    {agent.description}
                  </p>

                  {/* MCP Tools count + Effort */}
                  <div style={{ display: "flex", gap: 10, fontSize: "0.65rem", fontFamily: "'JetBrains Mono', monospace" }}>
                    <span style={{ color: "#58a6ff" }}>{agent.mcpTools.length} MCP</span>
                    <span style={{ color: "#bc8cff" }}>{agent.a2aSkills.length} A2A</span>
                    <span style={{ color: "#6e7681", marginLeft: "auto" }}>{agent.effort}</span>
                  </div>

                  {/* Quality Gate Bar (always visible for non-future agents) */}
                  {agent.status !== "future" && <QualityGateBar agent={agent} />}

                  {/* Expanded */}
                  {expandedAgent === agent.id && (
                    <div style={{ animation: "fadeUp 0.3s ease", marginTop: 12, paddingTop: 12, borderTop: `1px solid ${agent.color}20` }}>
                      {/* Audit Note */}
                      <div style={{
                        marginBottom: 10, padding: "6px 10px", borderRadius: 5,
                        background: agent.auditStatus.hitl ? "rgba(34,197,94,0.06)" : "rgba(239,68,68,0.06)",
                        border: `1px solid ${agent.auditStatus.hitl ? "#22c55e" : "#ef4444"}15`,
                        fontSize: "0.62rem", color: agent.auditStatus.hitl ? "#86efac" : "#fca5a5",
                        fontFamily: "'JetBrains Mono', monospace",
                      }}>
                        {agent.qualityNote}
                      </div>

                      {/* MCP Tools */}
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ fontSize: "0.55rem", color: "#58a6ff", fontWeight: 600, letterSpacing: "0.1em", marginBottom: 4 }}>MCP TOOLS (spec 2025-11-25)</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                          {agent.mcpTools.map(t => (
                            <code key={t} style={{
                              fontSize: "0.6rem", padding: "2px 6px", borderRadius: 4,
                              background: "rgba(88,166,255,0.08)", color: "#58a6ff",
                              border: "1px solid rgba(88,166,255,0.15)",
                            }}>{t}</code>
                          ))}
                        </div>
                      </div>
                      {/* A2A Skills */}
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ fontSize: "0.55rem", color: "#bc8cff", fontWeight: 600, letterSpacing: "0.1em", marginBottom: 4 }}>A2A SKILLS (v0.3.0 agent-card.json)</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                          {agent.a2aSkills.map(s => (
                            <code key={s} style={{
                              fontSize: "0.6rem", padding: "2px 6px", borderRadius: 4,
                              background: "rgba(188,140,255,0.08)", color: "#bc8cff",
                              border: "1px solid rgba(188,140,255,0.15)",
                            }}>{s}</code>
                          ))}
                        </div>
                      </div>
                      {/* Existing files */}
                      {agent.existingFiles.length > 0 && (
                        <div style={{ marginBottom: 10 }}>
                          <div style={{ fontSize: "0.55rem", color: "#00FF88", fontWeight: 600, letterSpacing: "0.1em", marginBottom: 4 }}>已有代码可复用</div>
                          <div style={{ fontSize: "0.62rem", color: "#3fb950", fontFamily: "'JetBrains Mono', monospace" }}>
                            {agent.existingFiles.join(" · ")}
                          </div>
                        </div>
                      )}
                      {/* Key Tech */}
                      {agent.keyTech && (
                        <div style={{
                          fontSize: "0.68rem", color: "#8b949e", lineHeight: 1.6,
                          padding: "8px 10px", background: "rgba(0,0,0,0.3)", borderRadius: 6,
                        }}>
                          {agent.keyTech}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ══════════ AUDIT VIEW ══════════ */}
        {view === "audit" && (
          <div>
            <AuditRoundTabs />
            <FrameworkLandscape />
            <CoreFindings />

            {/* Score bar */}
            <div style={{ display: "flex", gap: 1, marginBottom: 14, height: 24, borderRadius: 5, overflow: "hidden" }}>
              {Object.entries(auditStats).sort((a, b) => Number(b[0]) - Number(a[0])).map(([k, v]) => {
                const d = DELTA_MAP[k] || DELTA_MAP["0"];
                return (
                  <div key={k} style={{
                    flex: v, display: "flex", alignItems: "center", justifyContent: "center",
                    background: d.color + "22",
                    fontSize: "0.6rem", fontWeight: 700, color: d.color,
                    fontFamily: "'JetBrains Mono', monospace",
                  }}>
                    {v}{d.label}
                  </div>
                );
              })}
            </div>

            {/* Audit sections */}
            {sections.map(sec => {
              const info = SECTION_MAP[sec];
              return (
                <div key={sec} style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#f1f5f9", marginBottom: 5 }}>
                    {info.icon} {info.label}
                  </div>
                  {AUDIT_ITEMS.filter(i => i.section === sec).map((item) => {
                    const idx = AUDIT_ITEMS.indexOf(item);
                    return (
                      <AuditDetailRow
                        key={idx}
                        item={item}
                        isOpen={!!auditOpen[idx]}
                        onToggle={() => toggleAudit(idx)}
                      />
                    );
                  })}
                </div>
              );
            })}

            {/* Roadmap */}
            <div style={{ padding: 14, borderRadius: 8, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
              <h3 style={{ fontSize: "0.82rem", fontWeight: 700, color: "#f1f5f9", margin: "0 0 10px" }}>断档式领先路径</h3>
              <div style={{ fontSize: "0.7rem", lineHeight: 2 }}>
                {[
                  { p: "P0", c: "#ef4444", t: "30分钟: langgraph>=1.0 · langchain>=1.0 · 清理CrewAI幽灵import" },
                  { p: "P0", c: "#ef4444", t: "1小时: agent.json→agent-card.json (5处)" },
                  { p: "P0", c: "#ef4444", t: "3-5天: DataBridge接真实数据到4个V10 Agent" },
                  { p: "P0", c: "#ef4444", t: "2-3天: 4 V10 Agent接入LangGraph统一编排+CriticAgent+HITL" },
                  { p: "P1", c: "#f59e0b", t: "1周: create_agent+Middleware重构 · MCP升级2025-11-25 Tasks · Pydantic" },
                  { p: "P1", c: "#f59e0b", t: "3-5天: 全7 Agent接Langfuse · 云端部署 · docker-compose V10" },
                  { p: "P2", c: "#38bdf8", t: "1-2周: 评估Deep Agents · A2A官方SDK+gRPC · MCP Registry" },
                  { p: "P3", c: "#22c55e", t: "持续: ADK封装 · A2UI/AG-UI · Node Caching · Durable State · reasoning models" },
                ].map((r, i) => (
                  <div key={i}>
                    <span style={{
                      background: r.c + "20", color: r.c, padding: "2px 6px",
                      borderRadius: 3, fontWeight: 700, fontSize: "0.58rem",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>{r.p}</span>
                    <span style={{ marginLeft: 6, color: "#e2e8f0" }}>{r.t}</span>
                  </div>
                ))}
              </div>

              <div style={{
                marginTop: 10, padding: "10px 12px", borderRadius: 6,
                background: "rgba(167,139,250,0.04)", border: "1px solid rgba(167,139,250,0.08)",
                fontSize: "0.68rem", lineHeight: 1.8,
              }}>
                <strong style={{ color: "#a78bfa" }}>断档式领先 = 方向对了 + 版本跟上 + 数据灌入 + 编排统一</strong><br />
                <span style={{ color: "#22c55e" }}>✅ 做对了: LangGraph选型·MCP SDK·A2A概念·3D记忆·多模型·安全·45,437行</span><br />
                <span style={{ color: "#ef4444" }}>❌ 要修: 版本(0.3→1.0)·编排(3/7→7/7)·路径·数据·幽灵依赖</span><br />
                <span style={{ color: "#f1f5f9" }}>P0完成(~1周) → ODM/OEM领域断档式领先 · 无竞品可比</span><br />
                <span style={{ color: "#f1f5f9" }}>P0+P1完成(~3周) → BP可写:「对标LangGraph 1.0+MCP 2025-11-25+A2A v0.3」</span>
              </div>
            </div>
          </div>
        )}

        {/* ══════════ ARCHITECTURE VIEW ══════════ */}
        {view === "arch" && (
          <div>
            <div style={{ textAlign: "center", fontSize: "0.68rem", color: "#6e7681", marginBottom: 16, fontFamily: "'JetBrains Mono', monospace" }}>
              用户请求 → Gateway (MCP 2025-11-25) → LangGraph 1.0 Orchestrator → Agent → MCP Tools → 数据源
            </div>
            {ARCH_LAYERS.map((layer, i) => (
              <div key={layer.name} style={{
                animation: `fadeUp 0.4s ease ${i * 0.1}s both`,
                margin: "0 0 8px",
                padding: "14px 18px",
                background: `${layer.color}08`,
                border: `1px solid ${layer.color}20`,
                borderLeft: `3px solid ${layer.color}`,
                borderRadius: 10,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
                  <span style={{
                    fontSize: "0.58rem", padding: "2px 8px", borderRadius: 4,
                    background: `${layer.color}15`, color: layer.color,
                    fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
                  }}>
                    {layer.name}
                  </span>
                  <span style={{ fontSize: "0.85rem", fontWeight: 700, color: "#f0f6fc" }}>{layer.nameCn}</span>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {layer.items.map(item => (
                    <span key={item} style={{
                      fontSize: "0.68rem", padding: "3px 10px", borderRadius: 6,
                      background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
                      color: item.includes("已有") || item.includes("✅") ? "#00FF88"
                        : item.includes("⚠") ? "#f59e0b"
                        : item.includes("规划中") ? "#64748b"
                        : "#8b949e",
                    }}>
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            ))}

            {/* Protocol diagram */}
            <div style={{
              marginTop: 20, padding: 16,
              background: "rgba(88,166,255,0.04)", border: "1px solid rgba(88,166,255,0.12)",
              borderRadius: 10, textAlign: "center",
            }}>
              <div style={{ fontSize: "0.68rem", color: "#58a6ff", fontWeight: 600, letterSpacing: "0.1em", marginBottom: 10 }}>
                协议标准 (R5修正)
              </div>
              <div style={{
                display: "flex", justifyContent: "center", gap: 24, flexWrap: "wrap",
                fontSize: "0.72rem", fontFamily: "'JetBrains Mono', monospace",
              }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: "1.5rem", marginBottom: 4 }}>🔌</div>
                  <div style={{ color: "#58a6ff", fontWeight: 600 }}>MCP</div>
                  <div style={{ color: "#6e7681", fontSize: "0.58rem" }}>Agent ↔ Tools</div>
                  <div style={{ color: "#484f58", fontSize: "0.52rem" }}>spec 2025-11-25</div>
                  <div style={{ color: "#484f58", fontSize: "0.52rem" }}>Tasks + Extensions + CIMD</div>
                </div>
                <div style={{ color: "#30363d", fontSize: "1.5rem", alignSelf: "center" }}>+</div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: "1.5rem", marginBottom: 4 }}>🤝</div>
                  <div style={{ color: "#bc8cff", fontWeight: 600 }}>A2A v0.3.0</div>
                  <div style={{ color: "#6e7681", fontSize: "0.58rem" }}>Agent ↔ Agent</div>
                  <div style={{ color: "#484f58", fontSize: "0.52rem" }}>agent-card.json</div>
                  <div style={{ color: "#484f58", fontSize: "0.52rem" }}>gRPC + REST + JSON-RPC</div>
                </div>
                <div style={{ color: "#30363d", fontSize: "1.5rem", alignSelf: "center" }}>+</div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: "1.5rem", marginBottom: 4 }}>🛡️</div>
                  <div style={{ color: "#f778ba", fontWeight: 600 }}>OAuth 2.1</div>
                  <div style={{ color: "#6e7681", fontSize: "0.58rem" }}>认证 + 权限</div>
                  <div style={{ color: "#484f58", fontSize: "0.52rem" }}>mTLS + RBAC</div>
                  <div style={{ color: "#484f58", fontSize: "0.52rem" }}>Audit Trail</div>
                </div>
                <div style={{ color: "#30363d", fontSize: "1.5rem", alignSelf: "center" }}>+</div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: "1.5rem", marginBottom: 4 }}>🖥️</div>
                  <div style={{ color: "#22c55e", fontWeight: 600 }}>A2UI v0.8</div>
                  <div style={{ color: "#6e7681", fontSize: "0.58rem" }}>Agent → UI</div>
                  <div style={{ color: "#484f58", fontSize: "0.52rem" }}>声明式协议</div>
                  <div style={{ color: "#484f58", fontSize: "0.52rem" }}>Google + AG-UI</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════ PHASES VIEW ══════════ */}
        {view === "phases" && (
          <div>
            {PHASES.map((phase, i) => (
              <div key={`${phase.phase}-${i}`}
                onClick={() => setExpandedPhase(expandedPhase === i ? -1 : i)}
                style={{
                  animation: `fadeUp 0.4s ease ${i * 0.08}s both`,
                  margin: "0 0 10px", padding: "16px 20px",
                  background: expandedPhase === i ? `${phase.color}0a` : "rgba(255,255,255,0.02)",
                  border: expandedPhase === i ? `1px solid ${phase.color}30` : "1px solid rgba(255,255,255,0.06)",
                  borderRadius: 12, cursor: "pointer", transition: "all 0.25s",
                }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{
                      fontSize: "0.65rem", padding: "4px 10px", borderRadius: 6,
                      background: `${phase.color}15`, color: phase.color,
                      fontFamily: "'JetBrains Mono', monospace", fontWeight: 700,
                    }}>
                      {phase.phase}
                    </span>
                    {phase.priority && (
                      <span style={{
                        fontSize: "0.5rem", padding: "2px 6px", borderRadius: 3,
                        background: phase.priority === "CRITICAL" ? "#ef444420" : phase.priority === "HIGH" ? "#f59e0b20" : "#38bdf820",
                        color: phase.priority === "CRITICAL" ? "#ef4444" : phase.priority === "HIGH" ? "#f59e0b" : "#38bdf8",
                        fontFamily: "'JetBrains Mono', monospace", fontWeight: 700,
                      }}>
                        {phase.priority}
                      </span>
                    )}
                    <span style={{ fontSize: "0.95rem", fontWeight: 700, color: "#f0f6fc" }}>{phase.title}</span>
                  </div>
                  <span style={{ fontSize: "0.7rem", color: "#6e7681", fontFamily: "'JetBrains Mono', monospace" }}>{phase.duration}</span>
                </div>

                {expandedPhase === i && (
                  <div style={{ animation: "fadeUp 0.3s ease", marginTop: 14 }}>
                    {phase.tasks.map((task, j) => (
                      <div key={j} style={{
                        display: "flex", alignItems: "center", gap: 8,
                        padding: "6px 0", fontSize: "0.78rem", color: "#c9d1d9",
                        borderBottom: j < phase.tasks.length - 1 ? "1px solid rgba(255,255,255,0.03)" : "none",
                      }}>
                        <span style={{ color: phase.color, fontSize: "0.65rem", fontFamily: "'JetBrains Mono', monospace" }}>
                          {String(j + 1).padStart(2, "0")}
                        </span>
                        {task}
                      </div>
                    ))}
                    <div style={{
                      marginTop: 12, padding: "8px 12px",
                      background: `${phase.color}08`, borderRadius: 6,
                      fontSize: "0.75rem", color: phase.color,
                      border: `1px solid ${phase.color}15`,
                    }}>
                      交付物: {phase.outcome}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Timeline bar */}
            <div style={{
              marginTop: 20, padding: "12px 16px",
              background: "rgba(255,255,255,0.02)", borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.06)",
            }}>
              <div style={{ fontSize: "0.6rem", color: "#6e7681", fontFamily: "'JetBrains Mono', monospace", marginBottom: 8 }}>
                TIMELINE: P0 (1周) → P1 (1周) → P2 (5周) → P3 (6周)
              </div>
              <div style={{ display: "flex", gap: 2, height: 24, borderRadius: 6, overflow: "hidden" }}>
                <div style={{ flex: 1, background: "#ef444430", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <span style={{ fontSize: "0.55rem", color: "#ef4444", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>P0: 1w</span>
                </div>
                <div style={{ flex: 1, background: "#f59e0b30", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <span style={{ fontSize: "0.55rem", color: "#f59e0b", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>P1: 1w</span>
                </div>
                <div style={{ flex: 5, background: "#38bdf830", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <span style={{ fontSize: "0.55rem", color: "#38bdf8", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>P2: 5w</span>
                </div>
                <div style={{ flex: 6, background: "#22c55e30", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <span style={{ fontSize: "0.55rem", color: "#22c55e", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>P3: 6w</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════ COLLABORATION VIEW ══════════ */}
        {view === "collab" && (
          <div>
            <div style={{ textAlign: "center", fontSize: "0.72rem", color: "#6e7681", marginBottom: 16 }}>
              Agent之间通过A2A v0.3.0协议 (agent-card.json) 实时协作
            </div>

            {COLLAB_SCENARIOS.map((sc, i) => (
              <div key={i} style={{
                animation: `fadeUp 0.4s ease ${i * 0.08}s both`,
                margin: "0 0 10px", padding: "14px 18px",
                background: `${sc.color}06`, border: `1px solid ${sc.color}18`,
                borderRadius: 12,
              }}>
                <div style={{ fontSize: "0.88rem", fontWeight: 700, color: "#f0f6fc", marginBottom: 8 }}>
                  {sc.scenario}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
                  {sc.flow.map((agent, j) => (
                    <div key={j} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <span style={{
                        fontSize: "0.7rem", padding: "3px 8px", borderRadius: 6,
                        background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
                        color: "#e6edf3",
                      }}>
                        {agent}
                      </span>
                      {j < sc.flow.length - 1 && (
                        <span style={{ color: sc.color, fontSize: "0.8rem" }}>→</span>
                      )}
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: "0.72rem", color: "#8b949e", lineHeight: 1.6, margin: 0 }}>
                  {sc.desc}
                </p>
              </div>
            ))}

            {/* Value prop */}
            <div style={{
              marginTop: 20, padding: 16, textAlign: "center",
              background: "rgba(0,255,136,0.04)", border: "1px solid rgba(0,255,136,0.12)",
              borderRadius: 12,
            }}>
              <div style={{ fontSize: "0.92rem", fontWeight: 700, color: "#00FF88", marginBottom: 6 }}>
                平台价值 vs 单点工具
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, fontSize: "0.75rem", textAlign: "left", maxWidth: 600, margin: "0 auto" }}>
                <div>
                  <div style={{ color: "#6e7681", marginBottom: 4 }}>传统方式</div>
                  <div style={{ color: "#8b949e", lineHeight: 1.7 }}>
                    每个部门用不同工具<br />
                    数据孤岛无法联动<br />
                    问题发现靠人工巡检<br />
                    月度复盘手动做Excel
                  </div>
                </div>
                <div>
                  <div style={{ color: "#00FF88", marginBottom: 4 }}>Agent平台 (LangGraph 1.0)</div>
                  <div style={{ color: "#c9d1d9", lineHeight: 1.7 }}>
                    一个平台7/7 Agent协作<br />
                    A2A v0.3.0自动跨部门联动<br />
                    异常实时检测主动通知<br />
                    月报自动生成推送WeChat
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <div style={{
        maxWidth: 960, margin: "2rem auto 0", textAlign: "center",
        fontSize: "0.55rem", color: "#21262d", fontFamily: "'JetBrains Mono', monospace",
        lineHeight: 1.6,
      }}>
        MRARFAI V10.0 · R5审计修正 · {PLATFORM_AGENTS.length} Agents · {PLATFORM_AGENTS.reduce((s, a) => s + a.mcpTools.length, 0)} MCP Tools · {PLATFORM_AGENTS.reduce((s, a) => s + a.a2aSkills.length, 0)} A2A Skills
        <br />
        LangGraph 1.0 GA · LangChain 1.0 · MCP 2025-11-25 · A2A v0.3.0 · agent-card.json · Pydantic v2 · 2026.2.14
      </div>
    </div>
  );
}
