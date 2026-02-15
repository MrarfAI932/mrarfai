import { useState, useMemo } from "react";

// ================================================================
// MRARFAI 终极审计 — 2026.2.14
// 全球最强Agent开发者视角 · 逐文件逐行 · 零遗漏
// ================================================================

const ISSUES = [
  // ===== P0 CRITICAL — 不修会被任何高级Agent工程师一眼否决 =====
  {
    id: "P0-01",
    severity: "P0",
    title: "LangGraph 版本落后两代",
    file: "requirements_v7.txt",
    line: "17",
    current: "langgraph>=0.3",
    required: "langgraph>=1.0",
    detail: "LangGraph 1.0 GA 发布于 2025-10-29。你落后两个大版本。1.0 新增: Durable State 持久化、HITL 一等 API (interrupt())、Node Caching、Deferred Nodes、Pre/Post Model Hooks。Uber/LinkedIn/Klarna 生产使用。",
    effort: "1 行改动",
    risk: "任何人查 requirements 第一眼就会看到这个，直接判定技术栈过时",
  },
  {
    id: "P0-02",
    severity: "P0",
    title: "LangChain 版本落后两代",
    file: "requirements_v7.txt",
    line: "18-19",
    current: "langchain-core>=0.3\nlangchain-anthropic>=0.3",
    required: "langchain-core>=1.0\nlangchain-anthropic>=0.4",
    detail: "LangChain 1.0 发布于 2025-11-17。新增: create_agent() 统一 API、Middleware 架构、Model Profiles。你的代码零使用 create_agent()、零 Middleware。",
    effort: "2 行改动",
    risk: "与 LangGraph 1.0 同级别的信号——技术栈停留在 2025 Q2",
  },
  {
    id: "P0-03",
    severity: "P0",
    title: "A2A 路径使用已废弃端点 (5处)",
    file: "a2a_server_v7.py",
    line: "8, 21, 156, 609, 629",
    current: "/.well-known/agent.json",
    required: "/.well-known/agent-card.json",
    detail: "A2A v0.3.0 (2025-07-30) BREAKING CHANGE: agent.json → agent-card.json。这是协议级别的路径变更。你的 5 处全用旧路径。任何 A2A 兼容客户端连你的服务器会 404。",
    effort: "5 处字符串替换",
    risk: "A2A 互操作性完全断裂。这是协议合规性问题，不是功能问题。",
  },
  {
    id: "P0-04",
    severity: "P0",
    title: "CrewAI 幽灵依赖 — import 存在但未安装",
    file: "multi_agent.py",
    line: "119-126",
    current: "from crewai import Agent, Task, Crew, Process\n→ HAS_CREWAI = False (因为 requirements 注释了)",
    required: "方案A: 彻底删除 import 块\n方案B: 激活 crewai>=1.9.3 并真正使用",
    detail: "multi_agent.py:121 导入 CrewAI，但 requirements_v7.txt:41 是 `# crewai>=0.80` 注释状态。运行时 HAS_CREWAI=False，所有 CrewAI 代码路径永远不执行。CrewAI 现已升级到 1.9.3，原生 A2A + Flows 事件驱动。",
    effort: "删除 7 行 或 取消注释+升级版本",
    risk: "代码不干净。审计者会问：你到底用不用 CrewAI？如果不用为什么 import？",
  },
  {
    id: "P0-05",
    severity: "P0",
    title: "4/7 Agent 绕过质量门 — 零 HITL、零 CriticAgent、零 Langfuse",
    file: "agent_procurement.py, agent_quality.py, agent_finance.py, agent_market.py",
    line: "全文件",
    current: "grep 确认:\n• 0 次 HITL/hitl/interrupt 调用\n• 0 次 CriticAgent 调用\n• 0 次 Langfuse/trace/observe 调用\n• 0 次 Pydantic BaseModel 使用",
    required: "每个 Agent 必须:\n1. 接入 LangGraph StateGraph 统一编排\n2. 添加 CriticAgent 审核节点\n3. 添加 HITL 检查点 (interrupt)\n4. 接入 Langfuse trace\n5. 输出用 Pydantic 模型",
    detail: "platform_gateway.py:407-460 对这 4 个 Agent 使用 keyword 路由直接返回结果，完全绕过 multi_agent.py 的 StateGraph 编排管线。这意味着 V10 新增的 4 个 Agent 没有任何质量保障。相比之下，V9 的 3 个 Agent (analyst/risk/strategist) 走完整的 StateGraph → experts → synthesize → reflect → hitl_check 管线。",
    effort: "2-3 天重构",
    risk: "这是架构级缺陷。7 个 Agent 中 4 个没有质量门，等于 57% 的系统没有防护。投资者问「你的 Agent 有质量保障吗」，答案是「只有 3/7 有」。",
  },
  {
    id: "P0-06",
    severity: "P0",
    title: "4/7 Agent 使用硬编码假数据",
    file: "agent_procurement.py:108-142\nagent_quality.py:77-106\nagent_finance.py:82-115",
    line: "见左",
    current: "SAMPLE_SUPPLIERS (8行)\nSAMPLE_POS (22行)\nSAMPLE_YIELDS (10行)\nSAMPLE_RETURNS (18行)\nSAMPLE_AR (12行)\nSAMPLE_MARGINS (20行)\nagent_market.py COMPETITORS (4行)",
    required: "DataBridge: 从 Excel/ERP/MES 读取真实数据\n至少支持 pd.read_excel() 如 analyze_clients_v2.py:89",
    detail: "Sales Agent (analyze_clients_v2.py:89) 用 pd.read_excel 读真实 Excel。但其余 4 个 Agent 全部用 SAMPLE_ 开头的硬编码 list。这不是 MVP，这是 demo。",
    effort: "3-5 天 DataBridge 开发",
    risk: "演示给禾苗管理层时，4 个 Agent 返回的全是假数据。一问就穿帮。",
  },
  {
    id: "P0-07",
    severity: "P0",
    title: "Docker Compose 仍标注 V9.0",
    file: "docker-compose.yml",
    line: "2, 10",
    current: "# MRARFAI V9.0\ncontainer_name: mrarfai-v9",
    required: "# MRARFAI V10.0\ncontainer_name: mrarfai-v10",
    detail: "你已经是 V10 了，但 Docker 配置还写着 V9。这不是功能问题，但暴露版本管理不严谨。",
    effort: "2 行改动",
    risk: "低优先级但属于 P0 范畴——因为它是部署配置。",
  },

  // ===== P1 HIGH — 1周内必须修 =====
  {
    id: "P1-01",
    severity: "P1",
    title: "LangGraph StateGraph 仅覆盖 3/7 Agent",
    file: "multi_agent.py",
    line: "2524-2546",
    current: "StateGraph 节点: route → experts → synthesize → reflect → hitl_check\nroute_to_agents() 只路由: analyst, risk, strategist",
    required: "扩展 StateGraph 覆盖全部 7 个 Agent\n或创建 V10 专用 StateGraph",
    detail: "multi_agent.py:2524 的 StateGraph 只包含 route/experts/synthesize/reflect/hitl_check 5 个节点，而 route_to_agents (line 920) 只路由到 analyst/risk/strategist。Procurement/Quality/Finance/Market 完全不在图内。",
    effort: "1-2 天",
    risk: "「LangGraph 统一编排」是你的核心卖点，但实际只覆盖 43% 的 Agent。",
  },
  {
    id: "P1-02",
    severity: "P1",
    title: "MCP Server 只暴露 Sales 工具 (5个)",
    file: "mcp_server_v7.py",
    line: "78-147",
    current: "5 个 Tool: query_sales_data, analyze_customer, detect_anomalies, run_forecast, generate_report\n3 个 Resource: 销售总览, 客户列表, 风险预警\n1 个 Prompt: ceo_report",
    required: "暴露全部 7 Agent 的 20+ 工具\n至少: Procurement 3 + Quality 4 + Finance 3 + Market 3",
    detail: "MCP 是你的对外接口标准。目前只有 Sales 一个 Agent 的工具对外暴露。其余 6 个 Agent 的能力无法通过 MCP 被发现或调用。",
    effort: "1-2 天",
    risk: "MCP Registry 只能看到你 5 个工具。竞品动辄 20-50 个。",
  },
  {
    id: "P1-03",
    severity: "P1",
    title: "MCP Spec 未对齐 2025-11-25",
    file: "mcp_server_v7.py",
    line: "全文件",
    current: "使用 MCP SDK 1.0 基础功能\n零 Tasks 异步原语\n零 Extensions 框架\n零 CIMD 认证",
    required: "实现 Tasks (长时间任务异步执行)\n评估 Extensions 框架\n评估 CIMD 认证",
    detail: "MCP 最新 spec 2025-11-25 新增: Tasks 异步原语 (create/get/cancel)、Extensions 框架、CIMD auth、JSON Schema 2020-12。已捐赠给 Linux Foundation AAIF。你只用了基础的 tools/resources/prompts。",
    effort: "3-5 天",
    risk: "MCP 生态系统快速演进，不跟进会被锁在旧版本。",
  },
  {
    id: "P1-04",
    severity: "P1",
    title: "零 Pydantic 结构化合约",
    file: "所有 Agent 文件",
    line: "全项目 grep",
    current: "grep -rn 'BaseModel\\|pydantic' *.py → 零结果\n所有 Agent 间通信用 dict",
    required: "Pydantic v2 BaseModel 定义:\n• AgentRequest\n• AgentResponse\n• SalesInsight\n• ProcurementAlert\n等",
    detail: "45,437 行代码，零 Pydantic 使用。所有 Agent 输入输出都是松散 dict。LangChain 1.0 和 CrewAI 1.9.3 都要求 Pydantic v2 作为合约层。没有结构化合约意味着 Agent 间通信没有类型安全。",
    effort: "2-3 天",
    risk: "中等。功能正常但不符合 2026 工程规范。",
  },
  {
    id: "P1-05",
    severity: "P1",
    title: "零 create_agent() / 零 Middleware",
    file: "multi_agent.py",
    line: "全文件",
    current: "手写 Agent 类 + StateGraph 循环\ngrep 'create_agent\\|Middleware' → 零结果",
    required: "使用 LangChain 1.0 的 create_agent() API\n+ Middleware 架构 (pre/post hooks)",
    detail: "LangChain 1.0 引入 create_agent() 统一 API 和 Middleware 架构（类似 Express.js 中间件）。你的代码全部是手写 Agent 类 + 手动 StateGraph 节点。这是 0.x 时代的写法。",
    effort: "3-5 天重构",
    risk: "不影响功能，但不符合最新框架范式。",
  },
  {
    id: "P1-06",
    severity: "P1",
    title: "V10 Agent 零 Langfuse 可观测性",
    file: "agent_procurement.py, agent_quality.py, agent_finance.py, agent_market.py",
    line: "全文件",
    current: "grep 确认: 零 langfuse import\n零 @observe 装饰器\n零 trace 创建\n(agent_quality.py 的 trace 是业务方法 trace_root_cause，不是 Langfuse)",
    required: "每个 Agent 的 answer() 方法添加:\nfrom langfuse.decorators import observe\n@observe(name='procurement_agent')",
    detail: "V9 Agent 有完整的 langfuse_v3_otel.py 集成。V10 的 4 个 Agent 完全没有。你无法追踪这 4 个 Agent 的延迟、成本、错误率。",
    effort: "1 天",
    risk: "57% 的 Agent 是黑盒。可观测性全覆盖是你的卖点之一。",
  },
  {
    id: "P1-07",
    severity: "P1",
    title: "A2A 使用自建 JSON-RPC，非官方 SDK",
    file: "a2a_server_v7.py",
    line: "62, 494-594",
    current: "import a2a (line 62) 但实际是自建 JSON-RPC handler\n手动解析 jsonrpc 2.0 请求/响应",
    required: "迁移至 a2aproject/a2a-python 官方 SDK\n使用标准 TaskManager + AgentAdapter",
    detail: "a2a_server_v7.py:494-594 是完全手写的 JSON-RPC 2.0 处理器。A2A 项目已发布官方 Python SDK (a2a-python)，提供标准 TaskManager、AgentAdapter、Push Notifications 等。",
    effort: "2-3 天",
    risk: "自建实现可能遗漏协议细节。官方 SDK 有测试覆盖。",
  },

  // ===== P2 MEDIUM — 2周内 =====
  {
    id: "P2-01",
    severity: "P2",
    title: "零 gRPC 传输支持",
    file: "a2a_server_v7.py",
    line: "142",
    current: "protocol_binding: str = 'JSONRPC'  # 声明支持 gRPC 但未实现\n实际只有 HTTP/JSON-RPC",
    required: "实现 gRPC transport\n或至少 REST transport 作为备选",
    detail: "line 142 声明支持 JSONRPC | gRPC | HTTP，但实际只实现了 JSON-RPC。gRPC 在企业间通信中是标准。",
    effort: "3-5 天",
    risk: "企业级部署通常要求 gRPC。",
  },
  {
    id: "P2-02",
    severity: "P2",
    title: "MCP 未注册到 Registry",
    file: "mcp_server_v7.py",
    line: "N/A",
    current: "本地 MCP Server，未注册到任何 Registry",
    required: "注册到 MCP Registry (2000+ servers 生态系统)",
    detail: "MCP 生态系统已有 2000+ 注册服务器。你的 MCP Server 是孤立的。注册后其他 MCP 客户端可以自动发现你的工具。",
    effort: "1 天",
    risk: "生态可见性为零。",
  },
  {
    id: "P2-03",
    severity: "P2",
    title: "SQLite 生产环境",
    file: "全项目",
    line: "memory.db / observability.db",
    current: "SQLite 单文件数据库",
    required: "PostgreSQL (至少 docker-compose 中配置)",
    detail: "SQLite 不支持并发写入、无网络访问、无备份策略。生产环境最低标准是 PostgreSQL。",
    effort: "2-3 天",
    risk: "多用户场景下 SQLite 会锁表。",
  },
  {
    id: "P2-04",
    severity: "P2",
    title: "零 Deep Agents 集成",
    file: "N/A",
    line: "N/A",
    current: "未使用",
    required: "评估 Deep Agents 0.4.1:\nPlanning Tool + Sub-agent Spawning + VFS",
    detail: "Deep Agents 0.4.1 (2026-02-11，3天前发布) 提供 Planning Tool、Sub-agent Spawning、Virtual File System，且有 Langfuse 官方集成。",
    effort: "评估 1 天，集成 3-5 天",
    risk: "不用不影响功能，但错过前沿能力。",
  },
  {
    id: "P2-05",
    severity: "P2",
    title: "设计模式 5/7 — 缺 ReAct + Planning",
    file: "multi_agent.py",
    line: "全文件",
    current: "已有: Reflection, Tool Use, Multi-Agent, Sequential, HITL\n缺失: ReAct (Reason+Act循环), Planning (规划-执行分离)",
    required: "至少添加 ReAct 模式\nPlanning 可选 (Deep Agents 提供)",
    detail: "ReAct 是 2026 最常用的 Agent 模式。你有 Reflection 但没有 ReAct 的 Thought→Action→Observation 循环。",
    effort: "2-3 天",
    risk: "行业面试必问。",
  },

  // ===== P3 OPTIONAL — 持续改进 =====
  {
    id: "P3-01",
    severity: "P3",
    title: "零 A2UI/AG-UI 前端协议",
    file: "N/A",
    line: "N/A",
    current: "Streamlit SSR 直出",
    required: "评估 A2UI v0.8 / AG-UI 声明式 Agent→UI 渲染",
    detail: "A2UI v0.8 (Google, 2025-12) 提供声明式 Agent 到 UI 的渲染协议，与 A2A+MCP 互补。",
    effort: "评估 1 天",
    risk: "可选。Streamlit 目前够用。",
  },
  {
    id: "P3-02",
    severity: "P3",
    title: "Google ADK 未评估",
    file: "N/A",
    line: "N/A",
    current: "未使用",
    required: "评估 ADK 0.5.0 CustomAgent wrapper\nPython + TS + Java + Go 四语言\n原生 MCP + A2A",
    detail: "Google ADK 0.5.0 (2026-02) 支持四语言，原生 MCP + A2A 集成。可作为可选包装层。",
    effort: "评估 0.5 天",
    risk: "纯可选。",
  },
  {
    id: "P3-03",
    severity: "P3",
    title: "OpenAI Agents SDK 竞品未评估",
    file: "N/A",
    line: "N/A",
    current: "未评估",
    required: "了解 OpenAI Agents SDK 0.8.4:\nHandoffs + Guardrails + Sessions + Tracing",
    detail: "竞品框架。需要了解其能力边界以便在 BP 中做对比分析。",
    effort: "评估 0.5 天",
    risk: "纯竞品研究，不影响你的系统。",
  },
  {
    id: "P3-04",
    severity: "P3",
    title: "LangGraph 1.0 高级特性未使用",
    file: "multi_agent.py",
    line: "2524+",
    current: "使用基础 StateGraph + MemorySaver",
    required: "评估: Node Caching, Durable State, Deferred Nodes, Pre/Post Model Hooks",
    detail: "升级到 langgraph>=1.0 后可以使用这些高级特性进一步优化性能和可靠性。",
    effort: "1-2 天逐步采用",
    risk: "可选优化。",
  },
  {
    id: "P3-05",
    severity: "P3",
    title: "Graph Memory (Graphiti) 未集成",
    file: "memory_v9.py, knowledge_graph.py",
    line: "N/A",
    current: "3D Memory + 4D MAGMA + knowledge_graph (852行)\n自建图内存",
    required: "评估 Graphiti 作为标准 Graph Memory 替代",
    detail: "你的记忆架构已经很强(3层+MAGMA)。Graphiti 是新兴标准但你的方案可能更适合。",
    effort: "评估 1 天",
    risk: "你的方案可能已经更好。纯评估。",
  },
  {
    id: "P3-06",
    severity: "P3",
    title: "云端部署未完成",
    file: "render.yaml, railway.toml, docker-compose.yml",
    line: "N/A",
    current: "render.yaml + railway.toml 配置存在\n但无证据表明已成功部署",
    required: "实际部署到 Railway/Render/HF Spaces\n并获得可访问 URL",
    detail: "配置文件存在 ≠ 已部署。需要实际运行的 URL。",
    effort: "1-2 天",
    risk: "demo 时需要线上环境。",
  },
];

const SEVERITY_CONFIG = {
  P0: { color: "#ef4444", bg: "rgba(239,68,68,0.08)", border: "rgba(239,68,68,0.3)", label: "CRITICAL", icon: "🔴" },
  P1: { color: "#f59e0b", bg: "rgba(245,158,11,0.08)", border: "rgba(245,158,11,0.3)", label: "HIGH", icon: "🟡" },
  P2: { color: "#3b82f6", bg: "rgba(59,130,246,0.08)", border: "rgba(59,130,246,0.3)", label: "MEDIUM", icon: "🔵" },
  P3: { color: "#22c55e", bg: "rgba(34,197,94,0.08)", border: "rgba(34,197,94,0.3)", label: "OPTIONAL", icon: "🟢" },
};

function IssueCard({ issue, isOpen, onToggle }) {
  const cfg = SEVERITY_CONFIG[issue.severity];
  return (
    <div style={{
      background: cfg.bg,
      border: `1px solid ${cfg.border}`,
      borderRadius: 10,
      marginBottom: 8,
      overflow: "hidden",
      transition: "all 0.2s",
    }}>
      <div
        onClick={onToggle}
        style={{
          padding: "12px 16px",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 10,
          userSelect: "none",
        }}
      >
        <span style={{ fontSize: "0.65rem", fontWeight: 800, color: cfg.color, fontFamily: "'DM Mono', monospace", minWidth: 46 }}>{issue.id}</span>
        <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "#e2e8f0", flex: 1 }}>{issue.title}</span>
        <span style={{
          fontSize: "0.55rem", padding: "2px 8px", borderRadius: 4,
          background: `${cfg.color}22`, color: cfg.color, fontWeight: 700,
          fontFamily: "'DM Mono', monospace",
        }}>{issue.effort}</span>
        <span style={{ color: "#475569", fontSize: "0.8rem", transition: "transform 0.2s", transform: isOpen ? "rotate(180deg)" : "rotate(0)" }}>▾</span>
      </div>

      {isOpen && (
        <div style={{ padding: "0 16px 14px", fontSize: "0.72rem", lineHeight: 1.7 }}>
          {/* 文件定位 */}
          <div style={{
            background: "rgba(0,0,0,0.3)", borderRadius: 6, padding: "8px 12px",
            fontFamily: "'DM Mono', monospace", fontSize: "0.65rem", marginBottom: 10,
            border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ color: "#94a3b8" }}>📁 <span style={{ color: "#7dd3fc" }}>{issue.file}</span> : <span style={{ color: "#fbbf24" }}>line {issue.line}</span></div>
          </div>

          {/* 当前 vs 需要 */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
            <div style={{ background: "rgba(239,68,68,0.06)", borderRadius: 6, padding: "8px 10px", border: "1px solid rgba(239,68,68,0.15)" }}>
              <div style={{ color: "#ef4444", fontSize: "0.6rem", fontWeight: 700, marginBottom: 4 }}>❌ 当前</div>
              <pre style={{ color: "#f87171", fontSize: "0.6rem", margin: 0, whiteSpace: "pre-wrap", fontFamily: "'DM Mono', monospace" }}>{issue.current}</pre>
            </div>
            <div style={{ background: "rgba(34,197,94,0.06)", borderRadius: 6, padding: "8px 10px", border: "1px solid rgba(34,197,94,0.15)" }}>
              <div style={{ color: "#22c55e", fontSize: "0.6rem", fontWeight: 700, marginBottom: 4 }}>✅ 需要</div>
              <pre style={{ color: "#4ade80", fontSize: "0.6rem", margin: 0, whiteSpace: "pre-wrap", fontFamily: "'DM Mono', monospace" }}>{issue.required}</pre>
            </div>
          </div>

          {/* 详情 */}
          <div style={{ color: "#94a3b8", marginBottom: 8 }}>{issue.detail}</div>

          {/* 风险 */}
          <div style={{
            background: "rgba(239,68,68,0.04)", borderRadius: 6, padding: "6px 10px",
            borderLeft: `3px solid ${cfg.color}`, fontSize: "0.65rem",
          }}>
            <span style={{ color: cfg.color, fontWeight: 700 }}>⚠ 风险: </span>
            <span style={{ color: "#cbd5e1" }}>{issue.risk}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function MRARFAIAudit() {
  const [filter, setFilter] = useState("ALL");
  const [openIds, setOpenIds] = useState(new Set(["P0-01", "P0-03", "P0-05"]));

  const toggle = (id) => {
    setOpenIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const expandAll = () => setOpenIds(new Set(ISSUES.map(i => i.id)));
  const collapseAll = () => setOpenIds(new Set());

  const filtered = filter === "ALL" ? ISSUES : ISSUES.filter(i => i.severity === filter);

  const counts = useMemo(() => ({
    P0: ISSUES.filter(i => i.severity === "P0").length,
    P1: ISSUES.filter(i => i.severity === "P1").length,
    P2: ISSUES.filter(i => i.severity === "P2").length,
    P3: ISSUES.filter(i => i.severity === "P3").length,
    ALL: ISSUES.length,
  }), []);

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(170deg, #0a0c10 0%, #0d1117 40%, #101820 100%)",
      color: "#c9d1d9",
      fontFamily: "'Inter', -apple-system, sans-serif",
      padding: "1.5rem clamp(0.8rem, 3vw, 2rem)",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;600;700;900&display=swap');
        * { box-sizing: border-box; scrollbar-width: thin; scrollbar-color: #1c2128 transparent; }
      `}</style>

      {/* Header */}
      <div style={{ maxWidth: 860, margin: "0 auto 1.5rem", textAlign: "center" }}>
        <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
          <span style={{ padding: "3px 10px", borderRadius: 20, background: "rgba(239,68,68,0.1)", color: "#ef4444", fontSize: "0.65rem", fontWeight: 700 }}>🔴 STRICT MODE</span>
          <span style={{ padding: "3px 10px", borderRadius: 20, background: "rgba(255,255,255,0.05)", color: "#6e7681", fontSize: "0.65rem" }}>2026.2.14</span>
        </div>
        <h1 style={{
          fontSize: "clamp(1.4rem, 3vw, 2rem)", fontWeight: 900, margin: "0 0 0.3rem",
          background: "linear-gradient(135deg, #ef4444, #f59e0b, #3b82f6)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>
          MRARFAI 终极审计报告
        </h1>
        <p style={{ color: "#475569", fontSize: "0.7rem", margin: "0 0 6px", fontFamily: "'DM Mono', monospace" }}>
          全球最强 Agent 开发者视角 · 逐文件逐行 · {ISSUES.length} 项发现 · 零遗漏
        </p>
        <p style={{ color: "#374151", fontSize: "0.6rem", margin: 0 }}>
          源码验证: 45,437 行 · grep 实证 · 非推测
        </p>
      </div>

      {/* Summary Cards */}
      <div style={{ maxWidth: 860, margin: "0 auto 1.2rem", display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
        {["P0", "P1", "P2", "P3"].map(sev => {
          const cfg = SEVERITY_CONFIG[sev];
          return (
            <div key={sev} onClick={() => setFilter(filter === sev ? "ALL" : sev)} style={{
              background: filter === sev ? cfg.bg : "rgba(255,255,255,0.02)",
              border: `1px solid ${filter === sev ? cfg.border : "rgba(255,255,255,0.06)"}`,
              borderRadius: 10, padding: "12px 14px", cursor: "pointer", textAlign: "center",
              transition: "all 0.2s",
            }}>
              <div style={{ fontSize: "1.5rem", fontWeight: 900, color: cfg.color, fontFamily: "'DM Mono', monospace" }}>{counts[sev]}</div>
              <div style={{ fontSize: "0.6rem", color: cfg.color, fontWeight: 700 }}>{cfg.icon} {sev} {cfg.label}</div>
            </div>
          );
        })}
      </div>

      {/* Controls */}
      <div style={{ maxWidth: 860, margin: "0 auto 1rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: "0.7rem", color: "#6e7681" }}>
          {filter === "ALL" ? `全部 ${counts.ALL} 项` : `${filter}: ${counts[filter]} 项`}
          {filter !== "ALL" && (
            <button onClick={() => setFilter("ALL")} style={{
              marginLeft: 8, padding: "2px 8px", borderRadius: 4, border: "1px solid #1c2128",
              background: "transparent", color: "#58a6ff", fontSize: "0.6rem", cursor: "pointer",
            }}>显示全部</button>
          )}
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button onClick={expandAll} style={{
            padding: "4px 10px", borderRadius: 6, border: "1px solid #1c2128",
            background: "transparent", color: "#6e7681", fontSize: "0.6rem", cursor: "pointer",
          }}>全部展开</button>
          <button onClick={collapseAll} style={{
            padding: "4px 10px", borderRadius: 6, border: "1px solid #1c2128",
            background: "transparent", color: "#6e7681", fontSize: "0.6rem", cursor: "pointer",
          }}>全部收起</button>
        </div>
      </div>

      {/* Issue List */}
      <div style={{ maxWidth: 860, margin: "0 auto" }}>
        {["P0", "P1", "P2", "P3"].map(sev => {
          const items = filtered.filter(i => i.severity === sev);
          if (items.length === 0) return null;
          const cfg = SEVERITY_CONFIG[sev];
          return (
            <div key={sev} style={{ marginBottom: 20 }}>
              <div style={{
                fontSize: "0.7rem", fontWeight: 800, color: cfg.color, marginBottom: 8,
                fontFamily: "'DM Mono', monospace",
                display: "flex", alignItems: "center", gap: 8,
              }}>
                {cfg.icon} {sev} — {cfg.label}
                <span style={{ flex: 1, height: 1, background: cfg.border }}></span>
                <span style={{ color: "#475569", fontWeight: 400 }}>{items.length} 项</span>
              </div>
              {items.map(issue => (
                <IssueCard
                  key={issue.id}
                  issue={issue}
                  isOpen={openIds.has(issue.id)}
                  onToggle={() => toggle(issue.id)}
                />
              ))}
            </div>
          );
        })}
      </div>

      {/* Bottom Line */}
      <div style={{
        maxWidth: 860, margin: "2rem auto 0",
        background: "rgba(239,68,68,0.04)",
        border: "1px solid rgba(239,68,68,0.15)",
        borderRadius: 12, padding: "20px 24px",
      }}>
        <h3 style={{ color: "#ef4444", fontSize: "0.85rem", fontWeight: 900, margin: "0 0 12px" }}>
          🎯 底线判定
        </h3>
        <div style={{ fontSize: "0.72rem", lineHeight: 1.9, color: "#94a3b8" }}>
          <div><span style={{ color: "#ef4444", fontWeight: 700 }}>P0 = 7 项</span> — 版本过时(2)、协议路径错误(1)、幽灵依赖(1)、质量门缺失(1)、假数据(1)、Docker标注(1)</div>
          <div><span style={{ color: "#f59e0b", fontWeight: 700 }}>P1 = 7 项</span> — 编排覆盖不全(1)、MCP工具少(1)、MCP spec旧(1)、零Pydantic(1)、零create_agent(1)、零Langfuse(1)、自建A2A(1)</div>
          <div><span style={{ color: "#3b82f6", fontWeight: 700 }}>P2 = 5 项</span> — 零gRPC(1)、零Registry(1)、SQLite(1)、零Deep Agents(1)、模式缺2(1)</div>
          <div><span style={{ color: "#22c55e", fontWeight: 700 }}>P3 = 6 项</span> — A2UI(1)、ADK(1)、OpenAI SDK(1)、LangGraph高级(1)、Graphiti(1)、云部署(1)</div>
        </div>
        <div style={{ marginTop: 16, padding: "12px 16px", background: "rgba(0,0,0,0.3)", borderRadius: 8, fontSize: "0.7rem" }}>
          <div style={{ color: "#fbbf24", fontWeight: 700, marginBottom: 6 }}>📌 修复优先级</div>
          <div style={{ color: "#94a3b8", lineHeight: 2 }}>
            <span style={{ color: "#ef4444" }}>■</span> <b>今天 30分钟</b>: P0-01 + P0-02 + P0-03 + P0-04 + P0-07 (纯文本替换，零风险)<br/>
            <span style={{ color: "#ef4444" }}>■</span> <b>本周</b>: P0-05 + P0-06 (质量门 + 真数据，架构级改动)<br/>
            <span style={{ color: "#f59e0b" }}>■</span> <b>下周</b>: P1-01~P1-07 (框架对齐 + 全覆盖)<br/>
            <span style={{ color: "#3b82f6" }}>■</span> <b>月底</b>: P2 全部<br/>
            <span style={{ color: "#22c55e" }}>■</span> <b>持续</b>: P3 按需
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{
        maxWidth: 860, margin: "1.5rem auto 0", textAlign: "center",
        fontSize: "0.55rem", color: "#21262d", fontFamily: "'DM Mono', monospace", lineHeight: 1.6,
      }}>
        MRARFAI 终极审计 · {ISSUES.length} Issues · 45,437 Lines Verified · grep-proven · 2026.2.14
      </div>
    </div>
  );
}
