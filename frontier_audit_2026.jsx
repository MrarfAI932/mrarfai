import { useState } from "react";

const SECTIONS = [
  {
    id: "protocols",
    title: "🔗 协议层",
    subtitle: "A2A + MCP — 2026年2月最新",
    items: [
      {
        name: "A2A Protocol",
        latest: "v0.3 → Linux Foundation开源 (2025.6)\ngRPC支持 · 签名AgentCard · Proto单一权威定义\n150+企业 (AWS/Salesforce/SAP加入)\nGoogle ADK原生A2A集成 (2026.2.11发布)",
        yours: "自建JSON-RPC 2.0实现 (a2a_server_v7.py)\nAgentCard + Task + Artifact + Executor\n7个Agent全注册 · 但未用官方SDK",
        delta: -2,
        verdict: "方向对，实现旧",
        fix: "pip install a2a-sdk → 用官方Python SDK重写，获得gRPC+签名+ADK互操作",
      },
      {
        name: "MCP Protocol",
        latest: "2025-06-18版 (当前stable)\nStructured Tool Output · Elicitation · Resource Links\nOAuth Resource Server · RFC 8707强制\nStreamable HTTP传输 · 已捐Linux Foundation",
        yours: "自建MCP概念 (mcp_server_v7.py)\n工具注册框架 · 但无标准传输层\n未实现Elicitation/Resource Links",
        delta: -3,
        verdict: "概念在，标准差距大",
        fix: "用官方MCP Python SDK重写tool层，获得ChatGPT/Claude/ADK生态兼容",
      },
      {
        name: "A2A + MCP协同",
        latest: "2026共识：MCP管agent↔tool，A2A管agent↔agent\nO'Reilly明确定义分工 (2025.6)\nGoogle ADK同时原生支持两者",
        yours: "概念上理解分工\n但两个协议都是自建版本\n未实际实现跨协议联动",
        delta: -2,
        verdict: "认知对齐，实现待升级",
        fix: "按官方模式：MCP暴露工具 + A2A暴露Agent能力 → ADK可发现",
      },
    ],
  },
  {
    id: "frameworks",
    title: "🏗️ 框架层",
    subtitle: "编排 + 开发框架",
    items: [
      {
        name: "编排框架",
        latest: "2026 Top 3生产级：\n① LangGraph 0.3+ (StateGraph, 最主流)\n② Google ADK 0.5 (Sequential/Parallel/Loop原语)\n③ CrewAI 2.x (角色型)\nO'Reilly 2026课程用LangGraph+CrewAI+FastMCP组合",
        yours: "LangGraph StateGraph (multi_agent.py 2,780行)\n+ CrewAI提及 · 实质上LangGraph为主\nSTART→route→experts→synthesize→reflect→hitl→END",
        delta: 0,
        verdict: "✅ 框架选择完全正确",
        fix: "确认LangGraph版本≥0.3（API有breaking changes），可选引入ADK做A2A层",
      },
      {
        name: "Agent开发框架",
        latest: "Google ADK 0.5 (2026.2.11发布，3天前)\nBaseAgent → LlmAgent / WorkflowAgent / CustomAgent\n原生MCP+A2A · 内建Eval · Dev UI · Trace\nPython/TypeScript/Go/Java多语言",
        yours: "自建Agent基类 (各agent_*.py)\n自建Engine+Executor+NLP Router\n无标准Dev UI · 无内建Eval工具",
        delta: -1,
        verdict: "自建方案可用，但ADK更标准",
        fix: "可选：将Agent封装为ADK CustomAgent，获得Dev UI + Eval + 标准部署",
      },
      {
        name: "多Agent协调拓扑",
        latest: "O'Reilly 2026.2.9文章(5天前)：\n「Prompting Fallacy」— 系统级失败不能靠改prompt\n推荐混合拓扑：并行专家 + 慢聚合器\n关键：协调税随Agent数指数增长",
        yours: "multi_agent.py支持：\n并行Expert · 链式协作 · 辩论模式\nCriticAgent反思循环 · HITL人工介入\n已有route→experts→synthesize流程",
        delta: 0,
        verdict: "✅ 你的拓扑符合最新推荐",
        fix: "符合O'Reilly推荐的混合拓扑（并行专家+聚合器），保持现有架构",
      },
    ],
  },
  {
    id: "patterns",
    title: "🧩 设计模式",
    subtitle: "2026年7大Agent设计模式对标",
    items: [
      {
        name: "7大设计模式",
        latest: "MachineLearningMastery 2026.1.5：\n① ReAct (思考+行动循环)\n② Reflection (自我批评)\n③ Tool Use (外部工具调用)\n④ Planning (任务分解)\n⑤ Multi-Agent (多Agent协作)\n⑥ Sequential (顺序流水线)\n⑦ HITL (人机协作)",
        yours: "已实现5/7：\n✅ Reflection (CriticAgent)\n✅ Tool Use (tool_registry.py)\n✅ Multi-Agent (multi_agent.py)\n✅ Sequential (协作链)\n✅ HITL (hitl_engine.py)\n❌ ReAct · ❌ Planning",
        delta: 0,
        verdict: "✅ 5/7超过多数生产系统",
        fix: "可选：给复杂查询加ReAct循环（思考→行动→观察→再思考）",
      },
      {
        name: "3种Agent记忆",
        latest: "2026标准三层记忆：\nEpisodic (事件记忆) · Semantic (语义知识)\nProcedural (操作程序)\nO'Reilly「Agentic Graph RAG」专门讲Graph Memory",
        yours: "memory_v9.py: 短期/长期/元记忆\nmeta_memory.py + persistent_memory.py\nknowledge_graph.py (语义层)",
        delta: 0,
        verdict: "✅ 三层对齐",
        fix: "概念对齐。可选：引入Graph Memory（Graphiti/Letta模式）做实体关系记忆",
      },
      {
        name: "结构化Agent间通信",
        latest: "ClickIT 2026.2.11(3天前)：\n从自由文本→JSON Schema合约\nAgent间通信必须结构化\n「数据合约」是2026架构必备",
        yours: "Agent输出JSON · 但无Pydantic/Schema验证\n通信靠字符串匹配+json.dumps",
        delta: -2,
        verdict: "有JSON但无合约",
        fix: "给每个Engine.answer()定义Pydantic输出模型 → 类型安全+自动验证",
      },
    ],
  },
  {
    id: "infra",
    title: "⚙️ 基础设施",
    subtitle: "可观测 + 安全 + 部署",
    items: [
      {
        name: "可观测性",
        latest: "2026标准栈：OpenTelemetry + 专业平台\nDatadog ADK集成(2026.2发布)\nLangfuse/Langsmith开源替代\n全链路trace + token成本 + 决策路径可视化",
        yours: "Langfuse v3 OTEL集成 ✅\nAgentTracer + CostCalculator ✅\nobservability.py完整实现 ✅",
        delta: 0,
        verdict: "✅ 对齐主流",
        fix: "确保7个Agent都接入trace（V10新Agent可能还没接）",
      },
      {
        name: "安全护栏",
        latest: "2026趋势：Bounded Autonomy\n明确操作边界 · 升级路径到人类\n审计trail · 输入/输出guardrails\n「Governance Agent」监控其他Agent",
        yours: "guardrails.py + input_guardrails.py ✅\nAuditLog全链路审计 ✅\nCircuitBreaker熔断 ✅\nRateLimiter速率限制 ✅",
        delta: 0,
        verdict: "✅ 超过多数生产系统",
        fix: "锦上添花：可加Governance Agent（专门监控其他Agent行为合规）",
      },
      {
        name: "多模型路由",
        latest: "Databricks报告：59%企业已用3+LLM\n模型路由是2026标配\n推理模型(reasoning)用于复杂任务\n快速模型用于简单任务",
        yours: "GPT-4o / Claude / Gemini / DeepSeek\n多模型路由已实现\nadaptive_gate.py动态选择",
        delta: 0,
        verdict: "✅ 完全对齐",
        fix: "无需改动。可选：加入reasoning模型(o3/DeepSeek-R1)用于复杂分析",
      },
      {
        name: "部署架构",
        latest: "2026标准：\nContainer → Cloud Run/K8s\nAgent专用数据库（Databricks Lakebase概念）\nCI/CD with agent-starter-pack\nGoogle ADK → Agent Engine一键部署",
        yours: "Docker已有 · Railway/Render配置存在\n但从未实际跑通云端\nSQLite本地 · 未迁移PostgreSQL",
        delta: -3,
        verdict: "配置在，部署未跑通",
        fix: "P1优先：Railway/Render实际部署一次 + SQLite→PostgreSQL",
      },
    ],
  },
  {
    id: "data",
    title: "📊 数据层",
    subtitle: "唯一阻塞项",
    items: [
      {
        name: "真实数据接入",
        latest: "所有生产Agent系统的底线：\n必须接入真实业务数据\n无真实数据 = 不可Demo = 不是产品\nStack-AI 2026指南：先做数据接入再做架构",
        yours: "Sales Agent: 真实Excel ✅\nRisk/Strategist: 管道注入 ⚡\nProcurement/Quality/Finance/Market: 全SAMPLE_ ❌",
        delta: -4,
        verdict: "🚨 关键阻塞",
        fix: "DataBridge: Excel出货数据→反推应收/成本/产量→灌入4个Agent (3-5天)",
      },
    ],
  },
];

const DELTA_LABELS = {
  0: { text: "✅ 已对齐", color: "#22c55e", bg: "#22c55e12" },
  "-1": { text: "微差距", color: "#38bdf8", bg: "#38bdf812" },
  "-2": { text: "可改进", color: "#f59e0b", bg: "#f59e0b12" },
  "-3": { text: "明显差距", color: "#ef4444", bg: "#ef444412" },
  "-4": { text: "🚨 阻塞", color: "#ef4444", bg: "#ef444420" },
};

function ItemCard({ item, isOpen, onToggle }) {
  const d = DELTA_LABELS[String(item.delta)] || DELTA_LABELS["0"];
  return (
    <div style={{ marginBottom: 6 }}>
      <div
        onClick={onToggle}
        style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "10px 14px", cursor: "pointer",
          background: isOpen ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.015)",
          border: `1px solid ${isOpen ? d.color + "40" : "rgba(255,255,255,0.06)"}`,
          borderRadius: isOpen ? "8px 8px 0 0" : 8,
          transition: "all 0.15s",
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>{item.name}</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 4,
            background: d.bg, color: d.color,
          }}>{d.text}</span>
          <span style={{
            fontSize: 10, color: "#64748b", fontStyle: "italic",
          }}>{item.verdict}</span>
          <span style={{ fontSize: 12, color: "#64748b", transform: isOpen ? "rotate(180deg)" : "rotate(0)", transition: "0.15s" }}>▾</span>
        </div>
      </div>
      {isOpen && (
        <div style={{
          padding: "14px 16px", fontSize: 12, lineHeight: 1.7,
          background: "rgba(255,255,255,0.02)",
          border: `1px solid ${d.color}25`, borderTop: "none",
          borderRadius: "0 0 8px 8px",
        }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 12 }}>
            <div>
              <div style={{ color: "#64748b", fontWeight: 700, marginBottom: 4, fontSize: 11 }}>🌍 2026.2 行业最前沿</div>
              <div style={{ color: "#cbd5e1", whiteSpace: "pre-line" }}>{item.latest}</div>
            </div>
            <div>
              <div style={{ color: "#64748b", fontWeight: 700, marginBottom: 4, fontSize: 11 }}>📦 你的MRARFAI V10</div>
              <div style={{ color: "#cbd5e1", whiteSpace: "pre-line" }}>{item.yours}</div>
            </div>
          </div>
          <div style={{
            padding: "8px 12px", borderRadius: 6,
            background: `${d.color}08`, border: `1px solid ${d.color}15`,
          }}>
            <span style={{ color: "#38bdf8", fontWeight: 700 }}>→ 行动：</span>
            <span style={{ color: "#e2e8f0" }}>{item.fix}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function FrontierAudit() {
  const [openItems, setOpenItems] = useState({});
  const toggle = (id) => setOpenItems(p => ({ ...p, [id]: !p[id] }));

  const allItems = SECTIONS.flatMap(s => s.items);
  const aligned = allItems.filter(i => i.delta === 0).length;
  const minor = allItems.filter(i => i.delta === -1).length;
  const fixable = allItems.filter(i => i.delta === -2).length;
  const gap = allItems.filter(i => i.delta === -3).length;
  const blocker = allItems.filter(i => i.delta === -4).length;

  return (
    <div style={{
      fontFamily: "'Fira Sans', 'Noto Sans SC', sans-serif",
      background: "linear-gradient(160deg, #070b14 0%, #0f172a 50%, #0a101e 100%)",
      color: "#e2e8f0", minHeight: "100vh", padding: "28px 20px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <div style={{ marginBottom: 6, display: "flex", alignItems: "baseline", gap: 12 }}>
          <h1 style={{
            fontSize: 22, fontWeight: 700, margin: 0,
            background: "linear-gradient(135deg, #38bdf8, #a78bfa, #f472b6)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>MRARFAI vs 2026.2 最前沿</h1>
          <span style={{ fontSize: 11, color: "#64748b" }}>截至 2026年2月14日</span>
        </div>
        <p style={{ fontSize: 12, color: "#94a3b8", marginBottom: 20 }}>
          信源：O'Reilly (2.9/2.13) · Google ADK 0.5 (2.11) · A2A v0.3 · MCP 2025-06-18 · Databricks State of AI · ClickIT (2.11) · Stack-AI · MLMastery · Springer PRISMA Survey
        </p>

        {/* Score Bar */}
        <div style={{
          display: "flex", gap: 2, marginBottom: 20, height: 32, borderRadius: 8, overflow: "hidden",
        }}>
          {[
            { count: aligned, color: "#22c55e", label: `${aligned} 已对齐` },
            { count: minor, color: "#38bdf8", label: `${minor} 微差距` },
            { count: fixable, color: "#f59e0b", label: `${fixable} 可改进` },
            { count: gap, color: "#ef4444", label: `${gap} 明显差距` },
            { count: blocker, color: "#dc2626", label: `${blocker} 阻塞项` },
          ].filter(s => s.count > 0).map((s, i) => (
            <div key={i} style={{
              flex: s.count, display: "flex", alignItems: "center", justifyContent: "center",
              background: s.color + "30", borderLeft: i ? `1px solid ${s.color}50` : "none",
              fontSize: 11, fontWeight: 700, color: s.color,
            }}>
              {s.label}
            </div>
          ))}
        </div>

        {/* Key Insight */}
        <div style={{
          padding: "14px 18px", borderRadius: 10, marginBottom: 24,
          background: "linear-gradient(135deg, rgba(34,197,94,0.05), rgba(239,68,68,0.05))",
          border: "1px solid rgba(167,139,250,0.2)",
          fontSize: 13, color: "#e2e8f0", lineHeight: 1.8,
        }}>
          <strong style={{ color: "#a78bfa" }}>O'Reilly 2026.2.9 核心观点：</strong>
          「你不能靠改prompt来修复系统级失败。Agent性能是<em>架构</em>的结果，不是提示词的结果。」
          <br/>你的MRARFAI在架构层面（编排拓扑、设计模式、安全护栏、记忆、可观测性、多模型）<strong style={{ color: "#22c55e" }}>全部对齐2026最前沿</strong>。
          差距集中在<strong style={{ color: "#ef4444" }}>实现层面</strong>：协议用自建而非官方SDK、数据用模拟而非真实、部署停在本地。
        </div>

        {/* Sections */}
        {SECTIONS.map(section => (
          <div key={section.id} style={{ marginBottom: 20 }}>
            <div style={{ marginBottom: 8 }}>
              <span style={{ fontSize: 14, fontWeight: 700, color: "#f1f5f9" }}>{section.title}</span>
              <span style={{ fontSize: 11, color: "#64748b", marginLeft: 8 }}>{section.subtitle}</span>
            </div>
            {section.items.map((item, i) => {
              const key = `${section.id}-${i}`;
              return <ItemCard key={key} item={item} isOpen={!!openItems[key]} onToggle={() => toggle(key)} />;
            })}
          </div>
        ))}

        {/* Priority Matrix */}
        <div style={{
          marginTop: 8, padding: 20, borderRadius: 12,
          background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
        }}>
          <h3 style={{ fontSize: 15, fontWeight: 700, color: "#f1f5f9", marginBottom: 14 }}>🎯 2026.2最新认知下的优先级</h3>
          <div style={{ fontSize: 13, lineHeight: 2.2 }}>
            <div>
              <span style={{ background: "#ef444430", color: "#ef4444", padding: "2px 8px", borderRadius: 4, fontWeight: 700, fontSize: 11 }}>P0 阻塞</span>
              <span style={{ color: "#e2e8f0", marginLeft: 8 }}>DataBridge接真实数据 — Stack-AI: 「先数据后架构」(3-5天)</span>
            </div>
            <div>
              <span style={{ background: "#f59e0b30", color: "#f59e0b", padding: "2px 8px", borderRadius: 4, fontWeight: 700, fontSize: 11 }}>P1 短期</span>
              <span style={{ color: "#e2e8f0", marginLeft: 8 }}>云端部署跑通 + Pydantic输出合约 (2-3天)</span>
            </div>
            <div>
              <span style={{ background: "#38bdf830", color: "#38bdf8", padding: "2px 8px", borderRadius: 4, fontWeight: 700, fontSize: 11 }}>P2 中期</span>
              <span style={{ color: "#e2e8f0", marginLeft: 8 }}>迁移官方A2A SDK v0.3 + MCP SDK → 进入Google ADK生态 (1-2周)</span>
            </div>
            <div>
              <span style={{ background: "#22c55e30", color: "#22c55e", padding: "2px 8px", borderRadius: 4, fontWeight: 700, fontSize: 11 }}>P3 可选</span>
              <span style={{ color: "#e2e8f0", marginLeft: 8 }}>ADK重构(Agent→ADK CustomAgent) · ReAct模式 · Graph Memory · Governance Agent</span>
            </div>
          </div>
          <div style={{
            marginTop: 14, padding: "10px 14px", borderRadius: 8,
            background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.15)",
            fontSize: 12, color: "#cbd5e1", lineHeight: 1.7,
          }}>
            <strong style={{ color: "#22c55e" }}>底线评估：</strong>
            你的MRARFAI在2026.2的坐标系中，<strong>架构层80+分，实现层40分</strong>。
            架构选择（LangGraph + 混合拓扑 + 5/7模式 + Langfuse + 多模型）完全匹配O'Reilly/Google/Gartner的2026推荐。
            差距全在「最后一英里」：真实数据、官方SDK、实际部署。这些是工程问题，不是设计问题。
          </div>
        </div>
      </div>
    </div>
  );
}
