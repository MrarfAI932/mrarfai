import { useState } from "react";

const ITEMS = [
  {
    id: "P2-01",
    title: "gRPC 传输层",
    file: "a2a_server_v7.py + proto/a2a_service.proto",
    status: "done",
    score: 95,
    evidence: [
      { check: ".proto 服务定义文件", pass: true, detail: "proto/a2a_service.proto: line 48 — service A2AService { 5 RPCs }" },
      { check: "pb2 消息 + grpc stub 文件", pass: true, detail: "a2a_service_pb2.py (95行) + a2a_service_pb2_grpc.py (手动 stub，含 add_A2AServiceServicer_to_server)" },
      { check: "Servicer 注册到 grpc.Server", pass: true, detail: "a2a_server_v7.py line 713-741: add_A2AServiceServicer_to_server() (proto → generic handler 双模式)" },
      { check: "Health Checking endpoint", pass: true, detail: "line 752-759: grpc_health.v1.HealthServicer + set('mrarfai.a2a.A2AService', SERVING)" },
      { check: "StreamTask 流式 RPC", pass: true, detail: "line 702: def StreamTask() — yield StreamEvent for SSE events" },
      { check: "grpcio import 就绪", pass: true, detail: "line 87-90: import grpc ✔ (HAS_GRPC flag)" },
      { check: "A2AGrpcServicer 类完整", pass: true, detail: "line 658: class A2AGrpcServicer — SendTask/GetTask/CancelTask/StreamTask/GetAgentCard" },
      { check: "TLS/mTLS 安全传输", pass: false, detail: "grep ssl_server_credentials → 0 (可选项，生产环境需要)" },
    ],
    verdict: "✅ 基本完成。.proto 定义、Servicer 正式注册 (proto-based + generic handler 双回退)、Health Check 全部就绪。手动 stub 兼容 grpc_tools 编译覆盖。仅 TLS 未实现 (可选)。",
    remaining: [
      "（可选）pip install grpcio-tools → 编译正式 pb2 文件",
      "（可选）添加 ssl_server_credentials TLS 支持",
    ],
  },
  {
    id: "P2-02",
    title: "MCP Registry 自注册",
    file: "mcp_server_v7.py",
    status: "done",
    score: 100,
    evidence: [
      { check: "MCP_REGISTRY_MANIFEST 完整", pass: true, detail: "line 622-685: 含 name/version/tools/packages/transport (v0.1 格式)" },
      { check: "register_to_registry() 函数", pass: true, detail: "line 693: POST /v0/servers → registry.modelcontextprotocol.io" },
      { check: "CLI --register 支持", pass: true, detail: "line 776-786: python mcp_server_v7.py --register" },
      { check: "CLI --manifest 输出", pass: true, detail: "line 773-775: JSON 输出供手动提交" },
      { check: "/.well-known/mcp/server.json", pass: true, detail: "line 760-798: Starlette 路由注入 + .well-known/mcp/server.json 静态文件 (2552 bytes)" },
      { check: "packages 字段 (v0.1 新增)", pass: true, detail: "line 672-678: pypi name + version ✔" },
    ],
    verdict: "✅ 完成。manifest + 注册 API + CLI + /.well-known/mcp/server.json 自动发现端点全部就绪。",
    remaining: [],
  },
  {
    id: "P2-03",
    title: "SQLite → PostgreSQL",
    file: "db_connector.py",
    status: "done",
    score: 95,
    evidence: [
      { check: "PostgresConnector 类", pass: true, detail: "line 366: class PostgresConnector(BaseConnector) — 完整实现" },
      { check: "psycopg2 import", pass: true, detail: "line 382-384: import psycopg2 + psycopg2.pool.SimpleConnectionPool" },
      { check: "DatabaseConfig 支持 postgresql 类型", pass: true, detail: "line 35: type = 'none|sqlite|mysql|postgresql|api'" },
      { check: "docker-compose PG 服务", pass: true, detail: "postgres:16-alpine + healthcheck + volume 配置完整" },
      { check: "init_postgres.sql schema", pass: true, detail: "99 行: dim_customer + dim_product + fact_sales 表" },
      { check: "requirements psycopg2", pass: true, detail: "psycopg2-binary>=2.9 ✔" },
      { check: "连接池管理", pass: true, detail: "SimpleConnectionPool(minconn=1, maxconn=10) + auto-rollback on putconn" },
      { check: "create_connector 工厂路由", pass: true, detail: "line 535: _CONNECTORS['postgresql'] = PostgresConnector" },
      { check: "query_* 方法完整 (8个)", pass: true, detail: "query_suppliers/orders/ar/margins/yields/returns/competitors/sales + test_connection" },
    ],
    verdict: "✅ 完成。PostgresConnector 类完整实现，含连接池管理、auto-rollback、RealDictCursor 查询、8 个领域查询方法、工厂注册。",
    remaining: [
      "（可选）awm_env_factory.py 中 12 处 sqlite3 硬编码抽象化",
    ],
  },
  {
    id: "P2-04",
    title: "Deep Agents 0.4.1 集成",
    file: "deep_agent_adapter.py",
    status: "done",
    score: 95,
    evidence: [
      { check: "deep_agent_adapter.py 适配器", pass: true, detail: "281 行: create_mrarfai_deep_agent() + 7域子Agent + StateGraph集成" },
      { check: "from deepagents import create_deep_agent", pass: true, detail: "line 26: safe import with HAS_DEEP_AGENTS flag" },
      { check: "multi_agent.py 集成", pass: true, detail: "lines 262-267: HAS_DEEP_AGENTS + _get_deep_agent() 懒加载 (line 289)" },
      { check: "Subagent spawning 定义", pass: true, detail: "8处 subagent 引用: 7个领域子Agent" },
      { check: "requirements 依赖", pass: true, detail: "deepagents>=0.4.1 ✔ in requirements_v7.txt" },
      { check: "get_platform_capabilities 注册", pass: true, detail: "line 3324: 'deep_agents': HAS_DEEP_AGENTS" },
    ],
    verdict: "✅ 完成。适配器文件完整，multi_agent.py 已集成懒加载，能力矩阵已注册。仅需安装 deepagents 包即可激活。",
    remaining: [
      "（仅运行时）pip install deepagents>=0.4.1",
    ],
  },
  {
    id: "P2-05",
    title: "ReAct + Planning 模式",
    file: "react_planner.py + multi_agent.py",
    status: "done",
    score: 95,
    evidence: [
      { check: "react_planner.py 文件", pass: true, detail: "310 行: HierarchicalPlanner + PlanStep + create_react_sales_agent" },
      { check: "HierarchicalPlanner 类", pass: true, detail: "line 51: DAG 任务分解 + 7域关键词匹配" },
      { check: "create_react_sales_agent()", pass: true, detail: "line 192: 封装 langgraph create_react_agent()" },
      { check: "get_parallel_groups() 并行调度", pass: true, detail: "line 148: 拓扑排序 → 并行批次" },
      { check: "multi_agent.py 导入集成", pass: true, detail: "lines 276-285: from react_planner import HierarchicalPlanner, create_react_sales_agent, add_planning_nodes ✔" },
      { check: "LangGraph ReAct 优先路径", pass: true, detail: "line 1411-1413: create_react_sales_agent() 作为优先路径，失败回退 Claude native tool_use" },
      { check: "QueryPlanner 集成", pass: true, detail: "line 2428: class QueryPlanner — needs_planning() + create_plan() 集成到 node_route 三条路径" },
      { check: "能力矩阵注册", pass: true, detail: "line 3322: 'react_planner_langgraph': HAS_REACT_PLANNER" },
    ],
    verdict: "✅ 完成。react_planner.py 已通过 multi_agent.py 导入并集成。LangGraph ReAct 作为优先 tool_use 路径，QueryPlanner 嵌入路由逻辑。",
    remaining: [
      "（仅运行时）pip install langgraph>=0.3",
    ],
  },
];

const STATUS_CONFIG = {
  done: { label: "✅ 已完成", color: "#10B981", bg: "#064E3B", border: "#059669" },
  mostly_done: { label: "🟡 基本完成", color: "#F59E0B", bg: "#451A03", border: "#D97706" },
  partial: { label: "🟠 部分完成", color: "#F97316", bg: "#431407", border: "#EA580C" },
  not_fixed: { label: "❌ 未完成", color: "#EF4444", bg: "#450A0A", border: "#DC2626" },
};

export default function P2Evaluation() {
  const [expanded, setExpanded] = useState(new Set(ITEMS.map(i => i.id)));
  const toggle = (id) => setExpanded(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  const totalScore = Math.round(ITEMS.reduce((s, i) => s + i.score, 0) / ITEMS.length);
  const done = ITEMS.filter(i => i.status === "done").length;
  const partial = ITEMS.filter(i => i.status === "mostly_done" || i.status === "partial").length;
  const notFixed = ITEMS.filter(i => i.status === "not_fixed").length;

  return (
    <div style={{ background: "#0D1117", color: "#C9D1D9", minHeight: "100vh", padding: "24px", fontFamily: "'Inter', -apple-system, sans-serif" }}>
      {/* Header */}
      <div style={{ maxWidth: 820, margin: "0 auto 24px", textAlign: "center" }}>
        <div style={{ fontSize: 11, color: "#6E7681", fontFamily: "monospace", marginBottom: 8 }}>MRARFAI V10.1 · grep-verified · {new Date().toISOString().slice(0, 10)}</div>
        <h1 style={{ fontSize: 28, fontWeight: 900, color: "#58A6FF", margin: "0 0 4px" }}>P2 修改评估报告 (最终版)</h1>
        <div style={{ fontSize: 13, color: "#8B949E" }}>5项 P2 逐条审计 · 基于代码 grep 证据 · 全部完成</div>
      </div>

      {/* Score Cards */}
      <div style={{ maxWidth: 820, margin: "0 auto 24px", display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <div style={{ background: "#161B22", border: "1px solid #21262D", borderRadius: 12, padding: 16, textAlign: "center" }}>
          <div style={{ fontSize: 32, fontWeight: 900, color: totalScore >= 70 ? "#10B981" : totalScore >= 50 ? "#F59E0B" : "#EF4444" }}>{totalScore}%</div>
          <div style={{ fontSize: 11, color: "#6E7681" }}>总完成度</div>
        </div>
        <div style={{ background: "#161B22", border: "1px solid #21262D", borderRadius: 12, padding: 16, textAlign: "center" }}>
          <div style={{ fontSize: 32, fontWeight: 900, color: "#10B981" }}>{done}</div>
          <div style={{ fontSize: 11, color: "#6E7681" }}>✅ 已完成</div>
        </div>
        <div style={{ background: "#161B22", border: "1px solid #21262D", borderRadius: 12, padding: 16, textAlign: "center" }}>
          <div style={{ fontSize: 32, fontWeight: 900, color: "#F59E0B" }}>{partial}</div>
          <div style={{ fontSize: 11, color: "#6E7681" }}>🟡 部分完成</div>
        </div>
        <div style={{ background: "#161B22", border: "1px solid #21262D", borderRadius: 12, padding: 16, textAlign: "center" }}>
          <div style={{ fontSize: 32, fontWeight: 900, color: "#EF4444" }}>{notFixed}</div>
          <div style={{ fontSize: 11, color: "#6E7681" }}>❌ 未完成</div>
        </div>
      </div>

      {/* Items */}
      <div style={{ maxWidth: 820, margin: "0 auto" }}>
        {ITEMS.map(item => {
          const cfg = STATUS_CONFIG[item.status];
          const isOpen = expanded.has(item.id);
          const passCount = item.evidence.filter(e => e.pass).length;
          const totalChecks = item.evidence.length;

          return (
            <div key={item.id} style={{ marginBottom: 12, background: "#161B22", border: `1px solid ${isOpen ? cfg.border : "#21262D"}`, borderRadius: 12, overflow: "hidden", transition: "border-color 0.2s" }}>
              {/* Header row */}
              <div onClick={() => toggle(item.id)} style={{ display: "flex", alignItems: "center", gap: 12, padding: "14px 18px", cursor: "pointer" }}>
                <span style={{ fontSize: 11, fontWeight: 800, color: cfg.color, background: cfg.bg, padding: "3px 10px", borderRadius: 6, border: `1px solid ${cfg.border}`, whiteSpace: "nowrap" }}>{item.id}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: "#E6EDF3" }}>{item.title}</div>
                  <div style={{ fontSize: 11, color: "#6E7681", marginTop: 2 }}>{item.file} · {passCount}/{totalChecks} checks passed</div>
                </div>
                {/* Score bar */}
                <div style={{ width: 100, display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ flex: 1, height: 6, background: "#21262D", borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${item.score}%`, height: "100%", background: cfg.color, borderRadius: 3, transition: "width 0.5s" }} />
                  </div>
                  <span style={{ fontSize: 12, fontWeight: 800, color: cfg.color, minWidth: 32, textAlign: "right" }}>{item.score}%</span>
                </div>
                <span style={{ fontSize: 11, color: cfg.color, fontWeight: 700, minWidth: 72, textAlign: "right" }}>{cfg.label}</span>
                <span style={{ color: "#6E7681", fontSize: 16, transform: isOpen ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}>▾</span>
              </div>

              {isOpen && (
                <div style={{ padding: "0 18px 16px", borderTop: "1px solid #21262D" }}>
                  {/* Evidence table */}
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: "#8B949E", marginBottom: 8 }}>GREP 证据</div>
                    {item.evidence.map((ev, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "6px 0", borderBottom: i < item.evidence.length - 1 ? "1px solid #21262D33" : "none" }}>
                        <span style={{ fontSize: 13, minWidth: 20 }}>{ev.pass ? "✅" : "❌"}</span>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 12, color: "#C9D1D9", fontWeight: 600 }}>{ev.check}</div>
                          <div style={{ fontSize: 11, color: ev.pass ? "#3FB950" : "#F85149", fontFamily: "monospace", marginTop: 2 }}>{ev.detail}</div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Verdict */}
                  <div style={{ marginTop: 14, padding: "10px 14px", background: cfg.bg, border: `1px solid ${cfg.border}33`, borderRadius: 8 }}>
                    <div style={{ fontSize: 11, fontWeight: 800, color: cfg.color, marginBottom: 4 }}>判定</div>
                    <div style={{ fontSize: 12, color: "#C9D1D9", lineHeight: 1.6 }}>{item.verdict}</div>
                  </div>

                  {/* Remaining work */}
                  {item.remaining.length > 0 && (
                    <div style={{ marginTop: 10 }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: "#8B949E", marginBottom: 6 }}>可选优化</div>
                      {item.remaining.map((r, i) => (
                        <div key={i} style={{ display: "flex", gap: 6, alignItems: "flex-start", padding: "3px 0" }}>
                          <span style={{ color: "#6E7681", fontSize: 10, marginTop: 2 }}>▸</span>
                          <span style={{ fontSize: 11, color: "#6E7681", fontFamily: "monospace" }}>{r}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <div style={{ maxWidth: 820, margin: "20px auto 0", background: "#161B22", border: "1px solid #21262D", borderRadius: 12, padding: 20 }}>
        <h3 style={{ fontSize: 14, fontWeight: 800, color: "#58A6FF", margin: "0 0 12px" }}>📊 结论 — P2 全部完成</h3>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #21262D" }}>
              {["项目", "初始状态", "最终状态", "变化"].map(h => (
                <th key={h} style={{ textAlign: "left", padding: "8px 6px", color: "#8B949E", fontWeight: 700, fontSize: 11 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              ["P2-01 gRPC", "❌ 未完成 (25%)", "✅ 95% 完成", "+70%", "#10B981"],
              ["P2-02 MCP Registry", "🟡 80% 完成", "✅ 100% 完成", "+20%", "#10B981"],
              ["P2-03 SQLite→PG", "❌ 未完成 (30%)", "✅ 95% 完成", "+65%", "#10B981"],
              ["P2-04 Deep Agents", "✅ 95% 完成", "✅ 95% 完成", "维持", "#10B981"],
              ["P2-05 ReAct+Planning", "🟠 60% 完成", "✅ 95% 完成", "+35%", "#10B981"],
            ].map(([name, before, after, delta, color], i) => (
              <tr key={i} style={{ borderBottom: "1px solid #21262D22" }}>
                <td style={{ padding: "8px 6px", fontWeight: 600, color: "#C9D1D9" }}>{name}</td>
                <td style={{ padding: "8px 6px", color: "#6E7681" }}>{before}</td>
                <td style={{ padding: "8px 6px", color: "#10B981", fontWeight: 700 }}>{after}</td>
                <td style={{ padding: "8px 6px", fontWeight: 800, color }}>{delta}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={{ marginTop: 16, padding: "12px 16px", background: "#064E3B", border: "1px solid #059669", borderRadius: 8, fontSize: 12, lineHeight: 1.8 }}>
          <div style={{ fontWeight: 800, color: "#10B981", marginBottom: 4 }}>🎉 P2 里程碑达成</div>
          <div style={{ color: "#C9D1D9" }}>
            <b>5/5 项全部完成</b> — 平均完成度从 58% 提升到 96%。<br/>
            <span style={{ color: "#8B949E" }}>• P2-01: .proto 定义 + Servicer 正式注册 + Health Check</span><br/>
            <span style={{ color: "#8B949E" }}>• P2-02: /.well-known/mcp/server.json 自动发现端点</span><br/>
            <span style={{ color: "#8B949E" }}>• P2-03: PostgresConnector 完整实现 (连接池 + 8 域查询)</span><br/>
            <span style={{ color: "#8B949E" }}>• P2-04: Deep Agents 0.4.1 适配器 + 懒加载集成</span><br/>
            <span style={{ color: "#8B949E" }}>• P2-05: react_planner LangGraph 集成 + QueryPlanner 路由</span><br/>
            <br/>
            <span style={{ color: "#F59E0B" }}>可选后续:</span> TLS 配置 (P2-01)、awm_env_factory 抽象 (P2-03)、运行时依赖安装 (P2-04/05)
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 820, margin: "16px auto 0", textAlign: "center", fontSize: 10, color: "#21262D", fontFamily: "monospace" }}>
        P2 Final Evaluation · 5/5 done · {ITEMS.reduce((s, i) => s + i.evidence.length, 0)} grep checks · {new Date().toISOString()}
      </div>
    </div>
  );
}
