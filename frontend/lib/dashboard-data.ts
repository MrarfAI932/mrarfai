// ─── Agent definitions ────────────────────────────────────────────
export interface Agent {
  id: string
  name: string
  role: string
  description: string
  skills: string[]
  taskCount: number
  status: "online" | "offline" | "busy"
  color: string
  icon: string
}

export const agents: Agent[] = [
  {
    id: "sales",
    name: "销售智能体",
    role: "营收情报",
    description: "分析客户管道、预测成交率、跟踪所有 ODM/OEM 账户的出货量。",
    skills: ["管道分析", "营收预测", "客户评分", "交易跟踪"],
    taskCount: 342,
    status: "online",
    color: "#6366f1",
    icon: "TrendingUp",
  },
  {
    id: "risk",
    name: "风控智能体",
    role: "风险评估",
    description: "监控供应链中断、客户付款风险以及影响运营的地缘政治因素。",
    skills: ["风险评分", "供应链监控", "付款分析", "地缘风险"],
    taskCount: 187,
    status: "online",
    color: "#ef4444",
    icon: "ShieldAlert",
  },
  {
    id: "strategist",
    name: "策略智能体",
    role: "战略规划",
    description: "制定市场进入策略、竞争分析和长期增长规划。",
    skills: ["市场分析", "竞争情报", "增长策略", "情景规划"],
    taskCount: 156,
    status: "online",
    color: "#a855f7",
    icon: "Brain",
  },
  {
    id: "procurement",
    name: "采购智能体",
    role: "供应链",
    description: "优化组件采购、供应商谈判和跨工厂库存管理。",
    skills: ["供应商评分", "成本优化", "库存预测", "BOM 分析"],
    taskCount: 298,
    status: "online",
    color: "#f59e0b",
    icon: "Package",
  },
  {
    id: "quality",
    name: "质量智能体",
    role: "质量保证",
    description: "跟踪缺陷率、产线质量指标，确保符合国际标准。",
    skills: ["缺陷分析", "质检指标", "合规检查", "良率优化"],
    taskCount: 213,
    status: "online",
    color: "#10b981",
    icon: "CheckCircle",
  },
  {
    id: "finance",
    name: "财务智能体",
    role: "财务分析",
    description: "管理营收跟踪、成本分析、利润优化和财务报告。",
    skills: ["损益分析", "利润跟踪", "现金流", "财务建模"],
    taskCount: 276,
    status: "online",
    color: "#06b6d4",
    icon: "DollarSign",
  },
  {
    id: "market",
    name: "市场智能体",
    role: "市场情报",
    description: "跟踪行业趋势、竞争对手动态和移动设备新兴市场机会。",
    skills: ["趋势分析", "竞品监控", "市场规模", "需求预测"],
    taskCount: 189,
    status: "online",
    color: "#ec4899",
    icon: "Globe",
  },
]

// ─── Shipment data ─────────────────────────────────────────────────
export const monthlyShipments = [
  { month: "1月", planned: 2100, actual: 1980 },
  { month: "2月", planned: 2200, actual: 2050 },
  { month: "3月", planned: 2400, actual: 2280 },
  { month: "4月", planned: 2300, actual: 2150 },
  { month: "5月", planned: 2500, actual: 2380 },
  { month: "6月", planned: 2600, actual: 2420 },
  { month: "7月", planned: 2400, actual: 2310 },
  { month: "8月", planned: 2300, actual: 2180 },
  { month: "9月", planned: 2500, actual: 2350 },
  { month: "10月", planned: 2600, actual: 2490 },
  { month: "11月", planned: 2700, actual: 2580 },
  { month: "12月", planned: 2800, actual: 2670 },
]

// ─── Client distribution ───────────────────────────────────────────
export const clientDistribution = [
  { name: "HMD", value: 12700, color: "#ffffff" },
  { name: "ZTE", value: 6700, color: "#cccccc" },
  { name: "ZYB", value: 1500, color: "#999999" },
  { name: "LAVA", value: 930, color: "#666666" },
  { name: "BLU", value: 346, color: "#444444" },
]

// ─── Revenue comparison ────────────────────────────────────────────
export const revenueComparison = [
  { quarter: "Q1", revenue2025: 734, revenue2024: 612 },
  { quarter: "Q2", revenue2025: 680, revenue2024: 590 },
  { quarter: "Q3", revenue2025: 720, revenue2024: 640 },
  { quarter: "Q4", revenue2025: 790, revenue2024: 670 },
]

// ─── Audit trail data ──────────────────────────────────────────────
export interface AuditEntry {
  id: string
  timestamp: string
  agent: string
  query: string
  confidence: number
  latency: number
  status: "success" | "error" | "warning"
}

export const auditEntries: AuditEntry[] = [
  { id: "req-001", timestamp: "2025-03-15 14:23:01", agent: "销售智能体", query: "HMD 账户 Q1 出货预测", confidence: 96.2, latency: 420, status: "success" },
  { id: "req-002", timestamp: "2025-03-15 14:22:45", agent: "风控智能体", query: "深圳工厂供应链风险评估", confidence: 91.8, latency: 680, status: "success" },
  { id: "req-003", timestamp: "2025-03-15 14:22:30", agent: "财务智能体", query: "LAVA 合同续签利润分析", confidence: 94.5, latency: 530, status: "success" },
  { id: "req-004", timestamp: "2025-03-15 14:22:15", agent: "市场智能体", query: "非洲市场竞品定价更新", confidence: 88.3, latency: 750, status: "warning" },
  { id: "req-005", timestamp: "2025-03-15 14:22:00", agent: "策略智能体", query: "拉丁美洲扩张市场准入分析", confidence: 92.7, latency: 890, status: "success" },
  { id: "req-006", timestamp: "2025-03-15 14:21:45", agent: "质量智能体", query: "2月生产批次缺陷率分析", confidence: 97.1, latency: 340, status: "success" },
  { id: "req-007", timestamp: "2025-03-15 14:21:30", agent: "采购智能体", query: "Q2 BOM 组件成本优化", confidence: 93.4, latency: 610, status: "success" },
  { id: "req-008", timestamp: "2025-03-15 14:21:15", agent: "销售智能体", query: "ZTE 账户续约概率评估", confidence: 89.6, latency: 470, status: "success" },
  { id: "req-009", timestamp: "2025-03-15 14:21:00", agent: "风控智能体", query: "BLU 应收账款付款风险评分", confidence: 85.2, latency: 920, status: "warning" },
  { id: "req-010", timestamp: "2025-03-15 14:20:45", agent: "财务智能体", query: "Q1 营收预测偏差分析", confidence: 95.8, latency: 380, status: "success" },
  { id: "req-011", timestamp: "2025-03-15 14:20:30", agent: "市场智能体", query: "南亚功能手机需求趋势", confidence: 90.1, latency: 710, status: "success" },
  { id: "req-012", timestamp: "2025-03-15 14:20:15", agent: "策略智能体", query: "ODM 合作伙伴评估 - 新客户管道", confidence: 91.5, latency: 830, status: "success" },
]

// ─── Chat messages ──────────────────────────────────────────────────
export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  agent?: string
  confidence?: number
  latency?: number
  collaborators?: string[]
  timestamp: string
}

export const sampleMessages: ChatMessage[] = [
  {
    id: "msg-1",
    role: "user",
    content: "我们前三大客户的 Q1 出货情况如何？",
    timestamp: "14:20",
  },
  {
    id: "msg-2",
    role: "assistant",
    content: "根据实时数据分析，以下是 Q1 出货情况：\n\n**HMD Global**: 已出货 320 万台 (目标: 350 万) - 完成率 91.4%\n**ZTE**: 已出货 170 万台 (目标: 190 万) - 完成率 89.5%\n**ZYB**: 已出货 38 万台 (目标: 42 万) - 完成率 90.5%\n\n整体 Q1 完成率为 84.3%。HMD 仍然是表现最强的客户，月环比持续增长。",
    agent: "销售智能体",
    confidence: 96.2,
    latency: 420,
    collaborators: ["财务智能体"],
    timestamp: "14:20",
  },
  {
    id: "msg-3",
    role: "user",
    content: "有什么需要注意的供应链风险吗？",
    timestamp: "14:21",
  },
  {
    id: "msg-4",
    role: "assistant",
    content: "已识别 3 个活跃风险因素：\n\n1. **组件短缺 (中等)**: 高通芯片交付延迟，影响 2 条产线。预计恢复时间: 2 周。\n2. **物流 (低)**: 深圳港口轻微拥堵 - 出港货物延迟 1-2 天。\n3. **汇率 (低)**: 人民币/美元波动在可接受的 2% 范围内。\n\n建议: 提前预订 Q2 高通 QCM6490 芯片以降低短缺风险。预计成本影响: BOM +0.3%。",
    agent: "风控智能体",
    confidence: 91.8,
    latency: 680,
    collaborators: ["采购智能体", "财务智能体"],
    timestamp: "14:21",
  },
]

export const quickPrompts = [
  "Q1 出货概览",
  "头部客户分析",
  "供应链风险",
  "营收预测",
  "市场趋势",
  "质量指标",
]
