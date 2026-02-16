"use client"

import { useState, useEffect } from "react"
import {
  TrendingUp,
  CheckCircle,
  DollarSign,
  Users,
  ArrowUpRight,
  Circle,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import {
  agents as defaultAgents,
} from "@/lib/dashboard-data"
import type { Agent } from "@/lib/dashboard-data"
import { getDashboardData, type DashboardData } from "@/lib/api"

// ---- Default KPI Cards ----
const defaultKpis = [
  {
    title: "总出货量",
    value: "23.9M",
    unit: "台",
    change: "+12.5%",
    icon: TrendingUp,
  },
  {
    title: "完成率",
    value: "84.3",
    unit: "%",
    change: "+3.2%",
    icon: CheckCircle,
  },
  {
    title: "Q1 营收",
    value: "7.34",
    unit: "亿",
    change: "+19.9%",
    icon: DollarSign,
  },
  {
    title: "活跃客户",
    value: "25",
    unit: "+",
    change: "+4 新增",
    icon: Users,
  },
]

const defaultShipments = [
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

const defaultClients = [
  { name: "HMD", value: 12700, color: "#ffffff" },
  { name: "ZTE", value: 6700, color: "#cccccc" },
  { name: "ZYB", value: 1500, color: "#999999" },
  { name: "LAVA", value: 930, color: "#666666" },
  { name: "BLU", value: 346, color: "#444444" },
]

const defaultRevenue = [
  { quarter: "Q1", revenue2025: 734, revenue2024: 612 },
  { quarter: "Q2", revenue2025: 680, revenue2024: 590 },
  { quarter: "Q3", revenue2025: 720, revenue2024: 640 },
  { quarter: "Q4", revenue2025: 790, revenue2024: 670 },
]

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-[#1a1a1a] bg-[#111] px-3 py-2 shadow-xl">
      <p className="mb-1 font-mono text-xs text-[#666]">{label}</p>
      {payload.map((entry: { name: string; value: number; color: string }, i: number) => (
        <p key={i} className="font-mono text-xs" style={{ color: entry.color }}>
          {entry.name}: {typeof entry.value === "number" ? entry.value.toLocaleString() : entry.value}
        </p>
      ))}
    </div>
  )
}

export function OverviewTab() {
  const [kpis, setKpis] = useState(defaultKpis)
  const [monthlyShipments, setMonthlyShipments] = useState(defaultShipments)
  const [clientDistribution, setClientDistribution] = useState(defaultClients)
  const [revenueComparison, setRevenueComparison] = useState(defaultRevenue)
  const [agents, setAgents] = useState<Agent[]>(defaultAgents)

  useEffect(() => {
    getDashboardData()
      .then((data) => {
        if (data.kpis?.length) {
          setKpis(data.kpis.map((k, i) => ({
            ...k,
            icon: defaultKpis[i]?.icon || TrendingUp,
          })))
        }
        if (data.monthlyShipments?.length) setMonthlyShipments(data.monthlyShipments)
        if (data.clientDistribution?.length) setClientDistribution(data.clientDistribution)
        if (data.revenueComparison?.length) setRevenueComparison(data.revenueComparison)
        if (data.agents?.length) setAgents(data.agents)
      })
      .catch(() => { /* use defaults on error */ })
  }, [])

  return (
    <div className="flex flex-col gap-6">
      {/* KPI Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {kpis.map((kpi, i) => {
          const Icon = kpi.icon
          return (
            <Card
              key={kpi.title}
              className="animate-fade-in-up border-[#1a1a1a] bg-[#111]"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <CardContent className="p-5">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-xs font-medium text-[#666]">
                      {kpi.title}
                    </p>
                    <div className="mt-2 flex items-baseline gap-1">
                      <span className="font-mono text-2xl font-bold text-[#ededed]">
                        {kpi.value}
                      </span>
                      <span className="font-mono text-sm text-[#666]">
                        {kpi.unit}
                      </span>
                    </div>
                    <div className="mt-2 flex items-center gap-1">
                      <ArrowUpRight className="h-3 w-3 text-[#ccc]" />
                      <span className="font-mono text-xs text-[#ccc]">
                        {kpi.change}
                      </span>
                    </div>
                  </div>
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#1a1a1a]">
                    <Icon className="h-5 w-5 text-[#888]" />
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Charts Row 1: Shipment Trend + Client Distribution */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Area Chart - Shipment Trend */}
        <Card
          className="animate-fade-in-up border-[#1a1a1a] bg-[#111] lg:col-span-2"
          style={{ animationDelay: "320ms" }}
        >
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-[#ededed]">
              月度出货趋势
            </CardTitle>
            <p className="font-mono text-xs text-[#666]">
              计划 vs 实际 (千台)
            </p>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={monthlyShipments}>
                  <defs>
                    <linearGradient id="plannedGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ffffff" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#ffffff" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="actualGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#888888" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#888888" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                  <XAxis
                    dataKey="month"
                    tick={{ fontSize: 11, fill: "#666", fontFamily: "var(--font-jetbrains-mono)" }}
                    axisLine={{ stroke: "#1a1a1a" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "#666", fontFamily: "var(--font-jetbrains-mono)" }}
                    axisLine={{ stroke: "#1a1a1a" }}
                    tickLine={false}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="planned"
                    name="计划"
                    stroke="#ffffff"
                    fill="url(#plannedGrad)"
                    strokeWidth={2}
                  />
                  <Area
                    type="monotone"
                    dataKey="actual"
                    name="实际"
                    stroke="#666666"
                    fill="url(#actualGrad)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-[#ffffff]" />
                <span className="font-mono text-xs text-[#666]">计划</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-[#666]" />
                <span className="font-mono text-xs text-[#666]">实际</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Horizontal Bar Chart - Client Distribution */}
        <Card
          className="animate-fade-in-up border-[#1a1a1a] bg-[#111]"
          style={{ animationDelay: "400ms" }}
        >
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-[#ededed]">
              客户分布
            </CardTitle>
            <p className="font-mono text-xs text-[#666]">
              出货量 (千台)
            </p>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={clientDistribution} layout="vertical" barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11, fill: "#666", fontFamily: "var(--font-jetbrains-mono)" }}
                    axisLine={{ stroke: "#1a1a1a" }}
                    tickLine={false}
                    tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fontSize: 11, fill: "#666", fontFamily: "var(--font-jetbrains-mono)" }}
                    axisLine={{ stroke: "#1a1a1a" }}
                    tickLine={false}
                    width={40}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="出货量" radius={[0, 4, 4, 0]}>
                    {clientDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2: Revenue + Agent Status */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Revenue YoY */}
        <Card
          className="animate-fade-in-up border-[#1a1a1a] bg-[#111]"
          style={{ animationDelay: "480ms" }}
        >
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-[#ededed]">
              营收同比对比
            </CardTitle>
            <p className="font-mono text-xs text-[#666]">
              季度 (百万人民币)
            </p>
          </CardHeader>
          <CardContent>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={revenueComparison} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                  <XAxis
                    dataKey="quarter"
                    tick={{ fontSize: 11, fill: "#666", fontFamily: "var(--font-jetbrains-mono)" }}
                    axisLine={{ stroke: "#1a1a1a" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "#666", fontFamily: "var(--font-jetbrains-mono)" }}
                    axisLine={{ stroke: "#1a1a1a" }}
                    tickLine={false}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="revenue2024" name="2024" fill="#444444" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="revenue2025" name="2025" fill="#ffffff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-[#444]" />
                <span className="font-mono text-xs text-[#666]">2024</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-[#ffffff]" />
                <span className="font-mono text-xs text-[#666]">2025</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Agent Status Grid */}
        <Card
          className="animate-fade-in-up border-[#1a1a1a] bg-[#111] lg:col-span-2"
          style={{ animationDelay: "560ms" }}
        >
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-[#ededed]">
              智能体状态
            </CardTitle>
            <p className="font-mono text-xs text-[#666]">
              实时健康监控
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {agents.map((agent) => (
                <div
                  key={agent.id}
                  className="flex items-center gap-3 rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] px-3 py-3 transition-colors hover:border-[#2a2a2a]"
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#1a1a1a]">
                    <Circle
                      className="h-3 w-3 text-[#888] fill-[#888]"
                    />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-xs font-medium text-[#ccc]">
                      {agent.name.replace(" Agent", "")}
                    </p>
                    <div className="flex items-center gap-1.5">
                      <div className="h-1.5 w-1.5 rounded-full bg-[#ededed] animate-pulse-glow" />
                      <span className="font-mono text-[10px] text-[#888]">
                        在线
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
