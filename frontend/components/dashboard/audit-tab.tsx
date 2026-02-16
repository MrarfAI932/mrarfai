"use client"

import { useState, useEffect } from "react"
import {
  FileText,
  Users,
  Clock,
  CheckCircle,
} from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { auditEntries as defaultEntries, type AuditEntry } from "@/lib/dashboard-data"
import { getAuditLogs, getAuditStats } from "@/lib/api"

const defaultKpis = [
  {
    title: "总请求数",
    value: "2,419",
    icon: FileText,
  },
  {
    title: "协同率",
    value: "34%",
    icon: Users,
  },
  {
    title: "平均延迟",
    value: "680ms",
    icon: Clock,
  },
  {
    title: "成功率",
    value: "99.2%",
    icon: CheckCircle,
  },
]

export function AuditTab() {
  const [auditKpis, setAuditKpis] = useState(defaultKpis)
  const [entries, setEntries] = useState<AuditEntry[]>(defaultEntries)

  useEffect(() => {
    // Fetch audit logs from API
    getAuditLogs()
      .then((data) => { if (data?.length) setEntries(data) })
      .catch(() => { /* use defaults */ })

    // Fetch audit stats from API
    getAuditStats()
      .then((stats) => {
        setAuditKpis([
          { title: "总请求数", value: stats.totalRequests, icon: FileText },
          { title: "协同率", value: stats.collaborationRate, icon: Users },
          { title: "平均延迟", value: stats.avgLatency, icon: Clock },
          { title: "成功率", value: stats.successRate, icon: CheckCircle },
        ])
      })
      .catch(() => { /* use defaults */ })
  }, [])

  return (
    <div className="flex flex-col gap-6">
      {/* KPI Row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {auditKpis.map((kpi, i) => {
          const Icon = kpi.icon
          return (
            <Card
              key={kpi.title}
              className="animate-fade-in-up border-[#1a1a1a] bg-[#111]"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <CardContent className="flex items-center gap-4 p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#1a1a1a]">
                  <Icon className="h-5 w-5 text-[#888]" />
                </div>
                <div>
                  <p className="text-xs font-medium text-[#666]">
                    {kpi.title}
                  </p>
                  <p className="mt-0.5 font-mono text-xl font-bold text-[#ededed]">
                    {kpi.value}
                  </p>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Log Table */}
      <Card
        className="animate-fade-in-up border-[#1a1a1a] bg-[#111]"
        style={{ animationDelay: "320ms" }}
      >
        <CardContent className="p-0">
          <div className="overflow-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-[#1a1a1a] hover:bg-transparent">
                  <TableHead className="font-mono text-xs text-[#555]">
                    时间戳
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555]">
                    智能体
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555]">
                    查询内容
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555] text-right">
                    置信度
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555] text-right">
                    延迟
                  </TableHead>
                  <TableHead className="font-mono text-xs text-[#555] text-right">
                    状态
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {entries.map((entry) => (
                  <TableRow
                    key={entry.id}
                    className="border-[#1a1a1a] transition-colors hover:bg-[#0e0e0e]"
                  >
                    <TableCell className="font-mono text-xs text-[#666] whitespace-nowrap">
                      {entry.timestamp}
                    </TableCell>
                    <TableCell>
                      <span className="text-xs font-medium text-[#ccc]">
                        {entry.agent}
                      </span>
                    </TableCell>
                    <TableCell className="max-w-xs">
                      <span className="text-xs text-[#666] line-clamp-1">
                        {entry.query}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span
                        className={`font-mono text-xs ${
                          entry.confidence >= 95
                            ? "text-[#ededed]"
                            : entry.confidence >= 90
                              ? "text-[#aaa]"
                              : "text-[#666]"
                        }`}
                      >
                        {entry.confidence}%
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span
                        className={`font-mono text-xs ${
                          entry.latency <= 500
                            ? "text-[#ededed]"
                            : entry.latency <= 700
                              ? "text-[#aaa]"
                              : "text-[#666]"
                        }`}
                      >
                        {entry.latency}ms
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge
                        className={`font-mono text-[10px] ${
                          entry.status === "success"
                            ? "border-[#333] bg-[#1a1a1a] text-[#ccc] hover:bg-[#1a1a1a]"
                            : entry.status === "warning"
                              ? "border-[#333] bg-[#1a1a1a] text-[#888] hover:bg-[#1a1a1a]"
                              : "border-[#333] bg-[#1a1a1a] text-[#555] hover:bg-[#1a1a1a]"
                        }`}
                      >
                        {entry.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
