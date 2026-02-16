"use client"

import { useState, useEffect } from "react"
import {
  TrendingUp,
  ShieldAlert,
  Brain,
  Package,
  CheckCircle,
  DollarSign,
  Globe,
  ChevronRight,
  X,
  Zap,
  ListChecks,
} from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { agents as defaultAgents, type Agent } from "@/lib/dashboard-data"
import { getAgents } from "@/lib/api"

const iconMap: Record<string, React.ElementType> = {
  TrendingUp,
  ShieldAlert,
  Brain,
  Package,
  CheckCircle,
  DollarSign,
  Globe,
}

function AgentCard({
  agent,
  onSelect,
  index,
}: {
  agent: Agent
  onSelect: (agent: Agent) => void
  index: number
}) {
  const Icon = iconMap[agent.icon] || Brain

  return (
    <Card
      className="group animate-fade-in-up cursor-pointer border-[#1a1a1a] bg-[#111] transition-all duration-300 hover:border-[#333] hover:shadow-lg"
      style={{ animationDelay: `${index * 80}ms` }}
      onClick={() => onSelect(agent)}
    >
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#1a1a1a]">
            <Icon className="h-5 w-5 text-[#999]" />
          </div>
          <ChevronRight className="h-4 w-4 text-[#333] transition-all group-hover:translate-x-0.5 group-hover:text-[#666]" />
        </div>

        <div className="mt-4">
          <h3 className="text-sm font-semibold text-[#ededed]">
            {agent.name}
          </h3>
          <p className="mt-0.5 font-mono text-xs text-[#666]">
            {agent.role}
          </p>
        </div>

        <p className="mt-3 text-xs leading-relaxed text-[#555]">
          {agent.description}
        </p>

        <div className="mt-4 flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <Zap className="h-3 w-3 text-[#444]" />
            <span className="font-mono text-[10px] text-[#666]">
              {agent.skills.length} 项技能
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <ListChecks className="h-3 w-3 text-[#444]" />
            <span className="font-mono text-[10px] text-[#666]">
              {agent.taskCount} 个任务
            </span>
          </div>
        </div>

        <div className="mt-3 flex items-center gap-1.5">
          <div className="h-1.5 w-1.5 rounded-full bg-[#ededed] animate-pulse-glow" />
          <span className="font-mono text-[10px] text-[#888]">在线</span>
        </div>
      </CardContent>
    </Card>
  )
}

function AgentDetail({
  agent,
  onClose,
}: {
  agent: Agent
  onClose: () => void
}) {
  const Icon = iconMap[agent.icon] || Brain

  return (
    <div className="animate-fade-in-up">
      <Card className="border-[#1a1a1a] bg-[#111]">
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#1a1a1a]">
                <Icon className="h-6 w-6 text-[#999]" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-[#ededed]">
                  {agent.name}
                </h2>
                <p className="font-mono text-xs text-[#666]">
                  {agent.role}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="rounded-lg p-2 text-[#555] transition-colors hover:bg-[#1a1a1a] hover:text-[#ccc]"
              aria-label="关闭详情"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <p className="mt-4 text-sm leading-relaxed text-[#666]">
            {agent.description}
          </p>

          {/* Stats */}
          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] p-4 text-center">
              <p className="font-mono text-2xl font-bold text-[#ededed]">
                {agent.skills.length}
              </p>
              <p className="mt-1 text-xs text-[#666]">技能</p>
            </div>
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] p-4 text-center">
              <p className="font-mono text-2xl font-bold text-[#ededed]">
                {agent.taskCount}
              </p>
              <p className="mt-1 text-xs text-[#666]">任务</p>
            </div>
            <div className="rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] p-4 text-center">
              <p className="font-mono text-2xl font-bold text-[#ededed]">
                99.8%
              </p>
              <p className="mt-1 text-xs text-[#666]">在线率</p>
            </div>
          </div>

          {/* Capabilities */}
          <div className="mt-6">
            <h3 className="text-sm font-medium text-[#ccc]">
              能力
            </h3>
            <div className="mt-3 flex flex-wrap gap-2">
              {agent.skills.map((skill) => (
                <Badge
                  key={skill}
                  className="border-[#1a1a1a] bg-[#0e0e0e] px-3 py-1 font-mono text-xs text-[#888] hover:bg-[#1a1a1a]"
                >
                  {skill}
                </Badge>
              ))}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="mt-6">
            <h3 className="text-sm font-medium text-[#ccc]">
              近期活动
            </h3>
            <div className="mt-3 flex flex-col gap-2">
              {[
                { time: "2 分钟前", action: "处理查询: Q2 营收预测" },
                { time: "8 分钟前", action: "与风控智能体协同工作" },
                { time: "15 分钟前", action: "生成报告: 客户分析" },
              ].map((item, i) => (
                <div
                  key={i}
                  className="flex items-center gap-3 rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] px-3 py-2"
                >
                  <div className="h-1.5 w-1.5 shrink-0 rounded-full bg-[#888]" />
                  <span className="font-mono text-xs text-[#555]">
                    {item.time}
                  </span>
                  <span className="text-xs text-[#999]">
                    {item.action}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export function AgentsTab() {
  const [selected, setSelected] = useState<Agent | null>(null)
  const [agents, setAgents] = useState<Agent[]>(defaultAgents)

  useEffect(() => {
    getAgents()
      .then((data) => { if (data?.length) setAgents(data) })
      .catch(() => { /* use defaults */ })
  }, [])

  if (selected) {
    return <AgentDetail agent={selected} onClose={() => setSelected(null)} />
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {agents.map((agent, i) => (
        <AgentCard key={agent.id} agent={agent} onSelect={setSelected} index={i} />
      ))}
    </div>
  )
}
