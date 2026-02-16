"use client"

import { Circle, Bell, Search } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface TopBarProps {
  activeTab: string
}

const tabTitles: Record<string, string> = {
  overview: "总览仪表盘",
  agents: "智能体矩阵",
  chat: "智能对话",
  audit: "审计日志",
  files: "文件管理",
}

export function TopBar({ activeTab }: TopBarProps) {
  return (
    <header className="sticky top-0 z-50 flex h-16 items-center justify-between border-b border-[#1a1a1a] bg-[#090909]/80 px-6 backdrop-blur-xl">
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-[#ededed] tracking-tight">
          {tabTitles[activeTab] || "仪表盘"}
        </h1>
      </div>
      <div className="flex items-center gap-4">
        {/* Search placeholder */}
        <div className="hidden items-center gap-2 rounded-lg border border-[#1a1a1a] bg-[#0e0e0e] px-3 py-1.5 md:flex">
          <Search className="h-3.5 w-3.5 text-[#555]" />
          <span className="font-mono text-xs text-[#3a3a3a]">搜索...</span>
          <kbd className="ml-4 rounded border border-[#1a1a1a] bg-[#141414] px-1.5 py-0.5 font-mono text-[10px] text-[#3a3a3a]">
            {"/"}
          </kbd>
        </div>

        {/* Agents Online Badge */}
        <Badge className="gap-1.5 border-[#333] bg-[#1a1a1a] px-3 py-1 text-[#ccc] hover:bg-[#1a1a1a]">
          <Circle className="h-2 w-2 fill-[#ededed] text-[#ededed] animate-pulse-glow" />
          <span className="font-mono text-xs">7 个智能体在线</span>
        </Badge>

        {/* Notifications */}
        <button className="relative rounded-lg p-2 text-[#555] transition-colors hover:bg-[#141414] hover:text-[#ccc]" aria-label="通知">
          <Bell className="h-4 w-4" />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-[#ffffff]" />
        </button>
      </div>
    </header>
  )
}
