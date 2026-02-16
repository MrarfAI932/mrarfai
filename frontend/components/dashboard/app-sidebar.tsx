"use client"

import { useState } from "react"
import Image from "next/image"
import {
  LayoutDashboard,
  Bot,
  MessageSquare,
  FileText,
  Upload,
  ChevronLeft,
  ChevronRight,
  LogOut,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface AppSidebarProps {
  activeTab: string
  onTabChange: (tab: string) => void
  onLogout: () => void
}

const navItems = [
  { id: "overview", label: "总览", icon: LayoutDashboard },
  { id: "agents", label: "智能体矩阵", icon: Bot },
  { id: "chat", label: "智能对话", icon: MessageSquare },
  { id: "audit", label: "审计日志", icon: FileText },
  { id: "files", label: "文件管理", icon: Upload },
]

export function AppSidebar({ activeTab, onTabChange, onLogout }: AppSidebarProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        "flex flex-col border-r border-[#1a1a1a] bg-[#090909] transition-all duration-300 ease-in-out",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Logo area */}
      <div className="flex h-16 items-center gap-3 border-b border-[#1a1a1a] px-4">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center">
          <Image
            src="/images/logo_horse.png"
            alt="MRARFAI logo"
            width={28}
            height={28}
            className="invert"
          />
        </div>
        {!collapsed && (
          <div className="flex flex-col overflow-hidden">
            <span className="text-sm font-semibold text-[#ededed] tracking-tight">
              MRARFAI
            </span>
            <span className="text-[10px] font-mono text-[#666]">
              智能情报平台
            </span>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4">
        <div className="flex flex-col gap-1">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = activeTab === item.id
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                  isActive
                    ? "bg-[#ffffff]/10 text-[#ffffff]"
                    : "text-[#666] hover:bg-[#141414] hover:text-[#ccc]",
                  collapsed && "justify-center px-0"
                )}
              >
                <Icon className={cn("h-4 w-4 shrink-0", isActive && "text-[#ffffff]")} />
                {!collapsed && <span>{item.label}</span>}
              </button>
            )
          })}
        </div>
      </nav>

      {/* Bottom section */}
      <div className="border-t border-[#1a1a1a] p-3">
        {!collapsed && (
          <div className="mb-3 rounded-lg bg-[#0e0e0e] px-3 py-2">
            <p className="font-mono text-[10px] text-[#555]">
              SPROCOMM &middot; 01401.HK
            </p>
            <p className="font-mono text-[10px] text-[#3a3a3a]">V10.1</p>
          </div>
        )}
        <div className="flex items-center gap-1">
          <button
            onClick={onLogout}
            className={cn(
              "flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-[#555] transition-colors hover:bg-[#141414] hover:text-[#ccc]",
              collapsed ? "w-full justify-center px-0" : "flex-1"
            )}
            aria-label="退出登录"
          >
            <LogOut className="h-4 w-4 shrink-0" />
            {!collapsed && <span>退出登录</span>}
          </button>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="flex items-center justify-center rounded-lg p-2 text-[#555] transition-colors hover:bg-[#141414] hover:text-[#ccc]"
            aria-label={collapsed ? "展开侧栏" : "折叠侧栏"}
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </aside>
  )
}
