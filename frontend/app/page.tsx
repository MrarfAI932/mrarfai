"use client"

import { useState, useEffect } from "react"
import { LoginForm } from "@/components/login-form"
import { AppSidebar } from "@/components/dashboard/app-sidebar"
import { TopBar } from "@/components/dashboard/top-bar"
import { OverviewTab } from "@/components/dashboard/overview-tab"
import { AgentsTab } from "@/components/dashboard/agents-tab"
import { ChatTab } from "@/components/dashboard/chat-tab"
import { AuditTab } from "@/components/dashboard/audit-tab"
import { FilesTab } from "@/components/dashboard/files-tab"
import { ScrollArea } from "@/components/ui/scroll-area"
import { getToken, clearToken } from "@/lib/auth"
import { logout as apiLogout } from "@/lib/api"

const tabs: Record<string, React.ComponentType> = {
  overview: OverviewTab,
  agents: AgentsTab,
  chat: ChatTab,
  audit: AuditTab,
  files: FilesTab,
}

export default function Page() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [activeTab, setActiveTab] = useState("overview")

  // 检查 localStorage 中是否有 token (持久化登录)
  useEffect(() => {
    const token = getToken()
    if (token) {
      setIsAuthenticated(true)
    }
  }, [])

  const handleLogout = () => {
    apiLogout()
    clearToken()
    setIsAuthenticated(false)
  }

  const ActiveComponent = tabs[activeTab] || OverviewTab

  if (!isAuthenticated) {
    return <LoginForm onLogin={() => setIsAuthenticated(true)} />
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#0a0a0a]">
      <AppSidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onLogout={handleLogout}
      />
      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar activeTab={activeTab} />
        <ScrollArea className="flex-1">
          <main className="p-6">
            <ActiveComponent />
          </main>
        </ScrollArea>
      </div>
    </div>
  )
}
