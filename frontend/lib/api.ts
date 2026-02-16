/**
 * MRARFAI — API 客户端
 * 所有前端 → FastAPI 后端的调用集中在此
 */

import { getAuthHeaders, setToken, setUser, clearToken } from "./auth"
import type { Agent, ChatMessage, AuditEntry } from "./dashboard-data"

const BASE = "/api"

// ─── 通用请求 ─────────────────────────────────────────────────
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
      ...options?.headers,
    },
    ...options,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `API Error: ${res.status}`)
  }

  return res.json()
}

// ─── Auth ─────────────────────────────────────────────────────
export interface LoginResult {
  token: string
  user: {
    username: string
    displayName: string
    role: string
    company: string
  }
}

export async function login(email: string, password: string): Promise<LoginResult> {
  const result = await request<LoginResult>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  })
  setToken(result.token)
  setUser(result.user)
  return result
}

export async function getMe() {
  return request<{ username: string; displayName: string; role: string; company: string }>("/auth/me")
}

export function logout() {
  clearToken()
}

// ─── Agents ───────────────────────────────────────────────────
export async function getAgents(): Promise<Agent[]> {
  return request<Agent[]>("/agents/list")
}

export async function askAgent(message: string): Promise<ChatMessage> {
  return request<ChatMessage>("/agents/ask", {
    method: "POST",
    body: JSON.stringify({ message }),
  })
}

export async function getAgentStats() {
  return request<Record<string, unknown>>("/agents/stats")
}

// ─── Dashboard Data ───────────────────────────────────────────
export interface DashboardData {
  kpis: { title: string; value: string; unit: string; change: string }[]
  monthlyShipments: { month: string; planned: number; actual: number }[]
  clientDistribution: { name: string; value: number; color: string }[]
  revenueComparison: { quarter: string; revenue2024: number; revenue2025: number }[]
  agents: Agent[]
}

export async function getDashboardData(): Promise<DashboardData> {
  return request<DashboardData>("/data/dashboard")
}

// ─── Audit ────────────────────────────────────────────────────
export async function getAuditLogs(): Promise<AuditEntry[]> {
  return request<AuditEntry[]>("/audit/logs")
}

export interface AuditStats {
  totalRequests: string
  collaborationRate: string
  avgLatency: string
  successRate: string
}

export async function getAuditStats(): Promise<AuditStats> {
  return request<AuditStats>("/audit/stats")
}

// ─── File Upload ──────────────────────────────────────────────
export interface UploadResult {
  id: string
  name: string
  size: number
  assignedAgents: string[]
  processingStatus: string
  analysisResult: string | null
  confidence: number | null
}

export async function uploadFile(file: File): Promise<UploadResult> {
  const formData = new FormData()
  formData.append("file", file)

  const res = await fetch(`${BASE}/data/upload`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: formData,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Upload failed: ${res.status}`)
  }

  return res.json()
}
