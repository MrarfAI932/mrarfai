"""
MRARFAI API — Pydantic Models
精确匹配前端 TypeScript 类型定义 (lib/dashboard-data.ts)
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# ─── Agent ─────────────────────────────────────────────────────
class AgentSchema(BaseModel):
    id: str
    name: str
    role: str
    description: str
    skills: List[str]
    taskCount: int
    status: str  # "online" | "offline" | "busy"
    color: str
    icon: str


# ─── Chat ──────────────────────────────────────────────────────
class ChatMessageSchema(BaseModel):
    id: str
    role: str  # "user" | "assistant"
    content: str
    agent: Optional[str] = None
    confidence: Optional[float] = None
    latency: Optional[int] = None
    collaborators: Optional[List[str]] = None
    timestamp: str


class AskRequest(BaseModel):
    message: str
    conversationId: Optional[str] = None


# ─── Audit ─────────────────────────────────────────────────────
class AuditEntrySchema(BaseModel):
    id: str
    timestamp: str
    agent: str
    query: str
    confidence: float
    latency: int
    status: str  # "success" | "error" | "warning"


class AuditStatsSchema(BaseModel):
    totalRequests: str
    collaborationRate: str
    avgLatency: str
    successRate: str


# ─── Dashboard ─────────────────────────────────────────────────
class KPISchema(BaseModel):
    title: str
    value: str
    unit: str
    change: str


class DashboardDataSchema(BaseModel):
    kpis: List[KPISchema]
    monthlyShipments: List[Dict[str, Any]]
    clientDistribution: List[Dict[str, Any]]
    revenueComparison: List[Dict[str, Any]]
    agents: List[AgentSchema]


# ─── Auth ──────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


class UserInfo(BaseModel):
    username: str
    displayName: str
    role: str
    company: str


class LoginResponse(BaseModel):
    token: str
    user: UserInfo


# ─── File Upload ───────────────────────────────────────────────
class UploadResponse(BaseModel):
    id: str
    name: str
    size: int
    assignedAgents: List[str]
    processingStatus: str
    analysisResult: Optional[str] = None
    confidence: Optional[float] = None
