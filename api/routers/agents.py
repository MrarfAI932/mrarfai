"""
MRARFAI API — 智能体路由
GET  /api/agents/list   → 7 个智能体列表
POST /api/agents/ask    → 调用 PlatformGateway.ask()
GET  /api/agents/stats  → 平台统计
"""

import sys
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, Depends

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from api.schemas import AgentSchema, AskRequest, ChatMessageSchema
from api.deps import get_current_user

router = APIRouter()

# ── 智能体定义 (精确匹配前端 dashboard-data.ts) ──
AGENTS: List[AgentSchema] = [
    AgentSchema(
        id="sales", name="销售智能体", role="营收情报",
        description="分析客户管道、预测成交率、跟踪所有 ODM/OEM 账户的出货量。",
        skills=["管道分析", "营收预测", "客户评分", "交易跟踪"],
        taskCount=342, status="online", color="#6366f1", icon="TrendingUp",
    ),
    AgentSchema(
        id="risk", name="风控智能体", role="风险评估",
        description="监控供应链中断、客户付款风险以及影响运营的地缘政治因素。",
        skills=["风险评分", "供应链监控", "付款分析", "地缘风险"],
        taskCount=187, status="online", color="#ef4444", icon="ShieldAlert",
    ),
    AgentSchema(
        id="strategist", name="策略智能体", role="战略规划",
        description="制定市场进入策略、竞争分析和长期增长规划。",
        skills=["市场分析", "竞争情报", "增长策略", "情景规划"],
        taskCount=156, status="online", color="#a855f7", icon="Brain",
    ),
    AgentSchema(
        id="procurement", name="采购智能体", role="供应链",
        description="优化组件采购、供应商谈判和跨工厂库存管理。",
        skills=["供应商评分", "成本优化", "库存预测", "BOM 分析"],
        taskCount=298, status="online", color="#f59e0b", icon="Package",
    ),
    AgentSchema(
        id="quality", name="质量智能体", role="质量保证",
        description="跟踪缺陷率、产线质量指标，确保符合国际标准。",
        skills=["缺陷分析", "质检指标", "合规检查", "良率优化"],
        taskCount=213, status="online", color="#10b981", icon="CheckCircle",
    ),
    AgentSchema(
        id="finance", name="财务智能体", role="财务分析",
        description="管理营收跟踪、成本分析、利润优化和财务报告。",
        skills=["损益分析", "利润跟踪", "现金流", "财务建模"],
        taskCount=276, status="online", color="#06b6d4", icon="DollarSign",
    ),
    AgentSchema(
        id="market", name="市场智能体", role="市场情报",
        description="跟踪行业趋势、竞争对手动态和移动设备新兴市场机会。",
        skills=["趋势分析", "竞品监控", "市场规模", "需求预测"],
        taskCount=189, status="online", color="#ec4899", icon="Globe",
    ),
]

# Agent 中文名 → 英文ID 映射
AGENT_NAME_MAP = {
    "sales": "销售智能体",
    "risk": "风控智能体",
    "strategist": "策略智能体",
    "procurement": "采购智能体",
    "quality": "质量智能体",
    "finance": "财务智能体",
    "market": "市场智能体",
}


def _get_gateway():
    """延迟加载 PlatformGateway"""
    try:
        from platform_gateway import get_gateway
        return get_gateway()
    except Exception:
        return None


@router.get("/list")
async def list_agents():
    """返回 7 个智能体列表，精确匹配前端 Agent[] 类型"""
    return [a.dict() for a in AGENTS]


@router.post("/ask", response_model=ChatMessageSchema)
async def ask_agent(req: AskRequest, user: dict = Depends(get_current_user)):
    """
    调用 PlatformGateway.ask() 进行真实 AI 对话。
    返回 ChatMessage 格式。
    """
    start_time = time.time()

    gateway = _get_gateway()
    if gateway is None:
        # 无 gateway 时返回说明消息
        return ChatMessageSchema(
            id=f"msg-{uuid.uuid4().hex[:8]}",
            role="assistant",
            content="系统正在初始化中，请稍后再试。PlatformGateway 尚未加载。",
            agent="系统",
            confidence=0.0,
            latency=0,
            timestamp=datetime.now().strftime("%H:%M"),
        )

    # 获取 API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    provider = os.environ.get("AI_PROVIDER", "claude").lower()

    try:
        result = gateway.ask(
            query=req.message,
            user=user.get("username", "anonymous"),
            provider=provider,
            api_key=api_key,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # 提取结果
        if result.get("type") == "collaboration":
            # 协作结果
            collab_result = result.get("result", {})
            content = collab_result.get("synthesis", str(collab_result))
            agent_name = AGENT_NAME_MAP.get(
                result.get("routing", {}).get("agent", "sales"),
                "销售智能体"
            )
            collaborators = [
                AGENT_NAME_MAP.get(a, a)
                for a in collab_result.get("agent_results", {}).keys()
                if a != result.get("routing", {}).get("agent", "")
            ]
            confidence = result.get("routing", {}).get("confidence", 0.9) * 100
        elif result.get("type") == "error":
            content = f"抱歉，处理您的请求时出现错误: {result.get('error', '未知错误')}"
            agent_name = "系统"
            collaborators = []
            confidence = 0.0
        else:
            # 单 Agent 结果
            content = result.get("answer", str(result))
            agent_name = AGENT_NAME_MAP.get(
                result.get("agent", "sales"),
                "销售智能体"
            )
            confidence = result.get("confidence", 0.9) * 100
            collaborators = []

        return ChatMessageSchema(
            id=f"msg-{uuid.uuid4().hex[:8]}",
            role="assistant",
            content=content,
            agent=agent_name,
            confidence=round(confidence, 1),
            latency=elapsed_ms,
            collaborators=collaborators if collaborators else None,
            timestamp=datetime.now().strftime("%H:%M"),
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ChatMessageSchema(
            id=f"msg-{uuid.uuid4().hex[:8]}",
            role="assistant",
            content=f"处理请求时出错: {str(e)}",
            agent="系统",
            confidence=0.0,
            latency=elapsed_ms,
            timestamp=datetime.now().strftime("%H:%M"),
        )


@router.get("/stats")
async def get_stats(user: dict = Depends(get_current_user)):
    """返回平台统计"""
    gateway = _get_gateway()
    if gateway:
        return gateway.get_platform_stats()
    return {
        "total_requests": 0,
        "by_agent": {},
        "by_status": {},
        "avg_duration_ms": 0,
        "agent_count": 7,
        "collab_scenarios": 4,
    }
