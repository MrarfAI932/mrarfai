#!/usr/bin/env python3
"""
MRARFAI Sales Agent v4.0 - 对话式智能分析
==========================================
将Dashboard的12维分析能力封装为Agent工具，
通过自然语言交互，自主选择分析维度并生成回答。

架构：
  用户提问 → LLM意图识别 → 调用分析工具 → LLM生成回答
"""

import json
import os
from typing import Optional

# ============================================================
# 工具定义：每个工具对应一个分析能力
# ============================================================

TOOLS = {
    "客户概览": {
        "description": "查看总营收、总出货量、客户数量、月度趋势等全局指标",
        "keywords": ["总营收", "营收", "总体", "概览", "总量", "大盘", "整体", "多少钱", "revenue"],
    },
    "客户分级": {
        "description": "ABC分级、客户排名、集中度分析、各客户占比",
        "keywords": ["分级", "排名", "ABC", "top", "最大", "客户", "前几", "占比", "集中度", "大客户"],
    },
    "流失预警": {
        "description": "客户流失风险评估、预警信号、高危客户列表",
        "keywords": ["流失", "预警", "风险", "危险", "丢失", "下滑", "流失率", "churn", "警告"],
    },
    "增长机会": {
        "description": "高增长客户、超额完成客户、新兴放量客户、量价齐升客户",
        "keywords": ["增长", "机会", "潜力", "上升", "放量", "新客户", "增长点", "growth"],
    },
    "价量分解": {
        "description": "单价趋势、价格变动、以量换价分析、利润质量评估",
        "keywords": ["单价", "价格", "价量", "均价", "利润", "毛利", "价格趋势", "ASP"],
    },
    "异常检测": {
        "description": "月度环比异常波动、暴增暴跌事件检测",
        "keywords": ["异常", "波动", "暴增", "暴跌", "环比", "突变", "anomaly"],
    },
    "行业对标": {
        "description": "与华勤、闻泰、龙旗等竞争对手对比",
        "keywords": ["行业", "对标", "竞争", "华勤", "闻泰", "龙旗", "竞品", "benchmark", "对比"],
    },
    "预测": {
        "description": "Q1 2026营收预测、客户级别预测、情景分析",
        "keywords": ["预测", "forecast", "下季度", "Q1", "2026", "未来", "趋势", "预估"],
    },
    "区域分析": {
        "description": "区域出货分布、HHI集中度指数、区域风险评估",
        "keywords": ["区域", "地区", "市场", "印度", "非洲", "拉美", "HHI", "分布"],
    },
    "产品结构": {
        "description": "FP/SP/PAD产品类型分布、CKD/SKD/CBU订单模式",
        "keywords": ["产品", "FP", "SP", "PAD", "CKD", "SKD", "CBU", "手机", "平板", "结构"],
    },
    "目标达成": {
        "description": "年度目标完成率、各客户保底目标跟踪",
        "keywords": ["目标", "达成", "完成率", "保底", "target", "KPI"],
    },
    "销售绩效": {
        "description": "销售团队/个人业绩排名、稳定性评估",
        "keywords": ["销售", "团队", "绩效", "业绩", "负责人", "人员", "sales"],
    },
    "CEO建议": {
        "description": "本月CEO应关注的3件事、战略行动建议",
        "keywords": ["CEO", "建议", "战略", "行动", "决策", "管理层", "老板", "总裁"],
    },
}


# ============================================================
# 意图识别：匹配用户问题到工具
# ============================================================

def identify_tools(question: str) -> list:
    """基于关键词匹配识别用户意图，返回相关工具列表"""
    question_lower = question.lower()
    matched = []
    
    for tool_name, tool_info in TOOLS.items():
        score = 0
        for kw in tool_info["keywords"]:
            if kw.lower() in question_lower:
                score += 1
        if score > 0:
            matched.append((tool_name, score))
    
    # 按匹配度排序
    matched.sort(key=lambda x: x[1], reverse=True)
    
    # 如果没有匹配，返回概览
    if not matched:
        return ["客户概览"]
    
    # 返回top 3最相关的工具
    return [m[0] for m in matched[:3]]


# ============================================================
# 数据提取：从分析结果中提取工具所需数据
# ============================================================

def extract_tool_data(tool_name: str, data: dict, results: dict, 
                       benchmark: dict = None, forecast: dict = None) -> str:
    """根据工具名提取相关数据，返回格式化的上下文"""
    
    if tool_name == "客户概览":
        info = {
            "总营收(万元)": round(data.get('总营收', 0), 0),
            "月度营收": [round(v, 0) for v in data.get('月度总营收', [])],
            "客户数": len(data.get('客户金额', [])),
            "总出货量": data.get('数量汇总', {}).get('全年实际', 0),
            "YoY增长率": data.get('总YoY', {}).get('增长率', 'N/A'),
        }
        return json.dumps(info, ensure_ascii=False, indent=2)
    
    elif tool_name == "客户分级":
        tiers = results.get('客户分级', [])[:15]
        return json.dumps(tiers, ensure_ascii=False, indent=2)
    
    elif tool_name == "流失预警":
        alerts = results.get('流失预警', [])[:10]
        return json.dumps(alerts, ensure_ascii=False, indent=2)
    
    elif tool_name == "增长机会":
        opps = results.get('增长机会', [])
        return json.dumps(opps, ensure_ascii=False, indent=2)
    
    elif tool_name == "价量分解":
        pv = results.get('价量分解', [])[:10]
        return json.dumps(pv, ensure_ascii=False, indent=2)
    
    elif tool_name == "异常检测":
        anomalies = results.get('MoM异常', [])[:15]
        return json.dumps(anomalies, ensure_ascii=False, indent=2)
    
    elif tool_name == "行业对标":
        if benchmark:
            return json.dumps(benchmark, ensure_ascii=False, indent=2, default=str)
        return '{"message": "行业对标数据未加载"}'
    
    elif tool_name == "预测":
        if forecast:
            return json.dumps(forecast, ensure_ascii=False, indent=2, default=str)
        return '{"message": "预测数据未加载"}'
    
    elif tool_name == "区域分析":
        region = results.get('区域洞察', {})
        return json.dumps(region, ensure_ascii=False, indent=2)
    
    elif tool_name == "产品结构":
        products = results.get('产品结构', {})
        orders = results.get('订单模式', {})
        return json.dumps({"产品结构": products, "订单模式": orders}, ensure_ascii=False, indent=2)
    
    elif tool_name == "目标达成":
        targets = results.get('目标达成', [])
        return json.dumps(targets, ensure_ascii=False, indent=2)
    
    elif tool_name == "销售绩效":
        team = results.get('销售绩效', [])
        return json.dumps(team, ensure_ascii=False, indent=2)
    
    elif tool_name == "CEO建议":
        findings = results.get('核心发现', {})
        alerts = results.get('流失预警', [])[:3]
        opps = results.get('增长机会', [])[:3]
        return json.dumps({
            "核心发现": findings,
            "Top流失风险": alerts,
            "Top增长机会": opps,
        }, ensure_ascii=False, indent=2)
    
    return "{}"


# ============================================================
# Agent 核心：组装 prompt → 调用 LLM → 返回回答
# ============================================================

def build_agent_prompt(question: str, tool_names: list, tool_data: dict) -> str:
    """构建发送给LLM的完整prompt"""
    
    context_parts = []
    for name in tool_names:
        context_parts.append(f"=== {name} ===\n{tool_data.get(name, '{}')}")
    
    context = "\n\n".join(context_parts)
    
    return f"""你是禾苗通讯（Sprocomm, 01401.HK）的AI销售分析Agent。
你拥有公司2025年度全部销售数据的分析能力。

用户问题：{question}

我已经调用了以下分析工具获取相关数据：
工具：{', '.join(tool_names)}

{context}

请基于以上数据回答用户的问题。要求：
1. 直接回答问题，先给结论
2. 用具体数字支撑
3. 如果发现风险或机会，主动提醒
4. 语气简洁专业，像一个资深销售分析总监在汇报
5. 如果数据不足以回答，诚实说明
6. 回答用中文，金额单位为万元

回答："""


def call_llm(prompt: str, provider: str = "deepseek", api_key: str = None) -> str:
    """调用LLM生成回答"""
    
    if provider == "deepseek":
        key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if not key:
            return _fallback_response(prompt)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ AI调用失败：{str(e)}\n\n{_fallback_response(prompt)}"
    
    elif provider == "claude":
        key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not key:
            return _fallback_response(prompt)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            return f"⚠️ AI调用失败：{str(e)}\n\n{_fallback_response(prompt)}"
    
    return _fallback_response(prompt)


def _fallback_response(prompt: str) -> str:
    """无API时的本地回退：直接返回数据摘要"""
    # 从prompt中提取数据部分
    if "=== 客户概览 ===" in prompt:
        return "📊 数据已加载。请配置DeepSeek或Claude API密钥以启用AI对话分析。\n\n当前可在左侧各Tab中查看完整分析结果。"
    return "请配置AI API密钥（DeepSeek或Claude）以启用对话式分析。"


# ============================================================
# 主入口：一个函数搞定所有
# ============================================================

def ask_agent(question: str, data: dict, results: dict,
              benchmark: dict = None, forecast: dict = None,
              provider: str = "deepseek", api_key: str = None) -> dict:
    """
    Agent主入口
    
    Args:
        question: 用户的自然语言问题
        data: 原始加载数据
        results: 12维分析结果
        benchmark: 行业对标数据（可选）
        forecast: 预测数据（可选）
        provider: LLM提供商 (deepseek/claude)
        api_key: API密钥
    
    Returns:
        {
            "answer": "AI回答文本",
            "tools_used": ["工具1", "工具2"],
            "thinking": "意图识别过程"
        }
    """
    # Step 1: 意图识别
    tools = identify_tools(question)
    
    # Step 2: 提取数据
    tool_data = {}
    for t in tools:
        tool_data[t] = extract_tool_data(t, data, results, benchmark, forecast)
    
    # Step 3: 构建prompt
    prompt = build_agent_prompt(question, tools, tool_data)
    
    # Step 4: 调用LLM
    answer = call_llm(prompt, provider, api_key)
    
    return {
        "answer": answer,
        "tools_used": tools,
        "thinking": f"识别意图 → 调用工具：{', '.join(tools)} → 生成回答",
    }


# ============================================================
# 预设问题（引导用户使用）
# ============================================================

SUGGESTED_QUESTIONS = [
    "今年总营收多少？和去年比怎么样？",
    "哪些客户有流失风险？应该怎么应对？",
    "最值得投入的3个增长方向是什么？",
    "哪些客户在以量换价？利润质量怎么样？",
    "Q1 2026的营收预测是多少？",
    "我们对印度市场的依赖度是否过高？",
    "销售团队谁表现最好？谁需要关注？",
    "本月CEO应该关注什么？",
    "有哪些异常波动需要注意？",
    "和华勤闻泰比，我们的差距在哪？",
]
