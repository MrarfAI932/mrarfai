#!/usr/bin/env python3
"""
MRARFAI Multi-Agent System v2.1 (CrewAI + Memory + HITL)
==========================================================
基于 CrewAI 框架的专家团队协作系统

v2.1 新增：
  ✅ Agent 记忆 — 记住之前分析，多轮深入对话
  ✅ Human-in-the-loop — 风控发现高风险时暂停等人确认

4 Agents: 分析师 + 风控 + 策略师 + 报告员
"""

import json
import os
from datetime import datetime
from typing import Optional
from collections import deque

# CrewAI 导入
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from crewai import LLM
    HAS_CREWAI = True
except Exception:
    HAS_CREWAI = False


# ============================================================
# Agent 记忆系统
# ============================================================

class AgentMemory:
    """
    多轮对话记忆
    - 短期记忆: 最近N轮QA（session级别）
    - 实体记忆: 提到过的客户/数据点
    - 分析摘要: 每轮分析的核心结论
    """
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversation_history = deque(maxlen=max_turns)
        self.entity_mentions = {}  # {客户名: [提到的上下文]}
        self.analysis_summaries = deque(maxlen=max_turns)
        self.risk_confirmations = {}  # {客户名: True/False} HITL确认记录
    
    def add_turn(self, question: str, answer: str, agents_used: list = None, 
                 expert_outputs: dict = None):
        """记录一轮对话"""
        turn = {
            'time': datetime.now().isoformat(),
            'question': question,
            'answer_preview': answer[:200],
            'agents': agents_used or [],
        }
        self.conversation_history.append(turn)
        
        # 提取实体
        for name_candidate in self._extract_entities(question + " " + answer):
            if name_candidate not in self.entity_mentions:
                self.entity_mentions[name_candidate] = []
            self.entity_mentions[name_candidate].append(question[:50])
        
        # 存分析摘要
        if expert_outputs:
            for expert, output in expert_outputs.items():
                self.analysis_summaries.append({
                    'expert': expert,
                    'summary': output[:150],
                    'question': question[:50],
                })
    
    def add_risk_confirmation(self, customer: str, confirmed: bool):
        """记录HITL风险确认"""
        self.risk_confirmations[customer] = {
            'confirmed': confirmed,
            'time': datetime.now().isoformat(),
        }
    
    def get_context_prompt(self) -> str:
        """生成记忆上下文，注入到Agent prompt中"""
        if not self.conversation_history:
            return ""
        
        lines = ["[之前的对话记忆]"]
        
        # 最近对话
        for turn in list(self.conversation_history)[-5:]:
            lines.append(f"Q: {turn['question'][:80]}")
            lines.append(f"A: {turn['answer_preview'][:100]}...")
        
        # HITL确认
        if self.risk_confirmations:
            lines.append("\n[风险确认记录]")
            for cust, info in self.risk_confirmations.items():
                status = "已确认关注" if info['confirmed'] else "已标记为低优先"
                lines.append(f"- {cust}: {status}")
        
        # 频繁提到的实体
        if self.entity_mentions:
            top_entities = sorted(
                self.entity_mentions.items(), 
                key=lambda x: len(x[1]), reverse=True
            )[:5]
            if top_entities:
                lines.append("\n[用户关注的重点客户]")
                for name, mentions in top_entities:
                    lines.append(f"- {name} (提到{len(mentions)}次)")
        
        return "\n".join(lines)
    
    def _extract_entities(self, text: str) -> list:
        """简单的实体提取（客户名等）"""
        # 这里用简单规则；实际可以用NER
        entities = []
        # 如果文本里包含了之前见过的客户名，记录
        for name in list(self.entity_mentions.keys()):
            if name in text:
                entities.append(name)
        return entities
    
    def register_known_entities(self, customer_names: list):
        """注册已知客户名用于实体识别"""
        for name in customer_names:
            if name not in self.entity_mentions:
                self.entity_mentions[name] = []
    
    def clear(self):
        self.conversation_history.clear()
        self.entity_mentions.clear()
        self.analysis_summaries.clear()
        self.risk_confirmations.clear()


# 全局记忆实例（session级别，在chat_tab中通过st.session_state持久化）
_global_memory = AgentMemory()

def get_memory() -> AgentMemory:
    return _global_memory

def set_memory(mem: AgentMemory):
    global _global_memory
    _global_memory = mem


# ============================================================
# Human-in-the-loop 检测
# ============================================================

def detect_hitl_triggers(results: dict, health_scores: list = None) -> list:
    """
    检测需要人工确认的高风险情况
    
    返回: [{
        'customer': str,
        'risk_level': str,
        'reason': str,
        'amount': float,
        'action_required': str,
    }]
    """
    triggers = []
    
    alerts = results.get('流失预警', [])
    for a in alerts:
        if '高' in a.get('风险', ''):
            triggers.append({
                'customer': a['客户'],
                'risk_level': '🔴 高风险',
                'reason': a.get('原因', '趋势下滑'),
                'amount': a.get('年度金额', 0),
                'action_required': '需要确认是否立即安排拜访',
            })
    
    if health_scores:
        for s in health_scores:
            if s['等级'] == 'F' and s['年度金额'] > 100:
                # 大额F级客户
                triggers.append({
                    'customer': s['客户'],
                    'risk_level': '🔴 健康分F级',
                    'reason': f"健康评分仅{s['总分']}分，" + " ".join(s.get('风险标签', [])),
                    'amount': s['年度金额'],
                    'action_required': '需要确认是否启动客户挽回计划',
                })
    
    # 去重
    seen = set()
    unique = []
    for t in triggers:
        if t['customer'] not in seen:
            seen.add(t['customer'])
            unique.append(t)
    
    return unique


# ============================================================
# 自定义工具：让Agent访问销售数据
# ============================================================

# 数据存储（独立于CrewAI）
_sales_data_store = {}

def set_sales_data(data_store: dict):
    global _sales_data_store
    _sales_data_store = data_store

def query_sales_data(query: str) -> str:
    ds = _sales_data_store
    if not ds:
        return "数据未加载"
    q = query.lower()
    result = {}
    if any(k in q for k in ['总', '营收', '收入', '概览', '全部']):
        result['总营收'] = ds.get('总营收')
        result['总YoY'] = ds.get('总YoY')
        result['月度营收'] = ds.get('月度营收')
        result['核心发现'] = ds.get('核心发现')
    if any(k in q for k in ['客户', '分级', 'abc', '排名', 'top']):
        result['客户分级'] = ds.get('客户分级', [])[:15]
    if any(k in q for k in ['风险', '流失', '预警', '异常']):
        result['流失预警'] = ds.get('流失预警')
        result['异常检测'] = ds.get('异常检测', [])[:10]
    if any(k in q for k in ['增长', '机会', '潜力']):
        result['增长机会'] = ds.get('增长机会')
    if any(k in q for k in ['价', '单价', '量', '质量']):
        result['价量分解'] = ds.get('价量分解', [])[:10]
    if any(k in q for k in ['区域', '市场', '地区']):
        result['区域洞察'] = ds.get('区域洞察')
    if any(k in q for k in ['行业', '竞争', '对标', '华勤', '闻泰']):
        result['行业对标'] = ds.get('行业对标')
    if any(k in q for k in ['预测', '2026', '未来', '前景']):
        result['预测'] = ds.get('预测')
    if not result:
        result = {'总营收': ds.get('总营收'), '总YoY': ds.get('总YoY'), '核心发现': ds.get('核心发现')}
    return json.dumps(result, ensure_ascii=False, indent=1, default=str)[:5000]

# CrewAI Tool（仅在CrewAI可用时定义）
if HAS_CREWAI:
    class SalesDataTool(BaseTool):
        """让Agent查询禾苗销售数据"""
        name: str = "sales_data_query"
        description: str = (
            "查询禾苗通讯销售数据。可以查询：总营收、月度趋势、客户分级、"
            "流失预警、增长机会、价量分解、区域分析、行业对标、预测。"
        )
        def _run(self, query: str) -> str:
            return query_sales_data(query)


# ============================================================
# Agent 角色定义
# ============================================================

AGENT_PROFILES = {
    "analyst": {
        "name": "📊 数据分析师",
        "emoji": "📊",
        "role": "禾苗通讯资深数据分析师",
        "goal": "精准解读销售数据，用数字揭示业务真相，识别趋势和模式",
        "backstory": (
            "你在消费电子ODM行业有15年数据分析经验，曾服务华勤、闻泰等头部企业。"
            "你以数据驱动著称，每个结论必须有数字支撑。"
            "你擅长发现月度波动规律、客户集中度风险、同比环比异常。"
            "你的分析风格：精准、客观、量化。先给结论，再给数据。"
        ),
        "keywords": [
            "营收", "收入", "金额", "出货", "数量", "客户", "分级",
            "ABC", "排名", "top", "占比", "集中", "月度", "季度",
            "同比", "环比", "趋势", "总览", "概览", "多少",
            "产品", "结构", "区域",
        ],
    },
    "risk": {
        "name": "🛡️ 风控专家",
        "emoji": "🛡️",
        "role": "禾苗通讯风险控制专家",
        "goal": "识别客户流失风险和异常波动，量化风险金额，提供预防方案",
        "backstory": (
            "你是前安永风险咨询总监，专注TMT行业客户风险管理。"
            "你对数据异常极其敏感——断崖式下跌、连续N月衰退、大客户集中度过高，"
            "这些你一眼就能看出。你的风格：直言不讳，发现问题就说。"
            "输出格式：风险等级→影响金额→原因分析→应对建议。"
            "\n\n⚠️ 重要：如果发现高风险客户（年度金额>200万且持续下滑），"
            "请在输出开头标记 [HIGH_RISK_ALERT] 并列出客户名和金额。"
        ),
        "keywords": [
            "风险", "流失", "预警", "下降", "下滑", "丢失", "断崖",
            "异常", "暴跌", "危险", "警告", "关注", "问题",
            "波动", "偏离", "不正常",
        ],
    },
    "strategist": {
        "name": "💡 策略师",
        "emoji": "💡",
        "role": "禾苗通讯战略顾问",
        "goal": "发现增长机会，制定可执行的战略方案，优化资源配置",
        "backstory": (
            "你是前麦肯锡TMT行业合伙人，专注手机ODM/OEM赛道战略规划。"
            "你擅长竞争分析（vs华勤/闻泰/龙旗）、增长机会识别、"
            "产品组合优化、客户钱包份额提升策略。"
            "你的风格：前瞻性、实用主义、聚焦ROI。建议必须可执行。"
            "输出格式：机会/方向→潜在价值→具体行动→优先级。"
        ),
        "keywords": [
            "增长", "机会", "战略", "策略", "建议", "方向", "投入",
            "竞争", "对手", "华勤", "闻泰", "龙旗", "行业", "对标",
            "预测", "forecast", "2026", "未来", "前景",
            "CEO", "管理", "决策", "优化", "提升",
            "价格", "价量", "利润",
        ],
    },
}

REPORTER_PROFILE = {
    "role": "禾苗通讯高级报告撰写人",
    "goal": "综合多位专家分析，生成简洁有力的综合报告，适合CEO阅读",
    "backstory": (
        "你是前FT中文网资深编辑，现任禾苗通讯战略分析部负责人。"
        "你擅长将复杂的数据分析和多方观点提炼为管理层可直接行动的建议。"
        "规则：1.不简单拼凑 2.先核心结论 3.分模块展开 4.最后给行动项 5.控制500字"
    ),
}


# ============================================================
# 路由
# ============================================================

def route_to_agents(question: str) -> list:
    q = question.lower()
    agents_needed = set()
    for agent_id, profile in AGENT_PROFILES.items():
        score = sum(1 for kw in profile['keywords'] if kw in q)
        if score > 0:
            agents_needed.add(agent_id)
    if any(k in q for k in ['CEO', '总结', '全面', '概览', '怎么样', '建议', '报告']):
        agents_needed = {"analyst", "risk", "strategist"}
    if not agents_needed:
        agents_needed = {"analyst", "risk", "strategist"}
    return list(agents_needed)


# ============================================================
# 数据上下文
# ============================================================

def build_data_store(data, results, benchmark=None, forecast=None):
    store = {
        '总营收': data.get('总营收', 0),
        '总YoY': data.get('总YoY', {}),
        '月度营收': data.get('月度总营收', []),
        '核心发现': results.get('核心发现', []),
        '客户数': sum(1 for c in data.get('客户金额', []) if c.get('年度金额', 0) > 0),
        '客户分级': results.get('客户分级', [])[:15],
        '流失预警': results.get('流失预警', []),
        '异常检测': results.get('MoM异常', [])[:10],
        '增长机会': results.get('增长机会', []),
        '价量分解': results.get('价量分解', [])[:10],
        '区域洞察': results.get('区域洞察', {}),
    }
    if benchmark:
        store['行业对标'] = {
            '市场定位': benchmark.get('市场定位', {}),
            '竞争对标': benchmark.get('竞争对标', {}),
            '结构性风险': benchmark.get('结构性风险', []),
            '战略机会': benchmark.get('战略机会', []),
        }
    if forecast:
        store['预测'] = {
            '总营收预测': forecast.get('总营收预测', {}),
            '客户预测': forecast.get('客户预测', [])[:5],
            '情景分析': forecast.get('风险场景', {}),
        }
    return store


# ============================================================
# LLM
# ============================================================

def _get_llm(provider: str, api_key: str):
    if provider == "deepseek":
        os.environ["OPENAI_API_KEY"] = api_key
        return LLM(
            model="deepseek/deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
        )
    elif provider == "claude":
        os.environ["ANTHROPIC_API_KEY"] = api_key
        return LLM(
            model="anthropic/claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=0.3,
        )
    return LLM(model="openai/gpt-4o", api_key=api_key, temperature=0.3)


# ============================================================
# 主入口 (CrewAI版)
# ============================================================

def ask_multi_agent(
    question: str,
    data: dict,
    results: dict,
    benchmark: dict = None,
    forecast: dict = None,
    provider: str = "deepseek",
    api_key: str = "",
    memory: AgentMemory = None,
) -> dict:
    """
    CrewAI Multi-Agent 问答入口
    
    返回: {
        "answer": str,
        "agents_used": list,
        "thinking": list,
        "expert_outputs": dict,
        "hitl_triggers": list,  # 需要人工确认的高风险
    }
    """
    
    if not HAS_CREWAI:
        return _ask_fallback(question, data, results, benchmark, forecast, provider, api_key, memory)
    
    # CrewAI框架开销太大，默认用简化版（同样4专家，直接调LLM，快10倍）
    # 如需启用CrewAI原生模式，将下面这行注释掉
    return ask_multi_agent_simple(question, data, results, benchmark, forecast, provider, api_key, memory)
    
    if not api_key:
        return {"answer": "⚠️ 请先配置API Key", "agents_used": [], "thinking": [], 
                "expert_outputs": {}, "hitl_triggers": []}
    
    # 记忆
    mem = memory or get_memory()
    mem_context = mem.get_context_prompt()
    thinking = [f"📩 收到问题：{question}"]
    if mem_context:
        thinking.append(f"🧠 加载了 {len(mem.conversation_history)} 轮对话记忆")
    
    # 数据
    data_store = build_data_store(data, results, benchmark, forecast)
    set_sales_data(data_store)
    
    # 注册已知客户名到记忆
    for c in data.get('客户金额', [])[:50]:
        name = c.get('客户', '')
        if name and len(name) >= 2:
            mem.register_known_entities([name])
    
    # 路由
    agents_needed = route_to_agents(question)
    thinking.append(f"🎯 调度 {len(agents_needed)} 个专家")
    
    # LLM
    try:
        llm = _get_llm(provider, api_key)
    except Exception as e:
        return {"answer": f"⚠️ LLM配置失败: {e}", "agents_used": [], "thinking": [],
                "expert_outputs": {}, "hitl_triggers": []}
    
    sales_tool = SalesDataTool()
    
    # 数据上下文
    ctx_str = json.dumps(
        {k: v for k, v in data_store.items() if v},
        ensure_ascii=False, indent=1, default=str
    )[:4000]
    
    # 记忆注入到prompt
    memory_section = ""
    if mem_context:
        memory_section = f"\n\n[对话记忆]\n{mem_context}\n"
    
    # 创建Agents
    crew_agents = {}
    
    if "analyst" in agents_needed:
        p = AGENT_PROFILES["analyst"]
        crew_agents["analyst"] = Agent(
            role=p["role"], goal=p["goal"], backstory=p["backstory"],
            tools=[sales_tool], llm=llm, verbose=False, memory=True, max_iter=3,
        )
        thinking.append(f"   📊 分析师 就绪")
    
    if "risk" in agents_needed:
        p = AGENT_PROFILES["risk"]
        crew_agents["risk"] = Agent(
            role=p["role"], goal=p["goal"], backstory=p["backstory"],
            tools=[sales_tool], llm=llm, verbose=False, memory=True, max_iter=3,
        )
        thinking.append(f"   🛡️ 风控 就绪")
    
    if "strategist" in agents_needed:
        p = AGENT_PROFILES["strategist"]
        crew_agents["strategist"] = Agent(
            role=p["role"], goal=p["goal"], backstory=p["backstory"],
            tools=[sales_tool], llm=llm, verbose=False, memory=True, max_iter=3,
        )
        thinking.append(f"   💡 策略师 就绪")
    
    reporter = Agent(
        role=REPORTER_PROFILE["role"], goal=REPORTER_PROFILE["goal"],
        backstory=REPORTER_PROFILE["backstory"],
        llm=llm, verbose=False, memory=True, max_iter=3,
    )
    thinking.append(f"   🖊️ 报告员 就绪")
    
    # 创建Tasks
    tasks = []
    task_map = {}
    
    if "analyst" in crew_agents:
        t = Task(
            description=f"分析禾苗通讯销售数据。用户问题：{question}\n数据：\n{ctx_str}{memory_section}\n要求：用数字说话，200字内。",
            expected_output="数据分析报告",
            agent=crew_agents["analyst"],
        )
        tasks.append(t); task_map["analyst"] = t
    
    if "risk" in crew_agents:
        t = Task(
            description=f"从风控角度分析。用户问题：{question}\n数据：\n{ctx_str}{memory_section}\n要求：关注流失和异常，如有高风险标记[HIGH_RISK_ALERT]，200字内。",
            expected_output="风险评估报告",
            agent=crew_agents["risk"],
            context=[task_map["analyst"]] if "analyst" in task_map else [],
        )
        tasks.append(t); task_map["risk"] = t
    
    if "strategist" in crew_agents:
        t = Task(
            description=f"从战略角度分析。用户问题：{question}\n数据：\n{ctx_str}{memory_section}\n要求：可执行建议，聚焦ROI，200字内。",
            expected_output="战略建议报告",
            agent=crew_agents["strategist"],
            context=list(task_map.values()),
        )
        tasks.append(t); task_map["strategist"] = t
    
    report_task = Task(
        description=f"综合专家分析生成CEO报告。原始问题：{question}{memory_section}\n格式：核心结论→分析详情(📊🛡️💡)→下一步行动(最多3条)。500字内，中文。",
        expected_output="综合报告",
        agent=reporter,
        context=list(task_map.values()),
    )
    tasks.append(report_task)
    
    # 执行
    thinking.append("🚀 Crew启动...")
    
    try:
        crew = Crew(
            agents=list(crew_agents.values()) + [reporter],
            tasks=tasks, process=Process.sequential,
            verbose=False, memory=True,
        )
        result = crew.kickoff()
        thinking.append("✅ 完成")
        
        # 收集输出
        expert_outputs = {}
        agents_used = []
        for aid, task in task_map.items():
            profile = AGENT_PROFILES[aid]
            agents_used.append(profile["name"])
            if hasattr(task, 'output') and task.output:
                expert_outputs[profile["name"]] = str(task.output)
            else:
                expert_outputs[profile["name"]] = "(输出已传递给报告员)"
        agents_used.append("🖊️ 报告员")
        
        final_answer = str(result) if result else "分析完成"
        
        # HITL: 检测风控输出中的高风险标记
        hitl_triggers = []
        risk_output = expert_outputs.get("🛡️ 风控专家", "")
        if "[HIGH_RISK_ALERT]" in risk_output or "高风险" in risk_output:
            hitl_triggers = detect_hitl_triggers(results)
            if hitl_triggers:
                thinking.append(f"⚠️ HITL: 检测到 {len(hitl_triggers)} 个高风险需要人工确认")
        
        # 记录到记忆
        mem.add_turn(question, final_answer, agents_used, expert_outputs)
        
        return {
            "answer": final_answer,
            "agents_used": agents_used,
            "thinking": thinking,
            "expert_outputs": expert_outputs,
            "hitl_triggers": hitl_triggers,
        }
    
    except Exception as e:
        thinking.append(f"❌ 异常: {str(e)}")
        return _ask_fallback(question, data, results, benchmark, forecast, provider, api_key, memory)


# ============================================================
# 降级方案
# ============================================================

def _call_llm_raw(system_prompt, user_prompt, provider, api_key):
    if not api_key:
        return "[需要API Key]"
    try:
        if provider == "deepseek":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.3, max_tokens=800,
            )
            return resp.choices[0].message.content
        elif provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=800,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text
    except Exception as e:
        return f"[调用失败: {e}]"


def ask_multi_agent_simple(
    question: str, data: dict, results: dict,
    benchmark=None, forecast=None,
    provider="deepseek", api_key="",
    memory: AgentMemory = None,
) -> dict:
    """简化版（不依赖CrewAI）"""
    
    mem = memory or get_memory()
    mem_context = mem.get_context_prompt()
    
    data_store = build_data_store(data, results, benchmark, forecast)
    ctx_json = json.dumps(
        {k: v for k, v in data_store.items() if v},
        ensure_ascii=False, indent=1, default=str
    )[:5000]
    
    agents_needed = route_to_agents(question)
    thinking = [f"📩 问题：{question}", f"🎯 调度 {len(agents_needed)} 个专家 (简化模式)"]
    if mem_context:
        thinking.append(f"🧠 加载 {len(mem.conversation_history)} 轮记忆")
    
    memory_section = f"\n\n[对话记忆]\n{mem_context}" if mem_context else ""
    
    expert_outputs = {}
    agents_used = []
    
    for aid in agents_needed:
        profile = AGENT_PROFILES[aid]
        thinking.append(f"{profile['emoji']} {profile['name']} 分析中...")
        
        system = f"你是{profile['role']}。{profile['backstory']}"
        prompt = f"用户问题：{question}\n\n禾苗销售数据：\n{ctx_json}{memory_section}\n\n200字内回答。"
        
        output = _call_llm_raw(system, prompt, provider, api_key)
        expert_outputs[profile["name"]] = output
        agents_used.append(profile["name"])
        thinking.append(f"{profile['emoji']} 完成 ({len(output)}字)")
    
    # Reporter
    thinking.append("🖊️ 报告员综合中...")
    all_opinions = "\n---\n".join(f"{n}：\n{t}" for n, t in expert_outputs.items())
    reporter_sys = f"你是{REPORTER_PROFILE['role']}。{REPORTER_PROFILE['backstory']}"
    report = _call_llm_raw(
        reporter_sys,
        f"问题：{question}\n\n专家分析：\n{all_opinions}{memory_section}\n\n综合报告，500字内。",
        provider, api_key,
    )
    agents_used.append("🖊️ 报告员")
    thinking.append(f"🖊️ 完成")
    
    # HITL
    hitl_triggers = []
    risk_out = expert_outputs.get("🛡️ 风控专家", "")
    if "高风险" in risk_out or "[HIGH_RISK_ALERT]" in risk_out:
        hitl_triggers = detect_hitl_triggers(results)
        if hitl_triggers:
            thinking.append(f"⚠️ HITL: {len(hitl_triggers)} 个需确认")
    
    mem.add_turn(question, report, agents_used, expert_outputs)
    
    return {
        "answer": report,
        "agents_used": agents_used,
        "thinking": thinking,
        "expert_outputs": expert_outputs,
        "hitl_triggers": hitl_triggers,
    }


def _ask_fallback(question, data, results, benchmark, forecast, provider, api_key, memory=None):
    return ask_multi_agent_simple(question, data, results, benchmark, forecast, provider, api_key, memory)
