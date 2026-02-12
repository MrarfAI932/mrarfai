#!/usr/bin/env python3
"""
MRARFAI LLM-as-Judge v5.0
============================
Phase 1 核心升级：用 LLM 自动评估 Agent 输出质量

三个评估维度：
  ① Correctness — 数据引用准确吗？计算对吗？
  ② Relevance   — 回答切题吗？重点突出吗？
  ③ Hallucination — 有编造数据吗？超出原始数据范围吗？

设计原则：
  - 每个维度独立评分，互不干扰
  - 评分 0.0-1.0 + 一句话理由
  - 支持本地运行（Anthropic API）和 Langfuse 集成
  - 异步评估，不阻塞主流程
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("mrarfai.judge")


# ============================================================
# 评分维度和 Prompt 模板
# ============================================================

JUDGE_DIMENSIONS = {
    "correctness": {
        "name": "数据准确性",
        "system": "你是一个严格的数据分析质量评审专家。你只评估数据准确性，不评价文风或格式。",
        "template": """评估以下销售分析报告的**数据准确性**。

## 评分标准（严格执行）
- 1.0: 所有数据引用准确，计算正确，结论有数据支撑
- 0.8: 大部分准确，存在轻微误差但不影响核心结论
- 0.5: 有明显数据错误或计算失误，但框架基本正确
- 0.3: 多处数据错误，结论缺乏支撑
- 0.0: 数据严重错误或完全编造

## 重点检查
- 百分比和增长率计算是否正确
- 客户名称、金额等引用是否与原始数据一致
- 排名和对比是否准确

## 待评估内容
【用户问题】{question}
【原始数据（截取）】{context}
【Agent回答】{output}

请严格按照以下JSON格式返回（不要返回其他内容）：
{{"score": 0.0, "reasoning": "一句话说明扣分原因"}}""",
    },
    
    "relevance": {
        "name": "问题相关性",
        "system": "你是用户体验评审专家。你只评估回答是否直接回应了问题，不评价数据是否准确。",
        "template": """评估回答是否**直接回应**了用户的问题。

## 评分标准（严格执行）
- 1.0: 完全回应问题，重点突出，无冗余
- 0.8: 基本回应，有少量冗余信息但不影响阅读
- 0.5: 部分回应，遗漏了问题中的关键点，或有大量无关内容
- 0.3: 只触及问题边缘，核心问题未回答
- 0.0: 完全跑题

## 重点检查
- 用户问的是什么？回答覆盖了吗？
- 如果用户问"哪些客户流失"，回答里有具体客户列表吗？
- 回答的篇幅是否合理？太短缺信息？太长跑题？

【用户问题】{question}
【Agent回答】{output}

返回JSON：{{"score": 0.0, "reasoning": "一句话说明"}}""",
    },
    
    "hallucination": {
        "name": "幻觉检测",
        "system": "你是事实核查专家。你只检查回答中是否有超出原始数据的编造内容。",
        "template": """检查回答中是否有**超出原始数据的编造内容**。

## 评分标准（严格执行）
- 1.0: 所有声明都可在提供的数据中找到明确依据
- 0.8: 包含合理的计算推导（如从数据中算出比率），但推导步骤清晰
- 0.5: 有一些合理推断但超出数据明确记录的范围
- 0.3: 包含数据中没有的具体数字或事实
- 0.0: 大量编造数据或引用了不存在的数据

## 重点检查
- 回答中提到的具体数字，在原始数据中找得到吗？
- 有没有编造客户名称、金额、百分比？
- 趋势判断是否有数据支撑？

【原始数据（截取）】{context}
【Agent回答】{output}

返回JSON：{{"score": 0.0, "reasoning": "一句话说明"}}""",
    },
}


# ============================================================
# LLM 调用封装（支持多 provider）
# ============================================================

def _call_judge_llm(system: str, prompt: str, 
                    provider: str = "claude", api_key: str = "",
                    model: str = None) -> Dict:
    """调用 LLM 进行评分，返回 {score, reasoning}"""
    
    if not api_key:
        return {"score": -1, "reasoning": "无 API Key，跳过评估"}
    
    try:
        if provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            model = model or "claude-sonnet-4-20250514"
            
            resp = client.messages.create(
                model=model,
                max_tokens=150,
                temperature=0,  # 评分需要确定性
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            
        elif provider == "deepseek":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            model = model or "deepseek-chat"
            
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
        
        else:
            return {"score": -1, "reasoning": f"不支持的 provider: {provider}"}
        
        # 解析 JSON 响应
        return _parse_judge_response(raw)
        
    except Exception as e:
        logger.error(f"Judge LLM 调用失败: {e}")
        return {"score": -1, "reasoning": f"调用失败: {str(e)[:100]}"}


def _parse_judge_response(raw: str) -> Dict:
    """稳健地解析 LLM 返回的 JSON 评分"""
    # 尝试直接解析
    try:
        result = json.loads(raw)
        if "score" in result:
            score = float(result["score"])
            score = max(0.0, min(1.0, score))  # 钳位到 [0, 1]
            return {
                "score": score,
                "reasoning": str(result.get("reasoning", "")).strip(),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # 尝试从文本中提取 JSON
    import re
    json_match = re.search(r'\{[^}]+\}', raw)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if "score" in result:
                score = float(result["score"])
                score = max(0.0, min(1.0, score))
                return {
                    "score": score,
                    "reasoning": str(result.get("reasoning", "")).strip(),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    # 最后尝试提取数字
    numbers = re.findall(r'(?:score|分数)[^\d]*(\d+\.?\d*)', raw.lower())
    if numbers:
        score = float(numbers[0])
        if score > 1:
            score = score / 10  # 如果 LLM 返回了 0-10 的分数
        return {"score": min(score, 1.0), "reasoning": raw[:200]}
    
    return {"score": -1, "reasoning": f"无法解析响应: {raw[:200]}"}


# ============================================================
# 主评估函数
# ============================================================

def judge_output(
    question: str,
    output: str,
    context: str = "",
    dimensions: List[str] = None,
    provider: str = "claude",
    api_key: str = "",
    parallel: bool = True,
    context_max_chars: int = 3000,
) -> Dict:
    """
    对 Agent 输出进行多维度 LLM-as-Judge 评估
    
    参数:
        question: 用户原始问题
        output: Agent 生成的回答
        context: 原始数据上下文（用于幻觉检测）
        dimensions: 要评估的维度列表，默认全部
        provider: LLM provider ("claude" / "deepseek")
        api_key: API Key
        parallel: 是否并行评估
        context_max_chars: 上下文最大字符数
    
    返回:
        {
            "correctness": {"score": 0.85, "reasoning": "..."},
            "relevance": {"score": 0.9, "reasoning": "..."},
            "hallucination": {"score": 0.8, "reasoning": "..."},
            "overall": 0.85,
            "elapsed_ms": 1234,
            "provider": "claude",
        }
    """
    t0 = time.time()
    dimensions = dimensions or list(JUDGE_DIMENSIONS.keys())
    
    # 截断上下文
    ctx = context[:context_max_chars] if context else "(无原始数据提供)"
    
    # 构建评估任务
    def _eval_dimension(dim: str) -> tuple:
        config = JUDGE_DIMENSIONS.get(dim)
        if not config:
            return dim, {"score": -1, "reasoning": f"未知维度: {dim}"}
        
        prompt = config["template"].format(
            question=question[:1000],
            output=output[:2000],
            context=ctx,
        )
        
        result = _call_judge_llm(
            system=config["system"],
            prompt=prompt,
            provider=provider,
            api_key=api_key,
        )
        return dim, result
    
    # 并行或串行评估
    scores = {}
    if parallel and len(dimensions) > 1:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_eval_dimension, d): d for d in dimensions}
            for future in futures:
                dim, result = future.result()
                scores[dim] = result
    else:
        for dim in dimensions:
            _, result = _eval_dimension(dim)
            scores[dim] = result
    
    # 计算综合分数（忽略失败的维度）
    valid_scores = [s["score"] for s in scores.values() if s["score"] >= 0]
    overall = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else -1
    
    elapsed = (time.time() - t0) * 1000
    
    return {
        **scores,
        "overall": overall,
        "elapsed_ms": round(elapsed, 1),
        "provider": provider,
        "dimensions_evaluated": len(valid_scores),
    }


def judge_quick(question: str, output: str, context: str = "",
                provider: str = "claude", api_key: str = "") -> float:
    """快速评分 — 只返回综合分数（用于流水线中快速判断）"""
    result = judge_output(question, output, context, provider=provider, api_key=api_key)
    return result["overall"]


# ============================================================
# 批量评估（用于 Golden Dataset 回归测试）
# ============================================================

def judge_batch(
    test_cases: List[Dict],
    agent_fn: callable,
    provider: str = "claude",
    api_key: str = "",
    version_tag: str = "dev",
) -> Dict:
    """
    批量评估一组测试用例
    
    参数:
        test_cases: [{"question": "...", "data": {...}, "expected": "..."}]
        agent_fn: 调用 Agent 的函数，签名 fn(question, data) -> str
        provider: 评分用的 LLM provider
        api_key: API Key
        version_tag: 版本标签（用于对比）
    
    返回:
        {
            "version": "v4.3",
            "total": 20,
            "avg_scores": {"correctness": 0.82, "relevance": 0.88, ...},
            "results": [...],
            "summary": "..."
        }
    """
    results = []
    dim_totals = {}
    
    for i, case in enumerate(test_cases):
        logger.info(f"评估 [{i+1}/{len(test_cases)}] {case['question'][:50]}...")
        
        # 调用 Agent
        try:
            output = agent_fn(case["question"], case.get("data", {}))
        except Exception as e:
            results.append({
                "question": case["question"],
                "error": str(e),
                "scores": {},
            })
            continue
        
        # 评估
        scores = judge_output(
            question=case["question"],
            output=output if isinstance(output, str) else str(output),
            context=str(case.get("data", ""))[:3000],
            provider=provider,
            api_key=api_key,
        )
        
        # 汇总
        for dim in JUDGE_DIMENSIONS:
            if dim in scores and scores[dim]["score"] >= 0:
                if dim not in dim_totals:
                    dim_totals[dim] = []
                dim_totals[dim].append(scores[dim]["score"])
        
        results.append({
            "question": case["question"],
            "output_preview": output[:200] if isinstance(output, str) else str(output)[:200],
            "scores": scores,
        })
    
    # 平均分
    avg_scores = {
        dim: round(sum(vals) / len(vals), 3)
        for dim, vals in dim_totals.items()
        if vals
    }
    
    overall_avg = round(
        sum(avg_scores.values()) / len(avg_scores), 3
    ) if avg_scores else -1
    
    return {
        "version": version_tag,
        "total": len(test_cases),
        "evaluated": len([r for r in results if "error" not in r]),
        "avg_scores": avg_scores,
        "overall_avg": overall_avg,
        "results": results,
    }


# ============================================================
# Langfuse 集成 — 自动将评分写入 Langfuse
# ============================================================

def judge_and_trace(
    question: str,
    output: str,
    context: str = "",
    trace_ctx=None,
    provider: str = "claude",
    api_key: str = "",
) -> Dict:
    """
    评估 + 自动写入 Langfuse Trace
    
    这是推荐在 ask_multi_agent 中使用的函数。
    """
    scores = judge_output(question, output, context, provider=provider, api_key=api_key)
    
    # 写入 Langfuse
    if trace_ctx and hasattr(trace_ctx, 'score'):
        for dim in JUDGE_DIMENSIONS:
            if dim in scores and scores[dim]["score"] >= 0:
                trace_ctx.score(
                    name=f"judge-{dim}",
                    value=scores[dim]["score"],
                    comment=scores[dim].get("reasoning", ""),
                )
        
        if scores["overall"] >= 0:
            trace_ctx.score(
                name="judge-overall",
                value=scores["overall"],
                comment=f"综合评分: {scores['overall']:.2f}",
            )
    
    return scores


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("MRARFAI LLM-as-Judge 测试")
    print("=" * 60)
    
    # 测试用例
    test_question = "华东区今年整体表现如何？"
    test_context = """
    华东区2024年数据:
    - 总营收: 41.71亿元（同比+54.1%）
    - Top客户: A公司 12.3亿, B公司 8.5亿, C公司 6.2亿
    - Q4环比增长: 15.2%
    """
    
    # 好回答
    good_output = """
    华东区2024年整体表现强劲：
    1. 总营收达41.71亿元，同比增长54.1%
    2. Top3客户占比64.7%（A公司12.3亿+B公司8.5亿+C公司6.2亿）
    3. Q4季度环比增长15.2%，增长动能持续
    建议：关注客户集中度风险，Top3占比偏高。
    """
    
    # 坏回答（含幻觉）
    bad_output = """
    华东区表现不错。
    总营收大约50亿，增长了60%左右。
    D公司是最大客户，贡献了20亿。
    全国排名第二，仅次于华南区。
    """
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("⚠️  未设置 ANTHROPIC_API_KEY，使用模拟评估")
        print("\n📊 好回答（模拟评分）:")
        print("  correctness: 0.95 — 数据引用完全匹配")
        print("  relevance:   0.90 — 直接回应问题")
        print("  hallucination: 0.95 — 所有数字都能在数据中找到")
        print("\n📊 坏回答（模拟评分）:")
        print("  correctness: 0.15 — 50亿/60%/D公司均错误")
        print("  relevance:   0.40 — 回应了问题但缺乏细节")
        print("  hallucination: 0.05 — 编造了D公司、全国排名等信息")
    else:
        print("\n📊 评估「好回答」...")
        good_scores = judge_output(
            test_question, good_output, test_context,
            provider="claude", api_key=api_key,
        )
        for dim, data in good_scores.items():
            if isinstance(data, dict):
                print(f"  {dim}: {data['score']:.2f} — {data['reasoning']}")
            elif dim == "overall":
                print(f"  综合: {data:.2f}")
        
        print(f"\n📊 评估「坏回答」...")
        bad_scores = judge_output(
            test_question, bad_output, test_context,
            provider="claude", api_key=api_key,
        )
        for dim, data in bad_scores.items():
            if isinstance(data, dict):
                print(f"  {dim}: {data['score']:.2f} — {data['reasoning']}")
            elif dim == "overall":
                print(f"  综合: {data:.2f}")
        
        print(f"\n⏱ 好回答耗时: {good_scores['elapsed_ms']:.0f}ms")
        print(f"⏱ 坏回答耗时: {bad_scores['elapsed_ms']:.0f}ms")
