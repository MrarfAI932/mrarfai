#!/usr/bin/env python3
"""
MRARFAI V9.0 — Recursive Language Model Engine
=================================================
基于 MIT CSAIL "Recursive Language Models" (arXiv:2512.24601)

核心思路:
  长数据不再塞进 LLM context → 加载到 Python REPL 环境
  Agent 写代码按需切片、分组、递归调用 sub-LM 分析
  
解决的核心痛点:
  当前 multi_agent.py line 610: [:5000] 强制截断数据
  RLM: 数据作为 REPL 变量，Agent 通过代码精准访问任意片段

论文关键发现:
  - 处理超出 context window 100倍的输入
  - RLM-Qwen3-8B 比基础模型高 28.3%
  - 成本相当甚至更低（sub-calls 用小模型）
  - Prime Intellect: "2026年最重要的推理范式"

使用方式:
  from rlm_engine import RLMSalesAnalyzer
  
  analyzer = RLMSalesAnalyzer(
      root_model="gpt-4o",
      sub_model="deepseek-chat",  # 递归调用用便宜模型
  )
  result = analyzer.analyze(shipment_data, "分析各品牌月度趋势")

兼容:
  v4.0 multi_agent.py 接口
  v3.3 protocol_layer.py MCP/A2A
  v8.0 adaptive_gate.py 模型路由
  v9.0 search_engine.py EnCompass 搜索
"""

import json
import os
import time
import hashlib
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger("mrarfai.rlm")


# ============================================================
# 配置
# ============================================================

class RLMConfig:
    """RLM 引擎配置"""
    
    # 模型配置
    ROOT_MODEL = os.getenv("RLM_ROOT_MODEL", "gpt-4o")
    SUB_MODEL = os.getenv("RLM_SUB_MODEL", "deepseek-chat")
    
    # REPL 沙箱配置
    MAX_REPL_STEPS = 15          # 最大 REPL 交互轮数
    MAX_CODE_LENGTH = 4000       # 单次代码最大长度
    REPL_TIMEOUT_SECONDS = 30    # 单次代码执行超时
    
    # 递归配置
    MAX_RECURSION_DEPTH = 3      # 最大递归深度
    MAX_SUB_CALLS = 20           # 最大 sub-LM 调用次数
    SUB_CALL_MAX_TOKENS = 1000   # sub-LM 每次最大输出 token
    
    # 数据分片配置
    CHUNK_SIZE_TOKENS = 2000     # 每个数据片段大小
    OVERLAP_TOKENS = 200         # 片段重叠
    
    # 成本控制
    MAX_TOTAL_COST_USD = 2.0     # 单次分析最大成本
    
    # 安全沙箱 — 禁止的模块和函数
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'importlib',
        'socket', 'http', 'urllib', 'requests', 'pathlib',
    }
    BLOCKED_BUILTINS = {
        'exec', 'eval', 'compile', '__import__', 'open',
        'input', 'breakpoint', 'exit', 'quit',
    }


# ============================================================
# REPL 沙箱环境
# ============================================================

class SandboxError(Exception):
    """沙箱安全错误"""
    pass


class SecureREPL:
    """
    安全的 Python REPL 沙箱
    
    出货数据加载为变量，Agent 写代码分析
    禁止文件/网络/系统操作
    """
    
    def __init__(self, config: RLMConfig = None):
        self.config = config or RLMConfig()
        self.namespace = {}
        self.execution_log = []
        self.step_count = 0
        
    def load_data(self, data: Any, var_name: str = "data"):
        """将数据加载到 REPL 命名空间"""
        self.namespace[var_name] = data
        self.namespace['json'] = json
        self.namespace['len'] = len
        self.namespace['type'] = type
        self.namespace['str'] = str
        self.namespace['int'] = int
        self.namespace['float'] = float
        self.namespace['list'] = list
        self.namespace['dict'] = dict
        self.namespace['set'] = set
        self.namespace['tuple'] = tuple
        self.namespace['sorted'] = sorted
        self.namespace['enumerate'] = enumerate
        self.namespace['zip'] = zip
        self.namespace['map'] = map
        self.namespace['filter'] = filter
        self.namespace['sum'] = sum
        self.namespace['min'] = min
        self.namespace['max'] = max
        self.namespace['abs'] = abs
        self.namespace['round'] = round
        self.namespace['range'] = range
        self.namespace['isinstance'] = isinstance
        self.namespace['print'] = print
        
        # 提供数据元信息
        if isinstance(data, str):
            data_info = f"字符串，长度 {len(data)} 字符"
        elif isinstance(data, dict):
            data_info = f"字典，{len(data)} 个键: {list(data.keys())[:10]}"
        elif isinstance(data, list):
            data_info = f"列表，{len(data)} 个元素"
            if data:
                data_info += f"，第一个元素类型: {type(data[0]).__name__}"
                if isinstance(data[0], dict):
                    data_info += f"，键: {list(data[0].keys())[:8]}"
        else:
            data_info = f"类型: {type(data).__name__}"
        
        self.namespace['_data_info'] = data_info
        
    def load_sub_lm(self, sub_lm_fn: Callable):
        """注入 sub-LM 调用函数"""
        self.namespace['llm_query'] = sub_lm_fn
        
    def execute(self, code: str) -> Dict[str, Any]:
        """
        在沙箱中执行代码
        
        Returns:
            {
                "success": bool,
                "output": str,      # stdout 输出
                "error": str,       # 错误信息
                "result": Any,      # 最后一个表达式的值
                "variables": dict,  # 新增/修改的变量
            }
        """
        self.step_count += 1
        
        if self.step_count > self.config.MAX_REPL_STEPS:
            return {
                "success": False,
                "error": f"达到最大 REPL 步数 ({self.config.MAX_REPL_STEPS})",
                "output": "",
                "result": None,
            }
        
        if len(code) > self.config.MAX_CODE_LENGTH:
            return {
                "success": False,
                "error": f"代码超长 ({len(code)} > {self.config.MAX_CODE_LENGTH})",
                "output": "",
                "result": None,
            }
        
        # 安全检查
        try:
            self._security_check(code)
        except SandboxError as e:
            return {
                "success": False,
                "error": f"安全拦截: {str(e)}",
                "output": "",
                "result": None,
            }
        
        # 执行
        stdout_capture = StringIO()
        old_vars = set(self.namespace.keys())
        
        try:
            with redirect_stdout(stdout_capture):
                # 尝试作为表达式求值
                try:
                    result = eval(code, {"__builtins__": {}}, self.namespace)
                except SyntaxError:
                    # 作为语句执行
                    exec(code, {"__builtins__": {}}, self.namespace)
                    result = None
            
            output = stdout_capture.getvalue()
            new_vars = set(self.namespace.keys()) - old_vars
            
            log_entry = {
                "step": self.step_count,
                "code": code[:200],
                "success": True,
                "output_preview": output[:500] if output else str(result)[:500],
            }
            self.execution_log.append(log_entry)
            
            return {
                "success": True,
                "output": output,
                "result": result,
                "error": "",
                "new_variables": list(new_vars),
            }
            
        except Exception as e:
            log_entry = {
                "step": self.step_count,
                "code": code[:200],
                "success": False,
                "error": str(e),
            }
            self.execution_log.append(log_entry)
            
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "result": None,
                "error": f"{type(e).__name__}: {str(e)}",
            }
    
    def get_state_summary(self) -> str:
        """获取当前 REPL 状态摘要"""
        vars_info = []
        for name, val in self.namespace.items():
            if name.startswith('_') or callable(val) or name in ('json',):
                continue
            if isinstance(val, (str, int, float, bool)):
                vars_info.append(f"  {name} = {str(val)[:100]}")
            elif isinstance(val, (list, dict)):
                vars_info.append(f"  {name} = {type(val).__name__}(len={len(val)})")
            else:
                vars_info.append(f"  {name} = {type(val).__name__}")
        
        return "\n".join([
            f"REPL 状态 (步骤 {self.step_count}/{self.config.MAX_REPL_STEPS}):",
            "变量:",
            *vars_info,
        ])
    
    def _security_check(self, code: str):
        """代码安全检查"""
        code_lower = code.lower()
        
        for mod in self.config.BLOCKED_MODULES:
            if f"import {mod}" in code_lower or f"from {mod}" in code_lower:
                raise SandboxError(f"禁止导入模块: {mod}")
        
        for builtin in self.config.BLOCKED_BUILTINS:
            if f"{builtin}(" in code:
                raise SandboxError(f"禁止调用: {builtin}")
        
        # 检查文件操作
        if any(op in code_lower for op in ['open(', 'file(', 'write(', 'unlink']):
            # 允许 json.dumps 等，但禁止文件操作
            if 'open(' in code_lower:
                raise SandboxError("禁止文件操作")


# ============================================================
# Sub-LM 调用管理
# ============================================================

class SubLMManager:
    """
    管理递归 sub-LM 调用
    
    - 调用计数和成本跟踪
    - 递归深度控制
    - 结果缓存
    """
    
    def __init__(self, config: RLMConfig = None):
        self.config = config or RLMConfig()
        self.call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.cache = {}
        self.call_log = []
        
    def create_sub_lm_fn(self, llm_call_fn: Callable, 
                          depth: int = 0) -> Callable:
        """
        创建可注入 REPL 的 sub-LM 调用函数
        
        Args:
            llm_call_fn: 底层 LLM 调用函数
            depth: 当前递归深度
            
        Returns:
            llm_query(prompt, max_tokens=500) -> str
        """
        
        def llm_query(prompt: str, max_tokens: int = 500) -> str:
            """
            在 REPL 中可用的 LLM 查询函数
            
            用法:
                result = llm_query("分析以下数据的趋势: " + data_slice)
            """
            # 检查调用限制
            if self.call_count >= self.config.MAX_SUB_CALLS:
                return f"[错误] 已达最大 sub-LM 调用次数 ({self.config.MAX_SUB_CALLS})"
            
            if depth >= self.config.MAX_RECURSION_DEPTH:
                return "[错误] 已达最大递归深度"
            
            # 检查成本
            if self.total_cost >= self.config.MAX_TOTAL_COST_USD:
                return f"[错误] 已达成本上限 (${self.config.MAX_TOTAL_COST_USD})"
            
            # 缓存检查
            cache_key = hashlib.md5(prompt[:500].encode()).hexdigest()
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 执行调用
            self.call_count += 1
            max_tokens = min(max_tokens, self.config.SUB_CALL_MAX_TOKENS)
            
            try:
                start = time.time()
                result = llm_call_fn(
                    prompt=prompt,
                    model=self.config.SUB_MODEL,
                    max_tokens=max_tokens,
                )
                elapsed = time.time() - start
                
                # 记录
                self.call_log.append({
                    "call_id": self.call_count,
                    "depth": depth,
                    "prompt_preview": prompt[:100],
                    "result_preview": result[:100],
                    "elapsed": round(elapsed, 2),
                })
                
                # 缓存
                self.cache[cache_key] = result
                
                return result
                
            except Exception as e:
                return f"[LLM 错误] {str(e)}"
        
        return llm_query
    
    def get_stats(self) -> Dict:
        """获取调用统计"""
        return {
            "total_calls": self.call_count,
            "max_calls": self.config.MAX_SUB_CALLS,
            "total_cost": round(self.total_cost, 4),
            "cache_hits": len(self.cache),
            "avg_latency": (
                round(sum(c["elapsed"] for c in self.call_log) / len(self.call_log), 2)
                if self.call_log else 0
            ),
        }


# ============================================================
# RLM 核心引擎
# ============================================================

class RLMExecutionMode(Enum):
    """RLM 执行模式"""
    FULL_RECURSIVE = "full_recursive"     # 完整递归（大数据量）
    REPL_ONLY = "repl_only"              # 仅 REPL，无递归调用
    HYBRID = "hybrid"                     # 混合：先 REPL 探索，按需递归


@dataclass
class RLMResult:
    """RLM 执行结果"""
    answer: str
    success: bool
    mode: RLMExecutionMode
    repl_steps: int
    sub_lm_calls: int
    total_elapsed: float
    execution_trace: List[Dict] = field(default_factory=list)
    cost_estimate: float = 0.0
    confidence: float = 0.0
    

class RLMEngine:
    """
    Recursive Language Model 引擎
    
    核心流程:
    1. 将数据加载到 REPL 沙箱
    2. Root LM 获得系统提示 + REPL 状态
    3. Root LM 写代码探索数据（切片、分组、统计）
    4. 在需要语义理解时调用 sub-LM
    5. 迭代直到 Root LM 给出最终答案
    
    相比直接塞 context:
    - 无截断：数据再大也能完整访问
    - 精确：Agent 自己决定看哪些数据
    - 便宜：sub-calls 用小模型
    - 可追溯：REPL 执行日志完整
    """
    
    # RLM 系统提示 — 参考论文原始 prompt 风格
    SYSTEM_PROMPT = """你是 MRARFAI 销售数据分析 Agent。你在一个 Python REPL 环境中工作。

## 环境
- 变量 `data` 包含完整的出货/销售数据（{data_info}）
- 你可以写 Python 代码探索和分析数据
- 你可以调用 `llm_query(prompt, max_tokens=500)` 让 sub-LM 帮你分析数据片段

## 工作方式
1. 先用代码探索数据结构: `data.keys()`, `len(data)`, `data[:3]` 等
2. 按需切片数据，不要试图一次处理所有数据
3. 对需要语义理解的部分，用 llm_query() 分析
4. 汇总各部分结果，给出最终答案

## 输出格式
- 写代码时，用 ```python ... ``` 包裹
- 准备好最终答案时，用 ```answer ... ``` 包裹
- 每步代码应有明确目的

## 可用函数
- `llm_query(prompt, max_tokens=500)` — 调用 sub-LM 分析
- 标准 Python 内置函数（len, sorted, sum, min, max, round 等）
- json 模块

## 限制
- 最多 {max_steps} 步代码执行
- 最多 {max_sub_calls} 次 llm_query 调用
- 禁止文件/网络操作

当前任务: {task}
"""

    def __init__(self, 
                 root_llm_fn: Callable = None,
                 sub_llm_fn: Callable = None,
                 config: RLMConfig = None):
        """
        Args:
            root_llm_fn: Root LM 调用函数 (prompt, system, max_tokens) -> str
            sub_llm_fn:  Sub LM 调用函数 (prompt, model, max_tokens) -> str
            config: RLM 配置
        """
        self.config = config or RLMConfig()
        self.root_llm_fn = root_llm_fn
        self.sub_llm_fn = sub_llm_fn
        
    def analyze(self, data: Any, task: str,
                mode: RLMExecutionMode = RLMExecutionMode.HYBRID,
                data_var_name: str = "data") -> RLMResult:
        """
        使用 RLM 范式分析数据
        
        Args:
            data: 任意数据（dict/list/str）
            task: 分析任务描述
            mode: 执行模式
            data_var_name: 数据在 REPL 中的变量名
            
        Returns:
            RLMResult 包含答案和执行追踪
        """
        start_time = time.time()
        
        # 1. 初始化 REPL 沙箱
        repl = SecureREPL(self.config)
        repl.load_data(data, data_var_name)
        
        # 2. 初始化 Sub-LM 管理器
        sub_mgr = SubLMManager(self.config)
        if self.sub_llm_fn and mode != RLMExecutionMode.REPL_ONLY:
            sub_lm_fn = sub_mgr.create_sub_lm_fn(self.sub_llm_fn)
            repl.load_sub_lm(sub_lm_fn)
        
        # 3. 构建系统提示
        system_prompt = self.SYSTEM_PROMPT.format(
            data_info=repl.namespace.get('_data_info', '未知'),
            max_steps=self.config.MAX_REPL_STEPS,
            max_sub_calls=self.config.MAX_SUB_CALLS,
            task=task,
        )
        
        # 4. 迭代执行
        conversation = []
        execution_trace = []
        final_answer = None
        
        for step in range(self.config.MAX_REPL_STEPS):
            # 构建当前上下文
            context = self._build_context(
                system_prompt, conversation, repl
            )
            
            # Root LM 生成下一步
            try:
                response = self.root_llm_fn(
                    prompt=context,
                    system=system_prompt,
                    max_tokens=2000,
                )
            except Exception as e:
                logger.error(f"Root LM 调用失败: {e}")
                break
            
            # 解析响应
            code_blocks = self._extract_code_blocks(response)
            answer_block = self._extract_answer(response)
            
            if answer_block:
                final_answer = answer_block
                execution_trace.append({
                    "step": step + 1,
                    "type": "answer",
                    "content": answer_block[:500],
                })
                break
            
            if code_blocks:
                for code in code_blocks:
                    result = repl.execute(code)
                    
                    execution_trace.append({
                        "step": step + 1,
                        "type": "code",
                        "code": code[:300],
                        "success": result["success"],
                        "output": (result["output"] or str(result["result"]))[:300],
                        "error": result.get("error", ""),
                    })
                    
                    # 将执行结果添加到对话历史
                    conversation.append({
                        "role": "assistant",
                        "content": response,
                    })
                    conversation.append({
                        "role": "user",
                        "content": self._format_repl_result(result),
                    })
            else:
                # Root LM 没有输出代码也没有给答案
                conversation.append({
                    "role": "assistant",
                    "content": response,
                })
                conversation.append({
                    "role": "user",
                    "content": "请继续分析，写代码探索数据或给出最终答案。",
                })
        
        # 5. 如果没有明确的 answer block，用最后一次响应
        if final_answer is None:
            final_answer = response if 'response' in dir() else "分析未完成"
        
        elapsed = time.time() - start_time
        
        return RLMResult(
            answer=final_answer,
            success=final_answer is not None,
            mode=mode,
            repl_steps=repl.step_count,
            sub_lm_calls=sub_mgr.call_count,
            total_elapsed=round(elapsed, 2),
            execution_trace=execution_trace,
            cost_estimate=sub_mgr.total_cost,
        )
    
    def _build_context(self, system: str, conversation: List[Dict],
                       repl: SecureREPL) -> str:
        """构建当前上下文"""
        parts = []
        
        # 对话历史（最近几轮）
        recent = conversation[-6:]  # 最近 3 轮交互
        for msg in recent:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                parts.append(f"[你之前的输出]\n{content}")
            else:
                parts.append(f"[REPL 执行结果]\n{content}")
        
        # REPL 状态
        parts.append(f"\n{repl.get_state_summary()}")
        
        # 提示下一步
        parts.append("\n请写代码继续分析，或用 ```answer ``` 给出最终答案。")
        
        return "\n\n".join(parts)
    
    def _extract_code_blocks(self, response: str) -> List[str]:
        """从 Root LM 响应中提取 Python 代码块"""
        blocks = []
        in_block = False
        current = []
        
        for line in response.split('\n'):
            if line.strip().startswith('```python'):
                in_block = True
                current = []
            elif line.strip() == '```' and in_block:
                in_block = False
                if current:
                    blocks.append('\n'.join(current))
            elif in_block:
                current.append(line)
        
        return blocks
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """提取最终答案"""
        in_answer = False
        answer_lines = []
        
        for line in response.split('\n'):
            if line.strip().startswith('```answer'):
                in_answer = True
                continue
            elif line.strip() == '```' and in_answer:
                break
            elif in_answer:
                answer_lines.append(line)
        
        if answer_lines:
            return '\n'.join(answer_lines).strip()
        
        return None
    
    def _format_repl_result(self, result: Dict) -> str:
        """格式化 REPL 执行结果"""
        if result["success"]:
            output = result["output"] or str(result["result"])
            return f"[执行成功]\n{output[:2000]}"
        else:
            return f"[执行错误] {result['error']}"


# ============================================================
# 销售数据专用 RLM 分析器
# ============================================================

class RLMSalesAnalyzer:
    """
    MRARFAI 专用的 RLM 销售数据分析器
    
    封装了:
    1. 出货数据的标准化加载
    2. 销售分析专用的 REPL 工具
    3. 与 multi_agent.py 的兼容接口
    4. 分析结果缓存
    """
    
    def __init__(self,
                 root_model: str = None,
                 sub_model: str = None,
                 root_llm_fn: Callable = None,
                 sub_llm_fn: Callable = None):
        """
        Args:
            root_model: Root LM 模型名
            sub_model: Sub LM 模型名
            root_llm_fn: Root LM 调用函数
            sub_llm_fn: Sub LM 调用函数
        """
        config = RLMConfig()
        if root_model:
            config.ROOT_MODEL = root_model
        if sub_model:
            config.SUB_MODEL = sub_model
            
        self.engine = RLMEngine(
            root_llm_fn=root_llm_fn,
            sub_llm_fn=sub_llm_fn,
            config=config,
        )
        self.analysis_cache = {}
        
    def analyze_shipment_data(self, data: Dict, question: str) -> Dict:
        """
        RLM 方式分析出货数据
        
        替代 multi_agent.py 中的 query_sales_data()[:5000]
        
        Args:
            data: 完整出货数据字典
            question: 用户问题
            
        Returns:
            {
                "answer": str,          # 分析结果
                "data_accessed": list,   # 访问了哪些数据片段
                "rlm_trace": list,       # 执行追踪
                "stats": dict,           # 统计信息
            }
        """
        # 选择执行模式
        mode = self._select_mode(data, question)
        
        # 执行 RLM 分析
        result = self.engine.analyze(
            data=data,
            task=question,
            mode=mode,
        )
        
        return {
            "answer": result.answer,
            "success": result.success,
            "mode": result.mode.value,
            "rlm_trace": result.execution_trace,
            "stats": {
                "repl_steps": result.repl_steps,
                "sub_lm_calls": result.sub_lm_calls,
                "elapsed": result.total_elapsed,
                "cost": result.cost_estimate,
            },
        }
    
    def _select_mode(self, data: Dict, question: str) -> RLMExecutionMode:
        """
        根据数据量和问题复杂度选择执行模式
        
        - 小数据 + 简单问题 → REPL_ONLY (快速，无递归)
        - 中等数据 → HYBRID (按需递归)
        - 大数据 + 复杂问题 → FULL_RECURSIVE (完整递归)
        """
        data_size = len(json.dumps(data, ensure_ascii=False, default=str))
        
        # 复杂问题关键词
        complex_keywords = ['所有', '每个', '对比', '趋势', '全面', '详细', 
                          '排名', '分析', '为什么', '原因']
        is_complex = any(kw in question for kw in complex_keywords)
        
        if data_size < 5000 and not is_complex:
            return RLMExecutionMode.REPL_ONLY
        elif data_size < 50000:
            return RLMExecutionMode.HYBRID
        else:
            return RLMExecutionMode.FULL_RECURSIVE


# ============================================================
# 与现有系统的集成适配器
# ============================================================

class RLMMultiAgentAdapter:
    """
    将 RLM 引擎集成到现有 multi_agent.py
    
    替代方式:
    旧: data_context = json.dumps(data)[:5000]  → 截断！
    新: rlm_result = rlm_adapter.query(data, question) → 无截断！
    """
    
    def __init__(self, rlm_analyzer: RLMSalesAnalyzer):
        self.analyzer = rlm_analyzer
        
    def query_sales_data_rlm(self, question: str, 
                              data_store: Dict = None) -> str:
        """
        替代 multi_agent.py 的 query_sales_data()
        
        不再截断，使用 RLM 精确访问
        """
        if not data_store:
            return "数据未加载"
        
        result = self.analyzer.analyze_shipment_data(data_store, question)
        
        if result["success"]:
            return result["answer"]
        else:
            # 降级到旧方式
            logger.warning("RLM 分析失败，降级到截断查询")
            return json.dumps(data_store, ensure_ascii=False, default=str)[:5000]
    
    def get_data_context_rlm(self, question: str, 
                              data_store: Dict) -> Tuple[str, Dict]:
        """
        为 Agent 构建数据上下文
        
        替代直接把数据 dump 成 string 的方式
        返回: (精选的数据摘要, RLM执行元数据)
        """
        result = self.analyzer.analyze_shipment_data(data_store, 
            f"为以下问题准备数据摘要（只提取相关数据，不做分析）: {question}")
        
        return result["answer"], {
            "rlm_mode": result["mode"],
            "data_points_accessed": result["stats"]["repl_steps"],
        }


# ============================================================
# 与 V9.0 其他模块的集成点
# ============================================================

class RLMEnCompassBridge:
    """
    RLM × EnCompass 集成
    
    在 RLM 的递归调用中引入 EnCompass 的 branchpoint 搜索:
    - REPL 探索阶段: 多种数据切片策略并行探索
    - Sub-LM 调用: 多种分析角度的 beam search
    - 汇总阶段: 多种综合策略的最优选择
    """
    
    def __init__(self, rlm_engine: RLMEngine, search_engine=None):
        self.rlm = rlm_engine
        self.search = search_engine  # EnCompass SearchEngine
        
    def analyze_with_search(self, data: Any, task: str) -> RLMResult:
        """RLM 递归 + EnCompass 搜索"""
        # 在每个 REPL 步骤后，用 EnCompass 评估并选择最优路径
        pass


class RLMAWMBridge:
    """
    RLM × AWM 集成
    
    AWM 合成环境 + RLM REPL = Agent 在合成的 SQLite 环境中
    用 RLM 方式递归分析数据
    """
    
    def __init__(self, rlm_engine: RLMEngine, awm_factory=None):
        self.rlm = rlm_engine
        self.awm = awm_factory
        
    def train_on_synthetic_envs(self, num_envs: int = 100):
        """
        在 AWM 合成环境中训练 RLM 策略
        
        流程:
        1. AWM 生成合成销售分析环境
        2. RLM 在环境中执行分析
        3. AWM 的 SQL 验证器提供 reward
        4. 用 GRPO 优化 RLM 策略
        """
        pass


class RLMMemoryBridge:
    """
    RLM × 三维记忆 集成
    
    跨 RLM 递归调用的记忆管理:
    - 当前递归上下文 → Working Memory
    - 成功的分析策略 → Skill Memory
    - 数据模式 → Strategy Memory
    """
    
    def __init__(self, rlm_engine: RLMEngine, memory_system=None):
        self.rlm = rlm_engine
        self.memory = memory_system
        
    def analyze_with_memory(self, data: Any, task: str) -> RLMResult:
        """RLM 递归 + 记忆增强"""
        # 1. 检索相关记忆
        # 2. 将记忆注入 REPL 系统提示
        # 3. 执行 RLM 分析
        # 4. 存储新的分析经验
        pass


# ============================================================
# 工具函数
# ============================================================

def estimate_data_tokens(data: Any) -> int:
    """估算数据的 token 数"""
    text = json.dumps(data, ensure_ascii=False, default=str)
    # 中文大约 1.5 char/token，英文大约 4 char/token
    # 混合估算: 2 char/token
    return len(text) // 2


def should_use_rlm(data: Any, question: str, 
                    threshold_tokens: int = 3000) -> bool:
    """
    判断是否应该使用 RLM 而非直接塞 context
    
    规则:
    - 数据量 > threshold → 用 RLM
    - 问题涉及全量数据分析 → 用 RLM
    - 简单单点查询 → 直接 context
    """
    tokens = estimate_data_tokens(data)
    
    if tokens > threshold_tokens:
        return True
    
    # 全量分析关键词
    if any(kw in question for kw in ['所有', '每个品牌', '全面', '对比分析', '排名']):
        return True
    
    return False


# ============================================================
# 入口
# ============================================================

def create_rlm_analyzer(
    root_model: str = "gpt-4o",
    sub_model: str = "deepseek-chat",
    root_llm_fn: Callable = None,
    sub_llm_fn: Callable = None,
) -> RLMSalesAnalyzer:
    """
    创建 RLM 销售分析器的便捷函数
    
    用法:
        analyzer = create_rlm_analyzer()
        result = analyzer.analyze_shipment_data(data, "分析各品牌趋势")
    """
    return RLMSalesAnalyzer(
        root_model=root_model,
        sub_model=sub_model,
        root_llm_fn=root_llm_fn,
        sub_llm_fn=sub_llm_fn,
    )


if __name__ == "__main__":
    # 演示: 用 RLM 分析模拟出货数据
    print("=" * 60)
    print("MRARFAI RLM Engine v9.0 Demo")
    print("=" * 60)
    
    # 模拟数据
    demo_data = {
        "总营收": 150000000,
        "月度总营收": [
            {"月份": f"2025-{m:02d}", "金额": 10000000 + m * 2000000}
            for m in range(1, 13)
        ],
        "客户金额": [
            {"客户": f"Brand_{chr(65+i)}", "年度金额": 20000000 - i * 3000000,
             "月度": [{"月": m, "金额": 1500000 + m * 100000 - i * 200000} 
                     for m in range(1, 13)]}
            for i in range(15)
        ],
    }
    
    # 测试 SecureREPL
    repl = SecureREPL()
    repl.load_data(demo_data)
    
    print("\n--- REPL 测试 ---")
    
    # 步骤1: 探索数据结构
    r1 = repl.execute("list(data.keys())")
    print(f"Step 1: {r1['result']}")
    
    # 步骤2: 查看客户数量
    r2 = repl.execute("len(data['客户金额'])")
    print(f"Step 2: {r2['result']} 个客户")
    
    # 步骤3: Top 5 客户
    r3 = repl.execute("""
top5 = sorted(data['客户金额'], key=lambda x: x['年度金额'], reverse=True)[:5]
[(c['客户'], c['年度金额']) for c in top5]
""")
    print(f"Step 3: Top 5 = {r3['result']}")
    
    # 步骤4: 月度趋势
    r4 = repl.execute("""
months = data['月度总营收']
growth = [(m['月份'], m['金额']) for m in months]
growth
""")
    print(f"Step 4: 月度趋势 = {r4['result'][:3]}...")
    
    # 步骤5: 安全测试 — 应该被拦截
    r5 = repl.execute("import os")
    print(f"Step 5 (安全): {r5['error']}")
    
    print(f"\n--- REPL 状态 ---")
    print(repl.get_state_summary())
    
    print(f"\n--- 执行日志 ---")
    for log in repl.execution_log:
        status = "✓" if log["success"] else "✗"
        print(f"  {status} Step {log['step']}: {log.get('output_preview', log.get('error', ''))[:60]}")
    
    # 测试模式选择
    print(f"\n--- 模式选择测试 ---")
    analyzer = RLMSalesAnalyzer()
    
    small_data = {"总营收": 100}
    large_data = {"customers": [{"name": f"c{i}"} for i in range(1000)]}
    
    print(f"  小数据+简单问题: {analyzer._select_mode(small_data, '总营收多少').value}")
    print(f"  小数据+复杂问题: {analyzer._select_mode(small_data, '分析所有客户趋势').value}")
    print(f"  大数据+复杂问题: {analyzer._select_mode(large_data, '对比分析每个客户').value}")
    
    # Token 估算
    print(f"\n--- Token 估算 ---")
    print(f"  demo_data: ~{estimate_data_tokens(demo_data)} tokens")
    print(f"  should_use_rlm(简单): {should_use_rlm(demo_data, '总营收')}")
    print(f"  should_use_rlm(复杂): {should_use_rlm(demo_data, '分析所有品牌月度趋势')}")
    
    print("\n✅ RLM Engine 初始化成功")
