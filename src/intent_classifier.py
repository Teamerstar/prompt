"""
意图分类模块
"""
import json
import re
from typing import Dict, Any, Optional
from src.llm_client import LLMClient


class IntentClassifier:
    def __init__(self, llm_client: LLMClient, prompt_template: str):
        self.llm = llm_client
        self.prompt_template = prompt_template
    
    def classify(self, user_input: str) -> Dict[str, Any]:
        """
        对用户输入进行意图分类
        
        Returns:
            {
                "category": 意图类别,
                "severity": 紧急程度,
                "need_human": 是否需要人工,
                "raw_output": 原始输出
            }
        """
        prompt = self.prompt_template.replace("{user_input}", user_input)
        
        result = self.llm.chat(prompt, temperature=0.1)
        raw_output = result["text"].strip()
        
        # 解析JSON输出
        parsed = self._parse_json(raw_output)
        
        return {
            "category": parsed.get("category"),
            "severity": parsed.get("severity"),
            "need_human": parsed.get("need_human"),
            "raw_output": raw_output,
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "latency_ms": result["latency_ms"]
            }
        }
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """从LLM输出中解析JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON部分
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {}
