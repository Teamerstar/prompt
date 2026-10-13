"""
话术生成模块
"""
from typing import Dict, Any
from src.llm_client import LLMClient


class ReplyGenerator:
    def __init__(self, llm_client: LLMClient, prompt_template: str):
        self.llm = llm_client
        self.prompt_template = prompt_template
    
    def generate(self, user_input: str, intent_category: str) -> Dict[str, Any]:
        """
        生成回复话术
        
        Returns:
            {
                "reply": 生成的回复,
                "raw_output": 原始输出,
                "usage": token使用情况
            }
        """
        if not intent_category:
            intent_category = "其他"  # 或你业务中合适的默认分类

        prompt = self.prompt_template.replace("{user_input}", user_input)
        prompt = prompt.replace("{intent_category}", intent_category)
        
        result = self.llm.chat(prompt, temperature=0.5)
        
        return {
            "reply": result["text"].strip(),
            "raw_output": result["text"],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "latency_ms": result["latency_ms"]
            }
        }
