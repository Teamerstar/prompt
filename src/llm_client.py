"""
LLM调用封装，兼容Ollama本地模型和OpenAI兼容API
"""
import json
import time
import requests
from typing import Dict, Any, Optional


class LLMClient:
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434",
                 api_type: str = "ollama", api_key: Optional[str] = None):
        """
        Args:
            model: 模型名称
            base_url: API地址
            api_type: "ollama" 或 "openai"
            api_key: OpenAI API Key（在线模型时必填）
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_type = api_type
        self.api_key = api_key
        
    def chat(self, prompt: str, system_prompt: str = "", 
             temperature: float = 0.3, max_tokens: int = 2048) -> Dict[str, Any]:
        """
        调用LLM生成回复
        
        Returns:
            {
                "text": 生成的文本,
                "prompt_tokens": prompt token数,
                "completion_tokens": 生成token数,
                "total_tokens": 总token数,
                "latency_ms": 延迟(毫秒)
            }
        """
        start_time = time.time()
        
        if self.api_type == "ollama":
            result = self._call_ollama(prompt, system_prompt, temperature, max_tokens)
        else:
            result = self._call_openai(prompt, system_prompt, temperature, max_tokens)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "text": result["text"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            "latency_ms": latency_ms
        }
    
    def _call_ollama(self, prompt: str, system_prompt: str, 
                     temperature: float, max_tokens: int) -> Dict[str, Any]:
        """调用Ollama API"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        return {
            "text": data.get("response", ""),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0)
        }
    
    def _call_openai(self, prompt: str, system_prompt: str,
                     temperature: float, max_tokens: int) -> Dict[str, Any]:
        """调用OpenAI兼容API"""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        usage = data.get("usage", {})
        return {
            "text": data["choices"]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)
        }
