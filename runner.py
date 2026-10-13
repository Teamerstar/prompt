"""
主运行脚本：单条测试 + 批量评估
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

from datetime import datetime

ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from config.settings import LLM_CONFIG, PROMPTS_DIR, LOG_DIR, RESULT_DIR
from src.llm_client import LLMClient
from src.intent_classifier import IntentClassifier
from src.reply_generator import ReplyGenerator


def load_prompt(version: str, prompt_type: str) -> str:
    """加载指定版本的Prompt模板"""
    prompt_path = PROMPTS_DIR / version / f"{prompt_type}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def run_single_test(text: str, prompt_version: str = "v1") -> Dict[str, Any]:
    """
    单条测试
    
    Args:
        text: 用户输入文本
        prompt_version: Prompt版本号
    
    Returns:
        完整结果字典
    """
    # 初始化LLM客户端
    llm_client = LLMClient(**LLM_CONFIG)
    
    # 加载Prompt
    intent_prompt = load_prompt(prompt_version, "intent_classification")
    reply_prompt = load_prompt(prompt_version, "reply_generation")
    
    # 初始化模块
    classifier = IntentClassifier(llm_client, intent_prompt)
    generator = ReplyGenerator(llm_client, reply_prompt)
    
    # 意图分类
    cls_result = classifier.classify(text)
    
    # 话术生成
    reply_result = generator.generate(text, cls_result["category"])
    
    # 汇总结果
    result = {
        #"ts": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "ts": datetime.now().isoformat(),
        "prompt_version": prompt_version,
        "model": LLM_CONFIG["model"],
        "case_id": "smoke-01",
        "input": {
            "text": text
        },
        "expect": None,
        "output_raw": json.dumps({
            "category": cls_result["category"],
            "severity": cls_result["severity"],
            "need_human": cls_result["need_human"]
        }, ensure_ascii=False),
        "reply": reply_result["reply"],
        "prompt_tokens": cls_result["usage"]["prompt_tokens"] + reply_result["usage"]["prompt_tokens"],
        "completion_tokens": cls_result["usage"]["completion_tokens"] + reply_result["usage"]["completion_tokens"],
        "latency_ms": cls_result["usage"]["latency_ms"] + reply_result["usage"]["latency_ms"],
        "error": None
    }
    
    return result


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="电商客服意图识别与话术生成系统")
    parser.add_argument("--text", default="这台手机电池太不耐用了", help="测试文本")
    parser.add_argument("--version", default="v1", help="Prompt版本号")
    parser.add_argument("--eval", action="store_true", help="运行完整评估")
    parser.add_argument("--sample", type=int, default=None, help="评估样本数量")
    
    args = parser.parse_args()
    
    if args.eval:
        # 运行完整评估
        from eval import run_evaluation
        run_evaluation(prompt_version=args.version, sample_size=args.sample)
    else:
        # 单条测试
        result = run_single_test(args.text, args.version)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 保存日志
        log_path = LOG_DIR / f"smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
