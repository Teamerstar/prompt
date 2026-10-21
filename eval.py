"""
评估脚本：计算准确率、召回率、F1，对比不同Prompt版本
"""
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
EVAL_SET_PATH = ROOT_DIR / "data" / "eval_set.json"
sys.path.append(str(ROOT_DIR))

from config.settings import EVAL_SET_PATH, PROMPTS_DIR, RESULT_DIR, LLM_CONFIG, COST_CONFIG
from src.llm_client import LLMClient
from src.intent_classifier import IntentClassifier
from src.reply_generator import ReplyGenerator

# 合法类别枚举
VALID_CATEGORIES = {"售前咨询", "售后投诉", "物流查询", "退换货", "其他"}


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    """加载评估集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(version: str, prompt_type: str) -> str:
    """加载指定版本的Prompt模板"""
    prompt_path = PROMPTS_DIR / version / f"{prompt_type}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    计算准确率、召回率、F1

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        {
            "accuracy": 整体准确率,
            "per_class": {
                "售前咨询": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 10},
                ...
            },
            "macro_f1": 宏平均F1
        }
    """
    assert len(y_true) == len(y_pred), "标签数量不匹配"

    # 过滤掉 None 和空字符串，避免 sorted 报错
    clean_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t and p and isinstance(t, str) and isinstance(p, str)
    ]
    if not clean_pairs:
        return {"accuracy": 0.0, "per_class": {}, "macro_f1": 0.0}

    y_true = [t for t, _ in clean_pairs]
    y_pred = [p for _, p in clean_pairs]

    # 整体准确率
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0

    # 每个类别的指标
    classes = sorted(set(y_true) | set(y_pred))
    per_class = {}

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(1 for t in y_true if t == cls)

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support
        }

    # 宏平均F1
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4)
    }


def calculate_cost(total_tokens: int, api_type: str) -> float:
    """计算成本（元）"""
    cost_config = COST_CONFIG.get(api_type, {"input": 0, "output": 0})
    # 简化计算，假设输入输出各占一半
    input_tokens = total_tokens // 2
    output_tokens = total_tokens - input_tokens
    cost = (input_tokens / 1_000_000 * cost_config["input"] +
            output_tokens / 1_000_000 * cost_config["output"])
    return round(cost, 6)


def run_evaluation(prompt_version: str = "v1", sample_size: int = None) -> Dict[str, Any]:
    """
    运行评估

    Args:
        prompt_version: Prompt版本号
        sample_size: 评估样本数量（None表示全部）

    Returns:
        评估结果字典
    """
    print(f"开始评估 Prompt版本: {prompt_version}")

    # 加载评估集
    eval_set = load_eval_set(EVAL_SET_PATH)
    if sample_size:
        eval_set = eval_set[:sample_size]
    print(f"评估样本数: {len(eval_set)}")

    # 初始化LLM客户端
    llm_client = LLMClient(**LLM_CONFIG)

    # 加载Prompt
    intent_prompt = load_prompt(prompt_version, "intent_classification")
    reply_prompt = load_prompt(prompt_version, "reply_generation")

    # 初始化模块
    classifier = IntentClassifier(llm_client, intent_prompt)
    generator = ReplyGenerator(llm_client, reply_prompt)

    # 存储结果
    results = []
    y_true = []
    y_pred = []
    total_tokens = 0
    total_latency = 0

    # 逐条评估
    for i, case in enumerate(eval_set):
        print(f"处理第 {i+1}/{len(eval_set)} 条: {case['case_id']}")

        user_input = case["text"]
        true_intent = case["intent"]

        # 意图分类
        cls_result = classifier.classify(user_input)
        pred_intent = cls_result.get("category")  # 用 .get() 避免 KeyError

        # 兜底：如果解析失败或返回非法类别，归为"其他"
        if not pred_intent or pred_intent not in VALID_CATEGORIES:
            print(f"  ⚠ 警告: 第 {i+1} 条分类结果异常 ({pred_intent!r})，兜底为'其他'")
            pred_intent = "其他"

        # 话术生成
        reply_result = generator.generate(user_input, pred_intent)

        # 统计
        usage = cls_result.get("usage", {})
        reply_usage = reply_result.get("usage", {})
        case_tokens = usage.get("total_tokens", 0) + reply_usage.get("total_tokens", 0)
        case_latency = usage.get("latency_ms", 0) + reply_usage.get("latency_ms", 0)

        total_tokens += case_tokens
        total_latency += case_latency

        y_true.append(true_intent)
        y_pred.append(pred_intent)

        results.append({
            "case_id": case["case_id"],
            "input": user_input,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "is_correct": true_intent == pred_intent,
            "reply": reply_result.get("reply", ""),
            "tokens": case_tokens,
            "latency_ms": case_latency
        })

        # 打印当前结果
        status = "✓" if true_intent == pred_intent else "✗"
        print(f"  {status} 真实: {true_intent}, 预测: {pred_intent}")

    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)

    # 计算成本
    cost = calculate_cost(total_tokens, LLM_CONFIG["api_type"])

    # 汇总结果
    summary = {
        "prompt_version": prompt_version,
        "model": LLM_CONFIG["model"],
        "eval_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": len(eval_set),
        "metrics": metrics,
        "cost": {
            "total_tokens": total_tokens,
            "total_cost_yuan": cost,
            "avg_cost_per_case": round(cost / len(eval_set), 6) if eval_set else 0,
            "avg_latency_ms": round(total_latency / len(eval_set), 2) if eval_set else 0
        },
        "results": results
    }

    # 保存结果
    result_path = RESULT_DIR / f"eval_{prompt_version}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n评估完成！结果已保存到: {result_path}")
    print(f"准确率: {metrics['accuracy']}")
    print(f"宏平均F1: {metrics['macro_f1']}")
    print(f"总Token消耗: {total_tokens}")
    print(f"总成本: {cost} 元")
    print(f"平均延迟: {round(total_latency / len(eval_set), 2)}ms")

    return summary


def compare_versions(versions: List[str]) -> None:
    """
    对比多个Prompt版本的效果
    """
    print("=" * 60)
    print("Prompt版本对比")
    print("=" * 60)

    all_results = {}
    for version in versions:
        print(f"\n>>> 评估版本: {version}")
        result = run_evaluation(prompt_version=version)
        all_results[version] = result

    # 对比表格
    print("\n" + "=" * 60)
    print("版本对比结果")
    print("=" * 60)
    print(f"{'版本':<10} {'准确率':<10} {'宏F1':<10} {'Token数':<12} {'成本(元)':<10}")
    print("-" * 60)
    for version, result in all_results.items():
        metrics = result["metrics"]
        cost = result["cost"]
        print(f"{version:<10} {metrics['accuracy']:<10.4f} {metrics['macro_f1']:<10.4f} "
              f"{cost['total_tokens']:<12} {cost['total_cost_yuan']:<10.6f}")

    # 保存对比结果
    compare_path = RESULT_DIR / f"compare_{'_'.join(versions)}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存到: {compare_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估电商客服意图识别与话术生成系统")
    parser.add_argument("--version", default="v1", help="Prompt版本号")
    parser.add_argument("--sample", type=int, default=None, help="评估样本数量")
    parser.add_argument("--compare", nargs="+", help="对比多个版本，如: --compare v1 v2")

    args = parser.parse_args()

    if args.compare:
        compare_versions(args.compare)
    else:
        run_evaluation(prompt_version=args.version, sample_size=args.sample)