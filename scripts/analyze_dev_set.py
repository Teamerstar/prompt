"""
开发集分析脚本：用于调试 Prompt，不用于最终评估
"""
import json
import sys
import time
from pathlib import Path

# 项目根目录（根据你的实际路径修改）
PROJECT_DIR = Path(r"D:\3000-学习\3700_AI\Prompt\prompt_lab_261012")
sys.path.append(str(PROJECT_DIR))

from src.llm_client import LLMClient
from src.intent_classifier import IntentClassifier
from config.settings import LLM_CONFIG, PROMPTS_DIR

def main():
    print("=" * 60)
    print("开发集分析脚本启动")
    print("=" * 60)
    
    # 1. 检查 Ollama 服务
    print("\n[1/5] 检查 Ollama 服务...")
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"  ✓ Ollama 服务正常，可用模型: {model_names}")
        else:
            print(f"  ✗ Ollama 返回异常状态码: {resp.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("  ✗ 无法连接 Ollama 服务，请先运行: ollama serve")
        return
    
    # 2. 检查数据文件
    print("\n[2/5] 检查数据文件...")
    dev_set_path = PROJECT_DIR / "data" / "processed" / "dev_set.json"
    print(f"  路径: {dev_set_path}")
    
    if not dev_set_path.exists():
        print(f"  ✗ 文件不存在: {dev_set_path}")
        return
    
    with open(dev_set_path, "r", encoding="utf-8") as f:
        dev_set = json.load(f)
    
    if not dev_set:
        print("  ✗ 开发集为空，请检查数据文件内容")
        return
    
    print(f"  ✓ 加载开发集: {len(dev_set)} 条")
    
    # 3. 检查 Prompt 文件
    print("\n[3/5] 检查 Prompt 文件...")
    prompt_path = PROMPTS_DIR / "v5"/ "intent_classification.txt"
    if not prompt_path.exists():
        print(f"  ✗ Prompt 文件不存在: {prompt_path}")
        return
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    print(f"  ✓ 加载 Prompt: {prompt_path.name}")
    
    # 4. 初始化分类器
    print("\n[4/5] 初始化分类器...")
    llm = LLMClient(**LLM_CONFIG)
    classifier = IntentClassifier(llm, prompt_template)
    print(f"  ✓ 模型: {LLM_CONFIG['model']}")
    
    # 5. 遍历开发集 
    print("\n[5/5] 开始分析开发集...")
    errors = []
    correct = 0
    total_tokens = 0
    total_latency = 0
    
    for i, case in enumerate(dev_set):
        text = case.get("text", "")
        true_label = case.get("intent", "")
        
        print(f"\n处理 [{i+1}/{len(dev_set)}] case_id: {case.get('case_id', 'N/A')}")
        print(f"  文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            result = classifier.classify(text)
        except Exception as e:
            print(f"  ✗ 分类异常: {e}")
            errors.append({
                "case_id": case.get("case_id"),
                "text": text,
                "true": true_label,
                "pred": "ERROR",
                "raw_output": str(e)
            })
            continue
        
        pred = result.get("category")
        
        if not pred or not isinstance(pred, str):
            pred = "其他"
        
        is_correct = pred == true_label
        if is_correct:
            correct += 1
            print(f"  ✓ 预测: {pred} (正确)")
        else:
            print(f"  ✗ 预测: {pred} (错误, 真实: {true_label})")
            errors.append({
                "case_id": case.get("case_id"),
                "text": text,
                "true": true_label,
                "pred": pred,
                "raw_output": result.get("raw_output")
            })
        
        usage = result.get("usage", {})
        total_tokens += usage.get("total_tokens", 0)
        total_latency += usage.get("latency_ms", 0)
    
    # 统计结果
    n = len(dev_set)
    acc = correct / n if n else 0
    
    print("\n" + "=" * 60)
    print("分析结果汇总")
    print("=" * 60)
    print(f"开发集样本数: {n}")
    print(f"正确: {correct}")
    print(f"错误: {len(errors)}")
    print(f"开发集准确率: {acc:.2%}")
    print(f"总Token: {total_tokens}")
    print(f"平均延迟: {total_latency / n:.0f}ms")
    
    if errors:
        print("\n错误案例：")
        for e in errors:
            print("-" * 60)
            print(f"case_id: {e['case_id']}")
            print(f"文本: {e['text']}")
            print(f"真实: {e['true']}")
            print(f"预测: {e['pred']}")
            print(f"原始输出: {e['raw_output']}")
    
    # 保存错误案例
    error_path = PROJECT_DIR / "outputs" / "dev_errors_pv5.json"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    
    print(f"\n错误案例已保存到: {error_path}")
    print("\n分析完成！")


if __name__ == "__main__":
    main()
