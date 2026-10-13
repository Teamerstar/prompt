"""
全局配置
"""
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据路径
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "train_data.json"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"

# Prompt路径
PROMPTS_DIR = ROOT_DIR / "prompts"
PROMPT_VERSION = "v1"  # 当前使用的Prompt版本

# 输出路径
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
RESULT_DIR = OUTPUT_DIR / "results"

# 确保目录存在
for dir_path in [DATA_DIR, OUTPUT_DIR, LOG_DIR, RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# LLM配置
LLM_CONFIG = {
    "model": "qwen3:4b",  # 本地模型
    "base_url": "http://localhost:11434",
    "api_type": "ollama",  # 切换在线模型时改为 "openai"
    "api_key": None,  # 在线模型时填写
}

# 成本配置（每百万token价格，单位：元）
COST_CONFIG = {
    "ollama": {"input": 0, "output": 0},  # 本地模型免费
    "openai": {"input": 15, "output": 60}  # 示例价格，按实际调整
}
