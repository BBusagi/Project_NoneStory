import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pathlib import Path

# 测试用的脚本

# ====== 路径配置 ======
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
base_dir = Path(__file__).parent.resolve().parent

MODEL_DIR = os.path.join(base_dir, "sft-model-n1", "medium", "final_model")  # 训练完保存的模型目录
VAL_DATA_PATH = os.path.join(base_dir, "output", "val_data_with_prompt.jsonl") # 可选，验证集数据路径（如不评估可为空）
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 获取模型的配置
config = model.config

# 输出相关信息
print(f"原模型：{config._name_or_path}")
print(f"训练次数（epochs）：{config.num_train_epochs if hasattr(config, 'num_train_epochs') else '未提供'}")
