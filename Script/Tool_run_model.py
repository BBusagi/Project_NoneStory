import os
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置模型路径（请替换为你训练保存的目录）
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
base_dir = Path(__file__).parent.resolve().parent
MODEL_DIR = os.path.join(base_dir, "sft-model-n1", "medium", "checkpoint-2880")  # 训练完保存的模型目录
print(f"🔄 加载模型：{MODEL_DIR}")

# 加载训练好的模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 如果有GPU，使用GPU进行推理；否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 测试输入文本（你可以根据需要修改这个文本）
input_text = "1000字程度の恋愛小説を書いてください。"

# 使用tokenizer将输入文本编码成模型可以理解的格式
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 生成文本
output = model.generate(input_ids,
                        max_length=1024,   # 最大生成长度
                        num_return_sequences=3,  # 生成的文本数量
                        do_sample=True,   # 启用采样（否则是贪心搜索）
                        top_p=0.92,       # nucleus sampling (p-值)
                        temperature=1,   # 温度控制生成文本的多样性
                        top_k=50,         # top-k采样
                        repetition_penalty=1.2)  # 重复惩罚

# 解码生成的output，转回为可读的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印生成的文本
print("生成的文本：")
print(generated_text)
