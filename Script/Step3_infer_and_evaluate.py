import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# 测试用的脚本

# ====== 路径配置 ======
MODEL_DIR = "sft-model-n1/medium/final_model"  # 训练完保存的模型目录
VAL_DATA_PATH = "output/val_data_with_prompt.jsonl"  # 可选，验证集数据路径（如不评估可为空）
USE_EVAL = os.path.exists(VAL_DATA_PATH)

# ====== 加载模型与 tokenizer ======
print(f"🔄 加载模型：{MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ 模型已加载到 {device}")

# ====== 推理函数 ======
def generate(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== 简单交互模式 ======
print("✍️ 输入你想测试的 prompt（输入 q 退出）")
while True:
    user_input = input("🟢 Prompt >>> ")
    if user_input.lower().strip() in ["q", "quit", "exit"]:
        break
    result = generate(user_input)
    print("📘 模型生成：", result)
    print("------")

# ====== 自动评估验证集（可选） ======
if USE_EVAL:
    print("🔍 正在加载验证集进行评估...")
    dataset = load_dataset("json", data_files={"validation": VAL_DATA_PATH}, split="validation")

    def preprocess(batch):
        full_texts = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(preprocess, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["prompt", "completion"])

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir="./eval_tmp",
            per_device_eval_batch_size=2,
        ),
        eval_dataset=tokenized_dataset
    )

    metrics = trainer.evaluate()
    print("📊 验证集评估结果：", metrics)

print("✅ 推理 & 评估结束。")
