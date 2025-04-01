import os
import json
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset

# sft训练脚本

# ====== 读取 config.json 配置参数 ======
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

model_name = config["model_name"]
data_path = config["data_path"]
output_dir = config["output_dir"]

# ====== 加载 tokenizer 和模型 ======
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 默认无 pad token

model = AutoModelForCausalLM.from_pretrained(model_name)

# ====== 加载并预处理数据 ======
dataset = load_dataset("json", data_files={"train": data_path}, split="train")

def preprocess(batch):
    full_texts = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
    
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=config["max_seq_length"],
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["prompt", "completion"])

def is_valid(sample):
    try:
        return all(
            isinstance(sample[k], list) and all(isinstance(i, int) for i in sample[k])
            for k in ["input_ids", "attention_mask", "labels"]
        )
    except:
        return False
tokenized_dataset = tokenized_dataset.filter(is_valid)

# ===== 中途检测 =====
print(f"有效样本数量：{len(tokenized_dataset)}")
print("输出模型路径为：", output_dir)

# ====== 获取最近的 checkpoint（用于断点续训） ======
def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

last_checkpoint = get_last_checkpoint(output_dir)

# ====== 训练参数 ======
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=config["batch_size"],
    num_train_epochs=config["epochs"],
    learning_rate=config["learning_rate"],
    fp16=config["fp16"],
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    auto_find_batch_size=True,
)

# ====== 启动训练器 ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# ====== 设置自动中止训练时间（单位：秒） ======
MAX_TRAIN_TIME = 90 * 60  # 90分钟

# 训练时长结束后中断训练的函数
def interrupt_training():
    print(f"\n⏰ 达到 {MAX_TRAIN_TIME // 60} 分钟限制，尝试中断训练...")
    raise TimeoutError("训练时间已到，自动中断。")

# 启动定时器（在后台线程执行中断）
timer = threading.Timer(MAX_TRAIN_TIME, interrupt_training)
timer.start()

# ====== 启动训练器 ======
try:
    print(f"🚀 开始训练（从 checkpoint: {last_checkpoint}）")
    trainer.train(resume_from_checkpoint=last_checkpoint)

except TimeoutError as e:
    print(f"🛑 {str(e)}，保存中断模型...")
    trainer.save_model(os.path.join(output_dir, "interrupted_checkpoint"))

except Exception as e:
    print(f"❌ 训练出错：{e}")
    trainer.save_model(os.path.join(output_dir, "interrupted_checkpoint"))

finally:
    timer.cancel()  # 清除定时器
    print("✅ 训练完成✅")