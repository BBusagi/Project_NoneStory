import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from pathlib import Path

# ====== 读取 config.json 配置参数 ======
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

base_dir = Path(__file__).parent.resolve().parent
model_name = config["model_name"]
data_path = str((base_dir / config["data_path"]).resolve())
output_dir = str((base_dir / config["output_dir"]).resolve())

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

# ====== 加载 tokenizer 和模型 ======
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 默认无 pad token

model = None
try:
    model = AutoModelForCausalLM.from_pretrained(last_checkpoint if last_checkpoint else model_name)
    param_count = sum(p.numel() for p in model.parameters())
    model_dir_used = model.config._name_or_path
    print(f"✅ 模型加载成功，共 {param_count:,} 个参数，加载自：{model_dir_used}")
    if len(model.state_dict()) == 0:
        print("⚠️ 警告：model.state_dict() 为空，模型可能未加载完整！")
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    model = None

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

print(f"✅ 有效样本数量：{len(tokenized_dataset)}")
print("🗂️ 输出模型路径为：", output_dir)

if model is None or len(model.state_dict()) == 0:
    raise RuntimeError("❌ 模型未成功加载，无法开始训练")

# ====== 训练参数 ======
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=config["batch_size"],
    num_train_epochs=config["epochs"],
    learning_rate=config["learning_rate"],
    fp16=config["fp16"],
    logging_steps=20,
    save_strategy="steps",
    save_steps=500,
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

# ====== 启动训练流程 ======
try:
    if last_checkpoint is None:
        print("🆕 当前为新模型训练")
    else:
        print(f"🔁 检测到已有 checkpoint：{last_checkpoint}，将继续训练")
    trainer.train()

except Exception as e:
    print(f"❌ 训练出错：{e}")
    if model is not None and len(model.state_dict()) > 0:
        interrupted_path = os.path.join(output_dir, "interrupted_checkpoint")
        trainer.save_model(interrupted_path)
        print(f"💾 出错模型已保存至：{interrupted_path}")
    else:
        print("⚠️ 模型无效，未保存任何内容")

finally:
    if model is not None and len(model.state_dict()) > 0:
        final_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_path)
        print(f"✅ 最终模型已保存至：{final_path}")
    else:
        print("⚠️ 未保存最终模型（模型无效）")
    print("✅ 训练完成 ✅")
