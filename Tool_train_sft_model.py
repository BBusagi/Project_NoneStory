import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset


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

def preprocess(example):
    prompt = example["prompt"]
    completion = example["completion"]
    full_text = prompt + completion

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=config["max_seq_length"],
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # 设置 labels（等于 input_ids）
    labels = input_ids.copy()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

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

print(f"有效样本数量：{len(tokenized_dataset)}")

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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ====== 启动训练器 ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

tokenized_dataset[10]

print("开始第一次训练...")
try:
    trainer.train()
except Exception as e:
    print(f"训练过程中发生错误：{e}")
    trainer.save_model(output_dir + "/interrupted_checkpoint")

time.sleep(5)

print("✅ 训练完成✅ ")