from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: 模型和 tokenizer 载入
model_name = "rinna/japanese-gpt2-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: 加载数据集
dataset = load_dataset("json", data_files={"train": "output/test6-train1/train_data_with_prompt.jsonl"}, split="train")

# Step 3: Tokenize
def tokenize(batch):
    tokens = tokenizer(
        [p + c for p, c in zip(batch["prompt"], batch["completion"])],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()  # ✅ 添加 labels 字段
    return tokens


tokenized_dataset = dataset.map(tokenize, batched=True)

# Step 4: 训练参数
training_args = TrainingArguments(
    output_dir="./sft-model-n1",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,  # 若使用支持的 GPU
    logging_steps=10,
    save_strategy="epoch"
)

# Step 5: Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Step 6: 开始训练
trainer.train()
