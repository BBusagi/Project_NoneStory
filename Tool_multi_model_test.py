import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# ===== 模型列表（名称 → 路径） =====
MODEL_PATHS = {
    "rinna-original": "rinna/japanese-gpt2-medium",
    "sft-v1": "sft-model-n1/medium/final_model",
    # 你可以继续添加模型对比
}

# ===== 参数设置 =====
max_new_tokens = 1024
temperature = 0.8
top_p = 0.95
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 加载所有模型和 tokenizer =====
models = {}
print("🔄 加载模型中...")
for name, path in MODEL_PATHS.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(device).eval()
    models[name] = (tokenizer, model)
print("✅ 所有模型加载完毕\n")

# ===== 主循环：输入 Prompt 测试 =====
print("✍️ 输入 prompt（输入 q 退出）")
while True:
    prompt = input("🟢 Prompt >>> ").strip()
    if prompt.lower() in ["q", "quit", "exit"]:
        break

    print("\n🔍 多模型生成对比结果：")
    print("=" * 60)
    for name, (tokenizer, model) in models.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n🧠 {name}：")
        print(result)
        print("-" * 60)
