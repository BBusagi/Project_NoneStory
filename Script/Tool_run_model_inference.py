from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ===== 模型路径（你保存好的模型） =====
MODEL_PATH = "sft-model-n1/medium/final_model"  # 改成你的路径

# ===== 加载模型和 tokenizer =====
print(f"📦 加载模型中：{MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("✅ 模型加载完成，输入日语 Prompt 后按回车生成（输入 q 退出）")

# ===== 交互式输入循环 =====
while True:
    prompt = input("\n🟢 Prompt >>> ").strip()
    if prompt.lower() in ["q", "quit", "exit"]:
        print("👋 退出交互测试")
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("📘 生成结果：", result)
