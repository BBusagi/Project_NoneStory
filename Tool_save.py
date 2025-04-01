from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载最后一个 checkpoint
ckpt_path = "./sft-model-n1/medium/checkpoint-2880"
output_path = "./sft-model-n1/medium/final_model"

model = AutoModelForCausalLM.from_pretrained(ckpt_path)
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

# 保存为最终模型目录
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"✅ 模型已成功保存到：{output_path}")
