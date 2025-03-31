from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 替换为你的 checkpoint 路径
model_path = "./sft-model-n1/checkpoint-375"
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small", use_fast=False)
tokenizer.save_pretrained("./sft-model-n1/checkpoint-375")
model = AutoModelForCausalLM.from_pretrained(model_path)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "以下は恋愛ジャンルの文学風文体で書かれた、N1単語を多く含む短編小説です："
result = generator(
    prompt,
    max_new_tokens=1500,         # ✅ 明确指定“生成多少 tokens”
    early_stopping=False,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    repetition_penalty=1.2,     # 可防止重复词
    eos_token_id=tokenizer.eos_token_id  # 明确模型停在哪
)

print(result[0]["generated_text"])
