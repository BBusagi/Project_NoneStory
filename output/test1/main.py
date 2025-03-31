import pandas as pd
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ✅ 你的 OpenAI API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 词汇表文件路径
wordlist_path = r"E:\PROJECTS\Git\Project_NoneStory\resource\WordList_N1_10.xlsx"
save_dir = r"E:\PROJECTS\Git\Project_NoneStory\output"
os.makedirs(save_dir, exist_ok=True)

# ✅ 读取 Excel 文件
df = pd.read_excel(wordlist_path, header=None)

# ✅ Prompt 构造函数
def build_prompt(word_list):
    word_string = "、".join(word_list)
    return f"""
以下の語彙を使って、日本語能力試験N1レベルの短編小説を作成してください：

語彙リスト：
- {word_string}

ジャンル：現代都市 + 異能（超能力）
長さ：900文字前後
文体：自然で文学的、会話も含めてください
結末：少し謎が残る終わり方にしてください

出力形式：
タイトル：<タイトル>
本文：
<本文>
使われたN1語彙：
- ...
""".strip()

# ✅ 主处理流程
for i in tqdm(range(1, 11)):
    words = df.iloc[i].dropna().astype(str).tolist()
    prompt = build_prompt(words)
    print(f"第 {i+1}  {prompt}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは日本語の小説作家です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=1200  # 为900字左右预留足够空间
        )

        content = response.choices[0].message.content.strip()

        # 保存为 txt 文件，编号从 01 开始
        file_name = f"story_{i+1:02d}.txt"
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    except Exception as e:
        print(f"❌ 第 {i+1} 行生成失败: {e}❌")

print("✅✅✅ 所有故事已生成完毕！✅✅✅")
