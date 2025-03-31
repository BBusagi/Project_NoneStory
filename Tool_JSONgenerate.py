import pandas as pd
import json
import random

# === 输入路径 ===
excel_path = "resource/WordList_N1_1000.xlsx"  # 词汇文件
output_path = "requests_batch_1000.jsonl"      # 输出地址

STYLES = {
    "文学風": "比喩や抽象的な表現を交え、心理描写を丁寧に行う落ち着いた文体",
    "評論風": "論理的で情報密度が高く、現代日本語の文章構造に沿った書き言葉の文体",
    "論述風": "N1レベルの語彙や文末表現、形式的な表現を意識し、試験や教材でも使える文体"
}

GENRES = ["学園", "異能力", "恋愛", "バトル", "ハーレム", "乙女"]

# === 读取词汇 ===
df = pd.read_excel(excel_path, header=None)

# === Prompt 构造函数 ===
def build_prompt(word_list):
    word_string = "、".join(word_list)
    selected_style = random.choice(list(STYLES.keys()))
    selected_genres = random.sample(GENRES, 2)
    genre_string = "×".join(selected_genres)
    style_desc = STYLES[selected_style]
    return f"""
以下のN1語彙を**できるだけ多く自然に使って**、日本語能力試験N1レベルの短編小説（1500文字前後）を作成してください。  
※文字数は1500字前後に収めてください。それを超えると無効となります。
※語彙の使用数が**30語未満の場合、その作品は無効とみなされます，スコア0と表記します。**

語彙リスト：
{word_string}

---
文体スタイル：今回は「{selected_style}」を使用してください。  
説明：{style_desc}  
※会話表現は必要に応じて挿入して構いませんが、全体としては書き言葉を基本としてください。
※構成や文体は自由ですが、安易なテンプレート形式にならないように工夫してください。
ジャンル（物語の題材）：今回は「{genre_string}」を組み合わせて物語を構成してください。  
---
出力形式：
タイトル：<タイトル>  
文体分類：{selected_style}  
ジャンル：{genre_string}
本文：  
<小説本文を出力>  
""".strip()

# === 构造每一行 JSON ===
requests = []

for i in range(1000):
    row = df.iloc[i]
    vocab_id = str(row[0]).strip()
    word_list = [str(w).strip() for w in row[1:] if pd.notna(w) and str(w).strip()]

    prompt = build_prompt(word_list)

    request_obj = {
        "custom_id": f"{vocab_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "temperature": 1.0,
            "max_tokens": 1400,
            "messages": [
                {"role": "system", "content": "あなたは日本語の小説作家です。"},
                {"role": "user", "content": prompt}
            ]
        }
    }

    requests.append(request_obj)

# === 写入 JSONL 文件 ===
with open(output_path, "w", encoding="utf-8") as f:
    for r in requests:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ 已保存到 {output_path}")
