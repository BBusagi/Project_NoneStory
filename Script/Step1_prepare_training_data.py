import json
import re

# 用于预处理训练材料

input_file = "output/test12/batch_67ea849b157081909a71a289ffe6241c_output.jsonl"
output_file = "train_data_with_prompt.jsonl"

def extract_field(content, field_name):
    match = re.search(f"{field_name}：(.+)", content)
    return match.group(1).strip() if match else f"不明{field_name}"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            content = data["response"]["body"]["choices"][0]["message"]["content"]

            # 元信息提取
            genre = extract_field(content, "ジャンル")
            style = extract_field(content, "文体分類")
            purpose = "N1単語を多く含む"
            base_prompt = f"以下は{genre}ジャンルの{style}文体で書かれた、{purpose}短編小説です："

            # 正文拆分
            upper_match = re.search(
                r"本文（上半分）：\s*(.*?)(?=\n+本文（下半分）：)", content, re.DOTALL
            )

            lower_match = re.search(
                r"本文（下半分）：\s*【前情提要】：\s*(.*?)\s*\n+(.*)", content, re.DOTALL
            )

            if upper_match and lower_match:
                upper_text = upper_match.group(1).strip()
                summary = lower_match.group(1).strip()
                lower_text = lower_match.group(2).strip()

                # 上半部分训练样本
                record_upper = {
                    "prompt": base_prompt + "\nこれは短編小説の前半部分です：",
                    "completion": upper_text
                }

                # 下半部分训练样本（含前情提要）
                record_lower = {
                    "prompt": base_prompt + f"\nこれは短編小説の後半部分です。\n【前情提要】：{summary}",
                    "completion": lower_text
                }

                outfile.write(json.dumps(record_upper, ensure_ascii=False) + "\n")
                outfile.write(json.dumps(record_lower, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"处理失败：{e}")
