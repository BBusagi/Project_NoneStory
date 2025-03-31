import json

input_file = "output/test6-train1/batch_67e8b406a21c8190a335ea0ac42e75a3_output.jsonl"
output_file = "train_data_with_prompt.jsonl"

# """提取像 'ジャンル：恋愛・異能力' 的字段值"""
def extract_field(content, field_name):   
    if f"{field_name}：" in content:
        return content.split(f"{field_name}：")[1].split("\n")[0].strip()
    return None

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        try:
            # 提取正文内容
            content = data["response"]["body"]["choices"][0]["message"]["content"]    

            # 提取元信息
            genre = extract_field(content, "ジャンル") or "不明ジャンル"
            style = extract_field(content, "文体分類") or "不明文体"
            purpose = "N1単語を多く含む"

            # 生成prompt文本
            prompt = f"以下は{genre}ジャンルの{style}文体で書かれた、{purpose}短編小説です："

            if "本文：" in content:
                content = content.split("本文：")[1]
            # 去掉评分
            content = content.split("語彙")[0].strip()
            
            # 写入训练样本
            record = {
                "prompt": prompt,
                "completion": content
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"处理失败：{e}")
