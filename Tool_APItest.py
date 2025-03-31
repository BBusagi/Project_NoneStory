import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "短い日本語の物語を書いてください。"}
    ]
)

print("✅ 応答：\n")
print(response.choices[0].message.content)
