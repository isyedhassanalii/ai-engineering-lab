import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"

messages = [
    {
        "role": "user",
        "content": "Give 3 real-world use cases of AI",
    },
]

response = client.chat.completions.create(
    messages=messages,
    model=model_name,
)

print(response.choices[0].message.content)
