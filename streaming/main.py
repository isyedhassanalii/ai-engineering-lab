import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"

# asking the same thing twice to see the difference
prompt = "Write a short story about a cricket match in Lahore"
messages = [{"role": "user", "content": prompt}]

print("===== WITHOUT STREAMING =====")
response = client.chat.completions.create(
    messages=messages,
    model=model_name,
)
print(response.choices[0].message.content)

print("\n===== WITH STREAMING =====")
stream = client.chat.completions.create(
    messages=messages,
    model=model_name,
    stream=True,  # answer comes bit by bit
)

for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)

print()
