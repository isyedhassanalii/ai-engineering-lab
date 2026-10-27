import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"

# temperature = how much randomness in the answer
# low (0-0.3): same answer almost every time, good for facts/code/grading
# medium (0.7-1): normal chat
# high (1.2-2): more creative but can go off track
# groq allows 0 to 2, default is 1
# using 0 here because I want consistent scoring
temperature = 0

# roles in messages:
# system - tells the model how to behave (set once at the start)
# user - what the person actually asks
# assistant - model's previous replies, needed if you want chat history
system_prompt = "You are a strict technical interviewer for an AI Engineer role. Evaluate the candidate's answer. Give a score out of 10, then list 2 mistakes or missing points, then show a better short answer."

user_prompt = "Question: What is an LLM?\nMy answer: An LLM is a big AI model that can chat like a human."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

response = client.chat.completions.create(
    messages=messages,
    model=model_name,
    temperature=temperature,
)

print("===== TEMPERATURE =====")
print(temperature)

print("\n===== SYSTEM =====")
print(system_prompt)

print("\n===== USER =====")
print(user_prompt)

print("\n===== RESPONSE =====")
print(response.choices[0].message.content)
