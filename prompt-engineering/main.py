import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"

# bad prompt - too vague, model doesn't know who I am or what I want
# so it just dumps everything it knows
bad_prompt = "Tell me about AI use cases"

# how to write a good prompt:
# 1. role - who the model should act like, e.g. "you are an AI consultant". goes in system prompt
# 2. task - be specific, "give me 3 ideas" works way better than "tell me about"
# 3. constraints - word limit, no coding, how detailed etc
# 4. output format - show how you want the answer, otherwise it makes up its own format
# 5. zero/one shot - zero shot = just the instruction, no example
#    one shot = give it one example answer and it follows the same style
# 6. fallback - tell it what to do if it doesn't know, like "just say you don't know"
#    otherwise it might make stuff up
good_system_prompt = "You are an AI consultant who explains things simply to small business owners."

good_user_prompt = """I have a small online clothing store. it's just me and one other person and we don't know coding.

Can you give me 3 ways to use AI in my business?

Rules:
- no tools that need coding
- should be easy to set up within a week
- max 50 words for each idea

Answer like this:
1. idea name
   - what it does:
   - tool:
   - benefit:"""


def get_response(messages):
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    return response.choices[0].message.content


bad_messages = [
    {"role": "user", "content": bad_prompt},
]

good_messages = [
    {"role": "system", "content": good_system_prompt},
    {"role": "user", "content": good_user_prompt},
]

print("===== BAD PROMPT =====")
print(bad_prompt)

print("\n===== BAD RESPONSE =====")
print(get_response(bad_messages))

print("\n===== GOOD PROMPT =====")
print(good_system_prompt)
print(good_user_prompt)

print("\n===== GOOD RESPONSE =====")
print(get_response(good_messages))
