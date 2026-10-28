import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"

# small, medium and big prompt to see how tokens change
prompt1 = "What is a token?"
prompt2 = "Explain how LLMs generate text, step by step"
prompt3 = "Write a detailed guide on prompt engineering with examples for beginners"
prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=50  # limit output, long answers will get cut
    )

    # prompt = my input, completion = model reply, total = both
    usage = response.usage
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    # stop = finished normally, length = hit max_tokens
    print(f"Finish Reason: {response.choices[0].finish_reason}")
    print("-" * 40)
    



# print(response.choices[0].message.content)
