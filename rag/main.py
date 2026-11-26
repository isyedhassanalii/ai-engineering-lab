import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"

# rag = retrieval augmented generation
# the model doesn't know anything about my restaurant
# so first I search my own data for the line that matches the question
# then send that line with the question so it answers from my data
# and doesn't make up prices

knowledge_base = [
    "Chicken biryani is 650 rupees and beef biryani is 750 rupees.",
    "Zinger burger is 550 rupees. Extra cheese costs 100 rupees.",
    "We are open from 12pm to 12am, every day except Monday.",
    "Delivery is free above 1500 rupees, otherwise it is 150 rupees.",
]

def retrieve_info(question):
    words = question.lower().split()
    best_line = ""
    best_score = 0

    for line in knowledge_base:
        score = 0
        for word in words:
            if word in line.lower():
                score += 1
        if score > best_score:
            best_score = score
            best_line = line

    return best_line


question = "How much is the zinger burger?"
context = retrieve_info(question)

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": f"Answer only from this info:\n{context}"},
        {"role": "user", "content": question},
    ],
    model=model_name,
)

print(f"Question: {question}")
print(f"Found: {context}")
print(f"Answer: {response.choices[0].message.content}")
