import os
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"

# embed rag = rag that searches by meaning instead of by words
# my first rag only counted matching words, so if the question used
# different words than my data, it picked the wrong line
# here every line is turned into an embedding (a list of numbers)
# lines with similar meaning get similar numbers
# so "what time do you close" still finds the opening hours line

# small free model, gives 384 numbers for each text
model = SentenceTransformer("all-MiniLM-L6-v2")


# checks how close two embeddings are
# near 1 means same meaning, near 0 means not related
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


knowledge_base = [
    "Chicken biryani is 650 rupees and beef biryani is 750 rupees.",
    "Zinger burger is 550 rupees. Extra cheese costs 100 rupees.",
    "We are open from 12pm to 12am, every day except Monday.",
    "Delivery is free above 1500 rupees, otherwise it is 150 rupees.",
]

# embed all lines one time at the start
vectors = [model.encode(line) for line in knowledge_base]


# embed the question and return the line closest to it
def retrieve_info(question):
    q = model.encode(question)
    best_line = ""
    best_score = -1

    for i in range(len(knowledge_base)):
        score = cosine_similarity(q, vectors[i])
        if score > best_score:
            best_score = score
            best_line = knowledge_base[i]

    return best_line


question = "What time do you close at night?"
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
