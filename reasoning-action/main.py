import os
import re
from time import sleep
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "qwen/qwen3.8-27b"

# ReAct = reason + act
# Thought -> Action -> Observation, repeat until Final Answer


def get_food_price(item):
    item = item.lower()

    if item == "biryani":
        return 450
    elif item == "zinger burger":
        return 650
    elif item == "pizza":
        return 1200
    elif item == "cold drink":
        return 150
    else:
        return "item not on menu"


def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc error!"


tools = {
    "get_food_price": get_food_price,
    "calculator": calculator,
}

system_prompt = """You are a food ordering assistant for a restaurant.

You have these tools:
get_food_price(item)
calculator(expression)

Call tools exactly like these examples:
Action: get_food_price("Biryani")
Action: calculator("2 * 450 + 650")

Rules:
1. Call only one tool at a time.
2. After writing an Action, stop.
3. Never guess a price or a tool result.
4. Wait for the Observation, then decide the next step.
5. When the task is complete, give the Final Answer.

Format:
Thought: what you need to do
Action: tool_name(argument)

When finished:
Final Answer: your answer"""

user_prompt = """I have 2500 rupees. I want to order 2 biryani, 1 zinger burger and 2 cold drinks.
What is the total bill and how much money will I have left?"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

print("===== USER =====")
print(user_prompt)

for step in range(8):
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=0,
        stop=["Observation:"],
    )

    answer = response.choices[0].message.content

    print(f"\n===== STEP {step + 1} =====")
    print(answer)

    if "Final Answer:" in answer:
        break

    match = re.search(r"Action:\s*(\w+)\((.*?)\)", answer)

    if match:
        tool_name = match.group(1)
        tool_input = match.group(2).strip().strip('"')

        if tool_name in tools:
            observation = tools[tool_name](tool_input)
        else:
            observation = "Tool not found"
    else:
        observation = "No Action found, use the format"

    print("Observation:", observation)

    messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": "Observation: " + str(observation)})

    sleep(2)
