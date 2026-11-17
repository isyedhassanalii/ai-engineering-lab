import os
from time import sleep
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"

# prompt chaining
# basically instead of one big prompt, break the task into small steps
# output of one step goes as input to the next one
# good thing is I can print each step and see where it messed up

# review -> pull out complaints -> tag them -> reply to customer

restaurant_name = "Karachi Biryani House"

review = """Ordered 2 biryani and 1 zinger burger last night.
Delivery took almost 1 hour and the food was cold when it arrived.
Biryani was too salty and the burger was missing the cheese I paid extra for.
The rider was rude when I asked about the delay.
For 1550 rupees I expected much better."""


def get_response(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=0,
    )
    return response.choices[0].message.content


def get_complaints(review):
    system_prompt = """You work in customer support for a restaurant.
Read the review and list all the complaints.

Rules:
- only what the customer actually wrote, don't add anything
- one complaint per line, keep it short
- just the list, nothing else

Answer like this:
- complaint
- complaint"""

    user_prompt = "Review:\n" + review

    return get_response(system_prompt, user_prompt)


def tag_complaints(complaints):
    system_prompt = """You work in customer support for a restaurant.
Give each complaint a category: Food, Delivery, Staff or Price.

Rules:
- don't use any other category
- just the list, nothing else

Answer like this:
- complaint -> Category"""

    user_prompt = "Complaints:\n" + complaints

    return get_response(system_prompt, user_prompt)


def write_reply(review, tagged):
    system_prompt = f"""You are the support manager of {restaurant_name}.
Write a reply to this customer.

Rules:
- say sorry for each problem they had
- give them 20% off with code SORRY20
- keep it simple and polite, max 120 words
- don't promise anything else
- end with "Team {restaurant_name}", no placeholders like [Name]"""

    user_prompt = f"Review:\n{review}\n\nComplaints:\n{tagged}"

    return get_response(system_prompt, user_prompt)


print("===== REVIEW =====")
print(review)

complaints = get_complaints(review)
print("\n===== STEP 1 - COMPLAINTS =====")
print(complaints)

sleep(2) 

tagged = tag_complaints(complaints)
print("\n===== STEP 2 - TAGGED =====")
print(tagged)

sleep(2)

reply = write_reply(review, tagged)
print("\n===== STEP 3 - REPLY =====")
print(reply)
