import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=groq_api_key)

model_name = "openai/gpt-oss-20b"


class UseCase(BaseModel):
    title: str
    industry: str
    description: str


messages = [
    {
        "role": "user",
        "content": "Give 1 real-world use case of AI",
    },
]

response = client.chat.completions.create(
    messages=messages,
    model=model_name,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "use_case",
            "schema": UseCase.model_json_schema(),
        },
    },
)

use_case = UseCase.model_validate_json(response.choices[0].message.content)

print(use_case.title)
print(use_case.industry)
print(use_case.description)
