import os
from dotenv import load_dotenv
from openai import OpenAI

# Loads variables from a local .env file into the environment.
load_dotenv()

# OpenRouter speaks the OpenAI-compatible API, so we use the `openai` SDK
# but point it at OpenRouter's base_url and use an OpenRouter API key.
# Get a free key at: https://openrouter.ai/keys
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Pick any free model from https://openrouter.ai/models?max_price=0
# Set AI_MODEL in .env to override; falls back to the auto-router.
MODEL = os.environ.get("AI_MODEL", "openrouter/free")


def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    messages = []

    add_user_message(messages, "What is the capital of Pakistan?")
    answer = chat(messages)
    print(answer)

    add_assistant_message(messages, answer)
    add_user_message(messages, "What's its speciality?")
    answer = chat(messages)
    print(answer)
