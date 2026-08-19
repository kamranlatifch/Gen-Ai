import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

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

# Groq is also OpenAI-SDK-compatible. Used as a fallback when OpenRouter's
# free-tier daily rate limit (50 req/day) is hit. Get a free key at:
# https://console.groq.com/keys
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# OpenRouter/OpenAI's chat.completions API has no top-level `system` param
# (unlike Anthropic's Messages API) — the system prompt is just the first
# message in the list, with role "system".
SYSTEM_PROMPT = (
    "You are a patient math tutor. Do not directly answer a student's "
    "question. Guide them to a solution step by step."
)


def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def _stream_and_collect(stream):
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_text += delta
    print()
    return full_text


def chat(messages, temperature=0.7):
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        return _stream_and_collect(stream)
    except RateLimitError:
        print("(OpenRouter rate limit hit, falling back to Groq...)")
        stream = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        return _stream_and_collect(stream)


if __name__ == "__main__":
    messages = [
        # {"role": "system", "content": SYSTEM_PROMPT}
        ]

    while True:
        # 1. Prompt the user to enter some input
        user_input = input("You: ")

        # 2. Add it to the list of messages
        add_user_message(messages, user_input)

        # 3. Call the API (streams tokens to stdout as they arrive)
        print("Bot: ", end="")
        answer = chat(messages, temperature=0.1)
        print()

        # 4. Add generated text to the list of messages
        add_assistant_message(messages, answer)

        # 5. Repeat from #1
