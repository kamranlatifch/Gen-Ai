import json
import os
from datetime import datetime

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


# ////////////////////////////////////////////////////////////////
# Tool use
# ////////////////////////////////////////////////////////////////


def get_current_date_and_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# "parameters" describes the arguments the model has to supply when it calls
# this tool - not what the tool gives back. get_current_date_and_time() takes
# no arguments, so "properties" is empty. Declaring a "date" property here
# would tell the model to invent a date and pass it in, which is backwards:
# the date is the answer, and inventing it is exactly what the tool prevents.
get_current_date_and_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_date_and_time",
        "description": (
            "Get the current date and time on the user's machine, as a string "
            "formatted 'YYYY-MM-DD HH:MM:SS'. Call this whenever answering "
            "needs to know what day or time it is now, including questions "
            "about today, tomorrow, or how long until some future date."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

TOOLS = [get_current_date_and_time_schema]

# Maps the name the model asks for to the function we actually run.
TOOL_FUNCTIONS = {
    "get_current_date_and_time": get_current_date_and_time,
}


def chat_with_tools(messages, temperature=0.0):
    """Like chat(), but not streamed, and with the tools attached.

    Streaming and tool use do not mix comfortably: a streamed response arrives
    as text fragments, while a tool call arrives as structured data. This
    version waits for the whole reply so we can check message.tool_calls.
    """
    options = {"messages": messages, "temperature": temperature, "tools": TOOLS}
    try:
        return client.chat.completions.create(model=MODEL, **options).choices[0].message
    except RateLimitError:
        print("(OpenRouter rate limit hit, falling back to Groq...)")
        return (
            groq_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=4096,
                extra_body={"reasoning_effort": "low"},
                **options,
            )
            .choices[0]
            .message
        )


def add_assistant_tool_calls(messages, message):
    """Record what the model asked for, so it can see its own request later."""
    entry = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    messages.append(entry)


def run_tool(tool_call):
    """Run one tool the model asked for and return the result as a string."""
    name = tool_call.function.name
    if name not in TOOL_FUNCTIONS:
        return f"Error: no tool named {name}."

    # Arguments arrive as a JSON string because they travelled over the wire as
    # text. This tool takes none, but the model still sends "{}".
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for {name}."

    print(f"[calling {name}]")
    return str(TOOL_FUNCTIONS[name](**arguments))


def run_conversation(user_text, max_turns=5):
    """Ask a question, run whatever tools the model requests, return the answer."""
    messages = []
    add_user_message(messages, user_text)

    for _ in range(max_turns):
        message = chat_with_tools(messages)
        add_assistant_tool_calls(messages, message)

        # No tool requested means the model is finished and this is the answer.
        if not message.tool_calls:
            return message.content

        # Each result goes back in its own message, tagged with the id of the
        # call it answers, which is how the model matches them up.
        for call in message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": run_tool(call),
                }
            )

    return "Gave up: the model kept asking for tools without answering."


if __name__ == "__main__":
    # messages = [
    #     {"role": "user", "content": "What is the date and time right now?"},
    # ]
    # print(chat(messages))
    print(run_conversation("What is the date and time right now?"))
    