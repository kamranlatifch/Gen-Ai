import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

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


# ---------------------------------------------------------------- tool 2 ---


def add_duration_to_datetime(datetime_str, duration=0, unit="days"):
    """Add a duration to a date and return the result in a readable format."""
    date = datetime.strptime(datetime_str, "%Y-%m-%d")

    if unit == "months":
        # timedelta has no months, because months are not a fixed length.
        month_index = date.month - 1 + int(duration)
        year = date.year + month_index // 12
        month = month_index % 12 + 1
        day = min(date.day, _days_in_month(year, month))
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        new_date = date.replace(year=date.year + int(duration))
    elif unit in {"seconds", "minutes", "hours", "days", "weeks"}:
        new_date = date + timedelta(**{unit: duration})
    else:
        return f"Error: unsupported unit {unit!r}."

    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


def _days_in_month(year, month):
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]


add_duration_to_datetime_schema = {
    "type": "function",
    "function": {
        "name": "add_duration_to_datetime",
        "description": (
            "Add a duration to a date and return the resulting date, including "
            "the day of the week. Use this for any question about a date in the "
            "future or the past, such as '90 days from now' or 'what weekday is "
            "6 months after 2025-01-15'. Handles varying month lengths and leap "
            "years correctly, so prefer it over calculating dates yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "datetime_str": {
                    "type": "string",
                    "description": "The starting date, formatted 'YYYY-MM-DD'.",
                },
                "duration": {
                    "type": "number",
                    "description": (
                        "How much time to add. Negative values go backwards in "
                        "time. Defaults to 0."
                    ),
                },
                "unit": {
                    "type": "string",
                    "description": (
                        "Unit for the duration. One of: seconds, minutes, hours, "
                        "days, weeks, months, years. Defaults to days."
                    ),
                },
            },
            "required": ["datetime_str"],
        },
    },
}


# ---------------------------------------------------------------- tool 3 ---

# WMO weather codes. The API returns a number; humans want words.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
}


def _fetch_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode())


def get_weather(place):
    """Look up the current weather for a place.

    Uses Open-Meteo, which is free and needs no API key. Two requests: one to
    turn a place name into coordinates, one to fetch the weather there.
    """
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": place, "count": 1}
    )
    matches = _fetch_json(geocode_url).get("results")
    if not matches:
        # Returned as text, not raised: the model reads this and can retry with
        # a better spelled or more specific place name.
        return f"No place found matching {place!r}. Try a city name."

    location = matches[0]
    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        }
    )
    current = _fetch_json(forecast_url)["current"]

    description = WEATHER_CODES.get(current["weather_code"], "unknown conditions")
    return (
        f"{location['name']}, {location.get('country', '')}: {description}, "
        f"{current['temperature_2m']} C, humidity {current['relative_humidity_2m']}%, "
        f"wind {current['wind_speed_10m']} km/h"
    )


get_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city or town: conditions, "
            "temperature in Celsius, humidity, and wind speed. Use this for any "
            "question about weather right now in a named place. Only accepts a "
            "place name, not coordinates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": (
                        "City or town name, for example 'Lahore', 'Paris', or "
                        "'San Francisco'. Add a country if the name is ambiguous."
                    ),
                },
            },
            "required": ["place"],
        },
    },
}


# The menu the model sees, and the map from a requested name back to real code.
# Both must list every tool: the schema tells the model a name exists, the dict
# is what makes it reachable. Add to one and forget the other, and the model
# will ask for a tool that cannot be found.
TOOLS = [
    get_current_date_and_time_schema,
    add_duration_to_datetime_schema,
    get_weather_schema,
]

TOOL_FUNCTIONS = {
    "get_current_date_and_time": get_current_date_and_time,
    "add_duration_to_datetime": add_duration_to_datetime,
    "get_weather": get_weather,
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
    # text. Some tools take none, in which case the model sends "{}".
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for {name}."

    print(f"[calling {name} with {arguments}]")
    try:
        return str(TOOL_FUNCTIONS[name](**arguments))
    except Exception as error:
        # Hand the failure back as text instead of crashing. A wrong argument
        # name or an unreachable API is something the model can react to.
        return f"Error running {name}: {error}"


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
    QUESTIONS = [
        # One tool.
        """What is the weather in:
        * Lahore right now?
        * Islamabad right now?""",
        # Two tools, where the second needs the first one's answer: the model
        # cannot add 100 days to today until it asks what today is.
    #     "What day of the week is 100 days from today?",

    #    "What is capital of Pakistan?",
    #     # Two tools that do not depend on each other, so the model can request
    #     # both in one reply and they run in the same turn.
    #     "What is the weather in Paris, and what is today's date?",

    ]

    for question in QUESTIONS:
        print(f"\n=== {question}")
        print(run_conversation(question))
    # messages = []
    # add_user_message(messages, "What is the weather in Paris, and what is today's date??")
    # print(chat(messages))
