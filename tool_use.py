"""Tool use: letting the model call our Python functions.

A model on its own can only produce text. It cannot know today's date, do
reliable arithmetic, or cause anything to happen. Tools fix that: we describe
some functions, and when the model needs one it replies with a request to call
it. We run the function ourselves and hand back the result.

The model never runs any code. It only ever asks. The loop looks like this:

    user message -> model asks for a tool -> we run it -> we send the result
                 -> model answers (or asks for another tool)

This is the Anthropic tools lesson rewritten for OpenRouter and Groq, which
speak the OpenAI format. Three things differ from the Anthropic version:

    Anthropic                        OpenAI-compatible
    ---------                        -----------------
    "input_schema"                   "parameters", wrapped in a "function" key
    content blocks of type tool_use  message.tool_calls
    tool_result block in a user msg  a message with role "tool"
"""

import json
from datetime import datetime, timedelta

from openai import RateLimitError

from generate_datasets import GROQ_MODEL, MODEL, client, groq_client


# --------------------------------------------------------------------------
# The tools themselves - ordinary Python functions, nothing special about them
# --------------------------------------------------------------------------


def add_duration_to_datetime(datetime_str, duration=0, unit="days", input_format="%Y-%m-%d"):
    """Add a duration to a datetime string and return it in a readable format."""
    date = datetime.strptime(datetime_str, input_format)

    if unit == "seconds":
        new_date = date + timedelta(seconds=duration)
    elif unit == "minutes":
        new_date = date + timedelta(minutes=duration)
    elif unit == "hours":
        new_date = date + timedelta(hours=duration)
    elif unit == "days":
        new_date = date + timedelta(days=duration)
    elif unit == "weeks":
        new_date = date + timedelta(weeks=duration)
    elif unit == "months":
        month = date.month + duration
        year = date.year + month // 12
        month = month % 12
        if month == 0:
            month = 12
            year -= 1
        days_in_month = [31, 29 if _is_leap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day = min(date.day, days_in_month[month - 1])
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        new_date = date.replace(year=date.year + duration)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


def _is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def set_reminder(content, timestamp):
    """Stand-in for a real reminder service."""
    print(f"----\nSetting the following reminder for {timestamp}:\n{content}\n----")
    return f"Reminder set for {timestamp}."


# --------------------------------------------------------------------------
# Schemas - how we describe those functions to the model
# --------------------------------------------------------------------------

# The description is not documentation, it is the instruction manual the model
# reads to decide whether and how to call this. A vague description is the most
# common reason a model picks the wrong tool or fills in the wrong arguments,
# so these are deliberately long.

ADD_DURATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_duration_to_datetime",
        "description": (
            "Adds a specified duration to a datetime string and returns the resulting "
            "datetime in a detailed format. This tool converts an input datetime string "
            "to a Python datetime object, adds the specified duration in the requested "
            "unit, and returns a formatted string of the resulting datetime. It handles "
            "seconds, minutes, hours, days, weeks, months, and years, with special "
            "handling for month and year calculations to account for varying month "
            "lengths and leap years. The output always includes the day of the week, "
            "month name, day, year, and time with an AM/PM indicator, for example "
            "'Thursday, April 03, 2025 10:30:00 AM'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "datetime_str": {
                    "type": "string",
                    "description": (
                        "The input datetime string to which the duration will be added. "
                        "Formatted according to the input_format parameter."
                    ),
                },
                "duration": {
                    "type": "number",
                    "description": (
                        "The amount of time to add. Can be positive for future dates or "
                        "negative for past dates. Defaults to 0."
                    ),
                },
                "unit": {
                    "type": "string",
                    "description": (
                        "The unit of time for the duration. Must be one of: 'seconds', "
                        "'minutes', 'hours', 'days', 'weeks', 'months', 'years'. "
                        "Defaults to 'days'."
                    ),
                },
                "input_format": {
                    "type": "string",
                    "description": (
                        "The format string for parsing datetime_str, using Python's "
                        "strptime codes. For example '%Y-%m-%d' for dates like "
                        "'2025-04-03'. Defaults to '%Y-%m-%d'."
                    ),
                },
            },
            "required": ["datetime_str"],
        },
    },
}

SET_REMINDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_reminder",
        "description": (
            "Creates a timed reminder that will notify the user at the specified time "
            "with the provided content. Use this when a user wants to be reminded about "
            "something specific at a future point in time, such as meetings, tasks, "
            "medication schedules, or any other time-bound activity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The message shown in the reminder notification, such as 'Take "
                        "medication' or 'Join video call with team'."
                    ),
                },
                "timestamp": {
                    "type": "string",
                    "description": (
                        "When the reminder should fire, as an ISO 8601 timestamp "
                        "(YYYY-MM-DDTHH:MM:SS). Timezones are handled internally."
                    ),
                },
            },
            "required": ["content", "timestamp"],
        },
    },
}

TOOLS = [ADD_DURATION_SCHEMA, SET_REMINDER_SCHEMA]

# Maps the name the model asks for to the function we actually run. Without
# this, a model could only ever name functions, never reach them.
TOOL_FUNCTIONS = {
    "add_duration_to_datetime": add_duration_to_datetime,
    "set_reminder": set_reminder,
}


# --------------------------------------------------------------------------
# Talking to the model
# --------------------------------------------------------------------------


def chat(messages, tools=None, temperature=0.0):
    """Send the whole conversation and return the model's message object."""
    options = {"messages": messages, "temperature": temperature}
    if tools:
        options["tools"] = tools

    try:
        response = client.chat.completions.create(model=MODEL, **options)
    except RateLimitError:
        print("(OpenRouter limit reached, using Groq...)")
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL, max_tokens=4096, extra_body={"reasoning_effort": "low"}, **options
        )
    return response.choices[0].message


def run_tool(tool_call):
    """Run one tool the model asked for and return its result as a string."""
    name = tool_call.function.name

    # Arguments arrive as a JSON string, not a dict, because they travelled
    # over the wire as text. They are also model-generated, so they can be
    # malformed or name a tool that does not exist.
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for {name}."

    if name not in TOOL_FUNCTIONS:
        return f"Error: no tool named {name}."

    print(f"[calling {name} with {arguments}]")
    try:
        return str(TOOL_FUNCTIONS[name](**arguments))
    except Exception as error:
        # Hand the error back to the model rather than crashing. It can often
        # read the message and retry with corrected arguments.
        return f"Error running {name}: {error}"


def run_conversation(user_text, max_turns=5):
    """Talk to the model, running any tools it asks for, until it answers."""
    today = datetime.now().strftime("%Y-%m-%d") #Current Date Time
    messages = [
        # The model has no idea what day it is, so tell it. Without this it
        # invents a date and every relative calculation is wrong.
        {"role": "system", "content": f"You are a helpful assistant. Today's date is {today}."},
        {"role": "user", "content": user_text},
    ]

    for _ in range(max_turns):
        message = chat(messages, tools=TOOLS)

        # Rebuild the assistant message by hand instead of passing the SDK
        # object back. Some providers attach extra fields that others reject.
        assistant_message = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_message["tool_calls"] = [
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
        messages.append(assistant_message)

        # No tool requested means the model is done and this is the answer.
        if not message.tool_calls:
            return message.content

        # A model can request several tools at once, so run them all. Each
        # result goes back in its own message, tagged with the id of the call
        # it answers, which is how the model matches them up.
        for call in message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": run_tool(call),
                }
            )

    return "Gave up: the model kept asking for tools without producing an answer."


if __name__ == "__main__":
    QUESTIONS = [
        "What day of the week is 90 days from 2025-04-03?",
        "Set a reminder for my dentist appointment 3 days from today at 2pm.",
    ]

    for question in QUESTIONS:
        print(f"\n=== {question}")
        print(run_conversation(question))
