"""Prompt caching: reuse a long prefix so later calls are cheaper and faster.

Every chat call resends the whole prompt. If 90% of it never changes
(system rules, tool schemas, a big document) you pay to re-read it every time.

Caching says: remember this prefix. Next call, skip the work you already did.

Three places to mark, in this order (stable first, changing last):

    1. tools          schemas almost never change
    2. system prompt  the rules almost never change
    3. user message   a long document, then a short question that *does* change

cache_control marks a breakpoint: "cache everything up to here".
Only the last 4 breakpoints count. Put them on the big stable blocks.

Groq caches some models automatically (no cache_control needed).
OpenRouter / Anthropic need the marker. Same idea either way.
"""

import os

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, RateLimitError

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = os.environ.get("AI_MODEL", "openrouter/free")

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# "Cache everything up to this block." TTL is 5 minutes by default.
CACHE = {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# 1) System prompt  —  long, same every call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful assistant. Use only the policy document the user "
    "provides. If the answer is not in the document, say you do not know.\n\n"
    # Caches need a minimum size (often ~1024 tokens). Pad so a hit is possible.
    + ("Company policy: keep answers short. Cite the section name. ") * 80
)


# ---------------------------------------------------------------------------
# 2) Tools  —  schemas are huge and almost never change
# Put cache_control on the LAST tool. That caches all tools up to that one.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_section",
            "description": "Look up one named section of the policy document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "Section title, e.g. 'Remote work'.",
                    }
                },
                "required": ["section"],
            },
        },
        "cache_control": CACHE,
    }
]


# ---------------------------------------------------------------------------
# 3) User message  —  long document (cached) + short question (not cached)
# ---------------------------------------------------------------------------

DOCUMENT = (
    "POLICY DOCUMENT\n"
    "Remote work: employees may work from home up to 3 days per week.\n"
    "Expenses: meals over $50 need a receipt and manager approval.\n"
    "Time off: request PTO at least 5 business days in advance.\n"
    + ("Additional notes: follow the three rules above in all cases. ") * 80
)


def messages_for(question):
    """Same system + same document. Only the question at the end changes."""
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": CACHE},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DOCUMENT, "cache_control": CACHE},
                {"type": "text", "text": question},
            ],
        },
    ]


def cached_tokens(usage):
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "cached_tokens", 0) or 0


def cache_write_tokens(usage):
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return getattr(details, "cache_write_tokens", 0) or 0


def ask(question):
    kwargs = {
        "messages": messages_for(question),
        "tools": TOOLS,
        "max_tokens": 200,
        # Keep us on the same provider so the cache stays warm.
        "extra_body": {"session_id": "prompt-caching-lesson"},
    }
    try:
        response = client.chat.completions.create(model=MODEL, **kwargs)
    except (RateLimitError, APIStatusError):
        print("(OpenRouter unavailable, using Groq...)")
        response = groq_client.chat.completions.create(model=GROQ_MODEL, **kwargs)
    return response


def report(label, response):
    usage = response.usage
    print(f"{label}")
    print(f"  prompt tokens : {usage.prompt_tokens}")
    print(f"  cache write   : {cache_write_tokens(usage)}")
    print(f"  cache read    : {cached_tokens(usage)}")
    print(f"  answer        : {response.choices[0].message.content}")
    print()


if __name__ == "__main__":
    print("Call 1 writes the cache. Call 2 should read it.\n")

    first = ask("How many remote days per week are allowed?")
    report("CALL 1 (cold)", first)

    second = ask("Do I need a receipt for a $60 lunch?")
    report("CALL 2 (warm, different question, same prefix)", second)

    if cached_tokens(second.usage):
        print("Cache hit. The long prefix was not billed at full price.")
    else:
        print(
            "No cache hit this run. Common reasons: the model ignores "
            "cache_control, the prefix is still too short, or Groq only "
            "auto-caches some models (Kimi K2). The request shape is still "
            "the lesson: mark tools, system, and the long user block."
        )
