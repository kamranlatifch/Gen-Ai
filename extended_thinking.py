"""Extended thinking = the model thinks first, then answers.

Normal chat:  question -> answer
This:         question -> thinking (hidden) -> answer (shown)
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


def ask(prompt):
    """Send a question and print thinking, then the answer, as they arrive."""
    messages = [{"role": "user", "content": prompt}]
    options = {
        "messages": messages,
        "stream": True,
        "max_tokens": 4096,
    }

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            extra_body={"reasoning": {"effort": "high"}},
            **options,
        )
    except (RateLimitError, APIStatusError):
        print("(OpenRouter unavailable, using Groq...)")
        stream = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            extra_body={"reasoning_effort": "high"},
            **options,
        )

    in_answer = False
    print("THINKING (live):")

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        thinking = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        if thinking:
            print(thinking, end="", flush=True)

        if delta.content:
            if not in_answer:
                print("\n\nANSWER (live):")
                in_answer = True
            print(delta.content, end="", flush=True)

    print()


if __name__ == "__main__":
    question = """
You must solve this exactly. Do not skip any step. Recheck every count twice.

The string is:
strawberrydsefrrfrrrggbrrgrggrrererwdfvvrrrbvrr

Rules:
1. Count every letter 'r' or 'R' in the string. Call this N.
2. Count how many times two or more r's sit next to each other (rr, rrr, rrrr...).
   Call the number of such groups G. A run of 4 r's is ONE group, not four.
3. For each run of r's, write its length. Sum those lengths. This must equal N.
   If it does not, start over.
4. Count the letters that are NOT r. Call this M.
5. Compute: ((N * G) + M) modulo 17. Call this X.
6. Then compute: X squared, minus N, plus the number of distinct letters
   in the whole string. Call this Y.
7. Finally, is Y a prime number? Yes or no.

Show N, G, the list of run lengths, M, X, Y, and the prime answer.
After you finish, recount N from scratch once more and confirm it matches.
"""

    ask(question)
