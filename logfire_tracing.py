"""Lesson — Tracing with Pydantic Logfire.

DeepEval told you *how good* an answer was.
Logfire tells you *what happened* while producing it: timing, nesting, LLM
calls, token use, errors — as a tree of spans.

Vocabulary (memorize these two):

    TRACE  = one full request journey (the whole tree)
    SPAN   = one timed unit of work inside that tree (a node)

Example tree for a tiny RAG step:

    answer_question                  ← parent span (whole request)
      ├── retrieve_chunks            ← child span
      └── chat.completions           ← child span (auto, via instrument_openai)

Setup
-----
Works offline in the terminal with console traces. For the Logfire web UI:

    pip install -r requirements.txt
    logfire auth                 # browser login (once)
    logfire projects use <name>  # pick/create a project
    python logfire_tracing.py

Then open the Live view in https://logfire.pydantic.dev
"""

from __future__ import annotations

import os
import time

import logfire
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# 1) Connect Logfire (console always on; cloud only if you authenticated).
logfire.configure(
    service_name="logfire-tracing-lesson",
    send_to_logfire="if-token-present",
    inspect_arguments=False,  # quieter on some Python builds
)

# 2) Groq via OpenAI-compatible SDK — same pattern as the rest of this repo.
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

# 3) Auto-wrap every chat/embeddings call on THIS client with an LLM span.
logfire.instrument_openai(client)


# Fake "retriever" so the lesson shows nested spans without Voyage/vector DB.
FAKE_CHUNKS = [
    "INC-2023-Q4-011: eng patched ERR_MEM_ALLOC_FAIL; 40% fewer critical failures.",
    "XDR-471: medical research on Gene LOC73b (unrelated to the eng incident).",
]


def retrieve(query: str) -> list[str]:
    """Pretend search — still wrap it in a span so you see timing in the tree."""
    with logfire.span("retrieve_chunks", query=query):
        time.sleep(0.05)  # fake latency so the span has a visible duration
        # Prefer the eng chunk for this demo query.
        hits = [FAKE_CHUNKS[0]]
        logfire.info("retrieved {n} chunks", n=len(hits))
        return hits


def generate_answer(query: str, chunks: list[str]) -> str:
    """LLM call — Logfire already instruments client.chat.completions.create."""
    context = "\n".join(chunks)
    messages = [
        {
            "role": "system",
            "content": "Answer using only the provided context. Be one short sentence.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]
    # No manual span required here — instrument_openai adds "Chat Completion ...".
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
        extra_body={"reasoning_effort": "low"},
    )
    return response.choices[0].message.content or ""


def answer_question(query: str) -> str:
    """Parent span = one user request. Children nest under it automatically."""
    with logfire.span("answer_question", query=query):
        chunks = retrieve(query)
        answer = generate_answer(query, chunks)
        logfire.info("answer ready: {answer}", answer=answer)
        return answer


def demo_manual_only():
    """Part A — spans with no LLM. Read the indented tree in your terminal."""
    print("\n=== Part A: manual spans only ===\n")
    with logfire.span("manual_demo"):
        logfire.info("hello from inside the parent span")
        with logfire.span("step_one"):
            time.sleep(0.02)
            logfire.info("did step one")
        with logfire.span("step_two"):
            time.sleep(0.02)
            logfire.info("did step two")


def demo_traced_rag():
    """Part B — parent span + retrieve span + auto LLM span."""
    print("\n=== Part B: traced mini-RAG (Groq) ===\n")
    query = "What did eng do for INC-2023-Q4-011?"
    answer = answer_question(query)
    print(f"Q: {query}")
    print(f"A: {answer}")


if __name__ == "__main__":
    demo_manual_only()
    demo_traced_rag()
    print(
        "\nLook at the indented tree above: parent spans wrap children.\n"
        "If you ran `logfire auth`, the same tree appears in Live view.\n"
    )
