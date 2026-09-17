"""Lesson — LangGraph orchestration: researcher + writer + human confirm.

Concepts wired into one runnable graph:

  State              shared notebook (topic, notes, draft, decision, ...)
  Checkpointing      InMemorySaver + thread_id (save-game between steps)
  Conditional branch enough notes?  /  approve vs rewrite?
  Human-in-the-loop  interrupt() pauses for your yes/no in the terminal

Flow:

  START → researcher → (enough notes?) → writer → human_review → (decision?)
              ↑______________|                         |
                    rewrite ←--------------------------┘

Run (needs GROQ_API_KEY in .env):

    pip install -r requirements.txt
    python langgraph_researcher_writer.py
"""

from __future__ import annotations

import os
import uuid
from typing import Literal, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Minimum note length before we allow writing (forces 1 extra research pass
# on very short first drafts — shows the research loop clearly).
MIN_NOTES_CHARS = 180
MAX_RESEARCH_ROUNDS = 2


# ---------------------------------------------------------------------------
# 1) STATE — the shared notebook every node reads/writes
# ---------------------------------------------------------------------------
class ArticleState(TypedDict):
    topic: str
    notes: str
    draft: str
    research_rounds: int
    decision: str  # "approve" | "rewrite" | "reject" | ""


def ask_llm(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=1024,
        extra_body={"reasoning_effort": "low"},
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# 2) NODES — each returns a partial state update
# ---------------------------------------------------------------------------
def researcher(state: ArticleState) -> dict:
    """Gather / expand notes about the topic."""
    round_n = state.get("research_rounds", 0) + 1
    existing = state.get("notes") or ""

    if existing:
        prompt = (
            f"Topic: {state['topic']}\n\nExisting notes:\n{existing}\n\n"
            "Add 2–3 NEW short bullet facts (do not repeat). Bullets only."
        )
    else:
        prompt = (
            f"Topic: {state['topic']}\n\n"
            "Write 3 short factual bullet points a writer could use. Bullets only."
        )

    new_bits = ask_llm(
        "You are a careful researcher. Be concise and concrete. No fluff.",
        prompt,
    )
    notes = (existing + "\n" + new_bits).strip() if existing else new_bits
    return {"notes": notes, "research_rounds": round_n}


def writer(state: ArticleState) -> dict:
    """Turn notes into a short draft paragraph."""
    draft = ask_llm(
        "You are a clear technical writer. One short paragraph. No title.",
        f"Topic: {state['topic']}\n\nNotes:\n{state['notes']}\n\nWrite the draft.",
    )
    return {"draft": draft, "decision": ""}


def human_review(state: ArticleState) -> dict:
    """Pause for a human. interrupt() requires a checkpointer.

    First time this node runs: raises interrupt with the draft payload.
    After you resume with Command(resume=...): that value is returned here.
    """
    decision = interrupt(
        {
            "message": "Type: approve | rewrite | reject",
            "topic": state["topic"],
            "draft": state["draft"],
        }
    )
    choice = str(decision).strip().lower()
    if choice not in {"approve", "rewrite", "reject"}:
        choice = "reject"
    return {"decision": choice}


# ---------------------------------------------------------------------------
# 3) CONDITIONAL BRANCHES — pick the next node from state
# ---------------------------------------------------------------------------
def route_after_research(
    state: ArticleState,
) -> Literal["researcher", "writer"]:
    notes = state.get("notes") or ""
    rounds = state.get("research_rounds", 0)
    if rounds < MAX_RESEARCH_ROUNDS and len(notes) < MIN_NOTES_CHARS:
        return "researcher"
    return "writer"


def route_after_human(
    state: ArticleState,
) -> Literal["writer", "__end__"]:
    if state.get("decision") == "rewrite":
        return "writer"
    return "__end__"


# ---------------------------------------------------------------------------
# 4) BUILD + COMPILE (checkpointing enabled)
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ArticleState)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("human_review", human_review)

    graph.add_edge(START, "researcher")
    graph.add_conditional_edges(
        "researcher",
        route_after_research,
        {"researcher": "researcher", "writer": "writer"},
    )
    graph.add_edge("writer", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"writer": "writer", "__end__": END},
    )

    # Without a checkpointer, interrupt() cannot pause/resume.
    return graph.compile(checkpointer=InMemorySaver())


def run_until_done(graph, topic: str) -> ArticleState:
    """Invoke, and whenever we hit an interrupt, ask the human in the terminal."""
    # thread_id = which save-game slot to use
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(
        {
            "topic": topic,
            "notes": "",
            "draft": "",
            "research_rounds": 0,
            "decision": "",
        },
        config=config,
    )

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n--- Review draft ---\n\n{payload['draft']}\n")
        print(payload["message"])
        user_input = input("> ").strip()
        # Resume: Command(resume=...) becomes the return value of interrupt()
        result = graph.invoke(Command(resume=user_input), config=config)

    return result


def main():
    graph = build_graph()
    topic = input(
        "Topic (or Enter for default):\n> "
    ).strip() or "Why checkpointing matters in LangGraph?"

    final = run_until_done(graph, topic)

    print("\n--- Done ---")
    print(f"Decision: {final.get('decision')}")
    print(f"\nNotes:\n{final.get('notes')}")
    print(f"\nDraft:\n{final.get('draft')}")


if __name__ == "__main__":
    main()
