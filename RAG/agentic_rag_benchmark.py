"""Agentic RAG benchmark: retrieval-as-tool + multi-hop, benchmarked against
naive RAG and a long-context "load it all" baseline, on the same questions.

Three systems, same questions, same document (report.md):

    long-context   no retrieval at all - the whole document goes in the prompt
    naive RAG      exactly one fixed retrieval call (k=3), then answer
    agentic RAG    the model gets a search tool and decides itself how many
                   times to call it and what to search for each time - this
                   is what makes multi-hop questions (answer split across two
                   sections) answerable

Each question comes with a checklist of facts a correct answer must contain
(same idea as prompt_evaluator.py's solution_criteria). A judge model checks
off each fact, so a question's score IS the fraction of required facts it
covered - for a multi-hop question, each fact is one hop, so the score tells
you directly which hop a system missed, not just a vague quality rating.

Run from inside RAG/:  python agentic_rag_benchmark.py
"""

import json
import math
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import voyageai
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from voyageai.error import RateLimitError as VoyageRateLimitError

load_dotenv()

# --------------------------------------------------------------------------
# LLM client - same OpenRouter-first, Groq-fallback pattern used throughout
# this repo (see generate_datasets.py / tool_use.py). Kept self-contained
# here, same as every other lesson file in RAG/.
# --------------------------------------------------------------------------

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))
MODEL = os.environ.get("AI_MODEL", "openrouter/free")

groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Once OpenRouter's daily limit is spent, every later call would fail too, so
# remember it and go straight to Groq (same trick as generate_datasets.py).
openrouter_exhausted = False


def ask(prompt, temperature=0.0):
    """One-shot: send a prompt, get plain text back. Used for the long-context
    and naive-RAG systems, and for grading - none of those need tools."""
    global openrouter_exhausted
    messages = [{"role": "user", "content": prompt}]

    if not openrouter_exhausted:
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
            return r.choices[0].message.content or ""
        except RateLimitError:
            print("(OpenRouter daily limit reached, using Groq from here on...)")
            openrouter_exhausted = True

    r = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        extra_body={"reasoning_effort": "low"},
    )
    return r.choices[0].message.content or ""


def chat_with_tools(messages, tools, temperature=0.0):
    """Same shape as tool_use.py's chat(), factored out so the agentic system
    below can call it in a loop."""
    options = {"messages": messages, "temperature": temperature, "tools": tools}
    try:
        return client.chat.completions.create(model=MODEL, **options).choices[0].message
    except RateLimitError:
        print("(OpenRouter rate limit hit, falling back to Groq...)")
        return (
            groq_client.chat.completions.create(
                model=GROQ_MODEL, max_tokens=1024, extra_body={"reasoning_effort": "low"}, **options
            )
            .choices[0]
            .message
        )


# --------------------------------------------------------------------------
# Retrieval - copied from multi_index_rag.py: chunk report.md by section,
# index it with a VectorIndex (VoyageAI embeddings) and a BM25Index, fuse
# their rankings with Reciprocal Rank Fusion. No LLM reranker here - that is
# a separate lesson (reranking.py); this benchmark is about the agent loop.
# --------------------------------------------------------------------------

embedding_client = voyageai.Client()

# VoyageAI's free tier (no payment method on file) caps requests at 3 per
# minute - low enough that this benchmark hits it easily once naive/agentic
# RAG start issuing one query embedding per search call. Throttle proactively
# (remember recent call times, sleep before we'd exceed the limit) instead of
# firing requests we know will be rejected, and keep a retry as a safety net
# in case the estimate is still a little tight.
_embed_call_times: List[float] = []
_VOYAGE_RPM_LIMIT = 3


def _throttle_for_voyage():
    now = time.time()
    recent = [t for t in _embed_call_times if now - t < 60]
    if len(recent) >= _VOYAGE_RPM_LIMIT:
        wait = 60 - (now - recent[0]) + 1
        print(f"(Voyage rate limit, waiting {wait:.0f}s...)")
        time.sleep(max(wait, 0))
        now = time.time()
        recent = [t for t in _embed_call_times if now - t < 60]
    recent.append(now)
    _embed_call_times[:] = recent


def chunk_by_section(document_text):
    return re.split(r"\n## ", document_text)


def generate_embedding(chunks, model="voyage-3-large", input_type="query"):
    is_list = isinstance(chunks, list)
    for _ in range(3):
        _throttle_for_voyage()
        try:
            result = embedding_client.embed(
                chunks if is_list else [chunks], model=model, input_type=input_type
            )
            return result.embeddings if is_list else result.embeddings[0]
        except VoyageRateLimitError:
            print("(Voyage rate limit hit despite throttling, waiting 65s and retrying...)")
            time.sleep(65)
    raise RuntimeError("VoyageAI rate limit: gave up after 3 attempts.")


class VectorIndex:
    def __init__(self, distance_metric: str = "cosine", embedding_fn=None):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            return
        vectors = self._embedding_fn([d["content"] for d in documents])
        for vector, document in zip(vectors, documents):
            self._add_vector(vector, document)

    def _add_vector(self, vector, document):
        if not self.vectors:
            self._vector_dim = len(vector)
        self.vectors.append(list(vector))
        self.documents.append(document)

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if not self.vectors:
            return []
        query_vector = self._embedding_fn(query) if isinstance(query, str) else query

        dist_func = self._cosine_distance if self._distance_metric == "cosine" else self._euclidean_distance
        distances = [(dist_func(query_vector, v), doc) for v, doc in zip(self.vectors, self.documents)]
        distances.sort(key=lambda item: item[0])
        return [(doc, dist) for dist, doc in distances[:k]]

    def _euclidean_distance(self, v1, v2):
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(v1, v2)))

    def _cosine_distance(self, v1, v2):
        mag1 = math.sqrt(sum(x * x for x in v1))
        mag2 = math.sqrt(sum(x * x for x in v2))
        if mag1 == 0 or mag2 == 0:
            return 1.0
        cos_sim = sum(p * q for p, q in zip(v1, v2)) / (mag1 * mag2)
        return 1.0 - max(-1.0, min(1.0, cos_sim))


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.split(r"\W+", text.lower()) if t]

    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            tokens = self._tokenize(doc["content"])
            self.documents.append(doc)
            self._corpus_tokens.append(tokens)
            self._doc_len.append(len(tokens))
            for token in set(tokens):
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
        self._build_index()

    def _build_index(self):
        if not self.documents:
            return
        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        N = len(self.documents)
        self._idf = {
            term: math.log(((N - freq + 0.5) / (freq + 0.5)) + 1) for term, freq in self._doc_freqs.items()
        }

    def _score(self, query_tokens: List[str], i: int) -> float:
        counts = Counter(self._corpus_tokens[i])
        doc_len = self._doc_len[i]
        score = 0.0
        for token in query_tokens:
            if token not in self._idf:
                continue
            idf = self._idf[token]
            tf = counts.get(token, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self._avg_doc_len))
            score += idf * tf * (self.k1 + 1) / (denom + 1e-9)
        return score

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents or not isinstance(query, str):
            return []
        query_tokens = self._tokenize(query)
        scored = [(self._score(query_tokens, i), doc) for i, doc in enumerate(self.documents)]
        scored = [(s, doc) for s, doc in scored if s > 1e-9]
        scored.sort(key=lambda item: item[0], reverse=True)
        # Normalize to a "lower is better" distance, same convention as VectorIndex,
        # so a Retriever fusing both doesn't need to know which index is which.
        return [(doc, math.exp(-0.1 * s)) for s, doc in scored[:k]]


class Retriever:
    """Fuses rankings from several indexes with Reciprocal Rank Fusion (RRF):
    a document's score is the sum of 1/(k_rrf + rank) across every index it
    appeared in. Fusing ranks rather than raw scores is what lets a BM25 index
    and a VectorIndex - whose scores are on totally different scales - combine
    into one ranking. Same as multi_index_rag.py, minus the reranker."""

    def __init__(self, *indexes):
        self._indexes = list(indexes)

    def add_documents(self, documents: List[Dict[str, Any]]):
        for index in self._indexes:
            index.add_documents(documents)

    def search(self, query_text: str, k: int = 1, k_rrf: int = 60) -> List[Tuple[Dict[str, Any], float]]:
        all_results = [index.search(query_text, k=k * 5) for index in self._indexes]

        doc_ranks: Dict[int, Dict[str, Any]] = {}
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                entry = doc_ranks.setdefault(
                    id(doc), {"doc": doc, "ranks": [float("inf")] * len(self._indexes)}
                )
                entry["ranks"][idx] = rank + 1

        scored = [
            (e["doc"], sum(1.0 / (k_rrf + r) for r in e["ranks"] if r != float("inf")))
            for e in doc_ranks.values()
        ]
        scored = [(doc, s) for doc, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


with open("report.md", encoding="utf-8") as f:
    FULL_TEXT = f.read()

CHUNKS = chunk_by_section(FULL_TEXT)

RETRIEVER = Retriever(BM25Index(), VectorIndex(embedding_fn=generate_embedding))
RETRIEVER.add_documents([{"content": c} for c in CHUNKS])


# --------------------------------------------------------------------------
# The benchmark questions - hand-written from report.md's own cross-references
# (e.g. Section 6 literally says "Material Composite XT-5, discussed in
# Section 4..."), not LLM-generated, so we know exactly which section(s) each
# answer needs and can label each question single-hop or multi-hop with
# confidence instead of guessing.
# --------------------------------------------------------------------------

QUESTIONS = [
    # --- single-hop: everything needed is in one section ---
    {
        "question": "What sensitivity did the XDR-BioMk-v2 diagnostic panel achieve in early validation?",
        "hops": "single",
        "criteria": ["States the sensitivity as greater than 85% (>85%)"],
    },
    {
        "question": "What was the average tensile strength measured for Material Composite XT-5?",
        "hops": "single",
        "criteria": ["States 450 MPa, or a value within +/-15 MPa of 450 MPa"],
    },
    {
        "question": "What CPU clock speed was confirmed for the Zircon-5 Phase 3 prototype?",
        "hops": "single",
        "criteria": ["States 3.8 GHz (boost)"],
    },
    # --- multi-hop: report.md explicitly points to a second section ---
    {
        "question": (
            "Section 2 describes a software incident tracked under an incident ID. What is "
            "that ID, and which department did the attack behind it target, according to "
            "the cybersecurity section?"
        ),
        "hops": "multi",
        "criteria": [
            "Names the incident ID INC-2023-Q4-011 (from Section 2)",
            "States the finance department was targeted, via spear-phishing (from Section 10)",
        ],
    },
    {
        "question": (
            "Section 6 says it is evaluating a material for the chassis design that is "
            "discussed elsewhere in the report. Which material is it, and what specific "
            "weakness did testing reveal about it?"
        ),
        "hops": "multi",
        "criteria": [
            "Names Material Composite XT-5 (from Section 6)",
            "States that fatigue testing revealed micro-fracturing (from Section 4)",
        ],
    },
    {
        "question": (
            "The financial analysis section says investment tied to two other sections' R&D "
            "work may face review due to resource pressure. Name both topics and what each "
            "is working on."
        ),
        "hops": "multi",
        "criteria": [
            "Names Section 9 / Pharmaceutical Development and Compound CTX-204b",
            "Names Section 4 / Scientific Experimentation and Material Composite XT-5",
        ],
    },
]


# --------------------------------------------------------------------------
# System 1: long-context - no retrieval, the whole document goes in the prompt.
# --------------------------------------------------------------------------


def answer_long_context(question):
    prompt = f"""Answer the question using only the document below. Be concise.

<document>
{
    
}
</document>

Question: {question}"""
    return ask(prompt), 0  # 0 retrieval calls - nothing was retrieved, it all sat in context


# --------------------------------------------------------------------------
# System 2: naive RAG - exactly one fixed retrieval call, then answer. This is
# what complete_rag_flow.py reduces to once you add a generation step on top.
# --------------------------------------------------------------------------


def answer_naive_rag(question, k=3):
    results = RETRIEVER.search(question, k=k)
    context = "\n---\n".join(doc["content"] for doc, _ in results)
    prompt = f"""Answer the question using only the context below. If the context does not
contain the answer, say so plainly instead of guessing. Be concise.

<context>
{context}
</context>

Question: {question}"""
    return ask(prompt), 1  # exactly one retrieval call, always


# --------------------------------------------------------------------------
# System 3: agentic RAG - search is a tool. Same request/response loop as
# tool_use.py; the only difference is the tool is search_documents instead of
# add_duration_to_datetime. The model decides how many times to call it and
# what to search for each time - it can read the first result, realize it
# needs a second fact, and issue a second, differently-worded search. That is
# what makes multi-hop questions answerable without hardcoding a hop count.
# --------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the report for passages relevant to a query. Returns the top matching "
            "sections. Call this once per distinct piece of information you need - if the "
            "first result doesn't fully answer the question, search again with a more "
            "specific query rather than guessing."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for."}},
            "required": ["query"],
        },
    },
}

AGENT_SYSTEM_PROMPT = (
    "You answer questions about a report by searching it with the search_documents tool. "
    "Some questions need more than one search - if an answer references another section, "
    "search again for that section specifically before answering. Only answer once you "
    "have every fact the question asks for."
)


def answer_agentic_rag(question, max_turns=6):
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    search_calls = 0

    for _ in range(max_turns):
        message = chat_with_tools(messages, tools=[SEARCH_SCHEMA])

        entry = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in message.tool_calls
            ]
        messages.append(entry)

        if not message.tool_calls:
            return message.content, search_calls

        for call in message.tool_calls:
            search_calls += 1
            try:
                query = json.loads(call.function.arguments).get("query", question)
            except json.JSONDecodeError:
                query = question
            results = RETRIEVER.search(query, k=3)
            content = "\n---\n".join(doc["content"] for doc, _ in results)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content or "No results."}
            )

    return "Gave up: too many searches without an answer.", search_calls


# --------------------------------------------------------------------------
# Grading - a judge checks off each item in `criteria` against the answer.
# A question's score = fraction of criteria satisfied. This is more literal
# than prompt_evaluator.py's 1-10 holistic score on purpose: for a multi-hop
# question, each criterion IS one hop, so the fraction satisfied shows
# directly which hop a system missed.
# --------------------------------------------------------------------------


def grade(question, criteria, answer):
    checklist = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
    prompt = f"""Question: {question}

Answer to grade:
{answer}

Checklist - for each numbered item, does the answer satisfy it? Respond with ONLY a
JSON array of {len(criteria)} booleans, one per item, in order, nothing else.

{checklist}"""
    for _ in range(3):
        reply = ask(prompt)
        start, end = reply.find("["), reply.rfind("]")
        if start == -1 or end == -1:
            continue
        try:
            checks = json.loads(reply[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(checks, list) and len(checks) == len(criteria):
            return [bool(c) for c in checks]
    return [False] * len(criteria)  # judge never returned a usable answer - count as failed


# --------------------------------------------------------------------------
# Run all three systems against all questions, time each call, save the raw
# results so RESULTS.md can be written from real numbers, not guesses.
# --------------------------------------------------------------------------

SYSTEMS = {
    "long_context": answer_long_context,
    "naive_rag": answer_naive_rag,
    "agentic_rag": answer_agentic_rag,
}

RESULTS_FILE = "agentic_rag_benchmark_results.json"


def run_system(system_name, fn):
    results = []
    for case in QUESTIONS:
        start = time.time()
        answer, retrieval_calls = fn(case["question"])
        elapsed = time.time() - start

        checks = grade(case["question"], case["criteria"], answer)
        score = sum(checks) / len(checks)

        print(
            f"[{case['hops']:6s}] score={score:.2f} retrievals={retrieval_calls} "
            f"time={elapsed:.1f}s :: {case['question'][:60]}..."
        )

        results.append(
            {
                "system": system_name,
                "question": case["question"],
                "hops": case["hops"],
                "answer": answer,
                "criteria": case["criteria"],
                "checks": checks,
                "score": score,
                "retrieval_calls": retrieval_calls,
                "seconds": round(elapsed, 2),
            }
        )
    return results


def print_summary(all_results):
    print("\n=== Summary (avg score by system x hop type) ===")
    for system_name in SYSTEMS:
        for hop_type in ("single", "multi"):
            rows = [r for r in all_results if r["system"] == system_name and r["hops"] == hop_type]
            if not rows:
                continue
            avg_score = sum(r["score"] for r in rows) / len(rows)
            avg_calls = sum(r["retrieval_calls"] for r in rows) / len(rows)
            avg_secs = sum(r["seconds"] for r in rows) / len(rows)
            print(
                f"{system_name:14s} {hop_type:6s}  avg_score={avg_score:.2f}  "
                f"avg_retrievals={avg_calls:.1f}  avg_seconds={avg_secs:.1f}"
            )


if __name__ == "__main__":
    import sys

    # VoyageAI's free-tier rate limit (3 requests/min) makes a full run take
    # several minutes. Run one system per invocation, e.g.:
    #     python agentic_rag_benchmark.py naive_rag
    # Results accumulate in RESULTS_FILE (keyed by system) instead of being
    # overwritten each time, so a slow or interrupted run never loses work
    # already done. With no arguments, all three systems run one after another.
    requested = sys.argv[1:] or list(SYSTEMS)
    unknown = [s for s in requested if s not in SYSTEMS]
    if unknown:
        raise SystemExit(f"Unknown system(s) {unknown}. Choose from: {list(SYSTEMS)}")

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = []

    for system_name in requested:
        print(f"\n=== {system_name} ===")
        all_results = [r for r in all_results if r["system"] != system_name]
        all_results += run_system(system_name, SYSTEMS[system_name])

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    print_summary(all_results)
    print(f"\nSaved raw results to {RESULTS_FILE}")
