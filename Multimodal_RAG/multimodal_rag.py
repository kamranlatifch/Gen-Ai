"""Multimodal RAG (Day 05): retrieve the right chart PAGE from a PDF by
treating each page as a plain image - no OCR, no text extraction - then have
a vision-capable model read the retrieved chart and answer a data-trend
question about it.

Two phases, same shape as RAG/complete_rag_flow.py:

    indexing   render each PDF page to an image (PyMuPDF), embed it with a
               multimodal model (voyage-multimodal-3 - Voyage's hosted
               CLIP-style model: images and text land in the SAME vector
               space), store each (page_number, image, vector) in a
               VectorIndex - same class shape as every other RAG lesson here

    query      embed the question TEXT into that same space, cosine-search
               for the closest page image, then hand THAT IMAGE (never any
               extracted text) to a vision model and ask it to answer by
               looking at the chart

Grading checks two things separately, on purpose:
    retrieval correctness   did we fetch the page the question is actually
                             about? (a plain code check against GROUND_TRUTH's
                             page numbers - no LLM needed)
    answer correctness      did the vision model read the trend correctly?
                             (an LLM-judge checklist, same pattern as
                             RAG/agentic_rag_benchmark.py's grade())
Splitting them tells you which half of the pipeline broke, if either does.

Run from inside Multimodal_RAG/:
    python make_charts_pdf.py     # once, builds charts.pdf
    python multimodal_rag.py      # indexes it and runs the benchmark
"""

import base64
import json
import math
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Tuple

import pymupdf
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, RateLimitError
from PIL import Image
import voyageai
from voyageai.error import RateLimitError as VoyageRateLimitError

from make_charts_pdf import GROUND_TRUTH, PDF_PATH

load_dotenv()

# --------------------------------------------------------------------------
# LLM clients - same OpenRouter-first, Groq-fallback pattern as the rest of
# the repo. GROQ_VISION_MODEL matches image_support.py: gpt-oss cannot see
# images, so the vision fallback uses a model that can.
# --------------------------------------------------------------------------

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))
MODEL = os.environ.get("AI_MODEL", "openrouter/free")

groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"

openrouter_exhausted = False


def ask(prompt, temperature=0.0):
    """Text-only one-shot call, used for grading."""
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


def image_to_data_uri(image: Image.Image) -> str:
    """Vision APIs here take a URL or a data: URI, not a raw PIL image - see
    image_support.py, which only ever passed public URLs. These chart pages
    only exist locally, so base64-encode each one into a data URI instead."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def ask_about_image(image: Image.Image, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_to_data_uri(image)}},
            ],
        }
    ]
    try:
        r = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.0)
        return r.choices[0].message.content or ""
    except (RateLimitError, APIStatusError):
        print("(OpenRouter can't/won't handle the image, using Groq vision model...)")
        r = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=messages,
            max_tokens=512,
            extra_body={"reasoning_effort": "none"},
        )
        return r.choices[0].message.content or ""


# --------------------------------------------------------------------------
# Step 2: render PDF pages to images. This is the "treat the page as a
# picture" step - pymupdf rasterizes each page, no text is ever extracted.
# --------------------------------------------------------------------------


def render_pdf_pages(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    doc = pymupdf.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images


# --------------------------------------------------------------------------
# Step 3: multimodal ("CLIP-style") embeddings via VoyageAI. Same rate-limit
# story as RAG/agentic_rag_benchmark.py's Voyage calls - free tier without a
# payment method on file caps requests at 3/minute, so throttle proactively.
# --------------------------------------------------------------------------

embedding_client = voyageai.Client()
MULTIMODAL_MODEL = "voyage-multimodal-3"

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


def _multimodal_embed(inputs: List[list], input_type: str) -> List[List[float]]:
    for _ in range(3):
        _throttle_for_voyage()
        try:
            result = embedding_client.multimodal_embed(
                inputs=inputs, model=MULTIMODAL_MODEL, input_type=input_type
            )
            return result.embeddings
        except VoyageRateLimitError:
            print("(Voyage rate limit hit despite throttling, waiting 65s and retrying...)")
            time.sleep(65)
    raise RuntimeError("VoyageAI rate limit: gave up after 3 attempts.")


def embed_page_images(images: List[Image.Image]) -> List[List[float]]:
    # Each item in `inputs` is a list of segments (VoyageAI lets you mix text
    # and images per item). We stay OCR-free, so each page is just [image].
    return _multimodal_embed([[img] for img in images], input_type="document")


def embed_query_text(text: str) -> List[float]:
    return _multimodal_embed([[text]], input_type="query")[0]


# --------------------------------------------------------------------------
# Step 4: the index. Identical shape to VectorIndex everywhere else in this
# repo - store vectors + documents, cosine-search. The only thing that
# changed is what "content" means: a page image, not a text chunk.
# --------------------------------------------------------------------------


class PageIndex:
    def __init__(self):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []

    def add_pages(self, documents: List[Dict[str, Any]]):
        vectors = embed_page_images([d["image"] for d in documents])
        self.vectors.extend(vectors)
        self.documents.extend(documents)

    def search(self, query_text: str, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        qv = embed_query_text(query_text)
        scored = [(self._cosine_distance(qv, v), doc) for v, doc in zip(self.vectors, self.documents)]
        scored.sort(key=lambda item: item[0])
        return [(doc, dist) for dist, doc in scored[:k]]

    def _cosine_distance(self, v1, v2):
        mag1 = math.sqrt(sum(x * x for x in v1))
        mag2 = math.sqrt(sum(x * x for x in v2))
        if mag1 == 0 or mag2 == 0:
            return 1.0
        cos_sim = sum(p * q for p, q in zip(v1, v2)) / (mag1 * mag2)
        return 1.0 - max(-1.0, min(1.0, cos_sim))


def build_index() -> PageIndex:
    images = render_pdf_pages(PDF_PATH)
    index = PageIndex()
    index.add_pages([{"page": i + 1, "image": img} for i, img in enumerate(images)])
    print(f"Indexed {len(images)} pages from {PDF_PATH}")
    return index


# --------------------------------------------------------------------------
# Test questions - one per chart, each tagged with the page it MUST come
# from. That page number is the objective, code-level ground truth for
# retrieval; GROUND_TRUTH's sentence (from make_charts_pdf.py) is the basis
# for the answer-correctness checklist below.
# --------------------------------------------------------------------------

QUESTIONS = [
    {
        "page": 1,
        "question": "Looking at the revenue chart, did revenue go up, down, or stay flat "
        "across the 8 quarters shown? Give the starting and ending values.",
        "criteria": [
            "States revenue trended upward / increased",
            "Gives approximate start (~$2.1M) and end (~$5.8M) values",
        ],
    },
    {
        "page": 2,
        "question": "Did server latency increase or decrease over the year, and is there a "
        "month where it changed sharply?",
        "criteria": [
            "States latency decreased / trended downward overall",
            "Identifies July (or 'mid-year') as where it dropped sharply",
        ],
    },
    {
        "page": 3,
        "question": "Which month has the highest customer churn rate, and how does it compare "
        "to the other months?",
        "criteria": [
            "Identifies December as the spike month",
            "States it is roughly 2-3x (or clearly much higher than) other months",
        ],
    },
    {
        "page": 4,
        "question": "Monthly active users look flat for a while, then something changes. "
        "Describe the shape and roughly which month it happens.",
        "criteria": [
            "States MAU was flat (around 40k) before jumping",
            "Identifies September (or 'around month 9') as the jump point, to roughly 90-95k",
        ],
    },
    {
        "page": 5,
        "question": "Does website traffic follow a seasonal pattern across the year? When are "
        "the peaks and when are the troughs?",
        "criteria": [
            "States traffic is seasonal / cyclical",
            "Names November-December (or 'holiday season') as peak and summer (June-August) as the trough",
        ],
    },
    {
        "page": 6,
        "question": "What happened to the manufacturing defect rate over the year, and is "
        "there a point where the trend clearly changes?",
        "criteria": [
            "States defect rate declined / trended downward overall",
            "Identifies May (or 'mid-year', 'after the QA program') as the turning point",
        ],
    },
]


# --------------------------------------------------------------------------
# Answer-correctness grading - same checklist-judge pattern as
# RAG/agentic_rag_benchmark.py's grade().
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
    return [False] * len(criteria)


if __name__ == "__main__":
    index = build_index()

    results = []
    for case in QUESTIONS:
        retrieved_docs = index.search(case["question"], k=1)
        retrieved_page = retrieved_docs[0][0]["page"]
        retrieval_correct = retrieved_page == case["page"]

        answer = ask_about_image(retrieved_docs[0][0]["image"], case["question"])
        checks = grade(case["question"], case["criteria"], answer)
        answer_score = sum(checks) / len(checks)

        print(
            f"page {case['page']} -> retrieved page {retrieved_page} "
            f"[{'OK' if retrieval_correct else 'WRONG'}]  answer_score={answer_score:.2f}"
        )
        print(f"  Q: {case['question']}")
        print(f"  A: {answer}\n")

        results.append(
            {
                "expected_page": case["page"],
                "retrieved_page": retrieved_page,
                "retrieval_correct": retrieval_correct,
                "question": case["question"],
                "answer": answer,
                "criteria": case["criteria"],
                "checks": checks,
                "answer_score": answer_score,
            }
        )

    with open("multimodal_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    retrieval_acc = sum(r["retrieval_correct"] for r in results) / len(results)
    answer_acc = sum(r["answer_score"] for r in results) / len(results)
    passed = sum(r["retrieval_correct"] and r["answer_score"] == 1.0 for r in results)

    print("=== Summary ===")
    print(f"Retrieval accuracy : {retrieval_acc:.0%} ({sum(r['retrieval_correct'] for r in results)}/{len(results)})")
    print(f"Answer accuracy    : {answer_acc:.0%} (avg fraction of criteria met)")
    print(f"Fully passed       : {passed}/{len(results)} questions (right page AND fully correct answer)")
    print("\nSaved raw results to multimodal_rag_results.json")
