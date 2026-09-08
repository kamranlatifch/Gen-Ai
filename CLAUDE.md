# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal learning/experimentation sandbox for LLM API concepts (tool use, extended thinking, vision,
prompt caching, prompt evaluation/evals, and RAG). There is no application, package, or CLI here — each
top-level `.py` file is a standalone, runnable lesson script with its own `if __name__ == "__main__":`
demo. There is no test suite, build step, or lint config in this repo.

Most scripts are the Anthropic "building with Claude" course material rewritten to run against free
providers instead of the paid Anthropic API — see the module docstring at the top of each file for the
concept it demonstrates and the reasoning behind its design choices (e.g. `text_editor_tool.py`,
`prompt_caching.py`, `tool_use.py`).

## Running scripts

There is no virtualenv checked in. Install deps and run any file directly:

```bash
pip install -r requirements.txt
python <script>.py          # e.g. python tool_use.py
cd RAG && python <script>.py  # RAG scripts assume RAG/ as the cwd (they open ./report.md)
```

`requirements.txt` only lists `openai`, `python-dotenv`, and `voyageai`. Some RAG scripts
(`multi_index_rag.py`, `reranking.py`) additionally `import anthropic` (the real Anthropic SDK), which is
not in requirements.txt — install it separately if working on those files.

### Environment (`.env`, gitignored)

- `OPENROUTER_API_KEY`, `AI_MODEL` — primary provider, OpenAI-compatible endpoint at
  `https://openrouter.ai/api/v1`. `AI_MODEL` defaults to `openrouter/free` (a free-tier auto-router).
- `GROQ_API_KEY`, `GROQ_MODEL` — fallback provider (OpenAI-compatible too), used when OpenRouter's free
  daily/per-minute limits are hit. Defaults to `openai/gpt-oss-20b`.
- `VOYAGE_API_KEY` — used by RAG scripts for embeddings via `voyageai.Client()`.
- `multi_index_rag.py` / `reranking.py` instantiate `anthropic.Anthropic()` directly (model
  `claude-haiku-4-5`), which requires `ANTHROPIC_API_KEY` — not currently present in `.env`'s key set.

## Architecture / conventions to know before editing

**Two client patterns coexist, and most files duplicate the boilerplate rather than sharing it:**

1. **OpenAI-SDK-compatible pattern** (the majority of files): build an `OpenAI` client pointed at
   OpenRouter's `base_url`, plus a second `OpenAI` client pointed at Groq's `base_url` as a fallback.
   Nearly every file re-declares `client`, `MODEL`, `groq_client`, `GROQ_MODEL` at the top rather than
   importing them — `generate_datasets.py` is the closest thing to a shared module (`ask()`,
   `extract_json()`, `client`, `MODEL`, `groq_client`) and is imported by `tool_use.py`,
   `text_editor_tool.py`, `prompt_evaluator.py`, `run_evals.py`, and `prompt_engineering.py`. Other files
   (`toolschema.py`, `openrouter_test.py`, `extended_thinking.py`, `prompt_caching.py`,
   `multi_turn_conversion_with_tools.py`) keep their own independent copy of the same setup. When adding
   a new lesson script, prefer importing from `generate_datasets.py` over re-pasting the boilerplate.
2. **Direct Anthropic + VoyageAI pattern** (only `RAG/multi_index_rag.py`, `RAG/reranking.py`): uses the
   real `anthropic.Anthropic()` client for chat and `voyageai.Client()` for embeddings — no
   OpenRouter/Groq fallback logic.

**The OpenRouter → Groq fallback shape**, repeated throughout: try the OpenRouter call, catch
`openai.RateLimitError` (and sometimes `APIStatusError`), print a note, then retry the same request
against `groq_client`/`GROQ_MODEL`. Groq's `gpt-oss` model is a reasoning model, so Groq calls typically
pass `extra_body={"reasoning_effort": "low"}` (or `"none"`) to stop it from burning the whole output
budget on hidden thinking. `image_support.py` swaps in a vision-capable Groq model
(`qwen/qwen3.6-27b`) since `gpt-oss` can't see images.

**Tool-calling loop shape** (`tool_use.py`, `text_editor_tool.py`, `multi_turn_conversion_with_tools.py`):
messages list → call the model with `tools=` → if `message.tool_calls` is empty, return the answer;
otherwise run each requested tool locally, append one `{"role": "tool", "tool_call_id": ..., "content":
...}` message per call, and loop (bounded by `max_turns`). Tool schemas use OpenAI's
`{"type": "function", "function": {...}}` shape (not Anthropic's `input_schema`), since everything routes
through the OpenAI-compatible API.

**`text_editor_tool.py`** sandboxes all model-driven file writes under `editor_workspace/`: `_resolve()`
resolves the model-supplied path and rejects anything that escapes the sandbox directory, and every write
pushes the previous file contents onto an in-memory undo stack (`_history`) before applying the edit.

**Prompt evaluation pipeline** (`generate_datasets.py` → `prompt_evaluator.py` / `run_evals.py` →
`prompt_engineering.py`): generate a JSON dataset of test cases (each with `prompt_inputs` and
`solution_criteria`), run a candidate prompt against every case, then have a second LLM call grade the
output against the criteria and return `{strengths, weaknesses, reasoning, score}`. `PromptEvaluator` in
`prompt_evaluator.py` is the reusable version of this; `run_evals.py`/`generate_datasets.py` are the
earlier one-off version it grew out of, scoped to AWS code-generation tasks. Datasets and results are
cached to JSON files (`dataset.json`, `meal_dataset.json`, `results.json`, `meal_results.json`,
`model_grades.json`) so re-running an eval doesn't re-pay for dataset generation — delete the dataset file
to force regeneration. Grading always uses `temperature=0` so scores are comparable across runs, and judge
calls retry a few times since free models sometimes return prose instead of JSON.

**RAG scripts** (`RAG/`) build the same ideas up in layers, largely copy-pasted between files rather than
imported:
- `chunking_strategies.py` — three chunkers: by sentence, by fixed character count with overlap, by
  markdown `## ` section heading.
- `voyage_text_embedding.py` — thin wrapper around `voyageai.Client().embed()`.
- `complete_rag_flow.py` — the minimal end-to-end version: chunk `report.md` by section, embed each
  chunk, load into an in-memory `VectorIndex` (brute-force cosine/euclidean search, no external vector
  DB), embed a query, and search.
- `bm25-lexical-search.py` — a from-scratch `BM25Index` (k1/b params, IDF, term-frequency scoring) as the
  lexical-search counterpart to vector search.
- `multi_index_rag.py` — combines multiple `VectorIndex` instances plus the real Anthropic client to
  generate an answer from retrieved chunks, not just return the chunks.
- `reranking.py` — adds a reranking step on top of initial retrieval, also via the direct Anthropic
  client.

`RAG/report.md` is the fictional test document (a multi-department "annual report" full of fake incident
IDs, trial IDs, etc.) that every RAG script chunks/embeds/queries against — it doubles as the sample
corpus and as a source of easy-to-verify answers (e.g. "what happened with INC-2023-Q4-011").
