"""Lesson B — DeepEval RAG metrics.

RAG quality splits into two questions:

    Generation:  Is the answer good given what was retrieved?
                 → AnswerRelevancy, Faithfulness
    Retrieval:   Were the right chunks fetched, in a useful order?
                 → ContextualPrecision (needs expected_output)

Faithfulness asks: every claim in the answer must be supported by
`retrieval_context`. Hallucinated facts that aren't in the chunks → low score.

Demo data mirrors your RAG/report.md incident INC-2023-Q4-011. No live
retriever here — we hardcode chunks so the lesson stays about *scoring*.

    python deep_eval_rag.py
"""

from __future__ import annotations

import json
import os
from typing import Optional, Type, Union

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in judge reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class GroqJudge(DeepEvalBaseLLM):
    """Same Groq judge as deep_eval.py (duplicated so each lesson stays standalone)."""

    def __init__(self):
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model_name = GROQ_MODEL
        super().__init__(model=self.model_name)

    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return self.model_name

    def _complete(self, prompt: str, *, as_json: bool) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 4096,
            "extra_body": {"reasoning_effort": "low"},
        }
        if as_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def generate(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        text = self._complete(prompt, as_json=schema is not None)
        if schema is None:
            return text
        return schema.model_validate(extract_json(text))

    async def a_generate(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        return self.generate(prompt, schema=schema)


# Chunks a retriever might return for the eng / INC query (from report.md).
RELEVANT_CHUNK = (
    "The Software Engineering division dedicated considerable effort to "
    "improving the stability of Project Phoenix. Recurring issues, particularly "
    "ERR_MEM_ALLOC_FAIL_0x8007000E during peak loads, were prioritized at a cost "
    "of INC-2023-Q4-011. The deployment of a patch addressed the memory "
    "allocation error, resulting in a measured 40% reduction in critical "
    "failures under simulated stress tests (Test Case ID: INC-2023-Q4-011)."
)

IRRELEVANT_CHUNK = (
    "Medical Research made progress on XDR-471 syndrome. Analysis of patient "
    "cohort XDR-EU-03 linked symptom severity to Gene LOC73b marker expression."
)


def main():
    judge = GroqJudge()

    query = "What did the eng team do with INC-2023-Q4-011?"

    # Good RAG answer: grounded in retrieved chunks, on-topic.
    good_case = LLMTestCase(
        input=query,
        actual_output=(
            "For INC-2023-Q4-011, engineering patched a memory allocation failure "
            "(ERR_MEM_ALLOC_FAIL_0x8007000E) on Project Phoenix, which cut critical "
            "failures by about 40% in stress tests."
        ),
        expected_output=(
            "They patched ERR_MEM_ALLOC_FAIL related to INC-2023-Q4-011 and reduced "
            "critical failures by 40%."
        ),
        # Order matters for ContextualPrecision: relevant chunks should rank first.
        retrieval_context=[RELEVANT_CHUNK, IRRELEVANT_CHUNK],
    )

    # Bad RAG answer: invents a fact that is NOT in the retrieval context.
    hallucinated_case = LLMTestCase(
        input=query,
        actual_output=(
            "For INC-2023-Q4-011, the eng team migrated Phoenix to Kubernetes and "
            "fired the on-call engineer responsible for the outage."
        ),
        expected_output=(
            "They patched ERR_MEM_ALLOC_FAIL related to INC-2023-Q4-011 and reduced "
            "critical failures by 40%."
        ),
        retrieval_context=[RELEVANT_CHUNK, IRRELEVANT_CHUNK],
    )

    relevancy = AnswerRelevancyMetric(
        threshold=0.5, model=judge, async_mode=False
    )
    faithfulness = FaithfulnessMetric(
        threshold=0.5,
        model=judge,
        async_mode=False,
        # By default DeepEval treats "not in context" (idk) as OK. For RAG we
        # usually want unsupported claims to lower the score — turn this on.
        penalize_ambiguous_claims=True,
    )
    # Needs expected_output + retrieval_context: were useful chunks ranked high?
    precision = ContextualPrecisionMetric(
        threshold=0.5, model=judge, async_mode=False
    )

    print("=== Good grounded answer ===")
    evaluate(
        test_cases=[good_case],
        metrics=[relevancy, faithfulness, precision],
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=True, show_indicator=True),
    )

    print("\n=== Hallucinated answer (expect Faithfulness to drop) ===")
    evaluate(
        test_cases=[hallucinated_case],
        metrics=[relevancy, faithfulness, precision],
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=True, show_indicator=True),
    )


if __name__ == "__main__":
    main()
