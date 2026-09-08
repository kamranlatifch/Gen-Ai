"""Lesson A — DeepEval basics (test case → metrics → score).

DeepEval is pytest-style evaluation for LLM outputs. The core loop is:

    1. Build an LLMTestCase  (input + what the model said + optional expected answer)
    2. Pick metric(s)        (scoring rules — often LLM-as-judge)
    3. Run evaluate()        (get score 0–1, pass/fail vs threshold, and a reason)

This script uses Groq as the judge (not OpenAI). Run from the repo root:

    pip install -r requirements.txt
    python deep_eval.py

Lesson B (RAG metrics) lives in deep_eval_rag.py.
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
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def extract_json(text: str) -> dict:
    """Slice the first {...} block out of a model reply (fences / prose OK)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in judge reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval-compatible LLM that talks to Groq's OpenAI-compatible API.

    Metrics that need structured output pass a Pydantic `schema`. We ask Groq
    for JSON (`response_format`) and validate into that schema.
    """

    def __init__(self):
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model_name = GROQ_MODEL
        # Parent sets self.name and calls load_model().
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
        # Sync Groq client — fine for a lesson script; keep calls sequential.
        return self.generate(prompt, schema=schema)


def main():
    judge = GroqJudge()

    # --- 1. Test case -------------------------------------------------------
    # Think of this as one row in an eval spreadsheet.
    #   input           = user question
    #   actual_output   = what your app/model produced
    #   expected_output = gold answer (needed for correctness-style metrics)
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
    )
    # --- 2. Metrics ----------------------------------------------------------
    # GEval = you write the grading criteria in English; an LLM judges against them.
    correctness = GEval(
        name="Correctness",
        criteria=(
            "Is the actual output factually correct for the input? "
            "expected_output is the ground-truth key fact."
        ),
        evaluation_steps=[
            "Identify the key fact in expected_output.",
            "Check whether actual_output contains that same fact (paraphrase is fine).",
            "Ignore extra wording, full sentences vs short answers, and style.",
            "Give a high score if the fact is present and correct; low only if wrong or missing.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=judge,
        async_mode=False,
    )

    # Answer relevancy = does the answer stick to the question? (no expected needed)
    relevancy = AnswerRelevancyMetric(
        threshold=0.5,
        model=judge,
        async_mode=False,
    )

    # --- 3. Run --------------------------------------------------------------
    # evaluate() prints a table. Scores are 0–1; pass if score >= threshold.
    evaluate(
        test_cases=[test_case],
        metrics=[correctness, relevancy],
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=True, show_indicator=True),
    )


if __name__ == "__main__":
    main()
