"""A reusable prompt evaluation harness.

Our earlier scripts only evaluated one specific prompt. This class does the
same work for any prompt: describe the task, describe the inputs it takes, and
the evaluator generates test cases, runs your prompt on each one, and grades
the results.

    evaluator = PromptEvaluator()

    evaluator.generate_dataset(
        task_description="Write a 1 day meal plan for an athlete",
        prompt_inputs_spec={"height": "Athlete's height in cm", ...},
        output_file="dataset.json",
        num_cases=3,
    )

    results = evaluator.run_evaluation(
        run_prompt_function=run_prompt, dataset_file="dataset.json"
    )

Each test case in the dataset holds two things:

    prompt_inputs      the values handed to your prompt
    solution_criteria  what a correct answer must contain, used for grading

The criteria are generated alongside the inputs, so the answer key is written
before any answer exists. Without them the judge has to invent its own idea of
"good" for every case, and the scores stop meaning anything.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from generate_datasets import ask, extract_json

DATASET_PROMPT = """
Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that perform the following task:

<task_description>
{task_description}
</task_description>

Each test case supplies these inputs to the prompt:

<prompt_inputs>
{prompt_inputs_spec}
</prompt_inputs>

Generate an array of JSON objects, each representing one test case.

Example output:
```json
[
    {{
        "prompt_inputs": {{{example_inputs}}},
        "solution_criteria": [
            "Specific, checkable statement about what a correct answer contains",
            "Another such statement"
        ]
    }},
    ...additional
]
```

* Vary the inputs across test cases, including at least one difficult case
* Each solution_criteria entry must be specific enough that two people would
  grade it the same way
* Give 2-4 solution_criteria per test case
* Every prompt_inputs value must be a short string

Please generate {num_cases} objects.
"""

GRADER_PROMPT = """
You are an expert evaluator. Grade the following AI-generated solution.

Original Task:
<task>
{task_description}
</task>

Inputs Given:
<inputs>
{prompt_inputs}
</inputs>

Criteria A Correct Solution Must Meet:
<criteria>
{solution_criteria}
</criteria>

Solution to Evaluate:
<solution>
{output}
</solution>

Output Format
Provide your evaluation as a JSON object with these fields, in this order:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your overall assessment
- "score": A number between 1-10

Respond with JSON. Keep your response concise and direct.
"""


class PromptEvaluator:
    """Generates test cases for a prompt, then runs and grades it."""

    def __init__(self, max_concurrent_tasks=3):
        # Test cases are independent, so they can run at the same time. Keep
        # this low on free tiers: too many at once trips the rate limit and
        # every request fails instead of just being slow.
        self.max_concurrent_tasks = max_concurrent_tasks

    def generate_dataset(
        self,
        task_description,
        prompt_inputs_spec,
        output_file="dataset.json",
        num_cases=3,
    ):
        """Ask the LLM to invent test cases for this task."""
        spec_lines = "\n".join(f"- {k}: {v}" for k, v in prompt_inputs_spec.items())
        example_inputs = ", ".join(f'"{k}": "..."' for k in prompt_inputs_spec)

        prompt = DATASET_PROMPT.format(
            task_description=task_description,
            prompt_inputs_spec=spec_lines,
            example_inputs=example_inputs,
            num_cases=num_cases,
        )
        dataset = extract_json(ask(prompt))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        return dataset

    def grade_by_model(self, task_description, test_case, output):
        """Have a second LLM call score one output against the criteria."""
        prompt = GRADER_PROMPT.format(
            task_description=task_description,
            prompt_inputs=json.dumps(test_case["prompt_inputs"], indent=2),
            solution_criteria="\n".join(f"- {c}" for c in test_case["solution_criteria"]),
            output=output,
        )

        # Free models sometimes reply with nothing, with prose instead of
        # JSON, or with JSON that is missing the one field we actually need.
        # Give the judge a few tries before giving up.
        for _ in range(3):
            try:
                grade = extract_json_object(ask(prompt))
            except (ValueError, json.JSONDecodeError):
                continue

            if isinstance(grade.get("score"), (int, float)):
                return grade

        raise ValueError("The judge never returned valid JSON with a score.")

    def run_evaluation(
        self,
        run_prompt_function,
        dataset_file="dataset.json",
        output_file="results.json",
        task_description="",
    ):
        """Run the prompt on every test case and grade each result."""
        with open(dataset_file, encoding="utf-8") as f:
            dataset = json.load(f)

        def run_one(test_case):
            output = run_prompt_function(test_case["prompt_inputs"])
            grade = self.grade_by_model(task_description, test_case, output)
            return {
                "prompt_inputs": test_case["prompt_inputs"],
                "solution_criteria": test_case["solution_criteria"],
                "output": output,
                "strengths": grade.get("strengths", []),
                "weaknesses": grade.get("weaknesses", []),
                "reasoning": grade.get("reasoning", ""),
                "score": grade["score"],
            }

        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as pool:
            # map keeps results in dataset order, so run N always lines up with
            # run N+1 when you compare two versions of a prompt.
            results = list(pool.map(run_one, dataset))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results


def extract_json_object(text):
    """Pull a single {...} object out of a reply, ignoring any fences around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in the response:\n\n{text}")
    return json.loads(text[start : end + 1])


def print_report(results):
    """Print one line per test case, then the average score."""
    for i, result in enumerate(results, start=1):
        inputs = ", ".join(f"{k}={v}" for k, v in result["prompt_inputs"].items())
        print(f"--- Test case {i} (score: {result['score']}) ---")
        print(f"Inputs:    {inputs}")
        print(f"Reasoning: {result['reasoning']}")
        if result["weaknesses"]:
            print("Weaknesses:")
            for weakness in result["weaknesses"]:
                print(f"  - {weakness}")
        print()

    average = sum(r["score"] for r in results) / len(results)
    print(f"Average score: {average:.1f} across {len(results)} test cases")
