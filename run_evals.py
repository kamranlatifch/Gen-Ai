"""Step 2 of a prompt evaluation: run the prompt against every test case.

Step 1 (generate_datasets.py) produced dataset.json, a list of AWS tasks. Here we
feed each task to the prompt we want to evaluate and record what comes back.

The work is split into three layers, each with one job:

    run_eval       loops over the dataset
      run_test_case    runs one case and grades it
        run_prompt         builds the prompt and calls the LLM

Keeping them separate means you can change the prompt without touching the
grading, and change the grading without touching the loop.
"""

import json

from generate_datasets import ask

# Prompt v1, straight from the course slide. It is deliberately naive: no
# mention of AWS, no instruction to skip the explanation, no output format.
# Expect markdown fences and chatty prose. That is the point - we need a
# baseline bad score to prove the next version is better.
PROMPT_TEMPLATE = """
Please solve the following task:

{task}
"""


def load_dataset(path="dataset.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_prompt(test_case):
    """Merge the prompt template with one test case, then return the output."""
    prompt = PROMPT_TEMPLATE.format(task=test_case["task"])
    return ask(prompt)


def grade_by_model(test_case, output):
    """Grade the output of a test case by a model."""
    eval_prompt = f"""
        You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

        Original Task:
        <task>
        {test_case["task"]}
        </task>

        Solution to Evaluate:
        <solution>
        {output}
        </solution>

        Output Format
        Provide your evaluation as a structured JSON object with the following fields, in this specific order:
        - "strengths": An array of 1-3 key strengths
        - "weaknesses": An array of 1-3 key areas for improvement
        - "reasoning": A concise explanation of your overall assessment
        - "score": A number between 1-10

        Respond with JSON. Keep your response concise and direct.
        Example response shape:
        {{
            "strengths": string[],
            "weaknesses": string[],
            "reasoning": string,
            "score": number
        }}
    """
    # Free models sometimes reply with nothing or with prose instead of JSON,
    # so give the judge a few tries before giving up.
    for _ in range(3):
        eval_text = ask(eval_prompt)

        # The judge wraps its JSON in ```json fences, so slice from the first
        # "{" to the last "}" instead of parsing the raw reply.
        start = eval_text.find("{")
        end = eval_text.rfind("}")
        if start == -1 or end == -1:
            continue
        try:
            return json.loads(eval_text[start : end + 1])
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Judge never returned valid JSON. Last reply:\n\n{eval_text}")


def run_test_case(test_case):
    """Run one test case and grade the result."""
    output = run_prompt(test_case)
    grade = grade_by_model(test_case, output)

    return {
        "task": test_case["task"],
        "output": output,
        "score": grade["score"],
        "reasoning": grade["reasoning"],
    }


def run_eval(dataset):
    """Run every test case in the dataset."""
    results = []
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    return results


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Running {len(dataset)} test cases...\n")

    results = run_eval(dataset)

    # with open("results.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2)
    with open("model_grades.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for i, result in enumerate(results, start=1):
        print(f"--- Test case {i} (score: {result['score']}) ---")
        print(f"Task:      {result['task']}")
        print(f"Reasoning: {result['reasoning']}\n")

    average = sum(r["score"] for r in results) / len(results)
    print(f"Average score: {average:.1f}")
    print(f"Saved {len(results)} results to results.json")
