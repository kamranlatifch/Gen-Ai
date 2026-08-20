"""Prompt engineering practice: a 1-day meal plan for an athlete.

Goal: write a prompt that generates a 1-day meal plan based on an athlete's
height, weight, goal, and dietary restrictions.

All the machinery lives in prompt_evaluator.py. This file only holds the three
things that are specific to this task: what the task is, what inputs it takes,
and the prompt itself. To try a new version of the prompt, edit run_prompt and
run this file again - everything else stays put.
"""

import os

from generate_datasets import ask
from prompt_evaluator import PromptEvaluator, print_report

TASK_DESCRIPTION = "Write a compact, concise 1 day meal plan for a single athlete"

PROMPT_INPUTS_SPEC = {
    "height": "Athlete's height in cm",
    "weight": "Athlete's weight in kg",
    "goal": "Goal of the athlete",
    "restrictions": "Dietary restrictions of the athlete",
}

DATASET_FILE = "meal_dataset.json"


def run_prompt(prompt_inputs):
    """Version 1 - the straightforward attempt.

    No role, no output format, no handling of the dietary restriction beyond
    stating it. Run it, read the output, and note what is wrong. Those
    complaints become the rules that go into version 2.
    """
    prompt = f"""
What should this person eat?

- Height: {prompt_inputs["height"]}
- Weight: {prompt_inputs["weight"]}
- Goal: {prompt_inputs["goal"]}
- Dietary restrictions: {prompt_inputs["restrictions"]}
"""
    return ask(prompt)


if __name__ == "__main__":
    evaluator = PromptEvaluator(max_concurrent_tasks=3)

    # Generating the dataset costs API calls, and scores from different test
    # sets cannot be compared. So build it once and reuse it. Delete the file
    # when you deliberately want a fresh set.
    if not os.path.exists(DATASET_FILE):
        print(f"Generating {DATASET_FILE}...")
        evaluator.generate_dataset(
            task_description=TASK_DESCRIPTION,
            prompt_inputs_spec=PROMPT_INPUTS_SPEC,
            output_file=DATASET_FILE,
            num_cases=3,
        )

    print("Running evaluation...\n")
    results = evaluator.run_evaluation(
        run_prompt_function=run_prompt,
        dataset_file=DATASET_FILE,
        output_file="meal_results.json",
        task_description=TASK_DESCRIPTION,
    )

    print_report(results)
