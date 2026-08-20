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
    """Version 2 - written against the criteria in meal_dataset.json.

    Every rule below exists because v1 lost points on it. The pattern to take
    away: read the weaknesses in your results file, and turn each recurring
    complaint into an explicit instruction. A prompt cannot satisfy a
    requirement nobody told it about.
    """
    prompt = f"""
You are a sports dietitian. Write a 1 day meal plan for this athlete.

<athlete>
- Height: {prompt_inputs["height"]} cm
- Weight: {prompt_inputs["weight"]} kg
- Goal: {prompt_inputs["goal"]}
- Dietary restrictions: {prompt_inputs["restrictions"]}
</athlete>

Rules:
1. Structure: exactly 3 meals (Breakfast, Lunch, Dinner) and exactly 2 snacks.
   Five eating occasions in total. Do not add pre-workout, post-workout, or
   any other extra meals.
2. Calorie target: multiply body weight in kg by
   - 25 to 28 for weight loss
   - 30 to 33 for maintenance
   - 37 to 40 for muscle gain or endurance
   State the target on the first line, then make the meals add up to it.
3. Protein: at least 1.8 g per kg of body weight.
4. Every meal must contain a lean protein, a complex carbohydrate, and a
   vegetable.
5. Across the day include at least 2 servings of fruit, 5 servings of
   vegetables, and 4 servings of whole grains.
6. Dietary restrictions are absolute, including hidden sources. Whey, casein
   and yogurt are dairy. Marzipan and nut butters are nuts. Soy sauce and
   edamame are soy. Wheat, barley and rye are gluten. If a restriction rules
   out a protein source, use poultry, fish, legumes or soy instead.
7. For every item give the portion in grams and its calories. After the plan,
   give one totals line: calories, protein, carbs, fat.

Output only the calorie target, the five eating occasions, and the totals
line. No preamble, no cooking instructions, no closing advice.
"""
    return ask(prompt)


def run_prompt_v1(prompt_inputs):
    """Version 1 - the straightforward attempt.

    No role, no output format, no handling of the dietary restriction beyond
    stating it. Run it, read the output, and note what is wrong. Those
    complaints become the rules that go into version 2.
    """
    prompt = f"""
        Generate a 1 day meal plan for an athlete that meet their dietry restrictions::
    - Height: {prompt_inputs["height"]}
    - Weight: {prompt_inputs["weight"]}
    - Goal: {prompt_inputs["goal"]}
    - Dietary restrictions: {prompt_inputs["restrictions"]}

    Guidelines:
    1. Include accurate daily calorie amount
    2. Show protein,fat,and carb amount
    3. Specify when to eat each meal
    4. Use only food that fit restrictions
    5. List all protein sizes in grams
    6. Key budget friendly if mentioned



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
