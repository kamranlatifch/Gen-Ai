"""Step 1 of a prompt evaluation: build the test dataset.

The idea behind an eval: instead of eyeballing one output and deciding it
"looks fine", you collect a set of test inputs, run your prompt on all of them,
and score the results. Then "this prompt is better" becomes a number you can
compare instead of a feeling.

Here we generate those test inputs by asking an LLM for them.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

# Same setup as openrouter_test.py, minus the streaming and the chat loop.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = os.environ.get("AI_MODEL", "openrouter/free")

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


# Once OpenRouter's daily limit is spent, every later call would also fail, so
# remember it and skip straight to Groq instead of paying for a doomed request
# on each one.
openrouter_exhausted = False


def ask(prompt, temperature=0.0):
    """Send one prompt, get the reply back as a string.

    temperature=0 by default: an eval needs the same input to produce the same
    output, otherwise scores wobble between runs and you can't tell whether a
    prompt change helped or the model just rolled different dice.
    """
    global openrouter_exhausted

    messages = [{"role": "user", "content": prompt}]
    response = None

    if not openrouter_exhausted:
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=temperature
            )
        except RateLimitError:
            print("(OpenRouter daily limit reached, using Groq from here on...)")
            openrouter_exhausted = True

    if response is None:
        # gpt-oss is a reasoning model. Left alone it spends its whole 2048
        # token output budget thinking and returns a truncated (or empty)
        # answer. Low effort keeps the thinking short and the answer intact.
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
        )

    # Some free models occasionally return an empty message. Return "" rather
    # than None so callers can treat the result as a string either way.
    return response.choices[0].message.content or ""


def extract_json(text):
    """Pull the JSON array out of a reply.

    Models wrap JSON in ```json fences and often add a sentence before it, so
    json.loads() on the raw reply fails. Slicing from the first "[" to the last
    "]" ignores anything around it.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in the response:\n\n{text}")
    return json.loads(text[start : end + 1])


def generate_dataset():
    """Ask the LLM to invent AWS tasks to test our prompt against.

    Three things in this prompt are doing real work:
    - the example output block, which is the most reliable way to get JSON
      back in the exact shape you want
    - the constraints, which keep tasks small; without them you get
      "build a multi-region DR pipeline", useless as a test case
    - "3 objects", because you re-run this while the pipeline is still broken.
      Get it working on 3, then raise the number.
    """
    prompt = """
Generate a evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
    {
        "task": "Description of task",
    },
    ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression
* Focus on tasks that do not require writing much code

Please generate 3 objects.
"""
    return extract_json(ask(prompt))


if __name__ == "__main__":
    print("Generating dataset...\n")
    dataset = generate_dataset()

    # Saved to disk so later steps can reuse it without paying for the API call
    # again. You generate the dataset once and run the eval against it many
    # times as you improve the prompt.
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(json.dumps(dataset, indent=2))
    print(f"\nSaved {len(dataset)} test cases to dataset.json")
