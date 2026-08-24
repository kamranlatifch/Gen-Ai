"""Image handling: send a picture URL, the model looks at it, then answers.

Until now, content was always a string. For images it is a list of parts:

    content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://..."}},
    ]

The model fetches that URL itself. The URL must be public. You never download
the file in this lesson.

gpt-oss cannot see images. Vision needs a vision model, so Groq uses
qwen/qwen3.6-27b here.
"""

import os

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, RateLimitError

from openrouter_test import GROQ_MODEL

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = os.environ.get("AI_MODEL", "openrouter/free")

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"

# IMAGE_URL = "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png"
IMAGE_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRA1QmQ9Y_djRqouvtG_bmyhqbzr89iB0mpC_6Hc5F1dg&s=10"


def ask_about_image(url, question="What is in this image?"):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]

    try:
        response = client.chat.completions.create(model=MODEL, messages=messages)
    except (RateLimitError, APIStatusError):
        print("(OpenRouter unavailable, using Groq vision model...)")
        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=messages,
            max_tokens=1024,
            extra_body={"reasoning_effort": "none"},
        )

    return response.choices[0].message.content
prompt = """
Analyze the attached satellite image of a property with these specific steps:

1. Residence identification: Locate the primary residence on the property by looking for:
   - The largest roofed structure 
   - Typical residential features (driveway connection, regular geometry)
   - Distinction from other structures (garages, sheds, pools)
   Describe the residence's location relative to property boundaries and other features.

2. Tree overhang analysis: Examine all trees near the primary residence:
   - Identify any trees whose canopy extends directly over any portion of the roof
   - Estimate the percentage of roof covered by overhanging branches (0-25%, 25-50%, 50-75%, 75-100%)
   - Note particularly dense areas of overhang

3. Fire risk assessment: For any overhanging trees, evaluate:
   - Potential wildfire vulnerability (ember catch points, continuous fuel paths to structure)
   - Proximity to chimneys, vents, or other roof openings if visible
   - Areas where branches create a "bridge" between wildland vegetation and the structure
   
4. Defensible space identification: Assess the property's overall vegetative structure:
   - Identify if trees connect to form a continuous canopy over or near the home
   - Note any obvious fuel ladders (vegetation that can carry fire from ground to tree to roof)

5. Fire risk rating: Based on your analysis, assign a Fire Risk Rating from 1-4:
   - Rating 1 (Low Risk): No tree branches overhanging the roof, good defensible space around the structure
   - Rating 2 (Moderate Risk): Minimal overhang (<25% of roof), some separation between tree canopies
   - Rating 3 (High Risk): Significant overhang (25-50% of roof), connected tree canopies, multiple points of vulnerability
   - Rating 4 (Severe Risk): Extensive overhang (>50% of roof), dense vegetation against structure, numerous ember catch points, limited defensible space

For each item above (1-5), write one sentence summarizing your findings, with your final response being the numeric Fire Risk Rating (1-4) with a brief justification.
"""

if __name__ == "__main__":
    print("Image:", IMAGE_URL)
    print()
    print(ask_about_image(IMAGE_URL, prompt))
