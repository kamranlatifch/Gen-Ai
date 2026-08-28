from dotenv import load_dotenv
import voyageai
import re


load_dotenv()

client = voyageai.Client()

def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)

def generate_embedding(text, model="voyage-3-large", input_type="query"):
    result = client.embed([text], model=model, input_type=input_type)

    return result.embeddings[0]

with open("./report.md", "r") as f:
    text = f.read()

chunks = chunk_by_section(text)

print(generate_embedding(chunks[0]))