import openai
import csv
import re
import random
from openai import OpenAI
import csv
import json
from dotenv import load_dotenv
import os
import pandas as pd
# Load the .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GPT4O_MINI_API_KEY"), # OpenAI API Key
    base_url="https://aikey-gateway.ivia.ch" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)


# Function to extract the sentiment-responsible word
def extract_label(sentence):
    words = sentence.split()
    for word in words:
        if "**" in word:  # Assuming word is emphasized in markdown style
            return word.replace("**", "")
    return None  # If no emphasized word is found

# OpenAI API Call (assuming the API key is set in the environment)
def generate_sentences(sentiment, model="azure/gpt-4o-mini", n=500, temperature=0.8):
    prompt = (
        f"""Generate {n} {sentiment} texts (sentences and part of sentences) with between 4 and 10 words, 
        with only one word (adjective, verb, or noun) responsible for the {sentiment} sentiment. 
        Among the {n} input texts, make sure to uniformly distribute the size of the text generated between 4 and 10 words. 
        Mark the sentiment-responsible word with double asterisks (**word**) for extraction. 
        Provide only the texts, each on a new line, without numbering."""
    )
    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False
        )
    sentences = response["choices"][0]["message"]["content"].split("\n")
    print("responses: ", sentences)
    return [s.strip() for s in sentences if s.strip()]


sentiment = "positive"
# Generate sentences
sentences = generate_sentences(sentiment, n=5)

# Prepare data
rows = []
for idx, sentence in enumerate(sentences, 1):
    label = extract_label(sentence)
    if label:
        sentence = sentence.replace(f"**{label}**", label)  # Remove markdown
        rows.append([idx, sentence, label, sentiment])

# Write to CSV
with open("data/sentiment_classification_v2.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "input", "label", "aspect"])
    writer.writerows(rows)

print("CSV file generated successfully.")
