# Please install OpenAI SDK first: `pip3 install openai`
import os
import pandas as pd
import random
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
from model import ContentPolicyViolationError
import requests
import re

import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchInductor errors
torch._dynamo.reset()  # Reset inductor state

# Download Google's 1M common words dataset
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
VOCAB = requests.get(url).text.split("\n")

# Load the .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GPT4O_MINI_API_KEY"), # OpenAI API Key
    base_url="https://aikey-gateway.ivia.ch" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)



def create_prompt_for_neutral_replacement(sentence, input_concepts):             
    prompt = f"""
        You are an AI assistant that neutralizes concepts in sentences. Your task is to replace given concepts with neutral alternatives that neutralize their semantic importance while preserving grammatical correctness. The replacements must NOT be synonyms or somehow close in meaning.

        Example Input:
        "sentence": "Describe the ideal qualities of a leader in a team.",
        "input_concepts": ["Describe", "qualities", "leader", "team"]
        Example Output:
        "replacements": ["Mention", "aspects", "individual", "group"]

        Given the following sentence and concepts:

        Sentence: "{sentence}"
        Concepts: {input_concepts}

        For each concept, replace it with a new word that:
        - Neutralizes its semantic importance. This will strongly weaken their semantic importance in the sentence.
        - Preserves grammatical correctness.
        - Is NOT a synonym or somehow close in meaning.

        Return only a Python list of concepts in this format:
        ["neutralized_concept_1", "neutralized_concept_2", "neutralized_concept_3", ...]
        Please do not include any additional explanation, sentences, or content other than the list.
        """
    return prompt

def create_prompt_for_antonym_replacement(sentence, input_concepts):             
    prompt = f"""
        You are an AI assistant that neutralizes concepts in sentences. Your task is to replace given concepts with antonyms while preserving grammatical correctness. 
        
        Example Input:
        "sentence": "Describe the ideal qualities of a leader in a team.",
        "input_concepts": ["Describe", "qualities", "leader", "team"]
        Example Output:
        "replacements": ["Ignore", "flaws", "follower", "individual"]

        Given the following sentence and concepts:

        Sentence: "{sentence}"
        Concepts: {input_concepts}

        For each concept, replace it with a new word that:
        - Neutralizes its semantic importance by having the opposite meaning.
        - Preserves grammatical correctness.

        Return only a Python list of concepts in this format:
        ["antonym_concept_1", "antonym_concept_2", "antonym_concept_3", ...]
        Please do not include any additional explanation, sentences, or content other than the list.
        """
    return prompt


def get_multiple_completions(prompt, model="azure/gpt-4o-mini", num_sequences=3, temperature=1.0):
    responses = []
    
    for _ in range(num_sequences):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                stream=False
            )
            responses.append(response.choices[0].message.content)
        
        except BadRequestError as e:
            error_message = str(e)
            if "ResponsibleAIPolicyViolation" in error_message or "ContentPolicyViolationError" in error_message:
                raise ContentPolicyViolationError("Azure OpenAI blocked the request due to content policy violation.")
            else:
                raise  # Re-raise other errors
            
    return responses


def get_antonym_conceptnet(word):
    url = f"https://api.conceptnet.io/query?rel=/r/Antonym&start=/c/en/{word.lower()}&limit=10"
    response = requests.get(url)
    
    # Check if the response is OK
    if response.status_code != 200:
        print(f"Error querying ConceptNet for word '{word}': {response.status_code}")
        return None

    try:
        data = response.json()
    except ValueError:
        print(f"Invalid JSON response from ConceptNet for word '{word}'")
        return None
    
    antonyms = []
    for edge in data.get("edges", []):
        end = edge["end"]
        if end["@id"].startswith("/c/en/"):
            antonym = end["@id"].split("/c/en/")[-1]
            antonyms.append(antonym)
    if antonyms:
        return antonyms[0]
    else:
        return None

    
    
def remove_token(explanation):
    # Extract tokens, positions, and scores
    tokens_scores = [(key.rsplit('_', 1)[0], int(key.rsplit('_', 1)[1]), value) for key, value in explanation.items()]
    
    # Find the token with the highest score
    highest_token, highest_position, _ = max(tokens_scores, key=lambda x: x[2])
    # random_token = random.choice(VOCAB)
    sorted_tokens = sorted(tokens_scores, key=lambda x: x[1])
    sentence_highest = " ".join(
        token for token, token_position, _ in sorted_tokens
        if not (token == highest_token and token_position == highest_position)
    )
    return sentence_highest, highest_token



def replace_token_with_antonym(explanation):
    # Extract tokens, positions, and scores
    tokens_scores = [(key.rsplit('_', 1)[0], int(key.rsplit('_', 1)[1]), value) for key, value in explanation.items()]
    
    # Find the token with the highest score
    highest_token, highest_position, _ = max(tokens_scores, key=lambda x: x[2])

    # Try to get an antonym
    antonym = get_antonym_conceptnet(highest_token.lower())
    print(f"The antonym of {highest_token} is: ", antonym)
    if antonym is None:
        antonym = random.choice(VOCAB) # fallback: keep the original if no antonym found

    # Rebuild the sentence with the antonym replacing the harmful token
    sorted_tokens = sorted(tokens_scores, key=lambda x: x[1])
    sentence_with_antonym = " ".join(
        antonym if (token == highest_token and token_position == highest_position) else token
        for token, token_position, _ in sorted_tokens
    )

    return sentence_with_antonym, highest_token


def remove_label(sentence, word):
    if pd.isna(word):
        return sentence  # If word is NaN, do nothing and return the original sentence
    
    word = str(word)  # Convert to string just in case it's a float/int
    
    pattern = r'\b' + re.escape(word) + r'\b'
    cleaned_sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
    return cleaned_sentence

def replace_label_with_antonym(sentence, word):
    """Replace the given word in the sentence with its antonym from ConceptNet."""
    if pd.isna(word):
        return sentence  # If word is NaN, do nothing and return the original sentence
    
    antonym = get_antonym_conceptnet(word.lower())
    print(f"The antonym of '{word}' is: {antonym}")
    
    if antonym is None:
        antonym = random.choice(VOCAB)  # fallback
    
    # Regex replacement preserving surrounding punctuation
    pattern = r'\b' + re.escape(word) + r'\b'
    replaced_sentence = re.sub(pattern, antonym, sentence, flags=re.IGNORECASE)
    return replaced_sentence

    




if __name__=="__main__":
    sentence = "The artist created a stunning masterpiece that captivated the audience."
    input_concepts = ["artist", "stunning", "masterpiece", "captivated", "audience"]
    
    # Example usage
    prompt = create_prompt_for_replacement(sentence, input_concepts)
    completions = get_multiple_completions(prompt, num_sequences=3)

    # Print all k generated completions
    for i, completion in enumerate(completions):
        print(f"Completion {i+1}: {completion}\n")