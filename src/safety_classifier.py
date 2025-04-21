import random
import pandas as pd
import requests
import os
import copy
import math
import re
import numpy as np

from accelerate.utils import set_seed
from transformers import AutoTokenizer,  AutoModelForCausalLM
from scipy.special import softmax
import gc

from model import LLMPipeline, LLMAPI, ContentPolicyViolationError
from explainers import *
from utils import arg_parse, load_file, load_data, get_path, get_remaining_df
from accelerate.utils import set_seed
import requests

import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchInductor errors
torch._dynamo.reset()  # Reset inductor state

# Download Google's 1M common words dataset
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
VOCAB = requests.get(url).text.split("\n")


class SafetyClassifier:
    def __init__(self, model_name="OpenSafetyLab/MD-Judge-v0_2-internlm2_7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

    def __call__(self, input, output):
        # modified from modeling_internlm2.py: def chat()
        # chat with no system instruction
        prompt = self.tokenizer.apply_chat_template([
            {"role": "user", "content": input},
            {"role": "assistant", "content": output}
        ], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        resp = self.tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(resp.strip())
        return resp
        

def replace_harmful_token(explanation):
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

def remove_word(sentence, word):
    # Escape the word in case it contains regex special characters
    pattern = r'\b' + re.escape(word) + r'\b'
    # Remove the word regardless of case
    cleaned_sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
    # Remove extra whitespace
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
    return cleaned_sentence


def eval_label_safety(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
        
    classifier = SafetyClassifier()
    
    df_data = load_data(args)
    
    safety_path = get_path(args, folder_name="safety")
    file_exists = os.path.isfile(safety_path)  # Check if file exists
    df = get_remaining_df(df_data, safety_path)
    print("df: ", df.head())
    
    for _, row in df.iterrows():
        entry = {
            "id": row["id"],
            "input": row["input"],
            "aspect": row["aspect"],
            "label": row["label"]
        }

        sentence_label = remove_word(entry["label"])

        # Step 1: Classify the original input-response pair
        entry["pasr_0"] = classifier(entry["input"], row["response"])

        # Step 2: Generate a response to the sentence with the label removed
        try:
            response_label = llm.generate(sentence_label)
            entry["pasr_label"] = classifier(sentence_label, response_label)
        except ContentPolicyViolationError:
            continue

        # Save or print the result
        row_df = pd.DataFrame([entry])
        if save:
            row_df.to_csv(safety_path, mode="a", header=not file_exists, index=False)
            file_exists = True
        else:
            print(f"safety score at id {entry['id']}: ", row_df)

        # Clear cache to free memory
        del row, entry, row_df, sentence_label
        gc.collect()





def eval_safety(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
        
    classifier = SafetyClassifier()
    
    df_explanation = load_file(args, folder_name="explanations")
    
    safety_path = get_path(args, folder_name="safety")
    file_exists = os.path.isfile(safety_path)  # Check if file exists
    df = get_remaining_df(df_explanation, safety_path)
    print("df: ", df.head())
    
    for _, row in df.iterrows():
        entry = {
            "id": row["id"],
            "input": row["input"],
            "aspect": row["aspect"],
            "label": row["label"]
        }
        explanation = eval(row["explanation"], {"np": np, "nan": np.nan})
        contains_nan = any(np.isnan(value) for value in explanation.values())
        if contains_nan:
            continue
        sentence_highest, highest_token = replace_harmful_token(explanation)
        entry["highest_token"] = highest_token
        try:
            response_highest = llm.generate(sentence_highest)
            entry["pasr_highest"] = classifier(sentence_highest, response_highest)
        except ValueError as e:
            print(f"Skipping aspect due to error: {e}")
            continue
        except ContentPolicyViolationError:
            continue

        # Store in a DataFrame
        row_df = pd.DataFrame([entry])
        
        if save:
            # Append the single row to the CSV (write header only for the first instance)
            row_df.to_csv(safety_path, mode="a", header=not file_exists, index=False)
            file_exists = True  # Ensure header is not written again
        else: 
            print(f"safety score at id {row['id']}: ", row_df)
        # Clear cache to free memory
        del row, entry, row_df, explanation, label, aspect, sentence_highest, sentence_label, highest_token
        gc.collect()
        
        
        
        

       
# def attack_succes_rate():