import pandas as pd
import numpy as np
import random

from model import LLMPipeline, LLMAPI
from explainers import *
from utils import arg_parse, load_file, save_dataframe, load_vectorizer
from accelerate.utils import set_seed
import requests

# Download Google's 1M common words dataset
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
VOCAB = requests.get(url).text.split("\n")


def mask_tokens(score_dict, k):
    """Convert scores into a mask of 0s and 1s based on the top-k percentage."""
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    top_k_count = int(len(sorted_items) * k)
    threshold = sorted_items[top_k_count - 1][1] if top_k_count > 0 else float("inf")
    return {token: 1 if score >= threshold else 0 for token, score in score_dict.items()}


def transform_tokens(masked_dict, method="random"):
    """Replace masked tokens with random words from vocab or ellipsis."""
    transformed_tokens = []
    for token_pos, mask in masked_dict.items():
        token = token_pos.rsplit("_", 1)[0]  # Extract the token
        if mask == 0:
            if method == "random" and VOCAB:
                token = random.choice(VOCAB)
            elif method == "ellipsis":
                token = "..."
        transformed_tokens.append(token)
    return " ".join(transformed_tokens)


def evaluate_similarity(original_response, new_response, vectorizer):
    """Compute semantic similarity between original and new response."""
    # Use the configured vectorizer
    vectors = vectorizer.vectorize([original_response, new_response])
    cosine_similarity = vectorizer.calculate_similarity(vectors[0], vectors[1])
    return cosine_similarity


def process_dataframe(df, llm, vectorizer, thresholds=np.arange(0, 1.1, 0.1), method="random"):
    """Process dataframe to compute faithfulness scores across thresholds."""
    results = []
    for _, row in df.iterrows():
        entry = {"id": row["id"], "instruction": row["instruction"], "explanation": row["explanation"]}
        original_response = llm.generate(row["instruction"])
        
        for k in thresholds:
            masked_dict = mask_tokens(row["explanation"], k)
            new_instruction = transform_tokens(masked_dict, method)
            new_response = llm.generate(new_instruction)
            similarity = evaluate_similarity(original_response, new_response, vectorizer)
            entry[f"sim_{k:.1f}"] = similarity
        results.append(entry)
    
    return pd.DataFrame(results)


def eval_faithfulness(args, save=True):
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4", "deepseek"] else False 
    llm = LLMAPI(args) if api_required else LLMPipeline(args)
    vectorizer = load_vectorizer(args.vectorizer)
    
    df_explanations = load_file(args, folder_name="explanations")
    print(df_explanations.head())
    faithfulness_df = process_dataframe(df_explanations, llm, vectorizer, method=args.masking_method)
    print("Faithfulness Scores", faithfulness_df.head())
    if save:
        save_dataframe(faithfulness_df, args, folder_name="faithfulness")

if __name__=="__main__":
    parser, args = arg_parse()
    eval_faithfulness(args)

