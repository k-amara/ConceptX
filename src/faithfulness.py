import pandas as pd
import numpy as np
import random

from model import Model
from explainers import *
from utils import arg_parse, load_pkl, load_vectorizer, get_path
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


def process_dataframe(df, model, vectorizer, thresholds=np.arange(0, 1.1, 0.1), method="random"):
    """Process dataframe to compute faithfulness scores across thresholds."""
    results = []
    for _, row in df.iterrows():
        entry = {"id": row["id"], "instruction": row["instruction"], "explanation": row["explanation"]}
        original_response = model.generate(row["instruction"])
        
        for k in thresholds:
            masked_dict = mask_tokens(row["explanation"], k)
            new_instruction = transform_tokens(masked_dict, method)
            new_response = model.generate(new_instruction)
            similarity = evaluate_similarity(original_response, new_response, vectorizer)
            entry[f"sim_{k:.1f}"] = similarity
        results.append(entry)
    
    return pd.DataFrame(results)


def eval_faithfulness(args, save=True):
    if args.seed is not None:
        set_seed(args.seed)
        
    model = Model(args)
    vectorizer = load_vectorizer(args.vectorizer)
    
    df_explanations = load_pkl(args, folder_name="explanations")
    print(df_explanations.head())
    faithfulness_df = process_dataframe(df_explanations, model, vectorizer, method=args.masking_method)
    print("Faithfulness Scores", faithfulness_df.head())
    if save:
        faitfulness_path = get_path(args, folder_name="faithfulness", type="pkl")
        faithfulness_df.to_pickle(faitfulness_path)
        

if __name__=="__main__":
    parser, args = arg_parse()
    eval_faithfulness(args)

