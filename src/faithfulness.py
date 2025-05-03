import pandas as pd
import numpy as np
import random
import os
import gc

from model import LLMPipeline, LLMAPI, ContentPolicyViolationError
from explainers import *
from utils import arg_parse, merge_args, load_file, extract_args_from_filename, load_vectorizer, get_path, get_remaining_df
from accelerate.utils import set_seed
import requests

import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchInductor errors
torch._dynamo.reset()  # Reset inductor state

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
    original_response = str(original_response)
    new_response = str(new_response)
    vectors = vectorizer.vectorize([original_response, new_response])
    cosine_similarity = vectorizer.calculate_similarity(vectors[0], vectors[1])
    return cosine_similarity


def process_dataframe(row, llm, vectorizer, thresholds=np.arange(0, 1.1, 0.1), method="random"):
    """Process dataframe to compute faithfulness scores across thresholds."""
    entry = {"id": row["id"], "input": row["input"]}
    original_response = row["response"]
    
    for k in thresholds:
        masked_dict = mask_tokens(eval(row["explanation"]), k)
        new_input = transform_tokens(masked_dict, method)
        try:
            new_response = llm.generate(new_input)
        except ContentPolicyViolationError:
            continue  # Skip inputs that raise the error
        similarity = evaluate_similarity(original_response, new_response, vectorizer)
        entry[f"sim_{k:.1f}"] = similarity
    
    return entry


def eval_faithfulness(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
    
    vectorizer_kwargs = {}
    if args.vectorizer == "huggingface":
        vectorizer_kwargs["embedding_model"] = args.embedding_model
    vectorizer = load_vectorizer(args.vectorizer, **vectorizer_kwargs)
    
    fname = "faithfulness"
    df_explanation = load_file(args, folder_name="explanations")
    faithfulness_path = get_path(args, folder_name=fname)
    file_exists = os.path.isfile(faithfulness_path)  # Check if file exists
    df = get_remaining_df(df_explanation, faithfulness_path)
    print("remaining df: ", df.head())
    for i in range(len(df)):  # Process each input one by one
        row = df.iloc[i]
        input_id = df.iloc[i]["id"]
        # Check if any value is NaN
        row_dict = eval(row["explanation"], {"np": np, "nan": np.nan})
        contains_nan = any(np.isnan(value) for value in row_dict.values())
        if contains_nan:
            continue
        entry = process_dataframe(row, llm, vectorizer, method=args.masking_method)
        # Store in a DataFrame
        row_df = pd.DataFrame([entry])

        if save:
            # Append the single row to the CSV (write header only for the first instance)
            row_df.to_csv(faithfulness_path, mode="a", header=not file_exists, index=False)
            file_exists = True  # Ensure header is not written again
        else: 
            print(f"Faithfulness score at id {input_id}: ", row_df)
        # Clear cache to free memory
        del row, entry, row_df
        gc.collect()



                    

# Example usage
# args should contain result_save_dir and file_type at minimum
# process_explanations(args)


if __name__=="__main__":
    parser, args = arg_parse()
    eval_faithfulness(args)

