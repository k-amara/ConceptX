from explainers import *
import pandas as pd
import os
import gc
import numpy as np
import random
import copy
from scipy.stats import entropy
import pickle as pkl
from model import LLMPipeline, LLMAPI, ContentPolicyViolationError
from utils import arg_parse, load_data, load_vectorizer, get_remaining_df, merge_args
from accelerate.utils import set_seed
from faithfulness import process_dataframe
from accuracy import compute_acc_metrics

import requests
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()

def get_path_sampling_ratio(args, folder_name):
    save_dir = os.path.join(args.result_save_dir, f'sampling_ratio/{folder_name}/{args.model_name}/{args.dataset}/{args.explainer}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{folder_name}_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.explainer}_{args.seed}_{args.sampling_ratio}.{args.file_type}"
    return os.path.join(save_dir, filename)

def load_file_sampling_ratio(args, folder_name):
    # Load explanations from the specified path
    data_path = get_path_sampling_ratio(args, folder_name)
    print("data_path: ", data_path)
    if args.file_type == "pkl":
        with open(data_path, "rb") as f:
            data = pkl.load(f)
        df = pd.DataFrame(data)
    elif args.file_type == "csv":
        df = pd.read_csv(data_path)
    return df



def extract_args_from_filename_sampling_ratio(file):
    """
    Extract arguments from the file name and return a dictionary of arguments.
    
    Args:
        file (str): The file name.
        args (argparse.Namespace): The existing args to merge with extracted values.

    Returns:
        dict: Dictionary of extracted arguments.
    """
    parts = file.split("_")
    print("parts: ", parts)
    
    args_dict = {
        "num_batch": None,
        "dataset": None,
        "model_name": None,
        "explainer": None,
        "baseline": None,
        "seed": None
    }

    if "batch" in parts[1]:
        args_dict["num_batch"] = int(parts[2])
        dataset_idx = 3
    else:
        dataset_idx = 1
    
    args_dict["dataset"] = parts[dataset_idx]
    args_dict["model_name"] = parts[dataset_idx + 1]
    args_dict["explainer"] = parts[dataset_idx + 2]
    
    if args_dict["model_name"] == "gpt4o":
        return None  # Skip if model_name is "gpt4o"
    
    seed_idx = dataset_idx + 3
    args_dict["seed"] = int(parts[seed_idx])
    args_dict["sampling_ratio"] = float(parts[seed_idx+1].replace(".csv", ""))
    return args_dict

def eval_accuracy_sampling_ratio(args, save=True):
    
    label_args = copy.deepcopy(args)
    label_args.num_batch = None
    labels = load_data(label_args)[['id', 'label', 'aspect']]
    
    df_explanation = load_file_sampling_ratio(args, folder_name="explanations")
        
    accuracy_path = get_path_sampling_ratio(args, folder_name="accuracy")
    file_exists = os.path.isfile(accuracy_path)  # Check if file exists
    df = get_remaining_df(df_explanation, accuracy_path)
    
    accuracy_df = compute_acc_metrics(df, labels)
    if save:
        # Append the single row to the CSV (write header only for the first instance)
        accuracy_df.to_csv(accuracy_path, mode="a", header=not file_exists, index=False)
        file_exists = True  # Ensure header is not written again
    # Clear cache to free memory
    del accuracy_df
    gc.collect()

def compute_explanations(args, save=True):
    if args.seed is not None:
        set_seed(args.seed)

    api_required = args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"]
    rate_limit = args.model_name.startswith("gpt4")
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)

    vectorizer_kwargs = {}
    if args.vectorizer == "huggingface":
        vectorizer_kwargs["embedding_model"] = args.embedding_model
    vectorizer = load_vectorizer(args.vectorizer, **vectorizer_kwargs)

    df = load_data(args)
    print(df.head())

    # Choose appropriate explainer based on specified explainer
    kwargs = {}
    if args.explainer == "tokenshap":
        splitter = StringSplitter()
        explainer = TokenSHAP(llm, splitter, vectorizer, debug=False, sampling_ratio=args.sampling_ratio)
    elif args.explainer.startswith("conceptx"):
        splitter = ConceptSplitter()
        explainer = ConceptX(llm, splitter, vectorizer, debug=False, sampling_ratio=args.sampling_ratio, replace=args.coalition_replace)
        baseline_texts = None
        if args.target == "reference":
            baseline_texts = df['reference'].tolist()
        elif args.target == "aspect":
            baseline_texts = df['aspect'].tolist()
        kwargs = {"baseline_texts": baseline_texts} if baseline_texts is not None else {}
    else:
        raise ValueError(f"Unsupported explainer: {args.explainer}. Only 'tokenshap' and 'conceptx' are allowed.")

    explanations_path = get_path_sampling_ratio(args, folder_name="explanations")
    file_exists = os.path.isfile(explanations_path)
    df = get_remaining_df(df, explanations_path)

    for i in range(len(df)):
        input = df.iloc[i]["input"]
        if not isinstance(input, str):
            continue
        input_id = df.iloc[i]["id"]

        try:
            response = llm.generate(input)
            explanation_list = explainer([input], **kwargs)
            if not explanation_list:
                continue

            explanation = explanation_list[0]

            row_df = pd.DataFrame([{
                "id": input_id,
                "input": input,
                "response": response,
                "explanation": explanation
            }])

            if save:
                row_df.to_csv(explanations_path, mode="a", header=not file_exists, index=False)
                file_exists = True
            else:
                print(f"Explanation id {input_id}: ", row_df)

            # Memory cleanup
            del input, response, input_id, explanation, row_df
            gc.collect()

        except ContentPolicyViolationError:
            continue

    return

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
    df_explanation = load_file_sampling_ratio(args, folder_name="explanations")
    faithfulness_path = get_path_sampling_ratio(args, folder_name=fname)
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


def get_explanations_accuracy(args):
        
    explanations_dir = os.path.join(args.result_save_dir, "sampling_ratio/explanations")
    print("explanations_dir: ", explanations_dir)
    if not os.path.exists(explanations_dir):
        print(f"Explanations directory not found: {explanations_dir}")
        return
    
    # Walk through all subdirectories
    for root, _, files in os.walk(explanations_dir):
        for file in files:
            if file.endswith(args.file_type):
                # Extract arguments from filename
                args_dict = extract_args_from_filename_sampling_ratio(file)    
                print("args_dict: ", args_dict)
                if args_dict["dataset"].startswith("genderbias"):
                    # Convert dictionary to argparse.Namespace
                    updated_args = merge_args(args, args_dict)
                    print("updated_args: ", updated_args)
                    # Get expected accuracy file path
                    eval_accuracy_sampling_ratio(updated_args)





if __name__ == "__main__":
    parser, args = arg_parse()
    print('args do_sample:', args.do_sample)
    print('args result_save_dir:', args.result_save_dir)
    print('args sampling_ratio:', args.sampling_ratio)
    #compute_explanations(args)
    get_explanations_accuracy(args)
    #eval_faithfulness(args)
