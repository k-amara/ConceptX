import pandas as pd
import numpy as np
import os
import argparse
from utils import arg_parse, load_file, save_dataframe, load_labels, get_path

def get_explanation_ranks(explanation_scores):
    # Sort tokens based on their scores in descending order
    ranked_tokens = sorted(explanation_scores.items(), key=lambda x: x[1], reverse=True)
    rank_dict = {token.split('_')[0]: rank + 1 for rank, (token, _) in enumerate(ranked_tokens)}
    return rank_dict


def top_label_explanation_difference(explanation_scores, label):
    # Ensure the label exists in explanation_scores
    explanation_scores = {token.split('_')[0]: score for token, score in explanation_scores.items()}
    if label not in explanation_scores:
        return None  # Label not found in the explanation scores
    sorted_scores = sorted(explanation_scores.values(), reverse=True)
    # Ensure there is at least one top score to compare
    if len(sorted_scores) == 0:
        return None  # No scores available to compare
    
    top_score = sorted_scores[0]
    label_score = explanation_scores.get(label, None)
    if label_score is None:
        return None  # Label score is missing
    
    return top_score - label_score


def top_explanation_difference(explanation_scores):
    # Sort tokens by score in descending order
    sorted_scores = sorted(explanation_scores.values(), reverse=True)
    # Ensure there are at least two elements to compare
    if len(sorted_scores) < 2:
        return None  # Not enough data to compute a difference
    return sorted_scores[0] - sorted_scores[1]

def compute_acc_metrics(df, args):
    results = []
    labels = load_labels(args)
    # Merge df and labels on the 'id' column
    merged_df = pd.merge(df, labels[['id', 'label']], on='id', how='left')
    for _, row in merged_df.iterrows():
        entry = {"id": row["id"], "instruction": row["instruction"]}
        rank_dict = get_explanation_ranks(eval(row["explanation"]))
        entry["label_rank"] = rank_dict.get(row["label"], None)
        entry["top_3_tokens"] = [token for token, rank in sorted(rank_dict.items(), key=lambda item: item[1])[:3]]
        entry["top_difference"] = top_explanation_difference(eval(row["explanation"]))
        entry["top_label_difference"] = top_label_explanation_difference(eval(row["explanation"]), row["label"])
        results.append(entry)
    
    return pd.DataFrame(results)

def get_summary_scores(df):
    correct = df["label_rank"].notnull() & (df["label_rank"] == 1)
    results = {
        "accuracy": correct.sum() / len(df),
        "mean_top_difference": df["top_difference"].mean(),
        "mean_top_label_difference": df["top_label_difference"].mean()
    }
    return results

def eval_accuracy(args, save=True):
    df_explanations = load_file(args, folder_name="explanations")
    print(df_explanations.head())
    accuracy_df = compute_acc_metrics(df_explanations, args)
    summary_scores = get_summary_scores(accuracy_df)
    print("accuracy Scores", accuracy_df.head())
    print("Summary Scores", summary_scores)
    if save:
        save_dataframe(accuracy_df, args, folder_name="accuracy")
        

def get_explanations_accuracy(args):
    explanations_dir = os.path.join(args.result_save_dir, "explanations")
    
    if not os.path.exists(explanations_dir):
        print(f"Explanations directory not found: {explanations_dir}")
        return
    
    for root, _, files in os.walk(explanations_dir):
        for file in files:
            if file.endswith(args.file_type):
                # Extract model_name, dataset, explainer, and seed from folder structure
                path_parts = root.split(os.sep)
                print("path_parts: ", path_parts)
                try:
                    model_name = path_parts[-4]  # Example: explanations/model/dataset/explainer/0/
                    dataset = path_parts[-3]
                    explainer = path_parts[-2]
                    seed = path_parts[-1].split("_")[1]  # Extract seed (e.g., "0")
                except IndexError:
                    print(f"Skipping {root}, unexpected folder structure.")
                    continue
                print("seed")
                # Initialize args_dict with extracted values
                args_dict = {
                    "result_save_dir": args.result_save_dir,
                    "data_save_dir": args.data_save_dir,
                    "num_batch": None,
                    "dataset": dataset,
                    "model_name": model_name,
                    "explainer": explainer,
                    "baseline": None,
                    "seed": seed,
                    "file_type": args.file_type
                }

                # Extract optional num_batch and baseline from filename
                parts = file.split("_")
                
                # Check if 'batch' exists and handle accordingly
                if "batch" in parts[1]:
                    # If there's no 'batch_' in the string, skip split('batch_')
                    batch_part = parts[1].split("batch")
                    if len(batch_part) > 1:
                        args_dict["num_batch"] = batch_part[1]
                
                if len(parts) > 5:  # Check if baseline exists in filename
                    args_dict["baseline"] = parts[-2]

                print("args_dict", args_dict)
                # Convert args_dict to argparse.Namespace
                extracted_args = argparse.Namespace(**args_dict)

                # Get expected accuracy file path
                accuracy_path = get_path(extracted_args, folder_name="accuracy")

                if not os.path.exists(accuracy_path):
                    print(f"Processing: {file}")
                    eval_accuracy(extracted_args)
                else:
                    print(f"Skipping: {file} (already processed)")
                    

# Example usage
# args should contain result_save_dir and file_type at minimum
# process_explanations(args)


if __name__=="__main__":
    parser, args = arg_parse()
    get_explanations_accuracy(args)