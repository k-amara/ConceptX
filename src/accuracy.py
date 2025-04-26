import pandas as pd
import numpy as np
import os
import copy
from scipy.stats import entropy
from utils import arg_parse, merge_args, load_file, extract_args_from_filename, save_dataframe, load_data, get_path, get_remaining_df

import gc

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


def entropy(explanation_scores):
    positive_scores = np.array([v for v in explanation_scores.values() if v > 0])

    if positive_scores.sum() > 0:
        prob_distribution = positive_scores / positive_scores.sum()
        return -np.sum(prob_distribution * np.log2(prob_distribution))  # Use np.log2()
    else:
        print("No positive scores found, entropy is undefined.")
        return None
        

def compute_acc_metrics(df, labels):
    results = []
    # Merge df and labels on the 'id' column
    merged_df = pd.merge(df, labels[['id', 'label', 'aspect']], on='id', how='left')
    for _, row in merged_df.iterrows():
        entry = {"id": row["id"], "input": row["input"]}
        explanation = eval(row["explanation"])
        rank_dict = get_explanation_ranks(explanation)
        entry["label_rank"] = rank_dict.get(row["label"], None)
        entry["top_3_tokens"] = [token for token, rank in sorted(rank_dict.items(), key=lambda item: item[1])[:3]]
        entry["top_difference"] = top_explanation_difference(explanation)
        entry["top_label_difference"] = top_label_explanation_difference(explanation, row["label"])
        entry["entropy"] = entropy(explanation)
        entry["label"] = row["label"]
        entry["aspect"] = row["aspect"]
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
    
    label_args = copy.deepcopy(args)
    label_args.num_batch = None
    labels = load_data(label_args)[['id', 'label', 'aspect']]
    
    df_explanation = load_file(args, folder_name="explanations")
        
    accuracy_path = get_path(args, folder_name="accuracy")
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
        

def get_explanations_accuracy(args):
        
    explanations_dir = os.path.join(args.result_save_dir, "explanations")
    
    if not os.path.exists(explanations_dir):
        print(f"Explanations directory not found: {explanations_dir}")
        return
    
    # Walk through all subdirectories
    for root, _, files in os.walk(explanations_dir):
        for file in files:
            if file.endswith(args.file_type):
                # Extract arguments from filename
                args_dict = extract_args_from_filename(file)
                
                if args_dict["dataset"]=="genderbias":

                    # Convert dictionary to argparse.Namespace
                    updated_args = merge_args(args, args_dict)
                    # Get expected accuracy file path
                    eval_accuracy(updated_args)

                    

# Example usage
# args should contain result_save_dir and file_type at minimum
# process_explanations(args)


if __name__=="__main__":
    parser, args = arg_parse()
    get_explanations_accuracy(args)