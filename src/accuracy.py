import pandas as pd
import numpy as np
import os
from utils import arg_parse, merge_args, load_file, extract_args_from_filename, save_dataframe, load_labels, get_path

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
    
    # Walk through all subdirectories
    for root, _, files in os.walk(explanations_dir):
        for file in files:
            if file.endswith(args.file_type):
                # Extract arguments from filename
                args_dict = extract_args_from_filename(file)

                # Convert dictionary to argparse.Namespace
                updated_args = merge_args(args, args_dict)

                # Get expected accuracy file path
                accuracy_path = get_path(updated_args, folder_name="accuracy")

                if not os.path.exists(accuracy_path):
                    print(f"Processing: {file}")
                    eval_accuracy(updated_args)
                else:
                    print(f"Skipping: {file} (already processed)")

                    

# Example usage
# args should contain result_save_dir and file_type at minimum
# process_explanations(args)


if __name__=="__main__":
    parser, args = arg_parse()
    get_explanations_accuracy(args)