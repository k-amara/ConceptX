import pandas as pd
import numpy as np
import random

from utils import arg_parse, load_file, save_dataframe

def get_explanation_ranks(explanation_scores):
    # Sort tokens based on their scores in descending order
    ranked_tokens = sorted(explanation_scores.items(), key=lambda x: x[1], reverse=True)
    rank_dict = {token.split('_')[0]: rank + 1 for rank, (token, _) in enumerate(ranked_tokens)}
    return rank_dict


def top_explanation_difference(explanation_scores):
    # Sort tokens by score in descending order
    sorted_scores = sorted(explanation_scores.values(), reverse=True)
    # Ensure there are at least two elements to compare
    if len(sorted_scores) < 2:
        return None  # Not enough data to compute a difference
    return sorted_scores[0] - sorted_scores[1]

def compute_acc_metrics(df):
    results = []
    for _, row in df.iterrows():
        entry = {"id": row["id"], "instruction": row["instruction"], "explanation": row["explanation"]}
        rank_dict = get_explanation_ranks(row["explanation"])
        entry["label_rank"] = rank_dict.get(row["gender"], None)
        entry["difference"] = top_explanation_difference(row["explanation"])
        results.append(entry)
    
    return pd.DataFrame(results)

def get_summary_scores(df):
    correct = df["label_rank"].notnull() & (df["label_rank"] == 1)
    results = {
        "accuracy": correct.sum() / len(df),
        "mean_difference": df["difference"].mean()
    }
    return results

def eval_accuracy(args, save=True):
    df_explanations = load_file(args, folder_name="explanations")
    print(df_explanations.head())
    accuracy_df = compute_acc_metrics(df_explanations)
    summary_scores = get_summary_scores(accuracy_df)
    print("accuracy Scores", accuracy_df.head())
    print("Summary Scores", summary_scores)
    if save:
        save_dataframe(accuracy_df, args, folder_name="accuracy")

if __name__=="__main__":
    parser, args = arg_parse()
    eval_accuracy(args)