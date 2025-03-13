import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

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


def process_dataframe(df, model, thresholds=np.arange(0, 1.1, 0.1), method="random"):
    """Process dataframe to compute faithfulness scores across thresholds."""
    results = []
    for _, row in df.iterrows():
        entry = {"id": row["id"], "instruction": row["instruction"], "scores": row["scores"]}
        original_response = model.generate(row["instruction"])
        
        for k in thresholds:
            masked_dict = mask_tokens(row["scores"], k)
            new_instruction = transform_tokens(masked_dict, method)
            new_response = model.generate(new_instruction)
            similarity = evaluate_similarity(original_response, new_response)
            entry[f"sim_{k:.1f}"] = similarity
        results.append(entry)
    
    return pd.DataFrame(results)


def plot_faithfulness(folder, figure_save_dir):
    """Read all CSV files in folder and plot average similarity for each method."""
    data_list = []
    
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            method = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(folder, file))
            thresholds = np.arange(0, 1.1, 0.1)
            
            for k in thresholds:
                avg_similarity = df[f"sim_{k:.1f}"].mean()
                data_list.append({"Threshold": k, "Average Similarity": avg_similarity, "Method": method})
    
    plot_df = pd.DataFrame(data_list)
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=plot_df, x="Threshold", y="Average Similarity", hue="Method", marker="o")
    plt.xlabel("Threshold k")
    plt.ylabel("Average Similarity")
    plt.title("Faithfulness Evaluation: Similarity vs Threshold (Methods)")
    plt.grid(True)
    plt.legend(title="Method")
    plt.savefig(os.path.join(figure_save_dir, "faithfulness_comparison.png"))
    plt.show()


def eval_faithfulness(score_folder, faithfulness_folder, method):
    """Evaluate faithfulness for all score files in the given folder and save results."""
    if not os.path.exists(faithfulness_folder):
        os.makedirs(faithfulness_folder)
    
    for file in os.listdir(score_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(score_folder, file))
            faithfulness_df = process_dataframe(df, method=method)
            save_path = os.path.join(faithfulness_folder, f"{method}.csv")
            faithfulness_df.to_csv(save_path, index=False)
            print(f"Saved faithfulness scores for {file} to {save_path}")


#if __name__=="__main__":
    #faithfulness_df = process_dataframe(data, vocab, method="random", llm=mock_llm)
    #faithfulness_df.to_csv("faithfulness.csv", index=False)
    #print(faithfulness_df)
    #get_folder(args)
    # Plot similarity
    #plot_all_methods("faithfulness/model/dataset", "./figures")
    #eval_faithfulness(, , vocab, "random", llm=mock_llm)


