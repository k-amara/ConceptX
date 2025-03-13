import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import arg_parse
import pickle as pkl

def plot_faithfulness(folder, figure_save_dir):
    """Read all CSV files in folder and plot average similarity for each method."""
    data_list = []
    print("Folder", folder)
    # Walk through the folder and its subfolders
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            print("File:", file_path)
            if file.endswith(".pkl"):
                explainer = file.split("_")[-2]
                print("Explainer", explainer)
                # Open the file in binary read mode
                with open(file_path, 'rb') as file:
                    results = pkl.load(file)
                faithfulnness_df = pd.DataFrame(results)
                print("Faithfulness Scores", faithfulnness_df.head())
                thresholds = np.arange(0, 1.1, 0.1)
                
                for k in thresholds:
                    avg_similarity = faithfulnness_df[f"sim_{k:.1f}"].mean()
                    data_list.append({"Threshold": k, "Average Similarity": avg_similarity, "Explainer": explainer})
        
    plot_df = pd.DataFrame(data_list)
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=plot_df, x="Threshold", y="Average Similarity", hue="Explainer", marker="o")
    plt.xlabel("Threshold k")
    plt.ylabel("Average Similarity")
    plt.title("Faithfulness Evaluation: Similarity vs Threshold (Explainers)")
    plt.grid(True)
    plt.legend(title="Explainer")
    plt.savefig(os.path.join(figure_save_dir, "faithfulness_comparison.png"))
    plt.show()
    
if __name__ == "__main__":
    parser, args = arg_parse()
    plot_faithfulness(os.path.join(args.result_save_dir, f"faithfulness/{args.model_name}/{args.dataset}/"), args.fig_save_dir)