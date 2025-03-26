import pandas as pd
import os

# Define the root folder
root_folder = "results"

# Iterate over all CSV files in subfolders
for subdir, _, files in os.walk(root_folder):
    for file in files:
        file_path = os.path.join(subdir, file)
        
        # Rename files ending with '_concept_0.csv' to '_aspect_0.csv'
        if file.endswith("_concept_0.csv"):
            new_file_name = file.replace("_concept_0.csv", "_aspect_0.csv")
            new_file_path = os.path.join(subdir, new_file_name)
            os.rename(file_path, new_file_path)
            print(f"Renamed file: {file_path} -> {new_file_path}")

print("Processing complete.")
