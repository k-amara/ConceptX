import pandas as pd
import csv
import os


def load_data(dataset_name, data_save_dir):
    # Load dataset based on argument
    if dataset_name == "alpaca":
        df = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    elif dataset_name == "genderbias":
        pd.reas_csv(os.join(data_save_dir, "stereotypical_temp_0.8_responses.csv"))
        # ['id', 'instruction', 'reference', 'gender']
    else:
        raise ValueError("Unknown dataset type passed: %s!" % dataset_name)
    
  ## final dataset df.columns ['id', 'instruction', 'reference_text']