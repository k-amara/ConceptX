import pandas as pd
import os
import re

from explainers import ConceptSplitter

"""if __name__ == "__main__":
    data_save_dir = "/cluster/home/kamara/conceptx/data/"
    df_final = pd.read_csv(os.path.join(data_save_dir, "sst2_classification.csv"))
    df = df_final[:10]
    splitter = ConceptSplitter()
    
    for i in range(len(df)):  # Process each input one by one
        row = df.iloc[i]
        prompt = row["input"]
        prompt_cleaned = prompt.strip()
        prompt_cleaned = re.sub(r'\s+', ' ', prompt_cleaned)
        
        words = splitter.split(prompt_cleaned)
        print("Prompt: ", prompt)
        print("prompt cleaned: ", prompt_cleaned)
        concepts, indices = splitter.split_concepts(prompt_cleaned) # concepts are the samples in TokenSHAP
        replacements = splitter.get_replacements(concepts, prompt_cleaned, replace = "antonym")
        print("Concepts: ", concepts)
        print("Replacements: ", replacements)"""
        
if __name__ == "__main__":
    splitter = ConceptSplitter()
    prompt = "Describe the fear of flying."#"Create a personnification of the sun."#"Choose the synonym of amazing."
    prompt_cleaned = prompt.strip()
    prompt_cleaned = re.sub(r'\s+', ' ', prompt_cleaned)
    
    words = splitter.split(prompt_cleaned)
    print("Prompt: ", prompt)
    print("prompt cleaned: ", prompt_cleaned)
    concepts, indices = splitter.split_concepts(prompt_cleaned) # concepts are the samples in TokenSHAP
    replacements = splitter.get_replacements(concepts, prompt_cleaned, replace = "neutral")
    print("Concepts: ", concepts)
    print("Replacements: ", replacements)