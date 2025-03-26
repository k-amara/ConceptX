
import pandas as pd
import re

# Load the dataset
filename = "data/stereotypical_temp_0.8_responses.csv"
df = pd.read_csv(filename)

def replace_gender(text, original, replacement):
    """Replace gender-specific words while preserving case."""
    return re.sub(rf'\b{original}\b', replacement, text, flags=re.IGNORECASE)

# Create new versions of the dataset with replacements
df_female_male = df.copy()
df_female_male["input"] = df_female_male["input"].apply(lambda x: replace_gender(x, "woman", "female"))
df_female_male["input"] = df_female_male["input"].apply(lambda x: replace_gender(x, "man", "male"))
df_female_male["label"] = df_female_male["label"].replace({"woman": "female", "man": "male"})

# Save the new dataset
output_filename = "data/stereotypical_temp_0.8_responses_v2.csv"
df_female_male.to_csv(output_filename, index=False)

print(f"Modified dataset saved to {output_filename}")