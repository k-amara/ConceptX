from openai import OpenAI
import csv
from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset
from openai import OpenAI, BadRequestError
from model import ContentPolicyViolationError
from utils import get_remaining_df
import gc
import re
# Load the .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GPT4O_MINI_API_KEY"), # OpenAI API Key
    base_url="https://aikey-gateway.ivia.ch" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Keep only letters, numbers, and spaces
            text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

def get_sentiment_label(text, sentiment, model="azure/gpt-4o-mini", temperature=1.0):
    prompt = f"""
    Given the following text and its sentiment label, return ONLY the single word most responsible for conveying that sentiment.
    Text: {text}, Sentiment label: {sentiment}.
    Response: 
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False
        )
        # Check if response is valid and contains choices
        if response and hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if message and hasattr(message, "content") and message.content:
                return message.content.strip()
        
        # If response is invalid, return a default value or raise an error
        raise ValueError("Invalid response: No content received from API.")
    
    except BadRequestError as e:
        error_message = str(e)
        if "ContentPolicyViolationError" in error_message:
            raise ContentPolicyViolationError("Azure OpenAI blocked the request due to content policy violation.")
        else:
            raise  # Re-raise other errors
        


if __name__=="__main__":
    # Example usage:
    # Read instructions from the instructions_by_domain which already has id for each instruction and the domain associated
    ds = load_dataset("stanfordnlp/sst2")
    df = pd.DataFrame(ds['train'])
    df_filtered = df[(df['sentence'].str.len() > 29) & (df['sentence'].str.len() <= 56)]
    df_filtered["label"] = df_filtered["label"].map({0: "negative", 1: "positive"})
    df_final = df_filtered[['idx', 'sentence', 'label']].rename(columns={'idx': 'id', 'sentence': 'input', 'label': 'aspect'})

    fieldnames = ["id", "input", "aspect", "label"]
    filename = "data/sst2_classification.csv"
    
    file_exists = os.path.isfile(filename)  # Check if file exists
    df = get_remaining_df(df_final, filename)
    for index, row in df.iterrows():
        try: 
            row["label"] = get_sentiment_label(row["input"], row["aspect"])
        except ContentPolicyViolationError:
            continue
        except BadRequestError:
            continue
        except ValueError:
            continue  # Skip this example if the API returns an invalid response
        
         # Store in a DataFrame
        row_df = pd.DataFrame([row])
        # Append the single row to the CSV (write header only for the first instance)
        row_df.to_csv(filename, mode="a", header=not file_exists, index=False)
        file_exists = True  # Ensure header is not written again
        
        del row, row_df
        gc.collect()