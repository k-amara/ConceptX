import random
import pandas as pd
import requests
import os
import copy
import math
import numpy as np

from accelerate.utils import set_seed
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import gc

from explainers import *
from utils import arg_parse, load_file, load_data, get_sentiment_file_path, get_remaining_df, remove_token, remove_label, replace_token_with_antonym, replace_label_with_antonym
from accelerate.utils import set_seed
import requests

import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchInductor errors
torch._dynamo.reset()  # Reset inductor state

# Download Google's 1M common words dataset
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
VOCAB = requests.get(url).text.split("\n")


class SentimentClassifier:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def preprocess(self, text):
        return " ".join('@user' if t.startswith('@') and len(t) > 1 else 'http' if t.startswith('http') else t for t in text.split())
    
    def get_aspect_score(self, text, aspect):
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        
        scores = softmax(output[0][0].detach().numpy())
        
        # Convert id2label values to lowercase for comparison
        aspect_map = {v.lower(): k for k, v in self.config.id2label.items()}
        if isinstance(aspect, float) and math.isnan(aspect):
            raise ValueError("Aspect is NaN (missing value).")
        if not isinstance(aspect, str) or aspect.lower() not in aspect_map:
            raise ValueError(f"Aspect '{aspect}' not found in model aspects: {list(aspect_map.keys())}")
        
        aspect_index = aspect_map[aspect.lower()]
        return scores[aspect_index]

def sentiment_probability(classifier, sentence, sentence_highest, sentence_label, aspect):
    p0 = classifier.get_aspect_score(sentence, aspect)
    p_highest = classifier.get_aspect_score(sentence_highest, aspect)
    p_label = classifier.get_aspect_score(sentence_label, aspect)
    return p0, p_highest, p_label


def eval_classifier(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    classifier = SentimentClassifier()
    
    label_args = copy.deepcopy(args)
    label_args.num_batch = None
    labels = load_data(label_args)[['id', 'label', 'aspect']]
    
    df_explanation = load_file(args, folder_name="explanations")
    print("Lenght explanations: ", len(df_explanation))
    df_explanation = pd.merge(df_explanation, labels[['id', 'label', 'aspect']], on='id', how='left')
    print("Lenght explanations with label: ", len(df_explanation))
    
    sentiment_path = get_sentiment_file_path(args)
    file_exists = os.path.isfile(sentiment_path)  # Check if file exists
    df = get_remaining_df(df_explanation, sentiment_path)
    print("df: ", df.head())
    
    for _, row in df.iterrows():
        entry = {"id": row["id"], "input": row["input"]}
        explanation = eval(row["explanation"], {"np": np, "nan": np.nan})
        contains_nan = any(np.isnan(value) for value in explanation.values())
        if contains_nan:
            continue
        sentence, label, aspect = row["input"], row["label"], row["aspect"]
        
        sentence_highest, highest_token = replace_token_with_antonym(explanation) if args.steer_replace == "antonym" else remove_token(explanation)
        sentence_label = replace_label_with_antonym(sentence, label) if args.steer_replace == "antonym" else remove_label(sentence, label)
        try:
            entry["p0"], entry["p_highest"], entry["p_label"] = sentiment_probability(classifier, sentence, sentence_highest, sentence_label, aspect)
        except ValueError as e:
            print(f"Skipping aspect due to error: {e}")
            continue

        entry["aspect"], entry["highest_token"], entry["label"] = aspect, highest_token, label
        # Store in a DataFrame
        row_df = pd.DataFrame([entry])
        
        if save:
            # Append the single row to the CSV (write header only for the first instance)
            row_df.to_csv(sentiment_path, mode="a", header=not file_exists, index=False)
            file_exists = True  # Ensure header is not written again
        else: 
            print(f"sentiment score at id {row['id']}: ", row_df)
        # Clear cache to free memory
        del row, entry, row_df, explanation, label, aspect, sentence_highest, sentence_label, highest_token
        gc.collect()
        
        

if __name__=="__main__":
    parser, args = arg_parse()
    print("args do sample: ", args.do_sample)
    eval_classifier(args)

