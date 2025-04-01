import random
import pandas as pd
import requests
import os
import numpy as np

from accelerate.utils import set_seed
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import gc

from explainers import *
from utils import arg_parse, load_file, load_data, get_path, get_remaining_df
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
        
        if aspect.lower() not in aspect_map:
            raise ValueError(f"Aspect '{aspect}' not found in model aspects: {list(aspect_map.keys())}")
        
        aspect_index = aspect_map[aspect.lower()]
        return scores[aspect_index]


def replace_token(explanation, label):
    # Extract tokens, positions, and scores
    tokens_scores = [(key.rsplit('_', 1)[0], int(key.rsplit('_', 1)[1]), value) for key, value in explanation.items()]
    
    # Find the token with the highest score
    highest_token, highest_position, _ = max(tokens_scores, key=lambda x: x[2])
    random_token = random.choice(VOCAB)
    sorted_tokens = sorted(tokens_scores, key=lambda x: x[1])
    sentence_highest = " ".join(random_token if (token == highest_token) and (token_position == highest_position) else token for token, token_position, _ in sorted_tokens)
    
    # Replace the token corresponding to the label if provided
    # label_random_token = random.choice(VOCAB)
    sentence_label = " ".join(random_token if token == label else token for token, _, _ in sorted_tokens)
    
    return sentence_highest, sentence_label, highest_token, label

def sentiment_probability(classifier, sentence, sentence_highest, sentence_label, aspect):
    p0 = classifier.get_aspect_score(sentence, aspect)
    p_highest = classifier.get_aspect_score(sentence_highest, aspect)
    p_label = classifier.get_aspect_score(sentence_label, aspect)
    return p0, p_highest, p_label


def eval_classifier(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    classifier = SentimentClassifier()
        
    df = load_file(args, folder_name="explanations")
    args.num_batch = None
    labels = load_data(args)[['id', 'label', 'aspect']]
    df = pd.merge(df, labels[['id', 'label', 'aspect']], on='id', how='left')
    
    classification_path = get_path(args, folder_name="classification")
    file_exists = os.path.isfile(classification_path)  # Check if file exists
    df = get_remaining_df(df, classification_path)
    print("df: ", df)
    
    for _, row in df.iterrows():
        entry = {"id": row["id"], "input": row["input"]}
        explanation = eval(row["explanation"])
        label, aspect = row["label"], row["aspect"]
        sentence_highest, sentence_label, highest_token, label = replace_token(explanation, label)
        entry["p0"], entry["p_highest"], entry["p_label"] = sentiment_probability(classifier, row["input"], sentence_highest, sentence_label, aspect)
        entry["aspect"], entry["highest_token"], entry["label"] = aspect, highest_token, label
        # Store in a DataFrame
        row_df = pd.DataFrame([entry])

        if save:
            # Append the single row to the CSV (write header only for the first instance)
            row_df.to_csv(classification_path, mode="a", header=not file_exists, index=False)
            file_exists = True  # Ensure header is not written again
        else: 
            print(f"classification score at id {row['id']}: ", row_df)
        # Clear cache to free memory
        del row, entry, row_df, explanation, label, aspect, sentence_highest, sentence_label, highest_token
        gc.collect()
        
        

if __name__=="__main__":
    parser, args = arg_parse()
    eval_classifier(args)

