import random
import pandas as pd
import requests
import os
import copy
import math
import re
import numpy as np

from accelerate.utils import set_seed
from transformers import AutoTokenizer,  AutoModelForCausalLM
from scipy.special import softmax
import gc

from model import LLMPipeline, LLMAPI, ContentPolicyViolationError
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


class LLaMAGuard3:
    def __init__(self, model_name="meta-llama/Llama-Guard-3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        attn_implementation="sdpa",
                                                        torch_dtype=torch.float16, 
                                                        trust_remote_code=True).to("cuda")
        
    def moderate(self, chat):
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


    def __call__(self, input, output):
        # modified from modeling_internlm2.py: def chat()
        # chat with no system instruction
        resp = self.moderate([
            {"role": "user", "content": input},
            {"role": "assistant", "content": output},
            ])
        return resp
    
        

class SafetyClassifier:
    def __init__(self, model_name="OpenSafetyLab/MD-Judge-v0_2-internlm2_7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        attn_implementation="sdpa",
                                                        torch_dtype=torch.float16, 
                                                        trust_remote_code=True).to("cuda")

    def __call__(self, input, output):
        # modified from modeling_internlm2.py: def chat()
        # chat with no system instruction
        prompt = self.tokenizer.apply_chat_template([
            {"role": "user", "content": input},
            {"role": "assistant", "content": output}
        ], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        resp = self.tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(resp.strip())
        match = re.search(r"\[RESULT\]:\s*(\w+),\s*score\s*(\d+)", resp)
        return match.group(1), match.group(2)
    
        

def replace_harmful_token(explanation):
    # Extract tokens, positions, and scores
    tokens_scores = [(key.rsplit('_', 1)[0], int(key.rsplit('_', 1)[1]), value) for key, value in explanation.items()]
    
    # Find the token with the highest score
    highest_token, highest_position, _ = max(tokens_scores, key=lambda x: x[2])
    # random_token = random.choice(VOCAB)
    sorted_tokens = sorted(tokens_scores, key=lambda x: x[1])
    sentence_highest = " ".join(
        token for token, token_position, _ in sorted_tokens
        if not (token == highest_token and token_position == highest_position)
    )
    return sentence_highest, highest_token

def remove_word(sentence, word):
    # Escape the word in case it contains regex special characters
    pattern = r'\b' + re.escape(word) + r'\b'
    # Remove the word regardless of case
    cleaned_sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
    # Remove extra whitespace
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
    return cleaned_sentence


def eval_label_safety(args, save=True):
    if args.seed is not None:
        set_seed(args.seed)

    # Decide whether to use API-based LLM
    api_required = args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"]
    rate_limit = args.model_name.startswith("gpt4")
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)

    # Prepare paths and data
    df_data = load_data(args)
    
    folder_name = "safety"
    save_dir = os.path.join(args.result_save_dir, f'{folder_name}/{args.model_name}/{args.dataset}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{folder_name}_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.seed}.{args.file_type}"
    safety_path = os.path.join(save_dir, filename)
    file_exists = os.path.isfile(safety_path)

    df = get_remaining_df(df_data, safety_path)
    print("df: ", df.head())

    generated = []

    # === Pass 1: Generation with LLM ===
    for _, row in df.iterrows():
        entry = {
            "id": row["id"],
            "input": row["input"],
            "aspect": row["aspect"],
            "label": row["label"]
        }

        sentence_label = remove_word(entry["input"], entry["label"])

        try:
            entry["answer"] = llm.generate(entry["input"])
            entry["answer_label"] = llm.generate(sentence_label)
        except ContentPolicyViolationError:
            continue

        generated.append(entry)

    # Free GPU if needed
    if not api_required:
        del llm
        torch.cuda.empty_cache()

    # === Pass 2: Classification with MD-Judge ===
    classifier = SafetyClassifier()

    for entry in generated:
        try:
            entry["asr_0"], entry["score_0"] = classifier(entry["input"], entry["answer"])
            entry["asr_label"], entry["score_label"] = classifier(remove_word(entry["label"],entry["input"]), entry["answer_label"])
        except ValueError as e:
            print(f"Skipping entry {entry['id']} due to classifier error: {e}")
            continue

        row_df = pd.DataFrame([entry])

        if save:
            row_df.to_csv(safety_path, mode="a", header=not file_exists, index=False)
            file_exists = True
        else:
            print(f"safety score at id {entry['id']}: ", row_df)

        # Cleanup
        del row_df
        gc.collect()



def eval_safety(args, save=True):
    if args.seed is not None:
        set_seed(args.seed)

    # Decide on model usage
    api_required = args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"]
    rate_limit = args.model_name.startswith("gpt4")
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)

    # Load input data
    df_explanation = load_file(args, folder_name="explanations")
    safety_path = get_path(args, folder_name="safety")
    file_exists = os.path.isfile(safety_path)
    df = get_remaining_df(df_explanation, safety_path)
    print("df: ", df.head())

    generated = []

    # === Pass 1: Generation ===
    for _, row in df.iterrows():
        explanation = eval(row["explanation"], {"np": np, "nan": np.nan})
        if any(np.isnan(value) for value in explanation.values()):
            continue

        sentence_highest, highest_token = replace_harmful_token(explanation)
        try:
            response_highest = llm.generate(sentence_highest)
        except (ValueError, ContentPolicyViolationError):
            continue

        generated.append({
            "id": row["id"],
            "input": row["input"],
            "aspect": "harmful",
            "sentence_highest": sentence_highest,
            "highest_token": highest_token,
            "response_highest": response_highest
        })

    # Free GPU memory if needed
    if not api_required:
        del llm
        torch.cuda.empty_cache()

    # === Pass 2: Classification ===
    classifier = SafetyClassifier()

    for entry in generated:
        try:
            entry["asr_highest"], entry["score_highest"] = classifier(entry["sentence_highest"], entry["response_highest"])
        except ValueError as e:
            print(f"Skipping aspect due to error: {e}")
            continue

        row_df = pd.DataFrame([entry])

        if save:
            row_df.to_csv(safety_path, mode="a", header=not file_exists, index=False)
            file_exists = True
        else:
            print(f"safety score at id {entry['id']}: ", row_df)

        # Cleanup
        del row_df
        gc.collect()
        

       
# def attack_succes_rate():


if __name__=="__main__":
    parser, args = arg_parse()
    # eval_label_safety(args)
    eval_safety(args)
