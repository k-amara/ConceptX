import pandas as pd
import os
import re
import numpy as np

from accelerate.utils import set_seed
from transformers import AutoTokenizer,  AutoModelForCausalLM
import gc

from model import LLMPipeline, LLMAPI
from explainers import *
from utils import arg_parse, load_file, load_data, get_remaining_df, remove_token, remove_label, replace_token_with_antonym, replace_label_with_antonym
from accelerate.utils import set_seed
from defenders import SelfReminder, SelfParaphrase

import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchInductor errors
torch._dynamo.reset()  # Reset inductor state

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
    # "transformers_version": "4.41.2",
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
    


def get_safety_file_path(args):
    save_dir = os.path.join(args.result_save_dir, f'safety/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"safety_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.defender}"
    filename += f"antonym_" if args.steer_replace == "antonym" else ""
    filename += f"{args.seed}.{args.file_type}"
    safety_path = os.path.join(save_dir, filename)
    return safety_path


def eval_safety(args, save=True):
    torch.cuda.empty_cache()
    if args.seed is not None:
        set_seed(args.seed)

    # Choose LLM backend
    api_required = args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"]
    rate_limit = args.model_name.startswith("gpt4")
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)

    # Load input data
    if args.defender in ["conceptx", "tokenshap", "conceptshap", "random", "aconceptx"]:
        args.explainer = args.defender
        df = load_file(args, folder_name="explanations")
    else:
        df = load_data(args)

    df = df[:2]
    
    safety_file = get_safety_file_path(args)
    file_exists = os.path.isfile(safety_file)
    df = get_remaining_df(df, safety_file)
    print("df: ", df.head())
    
    generated = []

    for _, row in df.iterrows():
        entry = {
            "id": int(row["id"]),
            "input": row["input"],
            "aspect": row.get("aspect", "harmful")
        }

        if args.defender == "none":
            entry["answer"] = llm.generate(entry["input"])

        elif args.defender == "gpt4omini":
            input_clean = replace_label_with_antonym(entry["input"], row["label"]) if args.steer_replace == "antonym" else remove_label(entry["input"], row["label"])
            entry["answer"] = llm.generate(input_clean)

        elif args.defender in ["selfreminder", "selfparaphrase"]:
            defender_cls = SelfReminder if args.defender == "selfreminder" else SelfParaphrase
            defender_kwargs = {'model': llm}  # Always pass model
            if args.defender == "selfparaphrase":
                defender_kwargs['tokenizer'] = llm.tokenizer
            defender = defender_cls(**defender_kwargs)
            entry["answer"] = defender.get_response(entry["input"])

        elif args.defender in ["random", "tokenshap", "conceptshap", "conceptx", "aconceptx"]:
            explanation = eval(row["explanation"], {"np": np, "nan": np.nan})
            if any(np.isnan(v) for v in explanation.values()):
                continue
            sentence_highest, highest_token = replace_token_with_antonym(explanation) if args.steer_replace == "antonym" else remove_token(explanation)
            entry["explanatory_token"] = highest_token
            entry["answer"] = llm.generate(sentence_highest)
            
        generated.append(entry)
    
    # Free GPU memory if needed
    if not api_required:
        del llm
        torch.cuda.empty_cache()

    # === Pass 2: Classification ===
    classifier = SafetyClassifier()

    for entry in generated:
        try:
            entry["asr"], entry["hs"] = classifier(entry["input"], entry["answer"])
        except Exception as e:
            print(f"Skipping id {entry['id']} due to error: {e}")
            continue
        row_df = pd.DataFrame([entry])

        if save:
            row_df.to_csv(safety_file, mode="a", header=not file_exists, index=False)
            file_exists = True
        else:
            print(f"Score for id {entry['id']}: ", row_df)

        del row_df
        gc.collect()



if __name__=="__main__":
    parser, args = arg_parse()
    eval_safety(args)
