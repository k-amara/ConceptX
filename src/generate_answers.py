import os
import gc
import pandas as pd
import numpy as np
import torch
from accelerate.utils import set_seed
from model import LLMPipeline, LLMAPI
from utils import arg_parse, load_file, load_data, get_remaining_df, get_answers_file_path, replace_label_with_antonym, remove_label, replace_token_with_antonym, remove_token
from defenders import SelfReminder, SelfParaphrase


def generate_answers(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)

    api_required = args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"]
    rate_limit = args.model_name.startswith("gpt4")
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)

    if args.defender.startswith(("conceptx", "tokenshap", "random")):
        args.explainer = args.defender
        df_data = load_file(args, folder_name="explanations")
    else:
        df_data = load_data(args)

    answers_file = get_answers_file_path(args)
    file_exists = os.path.isfile(answers_file)
    df = get_remaining_df(df_data, answers_file)

    for _, row in df.iterrows():
        entry = {
            "id": int(row["id"]),
            "input": row["input"],
            "aspect": row.get("aspect", "harmful")
        }

        if args.defender == "none":
            entry["answer"] = llm.generate(entry["input"])

        elif args.defender == "gpt4o-mini":
            input_clean = replace_label_with_antonym(entry["input"], row["label"]) if args.steer_replace == "antonym" else remove_label(entry["input"], row["label"])
            entry["answer"] = llm.generate(input_clean)

        elif args.defender in ["selfreminder", "selfparaphrase"]:
            defender = SelfReminder() if args.defender == "selfreminder" else SelfParaphrase(tokenizer=llm.tokenizer)
            transformed_prompt = defender.transform_prompt(entry["input"])
            entry["answer"] = llm.generate(transformed_prompt)
            del transformed_prompt

        elif args.defender.startswith(("conceptx", "tokenshap", "random")):
            explanation = eval(row["explanation"], {"np": np, "nan": np.nan})
            if any(np.isnan(v) for v in explanation.values()):
                continue
            sentence_highest, highest_token = replace_token_with_antonym(explanation) if args.steer_replace == "antonym" else remove_token(explanation)
            entry["explanatory_token"] = highest_token
            entry["answer"] = llm.generate(sentence_highest)
            del sentence_highest, highest_token

        row_df = pd.DataFrame([entry])
        if save:
            row_df.to_csv(answers_file, mode="a", header=not file_exists, index=False)
            file_exists = True
        else:
            print(f"Score for id {entry['id']}: ", row_df)

        del row_df, entry, row
        gc.collect()
        
    print(f"Saved generated answers to {answers_file}")

if __name__ == "__main__":
    parser, args = arg_parse()
    generate_answers(args)