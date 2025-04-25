import argparse
import random

import numpy as np
import torch
from transformers import set_seed

# Define the path to the data, model, logs, results, and colors
#

CKPT_ROOT = "/cluster/home/kamara/conceptx/"
STORAGE = "/cluster/scratch/kamara/conceptx/"
#CKPT_ROOT = "/Users/kenzaamara/GithubProjects/conceptx/"
#STORAGE = "/Users/kenzaamara/GithubProjects/conceptx/"
DATA_DIR = CKPT_ROOT + "data/"
MODEL_DIR = STORAGE + "models/"
HIDDEN_STATE_DIR = STORAGE + "hidden_states/"
FIG_DIR = CKPT_ROOT + "figures/"
RESULT_DIR = CKPT_ROOT + "results/"



def fix_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_save_dir",
        help="Directory where benchmark is located",
        type=str,
        default=DATA_DIR,
    )

    parser.add_argument(
        "--model_save_dir",
        help="Directory where figures are saved",
        type=str,
        default=MODEL_DIR,
    )
    
    parser.add_argument(
        "--result_save_dir",
        help="Directory where results are saved",
        type=str,
        default=RESULT_DIR,
    )
    parser.add_argument(
        "--fig_save_dir",
        help="Directory where figures are saved",
        type=str,
        default="figures",
    )
    
    
    parser.add_argument(
        "--num_batch", type=int, default=None
    )
    parser.add_argument(
        "--batch_size", type=int, default=None
    )
    
    parser.add_argument(
        "--file_type", type=str, default="csv"
    )
    
    
    ### Language Model Parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Model type selected in the list of model classes",
    )
    parser.add_argument("--quantization", type=str, default=None, help="Quantization type (8bit, 4bit)")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--no_sample",
        action="store_false",
        dest="do_sample",
        help="If set, disables sampling. By default, sampling is enabled.",
    )
    
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Whether or not to use cpu. If set to False, " "we will use gpu/npu or mps device if available",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--jit", action="store_true", help="Whether or not to use jit trace to accelerate inference")
    
    
    ### Dataset
    
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Dataset of intructions selected in the list of datasets (alpaca, genderbias, sentiment)",
    )
    
    
    ### Explainability Method Parameters
    
    parser.add_argument(
        "--explainer",
        default=None,
        type=str,
        help="Explainer type selected in the list of explainer classes (random, tokenshap, conceptshap, conceptx)",
    )
    
    
    parser.add_argument(
        "--vectorizer",
        default="huggingface",
        type=str,
        help="Vectorizer type selected in the list of explainer classes (huggingface, openai, tfidf)",
    )
    
    parser.add_argument(
        "--baseline",
        default=None,
        type=str,
        help="The baseline for ConceptX - None (LLM initial response), reference (reference text), aspect (a specific aspect)",
    )
    
    ### Evaluation
    
    parser.add_argument(
        "--masking_method",
        default="ellipsis",
        type=str,
        help="Masking strategy when evaluating faithfulness (random, ellipsis)",
    )
    
    
    ### Safety Analysis
    parser.add_argument(
        "--safety_classifier",
        default="mdjudge",
        type=str,
        help="The safety classifier judging whether the answer given is safe; either mdjudge or llamaguard3",
    )
    
    parser.add_argument(
        "--defender",
        default="none",
        type=str,
        help="defender type selected in the list of explainer classes (selfreminder, selfparaphrase, gpt4omini, random, tokenshap, conceptshap, conceptx)",
    )
    
    parser.add_argument(
        "--steer_replace",
        default=None,
        type=str,
        help="Whther to replace the removed token with antonym to steer the model response; None is only removing the token; Only valid if defender is a token-level xai method",
    )
    
    args, unknown = parser.parse_known_args()
    return parser, args


def create_args_group(parser, args):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = group_dict
    return arg_groups


def merge_args(args, args_dict):
    # Convert args (Namespace) to a dictionary
    args_dict_existing = vars(args).copy()
    # Update with new values, giving priority to args_dict
    args_dict_existing.update({k: v for k, v in args_dict.items() if v is not None})
    # Reconstruct Namespace with updated values
    return argparse.Namespace(**args_dict_existing)