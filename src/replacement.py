from explainers import *
import pickle as pkl
import pandas as pd
from model import LLMPipeline, LLMAPI, ContentPolicyViolationError
from utils import arg_parse, load_data
from accelerate.utils import set_seed
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchInductor errors
torch._dynamo.reset()  # Reset inductor state


def compute_explanations(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
    
    df = load_data(args)
    print(df.head())


def get_replacements(self, concepts, text, replace="neutral"):
        if (replace=="neutral") or (replace=="antonym"):
            prompt = eval(f"create_prompt_for_{replace}_replacement")(text, concepts)
            try:
                completions = get_multiple_completions(prompt, num_sequences=1)
                print(completions)
                return completions[0]
            
            except ContentPolicyViolationError:
                print(f"Skipping prompt due to content policy violation during replacements: {text}")
                return None  # Return None to signal skipping
        else:
            return str([""]*len(concepts))
        
        
        # Clean the prompt to prevent empty tokens
        prompt_cleaned = prompt.strip()
        prompt_cleaned = re.sub(r'\s+', ' ', prompt_cleaned)
        
        self.words = self.splitter.split(prompt_cleaned)
        print("prompt: ", prompt)
        print("prompt cleaned: ", prompt_cleaned)
        self.concepts, self.indices = self.splitter.split_concepts(prompt_cleaned) # concepts are the samples in TokenSHAP
        
        # Skip processing if there is 1 or fewer concepts
        if len(self.concepts) <= 1:
            print("Skipping prompt due to insufficient concepts.")
            return None
        
        
        self.replacements = self.splitter.get_replacements(self.concepts, prompt_cleaned, replace = self.replace)
        if self.replacements is None:
            print("Skipping prompt due to content policy violation during replacements.")
            return None  # Skip explanation if replacements failed
        self.replacements = eval(self.replacements)
        if len(self.concepts) != len(self.replacements):
            print(f"Skipping prompt due to mismatch: concepts ({len(self.concepts)}) vs. replacements ({len(self.replacements)})")
            return None  # Avoid proceeding if lengths mismatch
        