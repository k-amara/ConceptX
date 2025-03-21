from explainers import *
import pickle as pkl
import pandas as pd
from model import LLMPipeline, LLMAPI, ContentPolicyViolationError
from utils import arg_parse, load_data, load_vectorizer, get_path, get_remaining_df
from accelerate.utils import set_seed
import os
import gc

def compute_explanations(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
    
    vectorizer = load_vectorizer(args.vectorizer)
    
    df = load_data(args)
    print(df.head())
    
    # Choose appropriate explainer based on specified explainer
    kwargs = {}
    if args.explainer == "random":
        splitter = StringSplitter()
        explainer = Random(llm, splitter)
    elif args.explainer == "svsampling":
        splitter = TokenizerSplitter(llm.tokenizer)
        explainer = SVSampling(llm, splitter)
    elif args.explainer == "ablation":
        splitter = TokenizerSplitter(llm.tokenizer)
        explainer = FeatAblation(llm, splitter)
    elif args.explainer == "tokenshap":
        splitter = StringSplitter()
        explainer = TokenSHAP(llm, splitter, vectorizer, debug=False, sampling_ratio=1.0)
    elif args.explainer == "conceptshap":
        splitter = ConceptSplitter()
        explainer = ConceptSHAP(llm, splitter, vectorizer, debug=False, sampling_ratio=1.0)
        # Determine baseline if needed
        baseline_texts = None
        if args.baseline == "reference":
            baseline_texts = df['reference'].tolist()
        elif args.baseline == "concept":
            baseline_texts = df['gender'].tolist() 
        # Add baseline to kwargs only if it's not None
        kwargs = {"baseline_texts": baseline_texts} if baseline_texts is not None else {}
    else:
        raise ("Unknown explainer type passed: %s!" % args.explainer)
    
    explanations_path = get_path(args, folder_name="explanations")
    file_exists = os.path.isfile(explanations_path)  # Check if file exists
    df = get_remaining_df(df, explanations_path)
    for i in range(len(df)):  # Process each instruction one by one
        instruction = df.iloc[i]["instruction"]
        instruction_id = df.iloc[i]["id"]
        try:
            response = llm.generate(instruction)
        except ContentPolicyViolationError:
            continue  # Skip instructions that raise the error
    
        # Get explanation for the single instruction
        explanation_list = explainer([instruction], **kwargs)
        if not explanation_list:  # Check if explanation is an empty list
            continue  # Skip this iteration and move to the next instruction -- probably becais
        explanation = explanation_list[0]  

        # Store in a DataFrame
        row_df = pd.DataFrame([{
            "id": instruction_id,
            "instruction": instruction,
            "response": response,
            "explanation": explanation
        }])

        if save:
            # Append the single row to the CSV (write header only for the first instance)
            row_df.to_csv(explanations_path, mode="a", header=not file_exists, index=False)
            file_exists = True  # Ensure header is not written again
        else: 
            print(f"Explanation id {instruction_id}: ", row_df)
        # Clear cache to free memory
        del instruction, response, instruction_id, explanation, row_df
        gc.collect()
        
    return
    


if __name__ == "__main__":
    parser, args = arg_parse()
    compute_explanations(args)