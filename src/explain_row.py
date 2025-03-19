from explainers import *
import pickle as pkl
import pandas as pd
from model import LLMPipeline, LLMAPI, process_instructions
from utils import arg_parse, load_data, load_vectorizer, get_path, get_remaining_df
from accelerate.utils import set_seed

def compute_explanations(args):
    
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
    
    
    i = 115
    instruction = df.iloc[i]["instruction"]
    instruction_id = df.iloc[i]["id"]
    
    # Get explanation for the single instruction
    explanation = explainer([instruction], **kwargs)[0]  

    # Store in a DataFrame
    row_df = pd.DataFrame([{
        "id": instruction_id,
        "instruction": instruction,
        "explanation": explanation
    }])
    print(f"Explanation id {instruction_id}: ", row_df)
        
    return
    


if __name__ == "__main__":
    parser, args = arg_parse()
    compute_explanations(args)