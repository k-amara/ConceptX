from explainers import *
import pickle as pkl
import pandas as pd
from model import LLMPipeline, LLMAPI, process_instructions
from utils import arg_parse, load_data, load_vectorizer, get_path
from accelerate.utils import set_seed


def compute_explanations(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4", "o1", "deepseek"] else False 
    llm = LLMAPI(args) if api_required else LLMPipeline(args)
    
    vectorizer = load_vectorizer(args.vectorizer)
    
    df = load_data(args)
    print(df.head())
    df = process_instructions(df, llm)
    
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
    
    explanations = explainer(df['instruction'].tolist(), **kwargs)
    # a list of dictionaries, each dictionary contains the explanation for a single instruction
    print("Explanations", explanations)
    
    if save:
        explanations_path = get_path(args, folder_name="explanations")
        # Store in a dictionary
        explanations_df = df[["id", "instruction", "response"]]
        explanations_df["explanation"] = explanations
        #with open(explanations_path, "wb") as f:
            #pkl.dump(explanations_dict, f)
        # Convert the dictionary to a pandas DataFrame
        # explanations_df = pd.DataFrame(explanations_dict)
        explanations_df.to_csv(explanations_path, index=False)
    
    return explanations
    


if __name__ == "__main__":
    parser, args = arg_parse()
    compute_explanations(args)