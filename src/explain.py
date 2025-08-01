from explainers import *
import pickle as pkl
import time
import pandas as pd
from model import LLMPipeline, LLMAPI, ContentPolicyViolationError
from utils import arg_parse, load_data, load_vectorizer, get_path, get_remaining_df
from accelerate.utils import set_seed
import os
import gc
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchInductor errors
torch._dynamo.reset()  # Reset inductor state

def compute_explanations(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
    
    vectorizer_kwargs = {}
    if args.vectorizer == "huggingface":
        vectorizer_kwargs["embedding_model"] = args.embedding_model
    vectorizer = load_vectorizer(args.vectorizer, **vectorizer_kwargs)
    
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
    elif args.explainer.startswith("conceptx"):
        splitter = ConceptSplitter()
        explainer = ConceptX(llm, splitter, vectorizer, debug=False, sampling_ratio=1.0, replace=args.coalition_replace)
        # Determine baseline if needed
        baseline_texts = None
        if args.target == "reference":
            baseline_texts = df['reference'].tolist()
        elif args.target == "aspect":
            baseline_texts = df['aspect'].tolist()
        kwargs = {"baseline_texts": baseline_texts} if baseline_texts is not None else {}
    else:
        raise ("Unknown explainer type passed: %s!" % args.explainer)
    
    explanations_path = get_path(args, folder_name="explanations")
    time_path = get_path(args, folder_name="computation_time")

    file_exists = os.path.isfile(explanations_path)  # Check if file exists
    df = get_remaining_df(df, explanations_path)
    
    time_logs = []

    for i in range(len(df)):  # Process each input one by one
        input = df.iloc[i]["input"]
        if not isinstance(input, str):
            continue
        input_id = df.iloc[i]["id"]

        try:
            start_time = time.time()

            response = llm.generate(input)
            explanation_list = explainer([input], **kwargs)
            if args.explainer.startswith("conceptx"):
                concepts = splitter.extract_meaningful_concepts(input) 
                print("Concepts: ", concepts)
                num_features = len(concepts)
            else:
                num_features = len(splitter.split(input))

            if not explanation_list:
                continue  # Skip this iteration if explanation is empty

            explanation = explanation_list[0]

            row_df = pd.DataFrame([{
                "id": input_id,
                "input": input,
                "response": response,
                "explanation": explanation
            }])

            if save:
                row_df.to_csv(explanations_path, mode="a", header=not file_exists, index=False)
                file_exists = True
            else:
                print(f"Explanation id {input_id}: ", row_df)

            # Update timing
            elapsed_time = time.time() - start_time
            input_length = len(input)

            time_logs.append({
                "id": input_id,
                "input_length": input_length,
                "num_features": num_features,
                "time_seconds": elapsed_time
            })


            # Clear cache to free memory
            del input, response, input_id, explanation, row_df
            gc.collect()

        except ContentPolicyViolationError:
            continue  # Skip inputs that raise the error
        
    if time_logs:
        os.makedirs(os.path.dirname(time_path), exist_ok=True)
        time_df = pd.DataFrame(time_logs)
        time_df.to_csv(time_path, index=False)

    return
    


if __name__ == "__main__":
    parser, args = arg_parse()
    print('args do_sample: ', args.do_sample)
    print('args result_save_dir: ', args.result_save_dir)
    compute_explanations(args)