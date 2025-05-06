from explainers import *
import pickle as pkl

from model import LLMPipeline, LLMAPI
from explainers import *
from utils import arg_parse, load_data, load_vectorizer
from accelerate.utils import set_seed
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchInductor errors
torch._dynamo.reset()  # Reset inductor state



def test_model(args):
    
    if args.seed is not None:
        set_seed(args.seed)
    
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
    
    df = load_data(args)
    print(df.head())
    inputs = df['input'].tolist()[:4]
    for instruction in inputs:
        print("Instruction:", instruction)
        response = llm.generate(instruction)
        print("Response:", response)

    return

def test_sentiment(args):
    
    if args.seed is not None:
        set_seed(args.seed)
    
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
    
    df = load_data(args)
    print(df.head())
    texts = df['text'].tolist()
    for text in texts:
        print("Text:", text)
        instruction = f"""Determine the sentiment of the following sentence: {text}. Your response must be either "positive" or "negative"."""
        response = llm.generate(instruction)
        print("Response:", response)

    return
    
def test_explainer(args):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    api_required = True if args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"] else False 
    rate_limit = True if args.model_name.startswith("gpt4") else False
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)
    
    df = load_data(args)
    print(df.head())
    inputs = df['input'].tolist()[:2]
    
    vectorizer_kwargs = {}
    if args.vectorizer == "huggingface":
        vectorizer_kwargs["embedding_model"] = args.embedding_model
    vectorizer = load_vectorizer(args.vectorizer, **vectorizer_kwargs)
    
    print("result save dir: ", args.result_save_dir)
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
    
    explanations = explainer(inputs, **kwargs)
    # a list of dictionaries, each dictionary contains the explanation for a single instruction
    print("Explanations", explanations)

    return


if __name__ == "__main__":
    parser, args = arg_parse()
    #test_model(args)
    test_explainer(args)
    