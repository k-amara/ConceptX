from explainers import *
import pickle as pkl

from model import LLMPipeline, LLMAPI
from explainers import *
from utils import arg_parse, load_data, load_vectorizer, get_path
from accelerate.utils import set_seed



def test_model(args):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    llm = LLMAPI(args) if args.model_type == "api" else LLMPipeline(args)
    
    df = load_data(args)
    print(df.head())
    instructions = df['instruction'].tolist()[:2]
    for instruction in instructions:
        prompt = f"""
        Given the following instruction, provide an answer as direct advice. Do not use bullet points.
        Instruction: "{instruction}"
        Response:
        """
        print("Instruction:", instruction)
        response = llm.generate(prompt)
        print("Response:", response)

    return
    
def test_explainer(args):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    llm = LLMAPI(args) if args.model_type == "api" else LLMPipeline(args)
    
    df = load_data(args)
    print(df.head())
    instructions = df['instruction'].tolist()[:2]
    
    vectorizer = load_vectorizer(args.vectorizer)
    
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
            baseline_texts = df['reference'].tolist()[:2]
        elif args.baseline == "concept":
            baseline_texts = df['gender'].tolist()[:2]
        # Add baseline to kwargs only if it's not None
        kwargs = {"baseline_texts": baseline_texts} if baseline_texts is not None else {}
    else:
        raise ("Unknown explainer type passed: %s!" % args.explainer)
    
    explanations = explainer(instructions, **kwargs)
    # a list of dictionaries, each dictionary contains the explanation for a single instruction
    print("Explanations", explanations)

    return


if __name__ == "__main__":
    parser, args = arg_parse()
    test_model(args)
    #test_explainer(args)