import os
from explainers import *
from utils import load_vectorizer


def compute_explanations(instructions, model, vectorizer, args):
    #### Explain the model ####
    # Choose appropriate explainer based on specified explainer
    if args.explainer == "random":
        splitter = TokenizerSplitter()
        explainer = Random(model, splitter)
    elif args.explainer == "svsampling":
        splitter = TokenizerSplitter()
        explainer = SVSampling(model, splitter)
    elif args.explainer == "ablation":
        splitter = TokenizerSplitter()
        explainer = FeatAblation(model, splitter)
    elif args.explainer == "tokenshap":
        splitter = StringSplitter()
        vectorizer = load_vectorizer("tfidf")
        explainer = TokenSHAP(model, splitter, vectorizer, debug=True)
    elif args.explainer == "conceptshap":
        splitter = ConceptSplitter()
        vectorizer = load_vectorizer("tfidf")
        explainer = ConceptSHAP(model, splitter, vectorizer, debug=True)
    else:
        raise ("Unknown explainer type passed: %s!" % args.explainer)
    
    scores = explainer(instructions)
    # a list of dictionaries, each dictionary contains the explanation for a single instruction
    print("Scores:", scores)
    return scores

def save_path(args):
    save_dir = os.path.join(args.result_save_dir, f'explanations/{args.model_name}/{args.dataset}/{args.explainer}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = "explanations_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.explainer}_{args.seed}.pkl"
    return os.path.join(save_dir, filename)
    