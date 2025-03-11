import os
import dill

from explainers import Random, SVSampling, Ablation, HEDGEOrig, TokenSHAP, ConceptSHAP
from explainer import ConceptSHAP, TokenSHAP, StringSplitter, HuggingFaceEmbeddings, ConceptProcessor


def compute_explanations(model, args):
    #### Explain the model ####
    # Choose appropriate explainer based on specified explainer
    if args.explainer == "random":
        explainer = Random()
    elif args.explainer == "svsampling":
        explainer = SVSampling()
    elif args.explainer == "ablation":
        explainer = Ablation()
    elif args.explainer == "tokenshap":
        splitter = StringSplitter()
        explainer = TokenSHAP(model, splitter, vectorizer, debug=True)
        df = explainer.analyze(prompt, sampling_ratio=0.1, print_highlight_text=True)
    elif args.explainer == "conceptshap":
        processor = ConceptProcessor()
        explainer = ConceptSHAP(model, processor, vectorizer, debug=True)
        df = explainer.analyze(prompt, sampling_ratio=0.1, print_highlight_text=True)  
    else:
        raise ("Unknown explainer type passed: %s!" % args.explainer)
    
    return df

def save_path(args):
    save_dir = os.path.join(args.result_save_dir, f'explanations/{args.model_name}/{args.dataset}/{args.explainer}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = "explanations_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.explainer}_{args.seed}.pkl"
    return os.path.join(save_dir, filename)
    