from explainers import *
import pickle as pkl

from model import Model
from explainers import *
from utils import arg_parse, load_data, load_vectorizer, get_path
from accelerate.utils import set_seed



def compute_explanations(args, save=True):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    model = Model(args)
    vectorizer = load_vectorizer(args.vectorizer)
    
    df = load_data(args)
    print(df.head())
    instructions = df['instruction'].tolist()[:3]
    ids = df["id"].tolist()[:3]
    
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
        explainer = TokenSHAP(model, splitter, vectorizer, debug=False, sampling_ratio=1.0)
    elif args.explainer == "conceptshap":
        splitter = ConceptSplitter()
        explainer = ConceptSHAP(model, splitter, vectorizer, debug=False, sampling_ratio=1.0)
    else:
        raise ("Unknown explainer type passed: %s!" % args.explainer)
    
    explanations = explainer(instructions)
    # a list of dictionaries, each dictionary contains the explanation for a single instruction
    print("Explanations", explanations)
    # save scores into args.results_dir
    
    if save:
        explanations_path = get_path(args, folder_name="explanations")
        # Store in a dictionary
        explanations_dict = {
            "instruction": instructions,
            "id": ids,
            "explanation": explanations
        }
        with open(explanations_path, "wb") as f:
            pkl.dump(explanations_dict, f)
    
    return explanations
    


if __name__ == "__main__":
    parser, args = arg_parse()
    compute_explanations(args)