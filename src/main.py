import os

from model import Model
from explainers import *
from utils import arg_parse,load_data, load_vectorizer
from accelerate.utils import set_seed



def main(prompt, args):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    model = Model(args)
    vectorizer = load_vectorizer(args.vectorizer)
    
    df = load_data(args)
    print(df.head())
    instructions = df['instruction'].tolist()[:3]
    
    vectorizer = load_vectorizer(args.vectorizer)
    
    if args.explainer == 'tokenshap':
        splitter = StringSplitter()
        explainer = TokenSHAP(model, splitter, vectorizer, debug=False, sampling_ratio=1.0)
        
    if args.explainer == 'conceptshap':
        splitter = ConceptSplitter()
        explainer = ConceptSHAP(model, splitter, vectorizer, debug=False, sampling_ratio=1.0)
        
    scores = explainer(instructions)
    # a list of dictionaries, each dictionary contains the explanation for a single instruction
    print("Scores:", scores)
    return
    
    
if __name__ == "__main__":
    parser, args = arg_parse()
    
    # Get the absolute path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Parent directory:", parent_dir)

    # Example
    prompt = "The artist created a stunning masterpiece that captivated the audience." #"Describe the ideal qualities of a leader in a team."
    main(prompt, args)
    