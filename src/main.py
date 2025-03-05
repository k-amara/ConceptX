import os

from model import Model
from explainer import ConceptSHAP, TokenSHAP, StringSplitter, HuggingFaceEmbeddings
from utils import arg_parse
from accelerate.utils import set_seed



def main(prompt, args):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    model = Model(args)
    splitter = StringSplitter()
    vectorizer = HuggingFaceEmbeddings()
    
    print("Prompt:", prompt)
    response = model.generate(prompt)
    print("Initial response:", response)
    
    if args.epxlainer == 'tokenshap':
        explainer = TokenSHAP(model, splitter, vectorizer, debug=True)
        df = explainer.analyze(prompt, sampling_ratio=0.1, print_highlight_text=True)
        
    if args.explainer == 'conceptshap':
        explainer = ConceptSHAP(model, splitter, vectorizer, debug=True)
        df = explainer.analyze(prompt, sampling_ratio=0.1, print_highlight_text=True)   
        
    print(df)
    explainer.plot_colored_text()
    explainer.print_colored_text()
    explainer.highlight_text_background()
    print(explainer.shapley_values)
    return
    
    
if __name__ == "__main__":
    parser, args = arg_parse()
    
    # Get the absolute path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Parent directory:", parent_dir)

    # Example
    prompt = "Describe the ideal qualities of a leader in a team."
    