import os

from text_generation import run_generation
from utils import arg_parse, get_main_concept, select_richest_concepts
from accelerate.utils import set_seed



def main(args):
    
    if args.seed is not None:
        set_seed(args.seed)
        
    print("Prompt:", args.prompt)
    generated_sequences = run_generation(args)
    response = generated_sequences[0]
    print("Response:", response)
    
    # Get target concept to explain
    target_concept = get_main_concept(response)
    print("Response Dominant Topic:", target_concept)
    
    # Get input concepts
    input_concepts, indexes = select_richest_concepts(args.prompt)
    print("Input Concepts:", input_concepts)
    
    return
    
    
if __name__ == "__main__":
    parser, args = arg_parse()
    
    # Get the absolute path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Parent directory:", parent_dir)

    # Execute main function
    main(args)