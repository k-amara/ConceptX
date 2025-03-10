import numpy as np
import torch
import random
import os
import pickle as pkl
from typing import List, Optional, Tuple, Union


# Explanatory Masks Generation -- Hard masking
def generate_explanatory_masks(
    instructions: List[str], 
    scores, 
    k: float, 
    tokenizer
) -> List[Optional[np.ndarray]]:
    """
    Generate explanatory masks based on SHAP values.

    Args:
        instructions (List[str]): List of instruction strings.
        scores: importance values.
        k (float): Percentage of important indices.
        tokenizer: Tokenizer object.
        next_token_id (int): Token ID.

    Returns:
        List: Explanatory masks.
    """
    masks = []
    for i, prompt in enumerate(instructions):
        n_token = len(tokenizer.tokenize(prompt))
        scores_i = np.array([x.tolist() for x in scores[i]])
        assert n_token == len(scores[i])
        if n_token != len(scores_i):
            masks.append(None)
        else:
            split_point = int(k * n_token)
            important_indices = (-scores_i).argsort()[:split_point]
            mask = np.zeros(n_token)
            mask[important_indices] = 1
            masks.append(mask)
    return masks

# Padding Left Mask
def padleft_mask(
    masks: List[Optional[np.ndarray]], 
    max_length: int
) -> torch.Tensor:
    """
    Pad masks on the left to match max length.

    Args:
        masks (List[Optional[np.ndarray]]): List of masks.
        max_length (int): Maximum length.

    Returns:
        torch.Tensor: Padded masks.
    """
    att_masks = torch.zeros((len(masks), max_length))
    for i, sub in enumerate(masks):
        att_masks[i][-len(sub):] = torch.Tensor(sub)
    return att_masks

def replace_words(sentence, mask, how="default"):
    """
    Replace words in a sentence based on a mask.

    Args:
        sentence (str): The input sentence.
        mask (list): List of 0s and 1s indicating which words to replace.
        how (str): "random" replaces with random words from the sentence, "default" replaces with "...".

    Returns:
        str: Modified sentence.
    """
    words = sentence.split()  # Split into words
    modified_words = words.copy()

    if how == "random":
        non_masked_words = [word for i, word in enumerate(words) if mask[i] == 1]
        for i, m in enumerate(mask):
            if m == 0 and non_masked_words:
                modified_words[i] = random.choice(non_masked_words)  # Replace with a random word from non-masked words

    elif how == "default":
        for i, m in enumerate(mask):
            if m == 0:
                modified_words[i] = "..."  # Replace with "..."

    return " ".join(modified_words)


# Function to calculate scores for the explanations
def get_scores(
    results,
    pipeline, 
    k: float,
    token_id: int = 0
) -> dict:
    """
    Calculates scores for the explanations.

    Args:
        instructions (List[str]): List of instruction strings.
        instruction_ids (List[int]): List of instruction IDs.
        scores: Shapley scores.
        pipeline: Pipeline object.
        k (float): The percentage of important indices.
        token_id (int, optional): Token ID. Defaults to 0.

    Returns:
        dict: Dictionary containing computed scores.
    """
    # Generate explanatory masks
    masks = generate_explanatory_masks(results["instruction"], results["explanation"], k, pipeline.tokenizer, token_id)

    # Initialize lists to store valid instruction ids and instructions
    valid_ids = []
    valid_instructions = []
    valid_tokens = []
    valid_token_ids = []
    
    N = len(results["instruction"])
    print("Number of explained instances", N)

    # Iterate through all instructions
    for i, instruction in enumerate(results["instruction"]):
        # Skip if mask is None
        if masks[i] is None:
            print("masks[i] is None for instruction", instruction, " - skipping...")
            N -= 1
            continue
        else:
            new_instruction = replace_token_ids(instruction, mask, tokenizer, how="random")
            new_response = model.generate(new_instruction)
            
            vectors = self.vectorizer.vectorize([base_response, new_response])
            base_vector = vectors[0]
            new_vector = vectors[1]
            
            # Calculate similarities
            cosine_similarity = self.vectorizer.calculate_similarity(
                base_vector, new_vector
            )

    print("Number of explained instances after removing None masks", N)

    return {
        "instruction_id": valid_ids,
        "instruction": valid_instructions,
        "tokens": valid_tokens,
        "token_ids": valid_token_ids,
    }

# Function to save scores
def save_scores(args, scores):
    """
    Saves the computed scores to a file.

    Args:
        args: Arguments object.
        scores: Dictionary containing computed scores.
    """
    save_dir = os.path.join(args.result_save_dir, f'scores/{args.model_name}/{args.dataset}/{args.algorithm}/seed_{args.seed}/')
    os.makedirs(save_dir, exist_ok=True)
    filename = "scores_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}_{args.threshold}.pkl"
    print(f"Saving scores to {os.path.join(save_dir, filename)}")
    with open(os.path.join(save_dir, filename), "wb") as f:
        pkl.dump(scores, f)

    