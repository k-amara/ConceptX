#from token_shap import *
import pandas as pd
from tqdm.auto import tqdm
import re
import numpy as np
import random
from typing import Optional

from explainers._explainer import Explainer
from explainers._splitter import Splitter
from explainers._vectorizer import TextVectorizer
from explainers._explain_utils import normalize_explanation
from model import ContentPolicyViolationError

# Refactored TokenSHAP class
class TokenSHAP(Explainer):
    def __init__(self, 
                 llm, 
                 splitter: Splitter, 
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False, 
                 sampling_ratio: float = 0.0):
        super().__init__(llm, splitter, vectorizer, debug)
        self.sampling_ratio = sampling_ratio

    def _generate_random_combinations(self, samples, k, exclude_combinations_set):
        n = len(samples)
        sampled_combinations_set = set()
        max_attempts = k * 10  # Prevent infinite loops in case of duplicates
        attempts = 0

        while len(sampled_combinations_set) < k and attempts < max_attempts:
            attempts += 1
            rand_int = random.randint(1, 2 ** n - 2)
            bin_str = bin(rand_int)[2:].zfill(n)
            combination = [samples[i] for i in range(n) if bin_str[i] == '1']
            indexes = tuple([i for i in range(n) if bin_str[i] == '1']) ## changes to source code (i+1 --> i)
            if indexes not in exclude_combinations_set and indexes not in sampled_combinations_set:
                sampled_combinations_set.add((tuple(combination), indexes))
        if len(sampled_combinations_set) < k:
            self._debug_print(f"Warning: Could only generate {len(sampled_combinations_set)} unique combinations out of requested {k}")
        return list(sampled_combinations_set)

    def _get_result_per_token_combination(self, prompt):
        samples = self.splitter.split(prompt)
        n = len(samples)
        self._debug_print(f"Number of samples (tokens): {n}")
        if n > 1000:
            print("Warning: the number of samples is greater than 1000; execution will be slow.")

        num_total_combinations = 2 ** n - 1
        self._debug_print(f"Total possible combinations (excluding empty set): {num_total_combinations}")

        num_sampled_combinations = int(num_total_combinations * self.sampling_ratio)
        self._debug_print(f"Number of combinations to sample based on sampling ratio {self.sampling_ratio}: {num_sampled_combinations}")

        # Always include combinations missing one token
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = samples[:i] + samples[i + 1:]
            indexes = tuple([j for j in range(n) if j != i]) ## changes to source code (j+1 --> j)
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)

        self._debug_print(f"Number of essential combinations (each missing one token): {len(essential_combinations)}")

        num_additional_samples = min(30, max(0, num_sampled_combinations - len(essential_combinations)))
        self._debug_print(f"Number of additional combinations to sample (capped at 30): {num_additional_samples}")

        sampled_combinations = []
        if num_additional_samples > 0:
            sampled_combinations = self._generate_random_combinations(
                samples, num_additional_samples, essential_combinations_set
            )
            self._debug_print(f"Number of sampled combinations: {len(sampled_combinations)}")
        else:
            self._debug_print("No additional combinations to sample.")

        # Combine essential and additional combinations
        all_combinations_to_process = essential_combinations + sampled_combinations
        self._debug_print(f"Total combinations to process: {len(all_combinations_to_process)}")

        prompt_responses = {}
        for idx, (combination, indexes) in enumerate(tqdm(all_combinations_to_process, desc="Processing combinations")):
            text = self.splitter.join(combination)
            self._debug_print(f"\nProcessing combination {idx + 1}/{len(all_combinations_to_process)}:")
            self._debug_print(f"Combination tokens: {combination}")
            self._debug_print(f"Token indexes: {indexes}")
            self._debug_print(f"Generated text: {text}")

            try:
                text_response = self.llm.generate(text)
                self._debug_print(f"Received response for combination {idx + 1}")

                prompt_key = text + '_' + ','.join(str(index) for index in indexes)
                prompt_responses[prompt_key] = (text_response, indexes)

            except ContentPolicyViolationError:
                self._debug_print(f"Skipping combination {idx + 1} due to content policy violation.")
                continue  # Skip this combination and move to the next one

        self._debug_print("Completed processing all combinations.")
        return prompt_responses

    def _get_df_per_token_combination(self, prompt_responses, baseline_text):
        df = pd.DataFrame(
            [(prompt.split('_')[0], response[0], response[1])
             for prompt, response in prompt_responses.items()],
            columns=['Prompt', 'Response', 'Token_Indexes']
        )

        all_texts = [baseline_text] + df["Response"].tolist()
        
        # Use the configured vectorizer
        vectors = self.vectorizer.vectorize(all_texts)
        base_vector = vectors[0]
        comparison_vectors = vectors[1:]
        
        # Calculate similarities
        cosine_similarities = self.vectorizer.calculate_similarity(
            base_vector, comparison_vectors
        )
        
        df["Cosine_Similarity"] = cosine_similarities

        return df

    def _calculate_shapley_values(self, df_per_token_combination, prompt):

        samples = self.splitter.split(prompt)
        n = len(samples)
        explanation = {}

        for i, sample in enumerate(samples, start=0): ## changes to source code (start=1 --> start=0)
            with_sample = np.average(
                df_per_token_combination[
                    df_per_token_combination["Token_Indexes"].apply(lambda x: i in x)
                ]["Cosine_Similarity"].values
            )
            without_sample = np.average(
                df_per_token_combination[
                    df_per_token_combination["Token_Indexes"].apply(lambda x: i not in x)
                ]["Cosine_Similarity"].values
            )

            explanation[sample + "_" + str(i)] = with_sample - without_sample

        return normalize_explanation(explanation)

    def analyze(self, prompt, print_highlight_text=False):
        # Clean the prompt to prevent empty tokens
        prompt_cleaned = prompt.strip()
        prompt_cleaned = re.sub(r'\s+', ' ', prompt_cleaned)

        self.baseline_text = self._calculate_baseline(prompt_cleaned)
        if self.baseline_text is None:
            print(f"Skipping prompt due to ContentViolation during baseline generation")
            return None  
        
        token_combinations_results = self._get_result_per_token_combination(
            prompt_cleaned
        )
        df_per_token_combination = self._get_df_per_token_combination(
            token_combinations_results, self.baseline_text
        )
        self.explanation = self._calculate_shapley_values(
            df_per_token_combination, prompt_cleaned
        )
        if print_highlight_text:
            self.highlight_text_background()

        return self.explanation