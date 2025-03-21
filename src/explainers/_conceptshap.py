#from token_shap import *
import pandas as pd
from tqdm.auto import tqdm
import re
import numpy as np
import random
from typing import Optional, Any

from explainers._explainer import Explainer
from explainers._splitter import ConceptSplitter
from explainers._vectorizer import TextVectorizer
from explainers._explain_utils import normalize_explanation
from model import ContentPolicyViolationError



class ConceptSHAP(Explainer):
    def __init__(self, 
                 llm, 
                 splitter: ConceptSplitter, 
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
            indexes = tuple([i + 1 for i in range(n) if bin_str[i] == '1'])
            if indexes not in exclude_combinations_set and indexes not in sampled_combinations_set:
                sampled_combinations_set.add((tuple(combination), indexes))
        if len(sampled_combinations_set) < k:
            self._debug_print(f"Warning: Could only generate {len(sampled_combinations_set)} unique combinations out of requested {k}")
        return list(sampled_combinations_set)

    def _get_result_per_concept_combination(self):
        n = len(self.concepts)
        self._debug_print(f"Number of concepts: {n}")
        if n > 1000:
            print("Warning: the number of concepts is greater than 1000; execution will be slow.")

        num_total_combinations = 2 ** n - 1
        self._debug_print(f"Total possible combinations (excluding empty set): {num_total_combinations}")

        num_sampled_combinations = int(num_total_combinations * self.sampling_ratio)
        self._debug_print(f"Number of combinations to sample based on sampling ratio {self.sampling_ratio}: {num_sampled_combinations}")

        # Always include combinations missing one concept
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = self.concepts[:i] + self.concepts[i + 1:]
            indexes = tuple([j for j in range(n) if j != i])
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)

        self._debug_print(f"Number of essential combinations (each missing one concept): {len(essential_combinations)}")

        # Compute the number of additional combinations (capped at 30)
        num_additional_concepts = min(30, max(0, num_sampled_combinations - len(essential_combinations)))
        self._debug_print(f"Number of additional combinations to sample (capped at 30): {num_additional_concepts}")

        sampled_combinations = []
        if num_additional_concepts > 0:
            sampled_combinations = self._generate_random_combinations(
                self.concepts, num_additional_concepts, essential_combinations_set
            )
            self._debug_print(f"Number of sampled combinations: {len(sampled_combinations)}")
        else:
            self._debug_print("No additional combinations to sample.")

        # Combine essential and additional combinations
        all_combinations_to_process = essential_combinations + sampled_combinations
        self._debug_print(f"Total combinations to process: {len(all_combinations_to_process)}")

        prompt_responses = {}
        for idx, (combination, indexes) in enumerate(tqdm(all_combinations_to_process, desc="Processing combinations")):
            # change to produce 
            new_concepts = self.splitter.replace_concepts_in_combination(self.concepts, self.replacements, indexes)
            new_words = self.splitter.replace_concepts_in_words(self.words, new_concepts, self.indices)
            text = self.splitter.join(new_words)
            self._debug_print(f"\nProcessing combination {idx + 1}/{len(all_combinations_to_process)}:")
            self._debug_print(f"Combination concepts: {combination}")
            self._debug_print(f"Concept indexes: {indexes}")
            self._debug_print(f"Generated text: {text}")

            try:
                text_response = self.llm.generate(text)
                self._debug_print(f"Received response for combination {idx + 1}")

                prompt_key = text + '_' + ','.join(str(index) for index in indexes)
                prompt_responses[prompt_key] = (text_response, indexes)

            except ContentPolicyViolationError:
                self._debug_print(f"Skipping combination {idx + 1} due to content policy violation.")
                continue  # Skip this combination and move to the next one


            prompt_key = text + '_' + ','.join(str(index) for index in indexes)
            prompt_responses[prompt_key] = (text_response, indexes)

        self._debug_print("Completed processing all combinations.")
        return prompt_responses

    def _get_df_per_concept_combination(self, prompt_responses, baseline_text):
        df = pd.DataFrame(
            [(prompt.split('_')[0], response[0], response[1])
             for prompt, response in prompt_responses.items()],
            columns=['Prompt', 'Response', 'Concept_Indexes']
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

    def _calculate_explanation(self, df_per_concept_combination):
        explanation = {}

        for i, concept in enumerate(self.concepts, start=0):
            with_concept = np.average(
                df_per_concept_combination[
                    df_per_concept_combination["Concept_Indexes"].apply(lambda x: i in x)
                ]["Cosine_Similarity"].values
            )
            without_concept = np.average(
                df_per_concept_combination[
                    df_per_concept_combination["Concept_Indexes"].apply(lambda x: i not in x)
                ]["Cosine_Similarity"].values
            )
            explanation[concept + "_" + str(self.indices[i])] = with_concept - without_concept

        print("Importance explanation: ", explanation)
        explanation = normalize_explanation(explanation)
        print("Normalized Importance explanation: ", explanation)
        
        for i, word in enumerate(self.words, start=0):
            if i not in self.indices:
                explanation[word + "_" + str(i)] = np.float32(-1.0)
        
        explanation = {k: v for k, v in sorted(explanation.items(), key=lambda x: int(x[0].split('_')[1]))}

        return explanation

    def analyze(self, prompt, baseline=None, print_highlight_text=False, **kwargs):
        # Clean the prompt to prevent empty tokens
        prompt_cleaned = prompt.strip()
        prompt_cleaned = re.sub(r'\s+', ' ', prompt_cleaned)
        
        self.words = self.splitter.split(prompt_cleaned)
        self.concepts, self.indices = self.splitter.split_concepts(prompt_cleaned) # concepts are the samples in TokenSHAP
        
        # Skip processing if there is 1 or fewer concepts
        if len(self.concepts) <= 1:
            print("Skipping prompt due to insufficient concepts.")
            return None
        
        self.replacements = self.splitter.get_replacements(self.concepts, prompt_cleaned)
        if self.replacements is None:
            print("Skipping prompt due to content policy violation during replacements.")
            return None  # Skip explanation if replacements failed
        
        
        print("Words: ", self.words)
        print("Concepts: ", self.concepts)
        print("Indices: ", self.indices)
        print("Replacements: ", self.replacements)
        
        self.baseline_text = baseline if baseline is not None else self._calculate_baseline(prompt_cleaned)
        print(f"Baseline Text: {self.baseline_text}") 

        concept_combinations_results = self._get_result_per_concept_combination()
        df_per_concept_combination = self._get_df_per_concept_combination(concept_combinations_results, self.baseline_text)
        self.explanation = self._calculate_explanation(df_per_concept_combination)
        print("ConceptSHAP values: ", self.explanation)
        if print_highlight_text:
            self.highlight_text_background()

        return self.explanation
    
    def __call__(self, prompts, **kwargs):
        explanation = []
        baseline_texts = kwargs.get("baseline_texts", None)
        
        for i, prompt in enumerate(prompts):
            baseline_text = baseline_texts[i] if baseline_texts else None
            
            exp = self.analyze(prompt, baseline=baseline_text)
            if exp is not None:
                explanation.append(exp)
            
        return explanation
    
# baseline_text, _ = self.splitter.get_main_concept(self._calculate_baseline(prompt_cleaned))
            