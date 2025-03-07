#from token_shap import *
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import re
import numpy as np
import random
from typing import Optional

import spacy
import requests

from shap_utils import get_text_before_last_underscore, TextVectorizer, HuggingFaceEmbeddings


nlp = spacy.load("en_core_web_sm")


import requests
import spacy

class ConceptProcessor:
    def __init__(self, split_pattern = ' '):
        self.split_pattern = split_pattern
        self.nlp = spacy.load("en_core_web_sm")
    
    def split(self, prompt):
        return re.split(self.split_pattern, prompt.strip())
    
    def join(self, words):
        return ' '.join(words)

    def extract_meaningful_concepts(self, text):
        """Extracts meaningful concepts (nouns, proper nouns, verbs) from a given text."""
        doc = self.nlp(text)
        return [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "VERB"} and not token.is_stop]

    def get_conceptnet_edges(self, word):
        """Fetches the number of ConceptNet edges (relations) for a given word."""
        url = f"http://api.conceptnet.io/c/en/{word}"
        response = requests.get(url).json()
        return len(response.get('edges', []))  # Count relations

    def split_concepts(self, text, concept_ratio=1.0):
        """Splits the text into meaningful concepts and their indices in the original text."""
        assert 0.1 <= concept_ratio <= 1.0, "The ratio of concepts in the input prompt must be between 0.1 and 1."
        
        words = self.split(text)  # Tokenize the text into words
        concepts = self.extract_meaningful_concepts(text)
        
        if not concepts:
            return [], [], []

        top_n = max(1, int(len(concepts) * concept_ratio))  # Number of top concepts to select
        
        # Get scores for concepts, assigning 0 if the word is not in ConceptNet
        concept_scores = {
            word: self.get_conceptnet_edges(word) if self.get_conceptnet_edges(word) > 0 else 0
            for word in concepts
        }

        # Sort by score (highest first) and take the top-N concepts
        sorted_list = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Extract top concepts and their scores
        concepts = [item[0] for item in sorted_list]
            
        # Find indices of the selected concepts in the word list
        indices = [words.index(concept) for concept in concepts if concept in words]
        
        # Reorder concepts based on their indices in ascending order.
        sorted_pairs = sorted(zip(indices, concepts))
        sorted_indices, sorted_concepts = zip(*sorted_pairs)
        
        return list(sorted_concepts), list(sorted_indices)

    def replace_concepts_in_combination(self, concepts, replacements, selected_concept_indices):
        """
        Replace unselected concepts with their replacement while keeping selected ones unchanged.
        """
        return [
            concepts[i] if i in selected_concept_indices else replacements[i]
            for i in range(len(concepts))
        ]

    def replace_concepts_in_words(self, words, new_concepts, concept_indices):
        """
        Integrate the modified concepts back into the original list of words.
        """
        new_words = words[:]
        for concept_idx, word_idx in zip(range(len(new_concepts)), concept_indices):
            new_words[word_idx] = new_concepts[concept_idx]
        return new_words
    
    # def get_replacements(self, concepts, text):
    
    def get_main_concept(self, text):
        """Identifies the most important concept in the text based on ConceptNet connections."""
        doc = self.nlp(text)
        candidate_concepts = {token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}}

        if not candidate_concepts:
            return None, 0

        concept_scores = {word: self.get_conceptnet_edges(word) for word in candidate_concepts}
        main_concept = max(concept_scores, key=concept_scores.get)

        return main_concept, concept_scores[main_concept]


class ConceptSHAP:
    def __init__(self, 
                 model, 
                 processor: ConceptProcessor, 
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False):
        self.model = model
        self.processor = processor
        self.vectorizer = vectorizer or HuggingFaceEmbeddings()
        self.debug = debug  # Add debug mode

    def _debug_print(self, message):
        if self.debug:
            print(message)

    def _calculate_baseline(self, prompt):
        baseline_text = self.model.generate(prompt)
        return baseline_text

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

    def _get_result_per_concept_combination(self, sampling_ratio):
        n = len(self.concepts)
        self._debug_print(f"Number of concepts: {n}")
        if n > 1000:
            print("Warning: the number of concepts is greater than 1000; execution will be slow.")

        num_total_combinations = 2 ** n - 1
        self._debug_print(f"Total possible combinations (excluding empty set): {num_total_combinations}")

        num_sampled_combinations = int(num_total_combinations * sampling_ratio)
        self._debug_print(f"Number of combinations to sample based on sampling ratio {sampling_ratio}: {num_sampled_combinations}")

        # Always include combinations missing one concept
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = self.concepts[:i] + self.concepts[i + 1:]
            indexes = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)

        self._debug_print(f"Number of essential combinations (each missing one concept): {len(essential_combinations)}")

        num_additional_concepts = max(0, num_sampled_combinations - len(essential_combinations))
        self._debug_print(f"Number of additional combinations to sample: {num_additional_concepts}")

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
            new_concepts = self.processor.replace_concepts(self.concepts, self.replacements, indexes)
            new_words = self.replace_concepts_in_words(self.words, new_concepts, self.indices)
            print("New Words: ", new_words)
            text = self.processor.join(new_words)
            print("Text: ", text)
            self._debug_print(f"\nProcessing combination {idx + 1}/{len(all_combinations_to_process)}:")
            self._debug_print(f"Combination concepts: {combination}")
            self._debug_print(f"Concept indexes: {indexes}")
            self._debug_print(f"Generated text: {text}")

            text_response = self.model.generate(text)
            self._debug_print(f"Received response for combination {idx + 1}")

            prompt_key = text + '_' + ','.join(str(index) for index in indexes)
            prompt_responses[prompt_key] = (text_response, indexes)

        self._debug_print("Completed processing all combinations.")
        return prompt_responses

    def _get_df_per_concept_combination(self, prompt_responses):
        df = pd.DataFrame(
            [(prompt.split('_')[0], response[0], response[1])
             for prompt, response in prompt_responses.items()],
            columns=['Prompt', 'Response', 'Concept_Indexes']
        )

        all_texts = [self.target_concept] + df["Response"].tolist()
        
        # Use the configured vectorizer
        vectors = self.vectorizer.vectorize(all_texts)
        target_concept_vector = vectors[0]
        comparison_vectors = vectors[1:]
        
        # Calculate similarities
        cosine_similarities = self.vectorizer.calculate_similarity(
            target_concept_vector, comparison_vectors
        )
        
        df["Cosine_Similarity"] = cosine_similarities

        return df

    def _calculate_shapley_values(self, df_per_concept_combination):
        def normalize_shapley_values(shapley_values, power=1):
            min_value = min(shapley_values.values())
            shifted_values = {k: v - min_value for k, v in shapley_values.items()}
            powered_values = {k: v ** power for k, v in shifted_values.items()}
            total = sum(powered_values.values())
            if total == 0:
                return {k: 1 / len(powered_values) for k in powered_values}
            normalized_values = {k: v / total for k, v in powered_values.items()}
            return normalized_values
        
        shapley_values = {}

        for i, concept in enumerate(self.concepts, start=1):
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

            shapley_values[concept + "_" + str(i)] = with_concept - without_concept

        return normalize_shapley_values(shapley_values)

    def print_colored_text(self):
        shapley_values = self.shapley_values
        min_value = min(shapley_values.values())
        max_value = max(shapley_values.values())

        def get_color(value):
            norm_value = (value - min_value) / (max_value - min_value)

            if norm_value < 0.5:
                r = int(255 * (norm_value * 2))
                g = int(255 * (norm_value * 2))
                b = 255
            else:
                r = 255
                g = int(255 * (2 - norm_value * 2))
                b = int(255 * (2 - norm_value * 2))

            return '#{:02x}{:02x}{:02x}'.format(r, g, b)

        for token, value in shapley_values.items():
            color = get_color(value)
            print(
                f"\033[38;2;{int(color[1:3], 16)};"
                f"{int(color[3:5], 16)};"
                f"{int(color[5:7], 16)}m"
                f"{get_text_before_last_underscore(token)}\033[0m",
                end=' '
            )
        print()

    def _get_color(self, value, shapley_values):
        norm_value = (value - min(shapley_values.values())) / (
            max(shapley_values.values()) - min(shapley_values.values())
        )
        cmap = plt.cm.YlOrRed
        return colors.rgb2hex(cmap(norm_value))

    def plot_colored_concept(self, new_line=False):
        num_items = len(self.shapley_values)
        fig_height = num_items * 0.5 + 1 if new_line else 2

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1
        x_pos = 0.1
        step = 1 / (num_items + 1)

        for concept, value in self.shapley_values.items():
            color = self._get_color(value, self.shapley_values)
            if new_line:
                ax.text(
                    0.5, y_pos, get_text_before_last_underscore(concept), color=color, fontsize=20,
                    ha='center', va='center', transform=ax.transAxes
                )
                y_pos -= step
            else:
                ax.text(
                    x_pos, y_pos, get_text_before_last_underscore(concept), color=color, fontsize=20,
                    ha='left', va='center', transform=ax.transAxes
                )
                x_pos += 0.1

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm,
            norm=plt.Normalize(
                vmin=min(self.shapley_values.values()),
                vmax=max(self.shapley_values.values())
            )
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.ax.set_position([0.05, 0.02, 0.9, 0.05])
        cbar.set_label('Shapley Value', fontsize=12)

        plt.tight_layout()
        plt.show()
        
    def plot_colored_text(self, new_line=False):
        
        num_items = len(self.words)
        fig_height = num_items * 0.5 + 1 if new_line else 2

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1
        x_pos = 0.1
        step = 1 / (num_items + 1)
        
        self.shapley_values
        
        all_values = [0] * num_items 
        for idx, value in zip(self.indices, self.shapley_values):
            all_values[idx] = value 

        for sample, value in all_values.items():
            color = self._get_color(value, all_values)
            if new_line:
                ax.text(
                    0.5, y_pos, get_text_before_last_underscore(sample), color=color, fontsize=20,
                    ha='center', va='center', transform=ax.transAxes
                )
                y_pos -= step
            else:
                ax.text(
                    x_pos, y_pos, get_text_before_last_underscore(sample), color=color, fontsize=20,
                    ha='left', va='center', transform=ax.transAxes
                )
                x_pos += 0.1

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm,
            norm=plt.Normalize(
                vmin=min(all_values.values()),
                vmax=max(all_values.values())
            )
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.ax.set_position([0.05, 0.02, 0.9, 0.05])
        cbar.set_label('Shapley Value', fontsize=12)

        plt.tight_layout()
        plt.show()

    def highlight_text_background(self):
        min_value = min(self.shapley_values.values())
        max_value = max(self.shapley_values.values())

        def get_background_color(value):
            norm_value = ((value - min_value) / (max_value - min_value)) ** 3
            r = 255
            g = 255
            b = int(255 - (norm_value * 255))
            return f"\033[48;2;{r};{g};{b}m"

        for token, value in self.shapley_values.items():
            background_color = get_background_color(value)
            reset_color = "\033[0m"
            print(f"{background_color}{get_text_before_last_underscore(token)}{reset_color}", end=' ')
        print()

    def analyze(self, prompt, sampling_ratio=0.0, print_highlight_text=False):
        # Clean the prompt to prevent empty tokens
        prompt_cleaned = prompt.strip()
        prompt_cleaned = re.sub(r'\s+', ' ', prompt_cleaned)
        
        self.words = self.processor.split(prompt_cleaned)
        self.concepts, self.indices = self.processor.split_concepts(prompt_cleaned) # concepts are the samples in TokenSHAP
        self.replacements = self.processor.get_replacements(self.concepts, prompt_cleaned)
        
        print("Words: ", self.words)
        print("Concepts: ", self.concepts)
        print("Indices: ", self.indices)
        print("Replacements: ", self.replacements)
        
        self.baseline_text = self._calculate_baseline(prompt_cleaned)
        print("Baseline Text: ", self.baseline_text)
        # Get target concept to explain
        self.target_concept = self.processor.get_main_concept(self.baseline_text)
        print("Response Dominant Topic:", self.target_concept)
        
        concept_combinations_results = self._get_result_per_concept_combination(sampling_ratio)
        df_per_concept_combination = self._get_df_per_concept_combination(concept_combinations_results)
        self.shapley_values = self._calculate_shapley_values(df_per_concept_combination)
        if print_highlight_text:
            self.highlight_text_background()

        return df_per_concept_combination