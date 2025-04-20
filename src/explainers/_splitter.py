import re
import spacy
import requests

from model import ContentPolicyViolationError
from utils import *

nlp = spacy.load("en_core_web_sm")

class Splitter:
    def split(self, prompt):
        raise NotImplementedError

    def join(self, tokens):
        raise NotImplementedError

class StringSplitter(Splitter):
    def __init__(self, split_pattern = r'\b\w+\b'):
        self.split_pattern = split_pattern
    
    def split(self, prompt):
        return [word for word in re.findall(self.split_pattern, str(prompt).strip()) if word]
    
    def join(self, tokens):
        return ' '.join(tokens)

class TokenizerSplitter(Splitter):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def split(self, prompt):
        return self.tokenizer.tokenize(prompt)

    def join(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)
    
class ConceptSplitter(Splitter):
    def __init__(self, split_pattern = r'\b\w+\b'):
        self.split_pattern = split_pattern
        self.nlp = spacy.load("en_core_web_sm")
    
    def split(self, prompt):
        return [word for word in re.findall(self.split_pattern, str(prompt).strip()) if word]
    
    def join(self, words):
        text = ' '.join(words)
        text = ' '.join(text.split()) # remove extra space (usefule when replace = True)
        return  text

    def extract_meaningful_concepts(self, text):
        """Extracts meaningful concepts (nouns, proper nouns, verbs, adjectives) from a given text,
        ensuring they exist in the tokenized word list."""
        words = self.split(text)  # Tokenize the text into words
        doc = self.nlp(text)
        
        # Extract concepts that are meaningful and exist in the tokenized words
        return [token.text for token in doc if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"} 
                and not token.is_stop and token.text in words]
        
    def get_conceptnet_edges(self, word):
        """Fetches the number of ConceptNet edges (relations) for a given word."""
        url = f"http://api.conceptnet.io/c/en/{word.lower()}"
        response = requests.get(url).json()
        return len(response.get('edges', []))  # Count relations

    def split_concepts(self, text, concept_ratio=1.0):
        """Splits the text into meaningful concepts and their indices in the original text."""
        assert 0.1 <= concept_ratio <= 1.0, "The ratio of concepts in the input prompt must be between 0.1 and 1."
        
        words = self.split(text)  # Tokenize the text into words
        concepts = self.extract_meaningful_concepts(text)
        print("concepts: ", concepts)
        
        if not concepts:
            return [], []

        top_n = max(1, int(len(concepts) * concept_ratio))  # Number of top concepts to select
        # Get scores for concepts, assigning 0 if the word is not in ConceptNet
        concept_scores = {
            word: self.get_conceptnet_edges(word) if self.get_conceptnet_edges(word) > 0 else 0
            for word in concepts
        }
        print("concept_scores: ", concept_scores)
        print("topn: ", top_n)
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
    
    def get_replacements(self, concepts, text, replace="neutral"):
        if (replace=="neutral") or (replace=="antonym"):
            prompt = eval(f"create_prompt_for_{replace}_replacement")(text, concepts)
            try:
                completions = get_multiple_completions(prompt, num_sequences=1)
                print(completions)
                return completions[0]
            
            except ContentPolicyViolationError:
                print(f"Skipping prompt due to content policy violation during replacements: {text}")
                return None  # Return None to signal skipping
        else:
            return str([""]*len(concepts))
    
    def get_main_concept(self, text):
        """Identifies the most important concept in the text based on ConceptNet connections."""
        doc = self.nlp(text)
        candidate_concepts = {token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}}

        if not candidate_concepts:
            return None, 0

        concept_scores = {word: self.get_conceptnet_edges(word) for word in candidate_concepts}
        main_concept = max(concept_scores, key=concept_scores.get)

        return main_concept, concept_scores[main_concept]
    
