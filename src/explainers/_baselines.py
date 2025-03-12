from explainers import Explainer
import random 
from captum.attr import (
    ShapleyValueSampling,
    FeatureAblation,
)


class Random(Explainer):
    def __init__(self, model, splitter, vectorizer=None, debug=False):
        super().__init__(model, splitter, vectorizer, debug)

    def analyze(self, prompt):
        tokens = self.splitter.split(prompt)
        random_scores = {token: random.random() for token in tokens}
        
        # Normalize scores
        min_value = min(random_scores.values())
        max_value = max(random_scores.values())
        self.shapley_values = {
            token: (score - min_value) / (max_value - min_value)
            for token, score in random_scores.items()
        }
        print("Random token scores:", self.shapley_values)
        return self.shapley_values


class SVSampling(Explainer):
    def __init__(self, model, splitter, vectorizer=None, debug=False):
        super().__init__(model, splitter, vectorizer, debug)
        self.sv = ShapleyValueSampling(self.model) 
    
    def analyze(self, prompt):
        tokens = self.splitter.split(prompt)
        target = self._calculate_baseline(prompt)
        print("Target:", target)
        # Is the target in 
        # Compute Shapley values
        attr = self.sv.attribute(tokens, target=target)
        self.shapley_values = {token: score.item() for token, score in zip(tokens, attr)}
        print("SVSampling token scores:", self.shapley_values)
        return self.shapley_values


class FeatAblation(Explainer):
    def __init__(self, model, splitter, vectorizer=None, debug=False):
        super().__init__(model, splitter, vectorizer, debug)
        self.fa = FeatureAblation(self.model) 
    
    def analyze(self, prompt):
        tokens = self.splitter.split(prompt)
        target = self._calculate_baseline(prompt)
        print("Target:", target)
        # Is the target in 
        # Compute Shapley values
        attr = self.fa.attribute(tokens, target=target)
        self.shapley_values = {token: score.item() for token, score in zip(tokens, attr)}
        print("FeatAblation token scores:", self.shapley_values)
        return self.shapley_values
