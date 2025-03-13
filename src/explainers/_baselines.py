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
        random_explanation = {f"{token}_{i}": random.random() for i, token in enumerate(tokens)}
        
        # Normalize explanation
        min_value = min(random_explanation.values())
        max_value = max(random_explanation.values())
        self.explanation = {
            token_key: (score - min_value) / (max_value - min_value)
            for token_key, score in random_explanation.items()
        }
        print("Random token explanation:", self.explanation)
        return self.explanation


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
        self.explanation = {f"{token}_{i}": score.item() for i, (token, score) in enumerate(zip(tokens, attr))}
        print("SVSampling token explanation:", self.explanation)
        return self.explanation


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
        self.explanation = {f"{token}_{i}": score.item() for i, (token, score) in enumerate(zip(tokens, attr))}
        print("FeatAblation token explanation:", self.explanation)
        return self.explanation
