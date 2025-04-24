from explainers import Explainer
import random 
from captum.attr import (
    ShapleyValueSampling,
    FeatureAblation,
    LLMAttribution, 
    TextTokenInput,
)
from explainers._explain_utils import normalize_explanation


class Random(Explainer):
    def __init__(self, llm, splitter, vectorizer=None, debug=False):
        super().__init__(llm, splitter, vectorizer, debug)

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
    def __init__(self, llm, splitter, vectorizer=None, debug=False):
        super().__init__(llm, splitter, vectorizer, debug)
        self.tokenizer = self.llm.tokenizer
        sv = ShapleyValueSampling(self.llm.model) 
        self.llm_attr = LLMAttribution(sv, self.tokenizer)
    
    def analyze(self, prompt):
        skip_tokens = [1]  # skip the special token for the start of the text <s>
        inp = TextTokenInput(
            prompt, 
            self.tokenizer,
            skip_tokens=skip_tokens,
        )
        target = self._calculate_baseline(prompt)
        print("Target:", target)
        attr_res = self.llm_attr.attribute(inp, target=target)
        attr = attr_res.seq_attr
        self.explanation = {f"{token}_{i}": score.item() for i, (token, score) in enumerate(zip(inp.values, attr))}
        print("SVSampling token explanation:", self.explanation)
        return normalize_explanation(self.explanation)



class FeatAblation(Explainer):
    def __init__(self, llm, splitter, vectorizer=None, debug=False):
        super().__init__(llm, splitter, vectorizer, debug)
        self.tokenizer = self.llm.tokenizer
        fa = FeatureAblation(self.llm.model) 
        self.llm_attr = LLMAttribution(fa, self.tokenizer)
    
    def analyze(self, prompt):
        skip_tokens = [1]  # skip the special token for the start of the text <s>
        inp = TextTokenInput(
            prompt, 
            self.tokenizer,
            skip_tokens=skip_tokens,
        )
        target = self._calculate_baseline(prompt)
        print("Target:", target)
        # Is the target in 
        # Compute Shapley values
        attr_res = self.llm_attr.attribute(inp, target=target)
        attr = attr_res.seq_attr
        self.explanation = {f"{token}_{i}": score.item() for i, (token, score) in enumerate(zip(inp.values, attr))}
        print("FeatAblation token explanation:", self.explanation)
        return normalize_explanation(self.explanation)
    