#from token_shap import *
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

from explainers._explain_utils import get_text_before_last_underscore
from explainers._splitter import Splitter
from explainers._vectorizer import TextVectorizer, TfidfTextVectorizer
from typing import Optional
from model import ContentPolicyViolationError

# Refactored TokenSHAP class
class Explainer:
    def __init__(self, 
                 llm, 
                 splitter: Splitter, 
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False):
        self.llm = llm
        self.splitter = splitter
        self.vectorizer = vectorizer or TfidfTextVectorizer()
        self.debug = debug  # Add debug mode

    def _debug_print(self, message):
        if self.debug:
            print(message)

    def _calculate_baseline(self, prompt):
        try:
            baseline_text = self.llm.generate(prompt)
        except ContentPolicyViolationError:
            baseline_text = None
        return baseline_text
    
    def analyze(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the analyze method")

    def __call__(self, prompts, *args, **kwargs):
        explanation = []
        for prompt in prompts:
            exp = self.analyze(prompt, *args, **kwargs)
            if exp is not None:
                explanation.append(exp)
        return explanation

    def print_colored_text(self):
        explanation = self.explanation
        min_value = min(explanation.values())
        max_value = max(explanation.values())

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

        for token, value in explanation.items():
            color = get_color(value)
            print(
                f"\033[38;2;{int(color[1:3], 16)};"
                f"{int(color[3:5], 16)};"
                f"{int(color[5:7], 16)}m"
                f"{get_text_before_last_underscore(token)}\033[0m",
                end=' '
            )
        print()

    def _get_color(self, value, explanation):
        norm_value = (value - min(explanation.values())) / (
            max(explanation.values()) - min(explanation.values())
        )
        cmap = plt.cm.coolwarm
        return colors.rgb2hex(cmap(norm_value))

    def plot_colored_text(self, new_line=False):
        num_items = len(self.explanation)
        fig_height = num_items * 0.5 + 1 if new_line else 2

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1
        x_pos = 0.1
        step = 1 / (num_items + 1)

        for sample, value in self.explanation.items():
            color = self._get_color(value, self.explanation)
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
                vmin=min(self.explanation.values()),
                vmax=max(self.explanation.values())
            )
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.ax.set_position([0.05, 0.02, 0.9, 0.05])
        cbar.set_label('explanation', fontsize=12)

        plt.tight_layout()
        plt.show()

    def highlight_text_background(self):
        min_value = min(self.explanation.values())
        max_value = max(self.explanation.values())

        def get_background_color(value):
            if pd.isna(value):  # Check for NaN values
                value = 0  # Assign a default value (e.g., 0 for minimum intensity)
            norm_value = ((value - min_value) / (max_value - min_value)) ** 3
            r = 255
            g = 255
            b = int(255 - (norm_value * 255))
            return f"\033[48;2;{r};{g};{b}m"

        for token, value in self.explanation.items():
            background_color = get_background_color(value)
            reset_color = "\033[0m"
            print(f"{background_color}{get_text_before_last_underscore(token)}{reset_color}", end=' ')
        print()
