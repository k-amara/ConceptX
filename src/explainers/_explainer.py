#from token_shap import *
import matplotlib.pyplot as plt
from matplotlib import colors

from explainers._explain_utils import get_text_before_last_underscore
from explainers._splitter import Splitter
from explainers._vectorizer import TextVectorizer, TfidfTextVectorizer
from typing import Optional

# Refactored TokenSHAP class
class Explainer:
    def __init__(self, 
                 model, 
                 splitter: Splitter, 
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False):
        self.model = model
        self.splitter = splitter
        self.vectorizer = vectorizer or TfidfTextVectorizer()
        self.debug = debug  # Add debug mode

    def _debug_print(self, message):
        if self.debug:
            print(message)

    def _calculate_baseline(self, prompt):
        baseline_text = self.model.generate(prompt)
        print("Baseline text:", baseline_text)
        return baseline_text
    
    def analyze(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the analyze method")

    def __call__(self, prompts, *args, **kwargs):
        scores = []
        for prompt in prompts:
            scores.append(self.analyze(prompt, *args, **kwargs))
        return scores

    def print_colored_text(self):
        scores = self.scores
        min_value = min(scores.values())
        max_value = max(scores.values())

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

        for token, value in scores.items():
            color = get_color(value)
            print(
                f"\033[38;2;{int(color[1:3], 16)};"
                f"{int(color[3:5], 16)};"
                f"{int(color[5:7], 16)}m"
                f"{get_text_before_last_underscore(token)}\033[0m",
                end=' '
            )
        print()

    def _get_color(self, value, scores):
        norm_value = (value - min(scores.values())) / (
            max(scores.values()) - min(scores.values())
        )
        cmap = plt.cm.coolwarm
        return colors.rgb2hex(cmap(norm_value))

    def plot_colored_text(self, new_line=False):
        num_items = len(self.scores)
        fig_height = num_items * 0.5 + 1 if new_line else 2

        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')

        y_pos = 1
        x_pos = 0.1
        step = 1 / (num_items + 1)

        for sample, value in self.scores.items():
            color = self._get_color(value, self.scores)
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
                vmin=min(self.scores.values()),
                vmax=max(self.scores.values())
            )
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.ax.set_position([0.05, 0.02, 0.9, 0.05])
        cbar.set_label('Scores', fontsize=12)

        plt.tight_layout()
        plt.show()

    def highlight_text_background(self):
        min_value = min(self.scores.values())
        max_value = max(self.scores.values())

        def get_background_color(value):
            norm_value = ((value - min_value) / (max_value - min_value)) ** 3
            r = 255
            g = 255
            b = int(255 - (norm_value * 255))
            return f"\033[48;2;{r};{g};{b}m"

        for token, value in self.scores.items():
            background_color = get_background_color(value)
            reset_color = "\033[0m"
            print(f"{background_color}{get_text_before_last_underscore(token)}{reset_color}", end=' ')
        print()
