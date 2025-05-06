<p align="center">
    <img src="FigMethod2.svg" alt="ConceptX Methodology"/>
  </p>
  
  # ConceptX: Concept-Level Explainability for Auditing and Steering LLM Responses
  
  **ConceptX** is a model-agnostic, concept-level attribution method that enables interpretable and controllable large language model (LLM) behavior. Built on a coalition-based Shapley framework, ConceptX:
  
  - Identifies semantically meaningful content words (concepts)
  - Assigns importance scores based on semantic similarity
  - Maintains contextual integrity via in-place replacement strategies
  - Supports **aspect-specific explanations** (e.g., targeting sentiment, bias, references)
  
  ConceptX can be used both for **auditing** (e.g., detecting bias or sentiment sources) and **steering** (e.g., prompt interventions to reduce harmfulness), all **without retraining** the model.
  
  See the [papers](#citation) for more details.
  
  ---
  
  ## Getting Started
  
  ### Prerequisites
  
  - Python 3.8.5
  
  ### Installation
  
  Install the required packages:
  
  ```bash
  pip install -r requirements.txt
  ```
  
  ---
  
  ## Datasets
  
  ### Public datasets
  - **[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)** – Instruction-following dataset
  - **[SST-2](https://huggingface.co/datasets/stanfordnlp/sst2)** – Binary sentiment classification
  - **[Sp1786-Sentiment](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset)** – Multiclass sentiment analysis
  
  ### ConceptX additional dataset
  - **GenderBias**: Download link TBD (anonymous for review)
  
  ---
  
  ## Pre-trained LLMs
  
  All models are sourced from [HuggingFace](https://huggingface.co/):
  
  - [**MistralAI 7B**](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - [**Gemma-3-4B**](https://huggingface.co/google/gemma-3-4b-it)
  - [**LLaMA-3.2-3B**](https://huggingface.co/meta-llama/Llama-3.2-3B)
  
  Also supported via API:
  - [**GPT-4o mini**](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence)
  
  ---
  
  ## Embedding Models
  
  From [SBERT](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html):
  
  - `all-MiniLM-L6-v2` — Compact, 384-dim embeddings
  - `all-mpnet-base-v2` — Larger, 768-dim embeddings
  
  ---
  
  ## Classifiers
  
  - **Sentiment Classifier**  
    RoBERTa-base, fine-tuned on TweetEval  
    [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
  
  - **Safety Classifier**  
    `MD-Judge-v0_2-internlm2_7b`  
    Labels: `safe` / `unsafe`, plus a 1–5 safety score  
    [OpenSafetyLab/MD-Judge-v0_2-internlm2_7b](https://huggingface.co/OpenSafetyLab/MD-Judge-v0_2-internlm2_7b)  
    ⚠️ Requires `transformers==4.41.2`
  
  ---
  
  ## Usage
  
  Generate explanations using:
  
  ```bash
  python src/explain.py --dataset [dataset] --model_name [model] --explainer [method] --seed [seed]
  ```
  
  ### Parameters:
  - `dataset`: One of `alpaca`, `sst2`, `genderbias`
  - `model_name`: Decoder LLM, e.g. `gpt2`, `mistral`
  - `explainer`: e.g. `random`, `tokenshap`, `conceptx-B-r`, `conceptx-B-n`, `conceptx-A-r`
  - `seed`: Random seed for reproducibility (e.g. `0`)
  
  ---
  
  ## ConceptX Naming Convention
  
  If using ConceptX, `args.explainer` should follow the format:  
  `conceptx-<T>-<R>` where:
  - `<T>` is one of:
    - `A` → aspect-based explanation
    - `R` → reference-based
    - `B` → base (general explanation)
  - `<R>` is one of:
    - `r` → remove
    - `n` → neutral
    - `a` → antonym
  
  Example: `conceptx-A-n` means *aspect-based explanation using neutral replacements*.
  
  **Code snippet to parse and validate:**
  
  ```python
  if args.explainer.startswith("conceptx"):
      parts = args.explainer.split("-")
      assert parts[1] in ("A", "R", "B"), f"Invalid code in parts[1]: {parts[1]}"
      assert parts[2] in ("r", "n", "a"), f"Invalid code in parts[2]: {parts[2]}"
  
      args.target = (
          "aspect" if parts[1] == "A" else
          "reference" if parts[1] == "R" else
          "base"
      )
  
      args.coalition_replace = (
          "neutral" if parts[2] == "n" else
          "antonym" if parts[2] == "a" else
          "remove"
      )
  ```
  
  ---
  
  ## Citation
  
  If you use ConceptX in your research, please cite:
  
  ```bibtex
  @article{Anonymous,
    title = {Concept-Level Explainability for Auditing and Steering LLM Responses},
    author = {Anonymous},
    journal = {Anonymous},
    year = {2025}
  }
  ```
  
  ---
  
  ## References
  
  1. Goldshmidt, Roni, and Miriam Horovicz. *"TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation."* arXiv preprint arXiv:2407.10114 (2024).
  
  ---
  
  ## Questions or Issues?
  
  Please [open a GitHub issue](https://github.com/) for support or feedback.