<p align="center">
    <img src = "FigMethod2.pdf" alt="ConceptX Methodology"/>
  </p>
  
  
  # ConceptX: Concept-Level Explainability for Auditing and Steering LLM Responses
  
  This is the code to implement ConceptX method, a model-agnostic, concept-level attribution-based explainability method that overcomes these limitations. 
  Built on a coalition-based Shapley framework, ConceptX filters for meaningful content words, assigns
  them importance based on semantic similarity, and maintains contextual integrity
  through in-place replacement strategies. It also supports aspect-specific explanation
  objectives. ConceptX enables both auditing, e.g., uncovering sources of bias, and
  steering, e.g., modifying prompts to shift sentiment or reduce harmfulness, without
  retraining. 
  (see [papers](#citations) for details and citations).
  
  ## Getting Started
  
  ### Prerequisites
  
  This code was tested with Python 3.8.5.
  
  ### Installation
  
  Load the Python packages:
  ```
  pip install -r requirements.txt
  ```
  
  ### Datasets
 
  - Alpaca dataset: https://huggingface.co/datasets/tatsu-lab/alpaca
  - SST-2 dataset available at \url{https://huggingface.co/datasets/stanfordnlp/sst2}
  - Sp1786-Sentiment:  https://huggingface.590co/datasets/Sp1786/multiclass-sentiment-analysis-dataset.

  
  ### Pre-trained Large Language Models
  
  The pre-trained language models were extracted from the Hugging Face model hub.
  - MistralAI 7B: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
  - Gemma-3-3B:

  API Calls:
  - GPT-4o mini: 
  
  
  ### Embedding models

  Library: SBERT.net, https://www.sbert.net/docs/sentence_transformer/pretrained_
models.html

  ### Classifiers

  - Sentiment classifier: RoBERTa-base model fine-tuned on the TweetEval sentiment benchmark: \url{https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest}
  - Safety classifier: MD-Judge-v0_2-internlm2_7b generates a label safe/unsafe as well as a safety score ranging from 1 (completely harmless) to 5 (extremely harmful): \url{https://huggingface.co/OpenSafetyLab/MD-Judge-v0_2-internlm2_7b}
  Note: MD-Judge-v0_2-internlm2_7b requires Transformers version: 4.41.2

  ## Usage
  
  To generate explanations by explainer, you can use the following commands:
  
  ```bash
  python src/explain.py --dataset [dataset] --model_name [model_name] --explainer [explainer] --seed [seed]
  ```
  
  The following parameters are available:
  - dataset: the name of the dataset to use "negation", "generics", "rocstories"
  - model_name: the name of the decoder language model to use "gpt2", "mistral"
  - explainer: the name of the explainability method to use "lime", "shap", "partition", "syntax", "syntax-w"
  - seed: the random seed to use for reproducibility (e.g. 0)
  
  
  ## Citation
  If you are using ConceptX code, please cite the following paper:
  ```
  @article{Anonymous,
    title={Concept-Level Explainability for Auditing and Steering LLM Responses},
    author={Anonymous},
    journal={Anonymous},
    year={2025}
  }
  ```
  For any questions about this code please file an github [issue](https://github.com/).
  
  
  ## References
  
  1. *LIME:* Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why should i trust you?: Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016.
  
  2. *Shapley sampling values:* Strumbelj, Erik, and Igor Kononenko. "Explaining prediction models and individual predictions with feature contributions." Knowledge and information systems 41.3 (2014): 647-665.
  
  3. *Partition* Chen, Hanjie, Guangtao Zheng, and Yangfeng Ji. "Generating hierarchical explanations on text classification via feature interaction detection." arXiv preprint arXiv:2004.02015 (2020).