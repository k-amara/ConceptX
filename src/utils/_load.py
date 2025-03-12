import os
import pandas as pd

from explainers import TextVectorizer, HuggingFaceEmbeddings, OpenAIEmbeddings, TfidfTextVectorizer


def load_vectorizer(vectorizer_name: str, **kwargs) -> TextVectorizer:
    """
    Load a vectorizer based on the provided vectorizer name.
    
    Args:
        vectorizer_name (str): Name of the vectorizer to load.
        **kwargs: Additional arguments required for specific vectorizers.
    
    Returns:
        TextVectorizer: An instance of the chosen vectorizer.
    """
    if vectorizer_name == "huggingface":
        return HuggingFaceEmbeddings(model_name=kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                                     device=kwargs.get("device", "cpu"))
    elif vectorizer_name == "openai":
        if "api_key" not in kwargs:
            raise ValueError("OpenAI vectorizer requires an API key.")
        return OpenAIEmbeddings(api_key=kwargs["api_key"],
                                model=kwargs.get("model", "text-embedding-3-small"))
    elif vectorizer_name == "tfidf":
        return TfidfTextVectorizer()
    else:
        raise ValueError(f"Unknown vectorizer name: {vectorizer_name}")
    


def load_data(dataset_name, data_save_dir):
    # Load dataset based on argument
    if dataset_name == "alpaca":
        df = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    elif dataset_name == "genderbias":
        pd.reas_csv(os.join(data_save_dir, "stereotypical_temp_0.8_responses.csv"))
        # ['id', 'instruction', 'reference', 'gender']
    else:
        raise ValueError("Unknown dataset type passed: %s!" % dataset_name)
    
  ## final dataset df.columns ['id', 'instruction', 'reference_text']