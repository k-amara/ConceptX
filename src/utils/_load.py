import os
import pandas as pd
from datasets import load_dataset
from explainers._vectorizer import TextVectorizer, HuggingFaceEmbeddings, OpenAIEmbeddings, TfidfTextVectorizer


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
    


def load_data(args):
    # Load dataset based on argument
    if args.dataset == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca")
        df = pd.DataFrame(ds['train'])
        df_filtered = df[df['input'].isna() | (df['input'] == '')]
        df_filtered['id'] = df_filtered.index
        return df_filtered[['id', 'instruction']]
    elif args.dataset == "genderbias":
        df = pd.reas_csv(os.join(args.data_save_dir, "stereotypical_temp_0.8_responses.csv"))
        # ['id', 'instruction', 'reference', 'gender']
        return df[['id', 'instruction', 'reference']]
    else:
        raise ValueError("Unknown dataset type passed: %s!" % args.dataset_name)
    
  ## final dataset df.columns ['id', 'instruction', 'reference_text']