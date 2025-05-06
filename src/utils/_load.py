import os
import pandas as pd
import pickle as pkl
from datasets import load_dataset
from explainers._vectorizer import TextVectorizer, HuggingFaceEmbeddings, OpenAIEmbeddings, TfidfTextVectorizer

def get_path(args, folder_name):
    save_dir = os.path.join(args.result_save_dir, f'{folder_name}/{args.model_name}/{args.dataset}/{args.explainer}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{folder_name}_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.explainer}_{args.seed}.{args.file_type}"
    return os.path.join(save_dir, filename)

def get_sentiment_file_path(args):
    save_dir = os.path.join(args.result_save_dir, f'sentiment/{args.model_name}/{args.dataset}/{args.explainer}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"sentiment_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.explainer}_"
    filename += f"antonym_" if args.steer_replace == "antonym" else ""
    filename += f"{args.seed}.csv"
    return os.path.join(save_dir, filename)

def get_answers_file_path(args):
    save_dir = os.path.join(args.result_save_dir, f'answers/{args.model_name}/seed_{args.seed}/{args.defender}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"answers_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.defender}_"
    filename += f"antonym_" if args.steer_replace == "antonym" else ""
    filename += f"{args.seed}.csv"
    return os.path.join(save_dir, filename)

def get_safety_file_path(args):
    save_dir = os.path.join(args.result_save_dir, f'safety/{args.safety_classifier}/{args.model_name}/seed_{args.seed}/{args.defender}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"safety_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.defender}_"
    filename += f"antonym_" if args.steer_replace == "antonym" else ""
    filename += f"{args.seed}.csv"
    return os.path.join(save_dir, filename)


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
        return HuggingFaceEmbeddings(model_name = "sentence-transformers/" + kwargs.get("embedding_model", "all-MiniLM-L6-v2"),
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
    
def split_batch(df, num_batch, batch_size):
    # Limit data to specified batch size and number
    n = len(df)
    # Check that num_batch and batch_size do not exceed available data
    total_batches = n // batch_size + (1 if n % batch_size != 0 else 0)
    assert num_batch < total_batches, f"Batch number {num_batch} is too large! Only {total_batches} batches available."

    n_min = batch_size * num_batch
    n_max = min(batch_size * (num_batch + 1), n)  # Ensure we don't exceed the dataset size
    return df[n_min:n_max]

def load_data(args):
    # Load dataset based on argument
    if args.dataset == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca")
        df = pd.DataFrame(ds['train'])
        df_filtered = df[df['input'].isna() | (df['input'] == '')]
        # Filter based on instruction length
        df_filtered = df_filtered[df_filtered['instruction'].str.len() <= 58]
        df_filtered['id'] = df_filtered.index
        df_final = df_filtered[['id', 'instruction']].rename(columns={'instruction': 'input'})
        df_final = df_final.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    elif args.dataset == "sst2":
        df_final = pd.read_csv(os.path.join(args.data_save_dir, "sst2_classification.csv"))
        df_final = df_final.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    elif args.dataset == "sp1786":
        df_final = pd.read_csv(os.path.join(args.data_save_dir, "sp1786_classification.csv"))[:1010]
    elif args.dataset == "genderbias":
        df_final = pd.read_csv(os.path.join(args.data_save_dir, "stereotypical_temp_0.8_responses.csv"))
    elif args.dataset == "genderbias2":
        df_final = pd.read_csv(os.path.join(args.data_save_dir, "stereotypical_temp_0.8_responses_v2.csv"))
    elif args.dataset == "saladbench":
        df_final = pd.read_csv(os.path.join(args.data_save_dir, "saladbench_labeled.csv"))
    else:
        raise ValueError("Unknown dataset type passed: %s!" % args.dataset)
    
    
    if args.num_batch is not None:
        print(f"Batch number {args.num_batch} of size {args.batch_size} is being used.")
        df_final = split_batch(df_final, args.num_batch, args.batch_size)
    else:
        print(f"Batch number is not specified. Using all {len(df_final)} examples.")
    
    return df_final
    
   
def load_file(args, folder_name):
    # Load explanations from the specified path
    data_path = get_path(args, folder_name)
    if args.file_type == "pkl":
        with open(data_path, "rb") as f:
            data = pkl.load(f)
        df = pd.DataFrame(data)
    elif args.file_type == "csv":
        df = pd.read_csv(data_path)
    return df

def save_file(data, args, folder_name):
    # Save data to the specified path
    save_path = get_path(args, folder_name)
    if args.file_type == "pkl":
        with open(save_path, "wb") as f:
            pkl.dump(data, f)
    elif args.file_type == "csv":
        data.to_csv(save_path, index=False)
    return

def save_dataframe(data, args, folder_name):
    # Save data to the specified path
    save_path = get_path(args, folder_name)
    if args.file_type == "pkl":
        data.to_pickle(save_path, index=False)
    elif args.file_type == "csv":
        data.to_csv(save_path, index=False)
    return




def get_remaining_df(df_full, path):
    """
    Resumes from the row after the row with the last processed ID.
    
    Args:
        df_full (pd.DataFrame): Full dataframe with all instructions (must have an "id" column).
        path (str): Path to the saved processed CSV.

    Returns:
        pd.DataFrame: Remaining dataframe to process.
    """
    df_full = df_full.reset_index(drop=True)
    if os.path.isfile(path):
        existing_df = pd.read_csv(path)
        print(f"Loaded existing processed data with {len(existing_df)} rows.")

        if not existing_df.empty:
            last_id = existing_df["id"].drop_duplicates().iloc[-1]
        else:
            last_id = None
    else:
        last_id = None

    if last_id is not None:
        # Find the index of the row with the last processed id
        match_idx = df_full.index[df_full["id"] == last_id]
        if len(match_idx) == 0:
            raise ValueError(f"Last processed id {last_id} not found in full dataframe.")
        start_idx = match_idx[0] + 1  # Resume from the next index
    else:
        start_idx = 0  # Start from beginning

    print(f"Resuming from last index: {start_idx-1} and last id: {last_id}")

    # Return the remaining dataframe starting from the next index
    return df_full.iloc[start_idx:]


def extract_args_from_filename(file):
    """
    Extract arguments from the file name and return a dictionary of arguments.
    
    Args:
        file (str): The file name.
        args (argparse.Namespace): The existing args to merge with extracted values.

    Returns:
        dict: Dictionary of extracted arguments.
    """
    parts = file.split("_")
    print("parts: ", parts)
    
    args_dict = {
        "num_batch": None,
        "dataset": None,
        "model_name": None,
        "explainer": None,
        "baseline": None,
        "seed": None
    }

    if "batch" in parts[1]:
        args_dict["num_batch"] = int(parts[2])
        dataset_idx = 3
    else:
        dataset_idx = 1
    
    args_dict["dataset"] = parts[dataset_idx]
    args_dict["model_name"] = parts[dataset_idx + 1]
    args_dict["explainer"] = parts[dataset_idx + 2]
    
    if args_dict["model_name"] == "gpt4o":
        return None  # Skip if model_name is "gpt4o"
    
    if len(parts) > dataset_idx + 4:
        args_dict["baseline"] = parts[dataset_idx + 3]
        seed_idx = dataset_idx + 4
    else:
        seed_idx = dataset_idx + 3

    args_dict["seed"] = int(parts[seed_idx].split(".")[0])

    return args_dict