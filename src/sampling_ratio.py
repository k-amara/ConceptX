from explainers import *
import pandas as pd
import os
import gc
from model import LLMPipeline, LLMAPI, ContentPolicyViolationError
from utils import arg_parse, load_data, load_vectorizer, get_remaining_df
from accelerate.utils import set_seed
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()

def get_path_sampling_ratio(args, folder_name):
    save_dir = os.path.join(args.result_save_dir, f'sampling_ratio/{folder_name}/{args.model_name}/{args.dataset}/{args.explainer}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{folder_name}_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.explainer}_{args.seed}_{args.sampling_ratio}.{args.file_type}"
    return os.path.join(save_dir, filename)


def compute_explanations(args, save=True):
    if args.seed is not None:
        set_seed(args.seed)

    api_required = args.model_name in ["gpt4o-mini", "gpt4o", "o1", "deepseek"]
    rate_limit = args.model_name.startswith("gpt4")
    llm = LLMAPI(args, rate_limit_enabled=rate_limit) if api_required else LLMPipeline(args)

    vectorizer_kwargs = {}
    if args.vectorizer == "huggingface":
        vectorizer_kwargs["embedding_model"] = args.embedding_model
    vectorizer = load_vectorizer(args.vectorizer, **vectorizer_kwargs)

    df = load_data(args)
    print(df.head())

    # Choose appropriate explainer based on specified explainer
    kwargs = {}
    if args.explainer == "tokenshap":
        splitter = StringSplitter()
        explainer = TokenSHAP(llm, splitter, vectorizer, debug=False, sampling_ratio=args.sampling_ratio)
    elif args.explainer.startswith("conceptx"):
        splitter = ConceptSplitter()
        explainer = ConceptX(llm, splitter, vectorizer, debug=False, sampling_ratio=args.sampling_ratio, replace=args.coalition_replace)
        baseline_texts = None
        if args.target == "reference":
            baseline_texts = df['reference'].tolist()
        elif args.target == "aspect":
            baseline_texts = df['aspect'].tolist()
        kwargs = {"baseline_texts": baseline_texts} if baseline_texts is not None else {}
    else:
        raise ValueError(f"Unsupported explainer: {args.explainer}. Only 'tokenshap' and 'conceptx' are allowed.")

    explanations_path = get_path_sampling_ratio(args, folder_name="explanations")
    file_exists = os.path.isfile(explanations_path)
    df = get_remaining_df(df, explanations_path)

    for i in range(len(df)):
        input = df.iloc[i]["input"]
        if not isinstance(input, str):
            continue
        input_id = df.iloc[i]["id"]

        try:
            response = llm.generate(input)
            explanation_list = explainer([input], **kwargs)
            if not explanation_list:
                continue

            explanation = explanation_list[0]

            row_df = pd.DataFrame([{
                "id": input_id,
                "input": input,
                "response": response,
                "explanation": explanation
            }])

            if save:
                row_df.to_csv(explanations_path, mode="a", header=not file_exists, index=False)
                file_exists = True
            else:
                print(f"Explanation id {input_id}: ", row_df)

            # Memory cleanup
            del input, response, input_id, explanation, row_df
            gc.collect()

        except ContentPolicyViolationError:
            continue

    return


if __name__ == "__main__":
    parser, args = arg_parse()
    print('args do_sample:', args.do_sample)
    print('args result_save_dir:', args.result_save_dir)
    print('args sampling_ratio:', args.sampling_ratio)
    compute_explanations(args)
