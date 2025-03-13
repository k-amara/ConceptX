import os

        
def save_pkl_path(args, folder_name):
    save_dir = os.path.join(args.result_save_dir, f'{folder_name}/{args.model_name}/{args.dataset}/{args.explainer}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{folder_name}_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.explainer}_{args.seed}.pkl"
    return os.path.join(save_dir, filename)