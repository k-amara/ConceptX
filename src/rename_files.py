import os

root_dir = "results/accuracy/gemma-2-2b"  # replace with your actual path
old_str = "gemma"
new_str = "gemma-2-2b"

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if old_str in filename:
            old_path = os.path.join(dirpath, filename)
            new_filename = filename.replace(old_str, new_str)
            new_path = os.path.join(dirpath, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
