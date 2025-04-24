import os
import shutil

def move_conceptx_a_files(root_dir="/cluster/home/kamara/conceptx/results"):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "conceptx":
                conceptx_path = os.path.join(dirpath, dirname)
                conceptx_a_path = os.path.join(dirpath, "conceptx-r")

                # Create the new directory if it doesn't exist
                os.makedirs(conceptx_a_path, exist_ok=True)

                conceptx_seed_path =  os.path.join(conceptx_path, "seed_0")
                conceptx_a_seed_path =  os.path.join(conceptx_a_path, "seed_0")
                os.makedirs(conceptx_a_seed_path, exist_ok=True)
                # Move matching files
                for filename in os.listdir(conceptx_seed_path):
                    if "conceptx-r" in filename:
                        src = os.path.join(conceptx_seed_path, filename)
                        dst = os.path.join(conceptx_a_seed_path, filename)
                        shutil.move(src, dst)
                        print(f"Moved: {src} -> {dst}")

# Run the function
move_conceptx_a_files()