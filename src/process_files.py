import os

# Path to the main folder
root_folder = '/cluster/home/kamara/conceptx/results/sentiment-antonym'

# Walk through all subfolders and files
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if 'classification' in filename:
            old_path = os.path.join(dirpath, filename)
            new_filename = filename.replace('classification', 'sentiment')
            new_path = os.path.join(dirpath, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} -> {new_path}')