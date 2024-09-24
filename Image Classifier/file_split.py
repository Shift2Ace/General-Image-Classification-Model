import os
import shutil
import random

def move_files(src_folder, dest_folder, percentage):
    new_folder_path = os.path.join(os.path.dirname(src_folder),dest_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    for subdir in os.listdir(src_folder):
        subdir_path = os.path.join(src_folder, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            random.shuffle(files)
            num_files_to_move = int(len(files) * percentage / 100)
            files_to_move = files[:num_files_to_move]

            new_subdir_path = os.path.join(new_folder_path, subdir)
            if not os.path.exists(new_subdir_path):
                os.makedirs(new_subdir_path)

            for file in files_to_move:
                shutil.move(os.path.join(subdir_path, file), os.path.join(new_subdir_path, file))

# Input parameters
src_folder = input("Source folder (train): ")  # Input folder path
dest_folder = input("New folder (valid/test): ")      # New folder name
percentage = int(input("Percentage (10): "))                # Percentage of files to move

# Move files
move_files(src_folder, dest_folder, percentage)
