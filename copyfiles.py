import os
import shutil

def copy_largest_files(source_folder, destination_folder, num_files_to_copy):
    for root, dirs, files in os.walk(source_folder):
        if root != source_folder:
            subfolder_name = os.path.basename(root)
            subfolder_destination = os.path.join(destination_folder, subfolder_name)
            os.makedirs(subfolder_destination, exist_ok=True)
            
            # Sort files by size (largest to smallest)
            files.sort(key=lambda f: os.path.getsize(os.path.join(root, f)), reverse=True)
            
            # Copy the five largest files
            for file in files[:num_files_to_copy]:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(subfolder_destination, file)
                shutil.copy(source_path, destination_path)
                print(f'Copied {file} to {subfolder_name}')

# Define the main folder
main_folder = 'train'

# Define the destination folder where you want to copy the files
destination_folder = 'smalltest'

# Define the number of largest files you want to copy from each subfolder
num_files_to_copy = 2

# Call the function
copy_largest_files(main_folder, destination_folder, num_files_to_copy)
