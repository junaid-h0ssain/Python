import os
import shutil

def compare_and_copy(folder1, folder2, review_folder):
    # Create the review folder if it doesn't exist
    os.makedirs(review_folder, exist_ok=True)

    # List of files in both folders
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Files in folder1 but not in folder2
    unique_files = files1 - files2

    # Copy those files to the review folder
    for file_name in unique_files:
        source_path = os.path.join(folder1, file_name)
        dest_path = os.path.join(review_folder, file_name)

        # Only copy if it's a file
        if os.path.isfile(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"Copied: {file_name}")
        else:
            print(f"Skipped (not a file): {file_name}")

# Example usage
folder2 = "D:/img_chk/New folder"
folder1 = "D:/rrr"
review_folder = "D:/img_chk"

compare_and_copy(folder1, folder2, review_folder)

