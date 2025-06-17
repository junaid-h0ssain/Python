import os
import shutil

def copy_images(source_folder, destination_folder):
    """
    Copies all image files from a source folder to a destination folder.

    Args:
        source_folder (str): The path to the folder from where images will be copied.
        destination_folder (str): The path to the folder where images will be copied.
    """
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Destination folder '{destination_folder}' created.")

    # image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_extensions = ('.mp4','.mkv','.webm')

    copied_count = 0
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(image_extensions):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            try:
                shutil.copy2(source_path, destination_path)
                print(f"Copied: '{filename}'")
                copied_count += 1
            except IOError as e:
                print(f"Error copying '{filename}': {e}")
            except Exception as e:
                print(f"An unexpected error occurred while copying '{filename}': {e}")

    print(f"\nFinished copying. Total images copied: {copied_count}")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace these paths with your actual source and destination folders
    source_directory = 'D:/EXIF_Correct'  # e.g., "C:/Users/YourUser/Pictures"
    destination_directory = "C:/Users/junu/Pictures/vids" # e.g., "C:/Users/YourUser/MyCopiedImages"
    # -------------------

    # --- Example Usage (Uncomment to run with example paths) ---
    # Create some dummy files for testing if the folders don't exist
    # if not os.path.exists("path/to/your/source/folder"):
    #     os.makedirs("path/to/your/source/folder")
    #     with open("path/to/your/source/folder/image1.jpg", "w") as f: f.write("dummy image content")
    #     with open("path/to/your/source/folder/document.txt", "w") as f: f.write("dummy text content")
    #     with open("path/to/your/source/folder/photo2.png", "w") as f: f.write("dummy image content")
    #     print("Created dummy source folder and files for testing.")

    copy_images(source_directory, destination_directory)