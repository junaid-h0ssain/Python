import os
from PIL import Image
import imagehash
from collections import defaultdict
import shutil

# Folders to compare
FOLDERS = ["D:/EXIF_Correct", "D:/Collection"]
PREFERRED_FOLDER = os.path.abspath("D:/EXIF_Correct")
OUTPUT_DUPLICATES = "D:/Duplicates"

os.makedirs(OUTPUT_DUPLICATES, exist_ok=True)

hashes = defaultdict(list)

# Step 1: Compute image hashes
for folder in FOLDERS:
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')):
                try:
                    path = os.path.join(root, file)
                    with Image.open(path) as img:
                        h = imagehash.average_hash(img)
                    hashes[str(h)].append(path)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

# Step 2: Process duplicates
for h, paths in hashes.items():
    if len(paths) > 1:
        # Find preferred file if it exists
        preferred = next((p for p in paths if p.startswith(PREFERRED_FOLDER)), paths[0])

        for dup_path in paths:
            if dup_path != preferred:
                filename = os.path.basename(dup_path)
                name, ext = os.path.splitext(filename)
                target_path = os.path.join(OUTPUT_DUPLICATES, filename)

                # Avoid overwriting
                counter = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(OUTPUT_DUPLICATES, f"{name}_{counter}{ext}")
                    counter += 1

                # Move or copy (use copy2 to test safely)
                shutil.move(dup_path, target_path)
                print(f"Moved duplicate: {dup_path} -> {target_path}")
