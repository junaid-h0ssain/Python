import os
from PIL import Image
import imagehash
from collections import defaultdict
import shutil

FOLDERS = ["D:/EXIF_Correct","D:/Collection"]
OUTPUT_DUPLICATES = ''
os.makedirs(OUTPUT_DUPLICATES, exist_ok=True)

hashes = defaultdict(list)

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

# Move duplicates safely
for h, paths in hashes.items():
    if len(paths) > 1:
        for dup_path in paths[1:]:
            filename = os.path.basename(dup_path)
            name, ext = os.path.splitext(filename)
            target_path = os.path.join(OUTPUT_DUPLICATES, filename)

            # Avoid overwriting by adding a counter to filename
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(OUTPUT_DUPLICATES, f"{name}_{counter}{ext}")
                counter += 1

            shutil.move(dup_path, target_path)
            print(f"Moved duplicate: {dup_path} -> {target_path}")
