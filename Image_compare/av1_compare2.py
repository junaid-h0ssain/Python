import os
import shutil
import re

# --- ⚙️ CONFIGURATION ---
# IMPORTANT: Replace these paths with your actual folder paths.
# Use raw strings (r"...") on Windows to handle backslashes correctly.

# Folder containing your smaller, AV1 encoded videos.
AV1_FOLDER = r"D:\img_chk\mega-video"

# Folder containing your larger, non-encoded videos.
NON_ENCODED_FOLDER = r"D:\av1_vids"

# Folder where the non-encoded videos will be moved if the condition is met.
# This folder will be created if it doesn't exist.
DESTINATION_FOLDER = r"D:\img_chk\ffff"
# -------------------------


def format_size(size_bytes):
    """Converts a size in bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.2f} MB"
    return f"{size_bytes / 1024**3:.2f} GB"


def clean_filename(filename):
    """
    Normalizes a filename for comparison by:
    1. Converting to lowercase.
    2. Removing common prefixes like dates and times.
    3. Removing spaces, hyphens, and other non-alphanumeric characters.
    """
    # Use regular expression to find the "PXL..." part
    match = re.search(r'PXL_\d{8}_\d{9}', filename, re.IGNORECASE)
    if match:
        # If a match is found, return only the matched string in lowercase
        return match.group(0).lower().replace('_', '')
    
    # Fallback if the pattern is not found
    # This cleans the entire string by removing special characters and spaces
    clean = re.sub(r'[^a-zA-Z0-9]', '', filename).lower()
    return clean


def main():
    """
    Compares individual files between two folders based on filename.
    If an AV1 file is smaller than its non-encoded counterpart, the
    non-encoded file is moved to a destination folder.
    """
    print("--- Individual File Comparison & Mover ---")

    # Create destination folder if it doesn't exist
    os.makedirs(DESTINATION_FOLDER, exist_ok=True)
    
    # 1. Create a map of non-encoded files for quick lookup.
    print(f"Scanning non-encoded folder: {NON_ENCODED_FOLDER}")
    non_encoded_map = {}
    try:
        for filename in os.listdir(NON_ENCODED_FOLDER):
            full_path = os.path.join(NON_ENCODED_FOLDER, filename)
            if os.path.isfile(full_path):
                # Use the new cleaning function to get a consistent key
                key = clean_filename(filename)
                if key:
                    non_encoded_map[key] = full_path
    except FileNotFoundError:
        print(f"❌ Error: The folder '{NON_ENCODED_FOLDER}' was not found.")
        return
    
    print(f"Found {len(non_encoded_map)} non-encoded files to check against.\n")
    
    moved_count = 0
    skipped_count = 0
    no_match_count = 0

    # 2. Iterate through the AV1 folder and compare each file
    print(f"--- Starting Comparison ---")
    try:
        for av1_entry in os.scandir(AV1_FOLDER):
            if not av1_entry.is_file():
                continue

            # Clean the AV1 filename to create a comparable key
            av1_key = clean_filename(av1_entry.name)

            # Check if a corresponding file exists using the cleaned key
            if av1_key in non_encoded_map:
                non_encoded_path = non_encoded_map[av1_key]
                non_encoded_filename = os.path.basename(non_encoded_path)

                av1_size = av1_entry.stat().st_size
                non_encoded_size = os.path.getsize(non_encoded_path)

                print(f"Comparing '{av1_entry.name}' ({format_size(av1_size)}) vs '{non_encoded_filename}' ({format_size(non_encoded_size)})")

                # 3. Compare sizes and move if condition is met
                if av1_size < non_encoded_size:
                    print(f"  ✅ AV1 is smaller. Moving '{non_encoded_filename}'...")
                    shutil.move(non_encoded_path, DESTINATION_FOLDER)
                    moved_count += 1
                else:
                    print(f"  ❌ AV1 is NOT smaller. Skipping.")
                    skipped_count += 1
            else:
                # 4. Handle cases where no match is found
                print(f"  ⚠️ No match found for '{av1_entry.name}' in the non-encoded folder.")
                no_match_count += 1
            
            print("-" * 25) # Separator for clarity

    except FileNotFoundError:
        print(f"❌ Error: The AV1 folder '{AV1_FOLDER}' was not found.")
        return

    # 5. Final summary
    print("\n--- 📊 Summary ---")
    print(f"🚚 Files Moved: {moved_count}")
    print(f"⏭️ Files Skipped (AV1 not smaller): {skipped_count}")
    print(f"❓ AV1 Files with No Match: {no_match_count}")
    print("\n🎉 Process complete.")


if __name__ == "__main__":
    main()
