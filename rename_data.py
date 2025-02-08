import os

def rename_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if "," in filename:
                new_filename = filename.replace(",", "_")  # Replace commas with underscores
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

# Update paths to your wildfire and nowildfire folders
wildfire_path = r"C:\Users\lufai\Downloads\archive\valid\wildfire"
nowildfire_path = r"C:\Users\lufai\Downloads\archive\valid\nowildfire"

# Rename images in both folders
rename_images(wildfire_path)
rename_images(nowildfire_path)
