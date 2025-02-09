import os




def wildfire(folder_path):
    # List all the files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # Sort files to ensure they are renamed in a consistent order (optional)
    files.sort()

    # Loop through each file and rename it
    for i, filename in enumerate(files, 1):
        # Define the new filename
        new_name = f"wildfire.{i}.jpg"
        
        # Create the full file paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_file, new_file)

    print("Renaming completed.")

def nowildfire(folder_path):
    # List all the files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # Sort files to ensure they are renamed in a consistent order (optional)
    files.sort()

    # Loop through each file and rename it
    for i, filename in enumerate(files, 1):
        # Define the new filename
        new_name = f"nowildfire.{i}.jpg"
        
        # Create the full file paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_file, new_file)


# Update paths to your wildfire and nowildfire folders
wildfire_path_a = r"C:\Users\lufai\Downloads\archive\test\wildfire"
nowildfire_path_a = r"C:\Users\lufai\Downloads\archive\test\nowildfire"
wildfire_path_v = r"C:\Users\lufai\Downloads\archive\valid\wildfire"
nowildfire_path_v = r"C:\Users\lufai\Downloads\archive\valid\nowildfire"
wildfire_path_t = r"C:\Users\lufai\Downloads\archive\train\wildfire"
nowildfire_path_t = r"C:\Users\lufai\Downloads\archive\train\nowildfire"

# Rename images in both folders
wildfire(wildfire_path_a)
nowildfire(nowildfire_path_a)
wildfire(wildfire_path_v)
nowildfire(nowildfire_path_v)
wildfire(wildfire_path_t)
nowildfire(nowildfire_path_t)
