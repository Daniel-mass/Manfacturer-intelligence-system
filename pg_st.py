import os

def print_tree(start_path, prefix="", hidden_dirs=None):
    if hidden_dirs is None:
        hidden_dirs = [".git", "__pycache__", "mlruns", "qdrant_data", "venv" , "octotools"]
    
    # Get files and folders, filtering out hidden ones
    files = [f for f in os.listdir(start_path) if f not in hidden_dirs]
    files.sort()

    for index, file in enumerate(files):
        path = os.path.join(start_path, file)
        
        # Skip artifact files for a clean README look
        if file.endswith(('.pyc', '.png', '.html', '.db','.jpg', '.jpeg')):
            continue

        connector = "├── " if index < len(files) - 1 else "└── "
        print(prefix + connector + file)

        if os.path.isdir(path):
            extension = "│   " if index < len(files) - 1 else "    "
            print_tree(path, prefix + extension, hidden_dirs)

if __name__ == "__main__":
    print("📂 Clean Project Structure:\n")
    print_tree(".")