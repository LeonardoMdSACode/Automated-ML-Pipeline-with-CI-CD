import os

root_path = os.getcwd()
# Define the directories to ignore
skip_dirs = {'__pycache__', '.pytest_cache', '.venv', '.git'}

print(f"Starting search in CWD: {root_path}\n")

for folderName, subfolders, filenames in os.walk(root_path):
    # Modify subfolders in-place to skip specific directories
    # This prevents os.walk from even looking inside them
    subfolders[:] = [d for d in subfolders if d not in skip_dirs]

    print('The current folder is ' + folderName)
    
    for subfolder in subfolders:
        print('SUBFOLDER OF ' + folderName + ': ' + subfolder)
        
    for filename in filenames:
        print('FILE INSIDE ' + folderName + ': ' + filename)
    
    print('')
