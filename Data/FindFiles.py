import os
import re

def find_files(folder, regex=".*", remove_empty = False):
    """
    Find all files matching the [regex] pattern in [folder]

    folder  :   string
                    folder to search (not recursive)
    regex   :   string (NOT regex object)
                    pattern to match
    """
    files = os.listdir(folder)
    matches = [os.path.abspath(os.path.join(folder, f))
               for f in files
               if re.search(regex, f, re.IGNORECASE)]

    if remove_empty:
        matches = [f for f in matches if os.path.getsize(f) > 0]
    matches.sort()
    return matches


def find_files_recursively(folder, fname_filter=None):
    """
    Find all files matching the filename in [folder]

    folder  :   string
                    folder to search recursive
    fname_filter : string
                    filename to match (if provided)
    """
    files = []
    for root, directories, filenames in os.walk(folder):
        for filename in filenames:
            if fname_filter is None or filename == fname_filter:
                files.append(os.path.join(root,filename))
    return sorted(set(files))