import os
import re

def find_files(folder, regex, remove_empty = False):
    """
    Find all files matching the [regex] pattern in [folder]

    folder  :   string
                    folder to search (no recursive)
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
