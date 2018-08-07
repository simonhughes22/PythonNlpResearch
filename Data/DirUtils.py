import os

def list_files(dir):
    from os import walk

    for (dirpath, dirnames, filenames) in walk(dir):
        return filenames
        
    return None

def dir_exists(folder):
    return (os.path.isdir(folder)) and (os.path.exists(folder))