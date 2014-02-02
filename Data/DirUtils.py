
def list_files(dir):
    from os import walk

    for (dirpath, dirnames, filenames) in walk(dir):
        return filenames
        
    return None