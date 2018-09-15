__author__ = 'simon.hughes'

import os, sys

def get_source_folders(src_root = None):
    """ Given a root folder, will build the python path by searching
        for all python files recursively in sub-directories
    """
    def valid_dir(dirpath):
        # ensure path in linux format (i.e. with /'s, also works on windows)
        parts = dirpath.replace("\\", "/").split("/")
        return not parts[-1].startswith(".") and not parts[-1].startswith("_")

    def is_python_file(fname):
        return fname.endswith(".py") or fname.endswith(".pyc") or fname.endswith(".pyo")

    if src_root is None:
        src_root = os.getcwd()

    folders = set()
    print("Searching:")
    for (fullpath, directories, files) in os.walk(src_root):
        python_files = list(filter(is_python_file, files))
        if len(python_files) > 0 and valid_dir(fullpath):
            folders.add(fullpath)
    return sorted(folders)

def update_path(folders):
    current = set(sys.path)
    for folder in folders:
        if folder not in current:
            sys.path.append(folder)

def import_python_files_from_path(src_root):
    src_folders = get_source_folders(src_root)
    print("Found source folders\n:{0}\n\nAdding to path".format(str(src_folders)))
    update_path(src_folders)

if __name__ == "__main__":

    #look up user
    user = os.environ["USER"]

    """ Writes a path.txt file to the profile dir. Copy into the ~/bash_profile """
    path = ("/Users/%s/GitHub/NlpResearch/PythonNlpResearch" % user)

    folders = get_source_folders(path)
    py_path = "'" + os.pathsep.join(folders) + "'"

    print("Python path:\n", py_path)
    # To write the path:
    # fname = path + "/launch_notebook.sh"
    # with open(fname, "w+") as f:
    #     f.write("export PYTHONPATH=" + py_path + "\n")
    #     f.write("cd Notebooks\n")
    #     f.write("ipython notebook --pylab inline\n")
    #
    # print "len:", len(py_path),"\n", "\n".join(folders)
