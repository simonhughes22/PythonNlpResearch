import os

class Settings(object):
    """ Bag of settings for the project """

    def __init__(self):

        cwd = os.getcwd()

        # Assumes you are running from the experiments folder or some other
        # doubly nested folder

        linux_delim = "/"
        path_delim = linux_delim

        if "\\" in cwd:
            path_delim = "\\"

        parts = cwd.split(path_delim)
        ix_root = parts.index("NlpResearch")
        # Ensure is in linux form, which works for both OS'
        root_path = linux_delim.join( parts[: ix_root + 1] ) + linux_delim
        self.root_path = root_path
        self.public_data_dir = self.root_path + "Data/PublicDatasets/"

        self.results_directory = "/Users/simon.hughes/Dropbox/Phd/Results/"
        self.data_directory    = "/Users/simon.hughes/Dropbox/Phd/Data/"

        print "Results Dir: "   + self.results_directory
        print "Data Dir:    "   + self.data_directory
        print "Root Dir:    "   + self.root_path
        print "Public Data: "   + self.public_data_dir

if __name__ == "__main__":

    stg = Settings()

    assert os.path.exists(stg.data_directory),      "Data Directory does not exist"
    assert os.path.exists(stg.results_directory),   "Results Directory does not exist"
    assert os.path.exists(stg.root_path),           "Root Path does not exist"
    assert os.path.exists(stg.public_data_dir),     "Public Datasets does not exist"

    i = raw_input("Press any key...")