class Settings(object):
    """ Bag of settings for the project """

    def __init__(self):
        import os
        
        cwd = os.getcwd()
        
        # Assumes you are running from the experiments folder or some other
        # doubly nested folder

        path_delim = "/"

        parts = cwd.split(path_delim)
        ix_root = parts.index("NlpResearch")
        root_path = path_delim.join( parts[: ix_root + 1] ) + path_delim

        self.results_directory = root_path + "Results" + path_delim
        self.data_directory    = root_path + "Data" + path_delim
        self.root_path = root_path

        print "Results Dir: "   + self.results_directory
        print "Data Dir:    "   + self.data_directory
    
if __name__ == "__main__":
    stg = Settings()
    
    i = raw_input("Press any key...")