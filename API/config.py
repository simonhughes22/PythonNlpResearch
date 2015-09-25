
import ConfigParser, os

class Config(object):

    def __init__(self, config_file):

        self.__verify_config_file__(config_file)
        self.__cfg__            = self.__load_config_file__(config_file)

        self.models_folder = self.__getfilename__("DEFAULT", "models_folder")
        self.temp_folder   = self.__getfilename__("DEFAULT", "temp_folder")
        self.essays_folder = self.__getfilename__("DEFAULT", "essays_folder")

    def __load_config_file__(self, config_file):
        config = ConfigParser.ConfigParser()
        config.read([config_file])
        return config

    def __getfilename__(self, section, key):
        fname = self.__getstring__(section, key)
        assert os.path.exists(fname), "File\Folder: %s does not exist" % fname
        return fname

    def __getstring__(self, section, key):
        value = self.__cfg__.get(section, key)
        assert (value != None and value.strip() != "")
        return value

    def __getint__(self, section, key):
        value = self.__cfg__.getint(section, key)
        return value

    def __getbool__(self, section, key):
        value = self.__cfg__.getboolean(section, key)
        return value

    def __getfloat__(self, section, key):
        value = self.__cfg__.getfloat(section, key)
        return value

    def __verify_config_file__(self, config_file):
        assert config_file.endswith(".cfg"), "Valid configuration file expected"
        assert os.path.isfile(config_file), "[{0}] is not a valid file path".format(config_file)

if __name__ == "__main__":

    f = "/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/API/Settings/simon.cfg"
    cfg = Config(f)
    pass
