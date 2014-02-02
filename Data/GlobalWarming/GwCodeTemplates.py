'''
Created on Apr 11, 2013
@author: Simon
'''
import WordTokenizer
import Settings
from Labeler import *

class GwCodeTemplates(object):
    '''
    Creates an object with .codes and .templates properties  
    '''
    def __init__(self, directory = None):
        '''
        Constructor
        '''
        if directory == None:
            directory = Settings.Settings().data_directory + "\GlobalWarming"

        if not directory.endswith("\\"):
            directory += "\\"
        self.directory = directory
        
        raw_lines = self.__loadLines__("CodeTemplates.txt")
        lines = []
        
        self.codes_per_document = []
        
        for l in raw_lines:
            ltrim = l.strip()
            if len(ltrim) == 0:
                continue
            split = ltrim.split("|")
            if len(split) != 2:
                raise Exception("Bad Template Data")
            
            code = to_parent_code( split[0] )
            self.codes_per_document.append(set([code]))
            lines.append(split[1].strip().lower())
        
        self.documents = WordTokenizer.tokenize(lines, min_word_count = 0)
        
    def __loadLines__(self, fName):
        """ Loads lines from a file and returns as a list 
            file => []
        """
        handle = open(self.directory + fName, "r")

        lines = []
        for line in handle.readlines():
            #print".",
            stripped = line.strip()
            if len(stripped) > 0 and not stripped.startswith("--"):
                lines.append(stripped)
        handle.close()
        return lines   

if __name__ == "__main__":
    
    dbnetwork = GwCodeTemplates()
    
    z = zip(dbnetwork.codes_per_document, dbnetwork.documents)
    match = filter(lambda tpl: "50" in tpl[0], z)

    i = raw_input("Any key..")
    