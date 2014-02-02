
from collections import defaultdict
import DataBaseClass
import logging

class SentenceFragmentData(DataBaseClass.DataBaseClass):
    '''
    Loads data split by sm code
    '''

    def __init__(self, loadSourceText = True):
        
        #Super call
        DataBaseClass.DataBaseClass.__init__(self, "Chicago")
        
        #LOAD SOURCE TEXT
        if loadSourceText:
            source_lines = self.__loadLines__("SourceText.txt")
            self.__process_documents_grouped_by_code__(source_lines)
        
        dataLines = self.__loadLines__("SentenceFragmentsToCodes.txt")
        dictLineToIndex = {}
        
        for line in dataLines:
            split = line.split("|")
            if len(split) != 2:
                raise Exception("Bad Data")
            
            code = split[1]
            line = split[0]
            self.__process_line__(dictLineToIndex, code, line)
        
        self.__compute_code_groupings__()
        logging.log(logging.INFO, "\tProcessed %i documents for %i codes", len(self.documents), len(self.allCodes))
  
"""
    if __name__ == "__main__":
        import SentenceFragmentData
        sd = SentenceFragmentData.SentenceFragmentData()
        print "Done"
        pass
"""