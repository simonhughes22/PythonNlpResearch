# Simply loads the data from the files, storing the codes
# and the sentences for each code
import logging
import DataBaseClass

class SentenceData(DataBaseClass.DataBaseClass):
    """ Loads the Dig Lit Data set
        >>  Populates the: 
            1. self.documents list
            2. self.codesForDocument list to set
    """
    
    def __init__(self, loadSourceText = True):
     
        #Super call
        DataBaseClass.DataBaseClass.__init__(self, "Chicago")
        
        #LOAD SENTENCES
        dataLines = self.__loadLines__("StudentSentences.txt")

        if loadSourceText:
            sourceLines = self.__loadLines__("SourceText.txt")
            rawLines = sourceLines + dataLines
            logging.log(logging.INFO, "\tIncluding source text")
        else:
            rawLines = dataLines

        # LOCALS
        # maps each document to its index in self.documents
        self.__process_documents_grouped_by_code__(rawLines)

        self.__compute_code_groupings__()
        
        logging.log(logging.INFO, "\tProcessed %i documents for %i codes", len(self.documents), len(self.allCodes))

"""
    if __name__ == "__main__":
        import SentenceData
        sd = SentenceData.SentenceData()
        print "Done"
        pass
"""