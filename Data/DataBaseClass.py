# Simply loads the data from the files, storing the codes
# and the sentences for each code

from collections import defaultdict
import CodeToCategory
import CodeToDocument
import logging

class DataBaseClass(object):

    def __init__(self, directory = None):
        
        baseDir = "C:\Users\Simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\src\Data"
        
        if directory == None:
            directory = ""
        
        if not directory.startswith("\\"):
            directory = "\\" + directory
        
        if not directory.endswith("\\"):
            directory += "\\"
        
        self.directory = baseDir + directory
        
        logging.log(logging.INFO, "GWData: Processing Data from directory \n\t'%s'", directory)

        #Properties
        # list of all docs
        self.documents = []
        # list of codes for each doc
        # list of sets, size is the same as number of docs
        self.codes_per_document = []
        
        #CODES
        self.sm_codes   = sorted(self.__loadLines__("SMCodes.txt"))
        self.text_codes   = sorted(self.__loadLines__("SourceTextCodes.txt"))        
        
        self.allCodes = set(self.sm_codes + self.text_codes)

        self.count_per_sm_code = defaultdict(int)

        #INDICES FOR CODE TYPE (over all codes for that type, not by code)
        self.sm_code_indices = []
        self.text_code_indices = []            
       
    def __loadLines__(self, fName):
        """ Loads lines from a file and returns as a list 
            file => []
        """
        handle = open(self.directory + fName, "r+")
        logging.log(logging.INFO, "\tReading: %s" , handle.name)

        lines = []
        for line in handle.readlines():
            #print".",
            stripped = line.strip()
            if len(stripped) > 0 and not stripped.startswith("--"):
                lines.append(stripped)
        handle.close()
        return lines   

    def __process_line__(self, dictLineToIndex, code, line, isSmCode = True):
        self.count_per_sm_code[code] = self.count_per_sm_code[code] + 1
        #Line already processed once then add SM code to existing
        if line in dictLineToIndex:
            # This does appear to work (i.e. no duplicates left over,
            # at least before stemming and stop word removal)
            lineNo = dictLineToIndex[line]
            self.codes_per_document[lineNo].add(code)
        else:
            doc_index = len(self.documents)
            # create hash map of doc to index
            dictLineToIndex[line] = doc_index
            # add line
            self.documents.append(line)
            # initialize set of codes for line
            self.codes_per_document.append(set([code])) # Need to pass a list to set otherwise see str as char array
            if isSmCode:
                self.sm_code_indices.append(doc_index)
            else:
                self.text_code_indices.append(doc_index) # get doc index by taking the len before we add it

    def __process_documents_grouped_by_code__(self, rawLines):
        dictLineToIndex = {}
        code = ""
        isSmCode = True
        for line in rawLines:
            if line in self.allCodes:
                #On an SM code line
                code = line.strip()
                isSmCode = code in self.sm_codes
            else:
                self.__process_line__(dictLineToIndex, code, line, isSmCode)

    def __compute_code_groupings__(self):
        #Compute the sm code categories for each doc
        self.categories_per_document = self.__map_codes_per_doc__(CodeToCategory.CodeToCategory())
        #Compute the source text for each doc based on the sm codes
        self.codes_per_document = self.__map_codes_per_doc__(CodeToDocument.CodeToDocument())


    def __map_codes_per_doc__(self, mapper):
        return [set(
                    [mapper.Map(code) 
                     for code in codes]
                     )
                for codes in self.codes_per_document]


    def labels_for(self, smcode):
        def matches_code(codes):
            #TODO return True or False instead?
            if smcode in codes:
                return 1
            return 0

        return [matches_code(codes) for codes in self.codes_per_document]
    
    def indices_for(self, smcode):
        """
        Returns the indexes into self.documents for
        docs that match the code
        """
        def matches_code(codes):
            if smcode in codes:
                return True
            return False

        return [i 
                for i, code in enumerate(self.codes_per_document) 
                    if matches_code(code) ]
   
    #Helper functions below (to assist data discovery, not used to compute right now)
    def sentences_for_code(self, code):
        """ Returns a list of sentences for a given code
        """
        sentences = []
        for i, sentence in enumerate(self.documents):
            if code in self.codes_per_document[i]:
                sentences.append(sentence)
        return sentences
    
    def sentences_not_for_code(self, code):
        """ Returns a list of sentences for a given code
        """
        sentences = []
        for i, sentence in enumerate(self.documents):
            if not code in self.codes_per_document[i]:
                sentences.append(sentence)
        return sentences
    
    def print_counts_per_code(self):
        """ Prints out an ordered list of counts per code 
            This data can be retrieved directly from counPerSmCode
        """ 
        total = 0
        keys = sorted(self.count_per_sm_code.keys())
        for k in keys:
            total = total + self.count_per_sm_code[k]
            print k.ljust(10) + str(self.count_per_sm_code[k])
        
        # Note that there are 3301  unique sentences
        # But on average ~2.3 codes per doc 
        print "\nTotal".ljust(10) + str(total)
        print "\n\nDistinct Docs " + str(len(self.documents))
    
    def print_sentences_for_code(self, code):
        """ For a given sm code, print the sentences
        """
        sentences = self.sentences_for_code(code)
        for s in sentences:
            print s
