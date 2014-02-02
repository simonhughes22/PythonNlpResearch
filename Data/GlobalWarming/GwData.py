# Simply loads the data from the files, storing the codesadd
# and the sentences for each code

from Labeler import *
from collections import defaultdict
import DataBaseClass
import logging
import Settings

CAUSAL = "Causal"
FOLDER = "GlobalWarming/Files"

class GwData(DataBaseClass.DataBaseClass):

    def __init__(self, directory = None, load_essays = True, load_source = True):

        if not load_essays and not load_source:
            raise Exception("You need to load either the source docs or the essays, or both")
        
        if directory == None:
            s = Settings.Settings()
            directory = s.data_directory + FOLDER + "/"

        if not directory.endswith("/"):
            directory += "/"

        self.directory = directory
        logging.log(logging.INFO, "GwDataClass: Processing Data from directory \n\t'%s'", directory)

        #Properties
        # list of all docs
        self.documents = []
        # list of codes for each doc
        # list of sets, size is the same as number of docs
        self.codes_per_document = []

        #CODES
        self.count_per_sm_code = defaultdict(int)

        self.raw_codes_per_document = []
        self.sm_codes = set()
        self.allCodes = set()
        self.causal_per_document = []

        #INDICES FOR CODE TYPE (over all codes for that type, not by code)
        self.sm_code_indices = []
        self.other_codes = set(['m', 'n', 'oth', 'other', 'n_co2', 'n_sun', 'nd', '51', '52'])

        if load_essays:

            self.__loadLines__("All_GW_Data.txt")
            #self.__loadLines__("CCM_DocSet_Coded2.txt")

            # >>> This is a duplicate of CCM_DocSet_Coded2.txt:
            #self.__loadLines__("Gem_Set2_1_ACM.txt")

            #self.__loadLines__("CM_essays_scoredF.txt")
            #self.__loadLines__("Gemini3Final.txt")
       
        if load_source:
            self.__loadLines__("CM_DocSet_Coded_1.0.txt", True)
       
        self.sm_codes = sorted(list(self.sm_codes))
        self.allCodes = sorted(self.sm_codes + [CAUSAL])

        d = defaultdict(int)
        for codes in self.codes_per_document:
            for c in codes:
                d[c] += 1

        self.sm_code_count = d
        self.sm_codes_for_classification = [c for c in self.sm_codes 
			if self.sm_code_count[c] > 1 
			and	c != "v"
			and not c.startswith("s")]

    def valid_code(self, code):
        code = code.strip()
        if code.startswith("(") and code.endswith(")"):
            if(any([x.isdigit() for x in code])):
                return True
            return False
        return True

    def remove_prefixes(self, code):

        # Note these are the 'vague' codes
        # Tried to remove these but classification
        # performance drops off without them
        # as there are fewer examples

        if code.startswith("pv") and len(code) > 2:
            code = "p" + code[2:]

        if len(code) > 1 and code.startswith("0"):
            code = code[1:]

        if len(code) > 1 and code.startswith("v"):
            rest = code[1:]
            if rest[0] == 'p':
                return rest

            if(len(rest) > 0 and all(map( lambda c: c.isdigit() or c == '.', rest))):
                return rest

            return code

        return code

    def collapse_periods(self, code):
        return to_parent_code(code)

    def collapse_other_codes(self, code):

        if code in self.other_codes:
            return 'oth'
        return code

    def fix_deprecated_codes(self, code):
#        if code in ["p15", "p31"]:
#           return "bck"
        return code

    def __process_codes__(self, rc):
        
        rawCodes = rc[:]
        rawCodes = rawCodes.lower()
        rawCodes = rawCodes.replace("summ:"," ").replace("sum:"," ").replace("\"", "").replace("cou"," ").replace("so"," ").replace("ozone"," ").replace("thin"," ")
        rawCodes = rawCodes.replace("->",",").replace(">"," ").replace("err","").replace("er","").replace(":","")
        rawCodes = rawCodes.replace("(", " (").replace("+", " ").replace("-", " ").replace("="," ").replace("err"," ").replace("p", " p").replace("v"," v")
        
        while rawCodes.count("  ") > 0:
            rawCodes = rawCodes.replace("  ", " ")

        rawCodes = rawCodes.replace(" ",",")

        split = rawCodes.split(",")
        split = [ s.replace("(","").replace(")","").strip()
                  for s in split
                  if self.valid_code(s)]

        split = [ self.collapse_other_codes( self.remove_prefixes(s) ) 
                  for s in split if len(s) > 0]

        # Can't map as an instance method
        split = [ self.collapse_periods(s) for s in split ]
        
        test = [c for c in split if len(c) > 2 and len(c) % 2 == 0 and all([chr.isdigit() for chr in c])]
        if len(test) > 0:
            print rc + " -> ",
            print test
        return split

    def __loadLines__(self, fName, isSource = False):
        """ Loads lines from a file
        """
        handle = open(self.directory + fName, "r")
        logging.log(logging.INFO, "\tReading: %s" , handle.name)

        lines = []
        fileLines = handle.readlines()

        LINE_INDEX = 3
        CONCEPT_CODE_INDEX = 4
        ONLY_CODES_INDEX = 9

        if isSource:
            LINE_INDEX -= 1
            CONCEPT_CODE_INDEX -= 1
            ONLY_CODES_INDEX -= 1

        dictLineToIndex = {}

        #debug
        #l_raw, l_codes = [] , []
        i = -1
        for line in fileLines[1:]: #Skip Header Line

            i += 1
            stripped = line.replace("/","").replace("\"","").strip()
            if len(stripped) > 0 and not stripped.startswith("--"):
                stripped = stripped.replace("\"","").replace("\x92","").replace("\x93","").replace("\x94","").replace("\xa0","").replace("-"," ")
                split = stripped.split("\t")
                if len(split) <= CONCEPT_CODE_INDEX:
                    #print "Skipping line: '" + line.strip().ljust(130) + "' as unable to parse"
                    continue

                line = split[LINE_INDEX].strip()
                
                if len(split) > ONLY_CODES_INDEX:
                    rawCodes = split[ONLY_CODES_INDEX].strip()
                    if rawCodes == "":
                        rawCodes = split[CONCEPT_CODE_INDEX]
                else:
                    rawCodes = split[CONCEPT_CODE_INDEX]
                
                splitCodes = self.__process_codes__(rawCodes)

                self.__process_line__(dictLineToIndex, splitCodes, rawCodes, line)

                #DEBUG
                #l_raw.append(rawCodes)
                #l_codes.append(splitCodes)

        handle.close()

        #z = zip(l_raw, l_codes)
        #for a,b in z:
        #    print a, b

    def __process_line__(self, dictLineToIndex, codes, complied_codes, line):

        if len(codes) == 0:
            return

        is_causal_code = is_causal(complied_codes)

        codes = set(codes)
        self.sm_codes = self.sm_codes.union(codes)

        # Don't include CAUSAL in sm_codes
        if is_causal_code:
            codes.add(CAUSAL)

        for code in codes:
            self.count_per_sm_code[code] = self.count_per_sm_code[code] + 1


        #Line already processed once then add SM code to existing
        if line in dictLineToIndex:
            # This does appear to work (i.e. no duplicates left over,
            # at least before stemming and stop word removal)
            lineNo = dictLineToIndex[line]
            self.codes_per_document[lineNo] = self.codes_per_document[lineNo].union(codes)
            if is_causal_code:
                self.causal_per_document[lineNo] = 1.0
        else:
            doc_index = len(self.documents)
            # create hash map of doc to index
            dictLineToIndex[line] = doc_index
            # add line
            self.documents.append(line)
            # initialize set of codes for line
            self.codes_per_document.append(codes) # Need to pass a list to set otherwise see str as char array
            self.sm_code_indices.append(doc_index)
            self.causal_per_document.append(is_causal_code)

    @classmethod
    def as_binary(cls):
        import WordTokenizer
        import TermFrequency
        import MatrixHelper
        import Converter
    
        xs = GwData()
        tokenized_docs = WordTokenizer.tokenize(xs.documents, min_word_count=5)
        tf = TermFrequency.TermFrequency(tokenized_docs)
    
        ts = MatrixHelper.gensim_to_numpy_array(tf.distance_matrix, None, 0, Converter.to_binary)
        return ts 
        
if __name__ == "__main__":
    #import GwData

    d = GwData()

    print d.sm_codes
    print "Done"
    inp = raw_input("Done. Press any key to continue")
