from collections import defaultdict
from DataBaseClass import DataBaseClass
import logging
import Settings
from IterableFP import compact

def is_causal(code):
    if code.replace(" ", "").find("-") > -1:
        return 1.0
    return 0.0

class CbData(DataBaseClass):
    """ Class that is equivalent to the GwData and classifies the data at the sentence level """
    def __init__(self, directory = None, include_vague = True):
        if directory == None:
            s = Settings.Settings()
            directory = s.data_directory + "CoralBleaching\\"

        self.directory = directory
        logging.log(logging.INFO, "GWData: Processing Data from directory \n\t'%s'", directory)

        #Properties
        # list of all docs
        self.documents = []
        # list of codes for each doc
        # list of sets, size is the same as number of docs
        self.codes_per_document = []

        #CODES

        self.count_per_sm_code = defaultdict(int)

        #INDICES FOR CODE TYPE (over all codes for that type, not by code)
        self.sm_code_indices = []
        self.sm_codes = set()
        #self.text_code_indices = []

        self.causal_per_document = []

        #LOAD SOURCE TEXT
        dataLines = self.__loadLines__("CbDatabase.csv")
        dictLineToIndex = {}

        to_replace = ["-", ";", ",", "  "]

        for line in dataLines[1:]: # skip header
            splt = line.split("|")
            if len(splt) == 4:
                essay, sentence_num, sentence, codes_str = splt
                compiled = ""
            elif len(splt) == 5:
                essay, sentence_num, sentence, codes_str, compiled = splt
            else:
                raise Exception("Bad Data")

            for repl_str in to_replace:
                codes_str = codes_str.replace(repl_str, " ")

            codes = self.__collapse_codes__(compact(codes_str.split(" ")))
            self.__process_line__(dictLineToIndex, codes, codes_str, sentence)

        self.allCodes = self.sm_codes.copy()
        logging.log(logging.INFO, "\tProcessed %i documents for %i codes", len(self.documents), len(self.allCodes))

    def __process_line__(self, dictLineToIndex, codes, rawCodes, line):
        if len(codes) == 0:
            return

        codes = set(codes)
        for code in codes:
            self.count_per_sm_code[code] = self.count_per_sm_code[code] + 1

        self.sm_codes = self.sm_codes.union(codes)
        #Line already processed once then add SM code to existing
        if line in dictLineToIndex:
            # This does appear to work (i.e. no duplicates left over,
            # at least before stemming and stop word removal)
            lineNo = dictLineToIndex[line]
            self.codes_per_document[lineNo].union(set(codes))
            if is_causal(rawCodes):
                self.causal_per_document[lineNo] = 1
        else:
            doc_index = len(self.documents)
            # create hash map of doc to index
            dictLineToIndex[line] = doc_index
            # add line
            self.documents.append(line)
            # initialize set of codes for line
            self.codes_per_document.append(set(codes)) # Need to pass a list to set otherwise see str as char array
            self.sm_code_indices.append(doc_index)
            self.causal_per_document.append(is_causal(rawCodes))

    def __collapse_codes__(self, codes):
        def remove_vague(code):
            code = code.strip().lower()
            if len(code) > 1 and code[0] == "v":
                return code[1:]
            return code
        return set(map(remove_vague, codes))


if __name__ == "__main__":

    data = CbData()