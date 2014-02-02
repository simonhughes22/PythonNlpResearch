import Lsa
import TfIdf
import WordTokenizer
import logging
import Settings
import nltk
import os.path

class GwLargeCorpus(object):
    '''
    classdocs
    '''
    def __init__(self, 
                 directory = None,
                 min_sentence_length = 3, tokenize = True):

        if directory == None:
            s = Settings.Settings()
            directory = s.data_directory + "GlobalWarming\\"

        if not directory.endswith("\\"):
            directory += "\\"

        self.directory = directory
        logging.log(logging.INFO, "GwLargeCorpus: Processing Data from directory \n\t'%s'", directory)

        lines = self.__loadLines__("globalwarming_specific_space11.txt")
        lines.append("")

        sentences = []
        current = ""
        for line in lines:
            current += line
            if(len(line.strip()) == 0 and len(current.strip()) > 0):
                sent = nltk.sent_tokenize(current.strip().lower())

                sentences.extend(sent)
                current = ""
        self.documents = sentences

        if tokenize:
            wt = WordTokenizer.WordTokenizer(min_word_count = 3)
            tokenized = wt.tokenize(sentences)

            tokenized_documents = []
            full_sentences = []
            for i in range(len(tokenized)):
                tokenized_docs = tokenized[i]
                if len(tokenized_docs) >= min_sentence_length:
                    tokenized_documents.append(tokenized_docs)
                    full_sentences.append(sentences[i])

            self.tokenized_docs = tokenized_documents
        pass

    def __loadLines__(self, fName, isSource = False):
        """ Loads lines from a file 
        """
        handle = open(self.directory + fName, "r")        
        logging.log(logging.INFO, "\tReading: %s" , handle.name)

        file_lines = handle.readlines()
        handle.close()          
        return file_lines
    
if __name__ == "__main__":
        
    l = GwLargeCorpus()
    raw_input("Press an key to terminate:")
