'''
Created on Apr 9, 2013

@author: simon
'''

import Lsa
import TfIdf
import WordTokenizer
import logging
import Settings
import nltk
import os.path
from gensim import corpora
from gensim.models.lsimodel import LsiModel

class GwLsa(object):
    '''
    classdocs
    '''
    def __init__(self, num_topics, directory = None, min_sentence_length = 3):
        if directory == None:
            directory = Settings.Settings().data_directory + "\GlobalWarming"

        if not directory.endswith("\\"):
            directory += "\\"

        self.directory = directory
        logging.log(logging.INFO, "GwLsaClass: Processing Data from directory \n\t'%s'", directory)
        
        lsa_file = "{0}lsa_{1}.lsi".format(directory, num_topics)
        id2Word_file = "{0}id2Word.txt".format(directory, num_topics)
        
        if os.path.isfile(lsa_file):
            pass
            #TODO
            #self.__lsa__ = LsiModel.load(lsa_file)
            #self.id2Word = corpora.Dictionary.load(id2Word_file)
            #return
            
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
                
                #if len(sentences) > 100:
                    #print " >> STOPPING EARLY TO SPEED DEBUGGING, PLEASE REMOVE"
                    #break
                
        documents = []
        wt = WordTokenizer.WordTokenizer(min_word_count = 3)
        tokenized = wt.tokenize(sentences)
        for tokenized_docs in tokenized:
            if len(tokenized_docs) >= min_sentence_length:
                documents.append(tokenized_docs)
        
        tfidf = TfIdf.TfIdf(documents)
        self.__lsa__ = Lsa.Lsa(tfidf, num_topics = num_topics)
        self.id2Word = tfidf.id2Word
        self.num_topics = num_topics
        
        #LsiModel.save(self.__lsa__, lsa_file)
        #corpora.Dictionary.save(self.id2Word, id2Word_file)
        
    def __loadLines__(self, fName, isSource = False):
        """ Loads lines from a file 
        """
        handle = open(self.directory + fName, "r")        
        logging.log(logging.INFO, "\tReading: %s" , handle.name)

        file_lines = handle.readlines()
        handle.close()          
        return file_lines
    
    def project(self, tokenized_doc):
        return self.__lsa__.to_lsa_vector(tokenized_doc)

    def project_matrix(self, tokenized_docs):
        return self.__lsa__.to_lsa_matrix(tokenized_docs)
    
if __name__ == "__main__":

        
    l = GwLsa(100)
    raw_input("Press an key to terminate:")
