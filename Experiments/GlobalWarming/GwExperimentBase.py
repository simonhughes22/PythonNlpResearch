'''
Created on Apr 14, 2013

@author: Simon
'''
from ExperimentBase import *
from GwLsa import *
import GwData

important_codes = set([
    "0", "1", "12", "20", "22", "3", "38", "40", "42", "50", "53", "bck", #Target change codes
    "p14", "p20", "p21", "p28", "p33", "p34", "p4", "p40", "p49", GwData.CAUSAL
])


class GwExperimentBase(ExperimentBase):
    __metaclass__ = ABCMeta

    def sub_dir(self):
        return GwData.FOLDER

    def get_data(self, settings):
        return GwData.GwData(directory=settings.data_directory + "\\" + self.sub_dir())

    def codes_to_filter(self):
        return important_codes

    def lsa_gw_vspace(self, tokenized_docs):

        lsa = GwLsa(num_topics = self.dimensions)
        distance_matrix = [lsa.project(tokenized_doc)
                  for tokenized_doc in tokenized_docs]
        return (distance_matrix, lsa.id2Word)