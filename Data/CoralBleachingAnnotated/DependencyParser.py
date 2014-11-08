from stanford_parser import parser
from WindowProcessor import WindowProcessor as WinProc

class Dependency(object):
    def __init__(self, relation, gov_tuple, dep_tuple):
        self.relation = relation
        self.gov_txt, self.gov_start, self.gov_end = gov_tuple
        self.dep_txt, self.dep_start, self.dep_end = dep_tuple
        pass

class DependencyParser(object):

    def __init__(self):
        #TODO inject
        self.__winProc_   = WinProc([], window_size=7)
        self.__ix2parsed_ = {}
        self.__parser_    = parser.Parser()

        processed_sent_ixs = set()
        for win_ix, win in enumerate(self.__winProc_.get_tagged_windows()):
            sent_ix = self.__winProc_.get_sentence_ix_from_window(win_ix)
            if sent_ix in processed_sent_ixs:
                continue

            processed_sent_ixs.add(sent_ix)
            tokens = self.__winProc_.get_tokenized_sentence(sent_ix)
            sentence_txt = " ".join(t for t in tokens if t.isalnum())
            dependencies = self.__parser_.parseToStanfordDependencies(sentence_txt)
            tupleResult  = [ Dependency(rel, (gov.text, gov.start, gov.end), (dep.text, dep.start, dep.end))
                                for rel, gov, dep in dependencies.dependencies
                            ]

import numpy as np
np.linspace()