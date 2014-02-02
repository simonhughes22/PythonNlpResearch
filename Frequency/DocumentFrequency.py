import math
'''
Created on Jan 21, 2013

@author: Simon
'''
from collections import defaultdict
import DictionaryHelper

def compute_document_frequency(lst_token_lists):
    '''
    @param param: a list of list of tokens
    @type tokens: list{list{string}}
    
    '''
    doc_freq = defaultdict(int)
    for lst in lst_token_lists:
        unique = set(lst)
        for token in unique:
            doc_freq[token] += 1
            
    return doc_freq

def document_frequency_ratio(lst_tokens, lst_class, predicate):
    """
        @param lst_tokens: a list of list of tokens
        @type lst_tokens: list{list{string}}
        
        @param lst_class: a list of class values (dependent var)
        @type lst_class: list{string}
        
        @param predicate: function returning true or false 
            for the membership of the target class
        @type predicate: function(string): Boolean
    """
    
    all_docs_freq = compute_document_frequency(lst_tokens)
    class_docs = [tpl[1]
                  for tpl in enumerate(lst_tokens) 
                  if predicate(lst_class[tpl[0]])]
    
    class_doc_freq = compute_document_frequency(class_docs)
    # Should NOT return DIV/0 as items will alwyas be present in
    # all docs if they are present in a subset 
    
    return dict(
                #Raw frequency ratio
                #(word, float(class_doc_freq[word]) / math.log( float(all_docs_freq[word]))) 
                
                #~TfIdf
                (word, float(class_doc_freq[word]) / float(all_docs_freq[word])) 
    
                for word in class_doc_freq.keys() )
    
    def document_frequency_distribution(lst_tokens):
        """ Computes the document frequency distribution
            for all tokens in the dataset, return a list
            of {token, pct doc occurence) pairs
        """
        num_docs = float(len(lst_tokens))
        sortedDocFreq = DictionaryHelper.sort_by_value(document_frequency_distribution(lst_tokens), 
                                                 reverse = True)
        return [ (kvp[0], kvp[1] / num_docs * 100) 
                 for kvp in sortedDocFreq ]
        