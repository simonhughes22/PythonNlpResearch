"""
 Module JMontyLingua.py

 Java API for MontyLingua, please see MontyLingua module
 for additional documentation and license information.

 EXPLANATION OF FUNCTIONS:

     @sig public String jist_predicates(String text)
       - returns lisp-style predicate argument structures
       - each structure should look something like this:
          - ("verb" "subject" "obj1" "obj2" ... )
       - words are all lemmatised, and determiners and
         modals are stripped out
       - obj's can be direct or indirect, but not
         subordinate clauses for now.
       - returns one pred-arg per line
       - multiple pred-args are possible for a sentence
       - blank line separates pred-args of each sentence

     @sig public String tag_text(String text)
       - takes in raw text.
       - tokenizes and POS tags text using Brill94
         tbl-based tagging and common sense
       - uses Penn Treebank tagset
         (http://www.cis.upenn.edu/~treebank/)
       - returns one tagged sentence per line

     @sig public String chunk_text(String text)
       - takes in raw text.
       - tokenizes, POS tags, and chunks tagset
         in adjective chunks, noun chunks, and verb
         chunks (AX, NX, and VX respectively)
       - returns one chunked sentence per line
     
     @sig public String lemmatise_text(String text)
       - lemmatises raw text and outputs the form:
         'These/DT/These sentences/NNS/sentence were/VBZ/be false/JJ/false'
         (lemma follows the pos tag)
       - returns one lemmatised sentence per line
"""

__author__  = "Hugo Liu <hugo@media.mit.edu>"
__version__ = "2.0"

import MontyLingua
import java
from jarray import array

class JMontyLingua(java.lang.Object):

    def __init__(self):
        "@sig public JMontyLingua()"
        self.theMontyLingua = MontyLingua.MontyLingua()

    def jist_predicates(self,text):
        "@sig public String jist_predicates(String text)"
        svoos_list = self.theMontyLingua.jist_predicates(text)
        return '\n\n'.join(map(lambda x:'\n'.join(x),svoos_list))

    def tag_text(self,text):
        "@sig public String tag_text(String text)"
        sentences = self.theMontyLingua.split_sentences(text)
        tokenized = map(self.theMontyLingua.tokenize,sentences)
        tagged = map(self.theMontyLingua.tag_tokenized,tokenized)
        return '\n\n'.join(tagged)

    
    def chunk_text(self,text):
        "@sig public String chunk_text(String text)"
        sentences = self.theMontyLingua.split_sentences(text)
        tokenized = map(self.theMontyLingua.tokenize,sentences)
        tagged = map(self.theMontyLingua.tag_tokenized,tokenized)
        chunked = map(self.theMontyLingua.chunk_tagged,tagged)
        return '\n\n'.join(chunked)

    def lemmatise_text(self,text):
        "@sig public String lemmatise_text(String text)"
        sentences = self.theMontyLingua.split_sentences(text)
        tokenized = map(self.theMontyLingua.tokenize,sentences)
        tagged = map(self.theMontyLingua.tag_tokenized,tokenized)
        lemmatised = map(self.theMontyLingua.lemmatise_tagged,tagged)
        return '\n\n'.join(lemmatised)

    
