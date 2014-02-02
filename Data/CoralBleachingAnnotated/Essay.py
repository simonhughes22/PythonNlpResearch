from collections import defaultdict
from itertools import imap
from xml.dom import minidom
from EssayElements import *


__ESSAY_START__  = "ESSAY_START"
__ESSAY_END__    = "ESSAY_END"

__SENTENCE_BREAK__ = "SENTENCE_BREAK"

class Essay(object):

    def __init__(self, fpath):
        self.full_path = fpath

        print "Processing Essay: %s" % str(self)

        xmldoc = minidom.parse(self.full_path)
        self.xmldoc = xmldoc

        """ Grab the tezt node,
            get the CDATA element's value,
            skip the first 2 junk chars,
            and rstrip the trailing \n (don't strip so that we preserve the offsets)
        """
        self.essay_text = self.xmldoc.getElementsByTagName("TEXT")[0].childNodes[0].nodeValue[2:].rstrip()

        self.id_to_element = {}
        self.sentences      = self.__get_sorted_elements__(Sentence, "Sentence")
        self.__connect_sentences__(self.sentences)

        self.concepts       = self.__get_sorted_elements__(Concept, "Concept")
        self.vague_concepts = self.__get_sorted_elements__(VagueConcept, "Vconcept")
        self.causal         = self.__get_sorted_elements__(Causal, "Causal")
        self.structures     = self.__get_sorted_elements__(Structure, "Structure")
        self.keywords = self.__get_sorted_elements__(KeyWord, "KeyWord")

        self.causal_links = self.__get_links__(CausalLink, "CausalLINK")
        self.structure_links = self.__get_links__(StructureLINK, "StructureLINK")

        #Single element
        compiled_score = self.__get_sorted_elements__(CompiledScore, "CompiledScore")
        if compiled_score:
            self.compiled_score = compiled_score[0]
        else:
            self.compiled_score = None

        self.tagged_words, self.tagged_sentences = self.__tag_essay__()
        pass

    def __connect_sentences__(self, sorted_sentences):
        """ Connects each sentence to the previous and next
            Requires sentences already in sorted order
        """
        last_ix = (len(sorted_sentences) - 1)
        for ix, sent in enumerate(sorted_sentences):
            is_first = (ix == 0)
            is_last = (ix == last_ix)

            if not is_first:
                sent.prev = sorted_sentences[ix - 1]
            if not is_last:
                sent.next = sorted_sentences[ix + 1]
        pass

    def __get_links__(self, fn, name):
        elements = self.__get_elements__(fn, name)
        last_ix = len(elements) - 1
        for i, el in enumerate(elements):
            is_first = (i == 0)
            is_last  = (i == last_ix)

            if not is_first:
                el.prev = elements[i -1]
            if not is_last:
                el.next = elements[i + 1]

            el.from_element = self.id_to_element[el.from_id]
            el.to_element   = self.id_to_element[el.to_id]
        pass

    def __get_elements__(self, fn, element_name):
        raw_elements = self.xmldoc.getElementsByTagName(element_name)
        elements = []
        for raw_el in raw_elements:
            element = fn(raw_el)
            element.parent = self
            # ensure all elements are indexed by id
            self.id_to_element[element.id] = element
            elements.append(element)
        return elements

    def __get_sorted_elements__(self, fn, element_name):
        elements = self.__get_elements__(fn, element_name)
        """ Sort by start and end positions """
        return sorted(elements, key=lambda snt: (snt.start, snt.end))

    def __tag_essay__(self):

        def tag_word(active_concept_ids, current_word):
            if len(active_concept_ids) > 0:
                active_concept_codes = \
                    set(
                        imap(lambda el: el.code,
                             imap(lambda id: self.id_to_element[id],
                                active_concept_ids)))
            else:
                active_concept_codes = set()
            # remove unicode, empty space and lower case it
            normalized = removeNonAscii(current_word).strip().lower()
            return (normalized, active_concept_codes)

        concept_starts = defaultdict(set)
        concept_ends = defaultdict(set)
        for c in self.concepts:
            concept_starts[c.start].add(c.id)
            concept_ends[c.end].add(c.id)

        sentence_starts = set(map(lambda sent: sent.start, self.sentences ))

        tagged_words = [] # list of tuples: (word, set(tags))
        current_word = ""

        active_concept_ids = set()
        last_ix = (len(self.essay_text) - 1)

        unique_tag_count = 0
        tagged_sentences = []
        current_sentence = []

        for ix, ch in enumerate(self.essay_text):
            if ch.isalnum():
                current_word += ch
            else:
                """ Finish processing previous word """
                if len(current_word) > 0:
                    tagged_pair = tag_word(active_concept_ids, current_word)
                    tagged_words.append(tagged_pair)
                    current_sentence.append(tagged_pair)
                    current_word = ""

            """ Process sentences """
            if ix in sentence_starts:
                """ Add sentence break """
                tagged_words.append( (__SENTENCE_BREAK__, set()) )

                if len(current_sentence) > 0:
                    tagged_sentences.append(current_sentence[:])
                current_sentence = []

            """ Store single chars """
            if not ch.isalnum():
                ch_stripped = removeNonAscii(ch.strip())
                if len(ch_stripped) > 0 and ch_stripped != "/":
                    """ Enter punctuation """
                    tagged_pair = tag_word(active_concept_ids, ch_stripped)
                    tagged_words.append(tagged_pair)
                    current_sentence.append(tagged_pair)

            """ Process concepts """
            if ix in concept_starts:
                concept_ids = concept_starts[ix]
                active_concept_ids.update(concept_ids)
                unique_tag_count += 1

            # remove any concept ids that have ended
            if ix != last_ix and ix in concept_ends :
                concept_ids = concept_ends[ix]
                active_concept_ids = active_concept_ids.difference(concept_ids)
            pass

        if len(current_word) > 0:
            tagged_pair = tag_word(active_concept_ids, ch_stripped)
            tagged_words.append(tagged_pair)
            current_sentence.append(tagged_pair)

        if len(current_sentence) > 0:
            tagged_sentences.append(current_sentence[:])

        assert unique_tag_count == len(self.concepts), \
            "Concepts count should match unique tag count"
        assert len(tagged_sentences) == len(self.sentences), \
            "Different number of tagged sentences to sentences in xml doc"

        # add essay start and end tags
        tagged_words.insert(0,  (__ESSAY_START__,   set())  )
        tagged_words.append(    (__ESSAY_END__  ,   set())  )
        return (tagged_words, tagged_sentences)

    def print_tags(self):
        max_len = max(map(lambda (wd, _): len(wd), self.tagged_words))
        for i, (wd, codes) in enumerate(self.tagged_words):
            str_codes = ""
            if len(codes) > 0:
                str_codes = str(list(codes))
            print str(i).ljust(3), wd.ljust(max_len + 1), str_codes

    def print_tagged_sentences(self):
        max_len = max(map(lambda (wd, _): len(wd), self.tagged_words))

        for i, (tagged_sentence) in enumerate(self.tagged_sentences):
            print "\nSENTENCE", i
            for wd, codes in tagged_sentence:
                str_codes = ""
                if len(codes) > 0:
                    str_codes = str(list(codes))
                print "\t", str(i).ljust(3), wd.ljust(max_len + 1), str_codes

    def __repr__(self):
        return self.full_path.split("/")[-1]
pass

def essay_loader():
    import Settings
    from os import listdir
    from os.path import isfile, join

    settings = Settings.Settings()
    root_folder = settings.data_directory + "CoralBleaching/Files/"

    onlyfiles = [f for f in listdir(root_folder) if isfile(join(root_folder, f))]
    full_paths = map(lambda f: join(root_folder, f), onlyfiles)

    # Make linux style path
    full_paths = map(lambda pth: pth.replace("\\", "/"), full_paths)
    assert len(full_paths) == 105, \
        "Wrong number of files found: %d. Expected  - 105" % len(full_paths)
    print "%d files found" % len(full_paths)
    return map(Essay, full_paths)

