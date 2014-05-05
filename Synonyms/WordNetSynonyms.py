
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
from IterableFP import flatten


class WordNetSynonyms(object):

    def __init__(self, sentences, emit_pos = False):
        self.emit_pos = emit_pos
        self.memoized_synonyms = defaultdict(list)

        self.__train__(sentences)

    def __convert2wn_pos__(self, pos):
        if pos.startswith("NN"):
            return wn.NOUN
        elif pos.startswith("VB"):
            return wn.VERB
        elif pos.startswith("JJ"):
            return wn.ADJ
        elif pos.startswith("RB"):
            return wn.ADV
        else:
            return None

    def __get_words_from_synset__(self, word, pos):
        synsets = wn.synsets(word, pos=pos)
        if len(synsets) == 0:
            return [word]

        words = set()
        for syn in synsets:
            for lemma in syn.lemmas:
                words.add(lemma.name)
        return list(words)

    def __get_word_synonyms__(self, word, pos):
        pair = (word, pos)
        if pair in self.memoized_synonyms:
            return self.memoized_synonyms[pair]

        words = self.__get_words_from_synset__(word, pos)
        self.memoized_synonyms[pair] = words
        return words

    def get_synonyms_for_word(self, word, pos):
        if pos is None:
            return [word]
        wn_pos = self.__convert2wn_pos__(pos)
        if not wn_pos:
            return [word]
        return self.__get_word_synonyms__(word, wn_pos)

    def __train__(self, sentences):

        unique_words = (flatten(sentences))
        syn_map = {}
        mapped = set()
        for sentence in sentences:
            tags = nltk.pos_tag(sentence)
            for wd, tag in tags:
                pair = (wd, tag)
                if pair in mapped:
                    continue
                synonyms = [(s, tag) for s in self.get_synonyms_for_word(wd, tag) if s in unique_words]
                if len(synonyms) >= 1:
                    matches = []
                    for spair in synonyms:
                        if spair in syn_map:
                            matches.append(syn_map[spair])
                    if len(matches) == 0:
                        synset = set(synonyms)
                        synset.add(pair)
                        for p in synset:
                            syn_map[p] = synset
                    elif len(matches) == 1:
                        matches[0].add(pair)
                        syn_map[pair] = matches[0]
                    else:
                        #merge existing synonym lists
                        new_synset = set()
                        for m in matches:
                            new_synset.update(m)
                        #update mapping to map to new larger set
                        for s in new_synset:
                            syn_map[s] = new_synset
                else: #length == 2
                    syn_map[pair] = set([pair])
                mapped.add(pair)
        self.synonym_map = {}

        processed = set()
        for values in syn_map.values():
            vid = id(values)
            if vid in processed:
                continue
            processed.add(vid)
            key = list(values)[0]
            for v in values:
                if v in self.synonym_map:
                    raise Exception("Duplicate key %s" % str(v))
                self.synonym_map[v] = key

    def get_synonyms_for_sentence(self, tokens):
        pos_tagged = nltk.pos_tag(tokens)
        pos_synonyms = []
        for from_wd, tag in pos_tagged:
            from_key = (from_wd, tag)
            if from_key in self.synonym_map:
                pos_synonyms.append(self.synonym_map[from_key])
            else:
                pos_synonyms.append(from_key)
        if self.emit_pos:
            return pos_synonyms
        else:
            return zip(*pos_synonyms)[0]

if __name__ == "__main__":

    sentences = ["the cat is in the hat",
                 "the felines are in the hat",
                 "the event will come soon",
                 "the outcome is the inevitable result, and effect overdue",
                 "The white clean and clear water was blue",
                 "They worked to go run the process",
                 "They process the wood to make it into paper",
                 "They act on the wood to construct paper"
                 ]

    sentences = map(lambda sen: sen.split(" "), sentences)

    wn_synonyms = WordNetSynonyms(sentences, emit_pos=True)
    for tokens in sentences:
        syns = wn_synonyms.get_synonyms_for_sentence(tokens)
        zipped = zip(tokens, syns)

        found = False
        for wd, (synonym,tag) in zipped:
            if wd != synonym:
                print wd, "->", synonym, tag
                found = True
        if found:
            print ""

    print "\n".join(map(str, sorted(wn_synonyms.synonym_map.items())))