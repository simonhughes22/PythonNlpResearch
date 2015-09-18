
from collections import defaultdict
from SpellingCorrector import SpellingCorrector
from Decorators import memoize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from IterableFP import flatten

class Essay(object):
    def __init__(self, name, sentences):
        self.name = name
        self.sentences = sentences

class Sentence(object):
    def __init__(self, tagged_words, sentence_tags):
        self.sentence_tags = sentence_tags
        self.tagged_words = tagged_words


def build_spelling_corrector(essays, lower_case, wd_sent_freq):
    all_words = []
    for essay in essays:
        for sentence in essay.tagged_sentences:
            unique_wds = set()
            for w, tags in sentence:
                if lower_case:
                    w = w.lower()
                all_words.append(w)  # retain frequencies for spelling correction
                if w not in unique_wds:
                    unique_wds.add(w)
                    wd_sent_freq[w] += 1

    return SpellingCorrector(all_words)


def process_essays(essays, min_df = 5,
                      remove_infrequent = False, spelling_correct = True,
                      replace_nums = True, stem = False, remove_stop_words = False,
                      remove_punctuation = True, lower_case=True, spelling_corrector=None, wd_sent_freq=None):

    """ returns a list of essays.

        Each essay is an Essay class consisting of the essay name, and the sentences, which are
        a list of tuples of word : set pairs. The set contains all the tags for the word.
    """

    INFREQUENT = "INFREQUENT"

    VALID_CHARS = {".", "?", "!", "=", "/", ":", ";", "&", "+", "-", "=", "%", "'", ",", "\\", "(", ")", "\""}
    if remove_stop_words:
        stop_wds = stopwords.words("english")

    if stem:
        stemmer = PorterStemmer()

    if spelling_corrector is None:
        wd_sent_freq = defaultdict(int)
        corrector = build_spelling_corrector(essays, lower_case, wd_sent_freq)
    else:
        corrector = spelling_corrector

    corrections = defaultdict(int)

    @memoize
    def correct_word(w):
        if len(w) > 2:
            if w.startswith("'") or w.startswith("\""):
                w = w[1:]
            if w.endswith("'") or w.endswith("\""):
                w = w[:-1]
        if w.endswith("n't") or w.endswith("n'"):
            cw = w[:-3] + "nt"
        elif w.endswith("'s"):
            cw = w[:-2]
        elif w == "&":
            cw = "and"
        else:
            cw = corrector.correct(w)
        return cw

    @memoize
    def is_valid_wd(wd):
        wd = wd.strip()
        if len(wd) > 1:
            return True
        if wd.isalpha() or wd.isdigit() or wd in VALID_CHARS:
            return True
        return False

    @memoize
    def process_word(w):

        # Remove quotes at the start and end
        w = w.strip()
        if lower_case:
            w = w.lower()
        if len(w) == 0:
            return None
        while len(w) > 1 and (w[0] == "\"" or w[0] == "'"):
            w = w[1:]
            if len(w) == 0:
                return None
        while len(w) > 1 and (w[-1] == "\"" or w[-1] == "'"):
            w = w[:-1]
            if len(w) == 0:
                return None

        if remove_stop_words and w in stop_wds:
            return None

        if remove_punctuation and not w.isalpha() and not w.isalnum() and not w.isdigit():
            return None

        if spelling_correct == True:
            cw = correct_word(w)
            if cw != w:
                corrections[(w, cw)] += 1
        else:
            cw = w

        # don't deem a numeric word as infrequent, so pre-process here instead
        if replace_nums and cw.isdigit():
            cw = "0" * len(cw)

        else: # has alpha chars

            if wd_sent_freq[cw] < min_df:
                if stem:
                    cwstemmed = stemmer.stem(cw)
                    if wd_sent_freq[cwstemmed] < min_df:
                        cw = INFREQUENT
                    else:
                        cw = cwstemmed
                        # could now also be all digits
                        if replace_nums and cw.isdigit():
                            cw = "0" * len(cw)
                else:
                    cw = INFREQUENT
            elif stem:
                cw = stemmer.stem(cw)
                if replace_nums and cw.isdigit():
                    cw = "0" * len(cw)
        if remove_stop_words and cw in stop_wds:
            return None
        return cw

    processed_essays = []
    for essay in essays:
        lst_sentences = []
        processed_essays.append(Essay(essay.file_name, lst_sentences))
        for i, sentence in enumerate(essay.tagged_sentences):
            new_sentence = []
            for j, (w, tags) in enumerate(sentence):
                # remove bad single chars
                if not is_valid_wd(w):
                    continue
                cw = process_word(w)
                if cw is None or cw == "" or (remove_infrequent and cw == INFREQUENT):
                    continue
                new_sentence.append((cw, tags))
            if len(new_sentence) > 0:
                lst_sentences.append(new_sentence)
    return processed_essays

def process_sentences(essays, min_df=5,
                      remove_infrequent=False, spelling_correct=True,
                      replace_nums=True, stem=False, remove_stop_words=False,
                      remove_punctuation=True, lower_case=True):
    """
    Flattens the processed essays by extracting just the sentences from the esays
    """

    processed_essays = process_essays(essays, min_df=min_df,
                                      remove_infrequent=remove_infrequent, spelling_correct=spelling_correct,
                                      replace_nums=replace_nums, stem=stem, remove_stop_words=remove_stop_words,
                                      remove_punctuation=remove_punctuation, lower_case=lower_case)
    sentences = []
    for essay in processed_essays:
        for sentence in essay.sentences:
            sentences.append(sentence)
    return sentences

if __name__ == "__main__":

    from BrattEssay import load_bratt_essays

    essays = load_bratt_essays()
    processed_sentences = process_sentences(essays, stem=True, spelling_correct=False)
    pass
