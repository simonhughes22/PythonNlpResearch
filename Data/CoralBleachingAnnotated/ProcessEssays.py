
from collections import defaultdict
from SpellingCorrector import SpellingCorrector
from Decorators import memoize
from nltk import PorterStemmer

def process_sentences(essays, min_df = 5, spelling_correct = True, replace_nums = True, stem = False):

    all_words = []
    wd_sent_freq = defaultdict(int)
    VALID_CHARS = {".", "?", "!", "=", "/", ":", ";", "&", "+", "-", "=", "%", "'", ",", "\\", "(", ")", "\""}

    if stem:
        stemmer = PorterStemmer()

    for essay in essays:
        for sentence in essay.tagged_sentences:
            unique_wds = set()
            for w, tags in sentence:
                all_words.append(w) # retain frequencies for spelling correction
                if w not in unique_wds:
                    unique_wds.add(w)
                    wd_sent_freq[w] += 1

    corrector = SpellingCorrector(all_words)
    corrections = defaultdict(int)

    @memoize
    def correct_word(w):
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
        if spelling_correct:
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
                        cw = "INFREQUENT"
                    else:
                        cw = cwstemmed
                        # could now also be all digits
                        if replace_nums and cw.isdigit():
                            cw = "0" * len(cw)
                else:
                    cw = "INFREQUENT"
            elif stem:
                cw = stemmer.stem(cw)
                if replace_nums and cw.isdigit():
                    cw = "0" * len(cw)
        return cw

    sentences = []
    for essay in essays:
        for i, sentence in enumerate(essay.tagged_sentences):
            new_sentence = []
            for j, (w, tags) in enumerate(sentence):
                # remove bad single chars
                if not is_valid_wd(w):
                    continue
                new_sentence.append((process_word(w), tags))
            sentences.append( new_sentence )
    return sentences

if __name__ == "__main__":
    from BrattEssay import load_bratt_essays

    essays = load_bratt_essays()
    processed_sentences = process_sentences(essays, stem=True)
    pass
