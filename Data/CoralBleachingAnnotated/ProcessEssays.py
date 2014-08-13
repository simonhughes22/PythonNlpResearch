
from collections import defaultdict
from SpellingCorrector import SpellingCorrector
from Decorators import memoize

def process_essays(essays, spelling_correct = True, min_df = 5, replace_nums = False):
    all_words = []
    wd_sent_freq = defaultdict(int)
    VALID_CHARS = {".", "?", "!", "=", "/", ":", ";", "&", "+", "-", "=", "%", "'", ",", "\\", "(", ")", "\""}

    for essay in essays:
        for sentence in essay.tagged_sentences:
            unique_wds = set()
            for w, tags in sentence:
                all_words.append(w)
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

        if replace_nums and cw.isdigit():
            cw = "0" * len(cw)
        # don't deem a numeric word as infrequent
        elif wd_sent_freq[cw] < min_df:
            cw = "INFREQUENT"
        return cw

    for essay in essays:
        for i, sentence in enumerate(essay.tagged_sentences):
            new_sentence = []
            for j, (w, tags) in enumerate(sentence):
                # remove bad single chars
                if not is_valid_wd(w):
                    continue
                new_sentence.append((process_word(w), tags))

            essay.tagged_sentences[i] = new_sentence
    return essays

if __name__ == "__main__":
    from BrattEssay import load_bratt_essays

    essays = load_bratt_essays()
    processed_essays = process_essays(essays)
