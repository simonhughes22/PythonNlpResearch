__author__ = 'simon.hughes'

import numpy as np
from WindowSplitter import split_into_windows
from collections import defaultdict
from IterableFP import flatten

SENTENCE_START = "<START>"
SENTENCE_END = "<END>"

class WindowProcessor(object):

    def __init__(self,  essays, window_size=7):
        self.window_size = window_size
        self.mid_ix = int(round(window_size / 2.0) - 1)
        self.__split_essays_(essays)

    def __bookend_(self, sentence):
        # make copy (don't mutate)
        standardized_sentence = list(sentence)
        # grow padding from middle outwards
        for i in range(self.mid_ix):
            standardized_sentence.insert(0, (SENTENCE_START, set()))
            standardized_sentence.append(   (SENTENCE_END,   set()))
        return standardized_sentence

    def __assert_windows_correct_(self, windows):
        lens = map(len, windows)
        assert min(lens) == max(lens) == self.window_size, \
            "Windows are not all the correct size"

    def split_tagged_sentence(self, tagged_sentence):
        """ Utility Method, not used below in split essays, but exposed to calling
            after construction
        """
        modified_sentence = list(tagged_sentence)
        self.bookend(modified_sentence)
        windows = split_into_windows(modified_sentence, window_size=self.window_size)
        self.assert_windows_correct(windows)
        return windows

    def __split_essays_(self, essays):
        """
            Takes a list of list of sentences (each an essay), and processes into a list of windows
        """

        # Indexes below refer to the window indices (not @ the sentence nor the essay level)
        # easiest to represent indices at the lowest level, the window, and then aggregate from there

        # top down mappings
        # ix's here refer to window indexes
        self.__essay2sentix_ = defaultdict(list)
        self.__sent2ix_ = defaultdict(list)
        self.__essay2ix_ = defaultdict(list)

        # bottom up, from the window
        # ix to ix
        self.__ix2essay_ = {}
        self.__ix2sent_  = {}
        self.__sent2essayix_ = {}

        # use lists below to retrieve raw data

        # list of raw data
        self.__std_sentences_ = []
        self.__orig_sentences_ = []
        # words only, no labels, and no padding
        self.__tokenized_sentences_ = []
        # flattened windows, one list, each containing a tagged window (list of word:lbl pairs)
        self.__tagged_windows_ = []
        # words only
        self.__word_windows_ = []
        # tags only (list of sets)
        self.__tag_windows_ = []

        # list of essays, which are a list of sentences, which are a list of windows (list of word:lbl pairs)
        self.__nested_essays_ = []
        # like above, but at the sentence level
        self.__nested_sentences_ = []

        sent_ix = -1
        win_ix  = -1

        self.all_codes = set()
        for e_ix, essay in enumerate(essays):
            nested_essay = []
            self.__nested_essays_.append(nested_essay)
            for sentence in essay:
                if len(sentence) == 0:
                    continue

                # pre-process and normalize sentence
                standardized_sentence = self.__bookend_(sentence)

                # tagged words
                self.__std_sentences_.append(standardized_sentence)
                self.__orig_sentences_.append(sentence)

                # words only
                self.__tokenized_sentences_.append(zip(*sentence)[0])
                sent_ix = len(self.__std_sentences_) - 1

                # fetch windows
                windows = split_into_windows(standardized_sentence, window_size=self.window_size)
                self.__assert_windows_correct_(windows)
                nested_essay.append(windows)

                # map essay to sentence
                self.__essay2sentix_[e_ix].append(sent_ix)
                self.__sent2essayix_[sent_ix] = e_ix
                self.__nested_sentences_.append(windows)
                for win in windows:
                    # map ix's to sentence and essay
                    self.__tagged_windows_.append(win)
                    words, tags = zip(*win)
                    for set_tags in tags:
                        for tag in set_tags:
                            self.all_codes.add(tag)
                    self.__word_windows_.append(words)
                    self.__tag_windows_.append(tags)

                    win_ix = len(self.__tagged_windows_) - 1

                    # grouped indexes, top down, essay -> sent -> win
                    self.__sent2ix_[sent_ix].append(win_ix)
                    self.__essay2ix_[e_ix].append(win_ix)

                    # 1:1, low level window ix to sent and essay
                    self.__ix2sent_[win_ix]  = sent_ix
                    self.__ix2essay_[win_ix] = e_ix
                    pass

        pass

    def get_windows_grouped_by_sentence(self):
        """ Returns a list of lists of windows. Each outer list item represents a sentence broken
            into windows of size self.window_size.

        """
        return self.__nested_sentences_

    def get_word_windows(self):
        """ A flattened list of word windows
        """
        return self.__word_windows_

    def get_tagged_windows(self):
        """ A flattened list of windows of word : tag set pairs
        """
        return self.__tagged_windows_

    def get_tag_windows(self):
        """ A flattened list of windows of tag sets
        """
        return self.__tag_windows_

    def __get_prev_sent_ix_(self, win_ix):
        """ win_ix  : int - window index
            returns : int - sentence index, or None
        """
        dfltRetval = None
        current_sent_ix = self.__ix2sent_[win_ix]
        b4 = current_sent_ix - 1
        if b4 < 0:
            return dfltRetval
        return b4

    def __get_next_sent_ix_(self, win_ix):
        """ win_ix  : int - window index
            returns : int - sentence index, or None
        """
        dfltRetval = None
        current_sent_ix = self.__ix2sent_[win_ix]
        after = current_sent_ix + 1
        if after >= len(self.__std_sentences_):
            return dfltRetval
        return after

    def __ix_in_range_(self, win_ix):
        return win_ix >= 0 and win_ix < len(self.__tagged_windows_)

    def __sent_ix_in_range_(self, sent_ix):
        return sent_ix >= 0 and sent_ix < len(self.__std_sentences_)

    def get_window_b4(self, win_ix):
        """ win_ix : int - window index
        """
        dfltRetVal = []
        # validate prev is within same sentece
        cur_sent_ix = self.__ix2sent_[win_ix]
        prev_win_ix = win_ix - 1
        # out of bounds, or from different sentences?
        if prev_win_ix < 0 or self.__ix2sent_[prev_win_ix] != cur_sent_ix:
            return dfltRetVal
        return self.__tagged_windows_[prev_win_ix]

    def get_window_at(self, win_ix):
        """ win_ix : int - window index
        """
        if not self.__ix_in_range_(win_ix):
            return []
        else:
            return self.__tagged_windows_[win_ix]

    def get_window_after(self, win_ix):
        """ win_ix : int - window index
        """
        dfltRetVal = []
        # validate prev is within same sentece
        cur_sent_ix = self.__ix2sent_[win_ix]
        next_win_ix = win_ix + 1
        # out of bounds, or from different sentences?
        if next_win_ix >= len(self.__tagged_windows_) or self.__ix2sent_[next_win_ix] != cur_sent_ix:
            return dfltRetVal
        return self.__tagged_windows_[next_win_ix]

    def get_sentence_ix_b4_window(self, win_ix):
        """ win_ix : int - window index
        """
        dfltRetVal = -1
        prev_sent_ix = self.__get_prev_sent_ix_(win_ix)
        if not prev_sent_ix:
            return dfltRetVal
        curr_sent_ix = self.__ix2sent_[win_ix]
        # From different essays?
        if self.__sent2essayix_[prev_sent_ix] != self.__sent2essayix_[curr_sent_ix]:
            return dfltRetVal
        return prev_sent_ix

    def get_sentence_ix_at_window(self, win_ix):
        """ win_ix : int - window index
        """
        if not self.__ix_in_range_(win_ix):
            return -1
        return self.__ix2sent_[win_ix]

    def get_sentence_ix_after_window(self, win_ix):
        """ win_ix : int - window index
        """
        dfltRetVal = -1
        next_sent_ix = self.__get_next_sent_ix_(win_ix)
        if not next_sent_ix:
            return dfltRetVal
        curr_sent_ix = self.__ix2sent_[win_ix]
        # From different essays?
        if self.__sent2essayix_[next_sent_ix] != self.__sent2essayix_[curr_sent_ix]:
            return dfltRetVal
        return next_sent_ix

    def get_sentence_ix_from_window(self, win_ix):
        if not self.__ix_in_range_(win_ix):
            return -1
        return self.__ix2sent_[win_ix]

    """ Methods to get sentences """
    def get_tokenized_sentence(self, sent_ix):
        if not self.__sent_ix_in_range_(sent_ix):
            return []
        return self.__tokenized_sentences_[sent_ix]

    def get_tagged_sentence(self, sent_ix):
        if not self.__sent_ix_in_range_(sent_ix):
            return []
        return self.__orig_sentences_[sent_ix]

    def get_padded_sentence(self, sent_ix):
        if not self.__sent_ix_in_range_(sent_ix):
            return []
        return self.__std_sentences_[sent_ix]

    def get_ys_by_code(self):
        ysByCode = defaultdict(list)
        for set_tags in self.__tagged_windows_:
            # Tags for middle word (target)
            tags4word = set_tags[self.mid_ix]
            for code in self.all_codes:
                ysByCode[code].append(1 if code in tags4word else 0)

        for key, lst in ysByCode.items():
            ysByCode[key] = np.asarray(lst)

        return ysByCode
