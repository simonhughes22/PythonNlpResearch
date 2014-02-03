__author__ = 'simon.hughes'

def middle_index(window_size):
    return int((window_size + 1) / 2.0) - 1

SENTENCE_START  = "SENTENCE_START"
SENTENCE_END    = "SENTENCE_END"

def add_bookends(sentence, tags):
    """
    sentence    :   list of str
                        sentence
    tags        :   list of str
                        tags for sentence
    returns (padded sentence, padded tags)

    Adds special start and end tags to a sentence
    """
    return (
        [SENTENCE_START]  + sentence + [SENTENCE_END],
        [None] + tags + [None]
    )

def split_into_windows(lst, window_size, pad_ends = False):
    """ Computes a series of windows of size
        window_size for a list

        lst             :   list of any
                                list of tokens to split into windows
        window_size     :   int
                                size of window
        pad_ends        :   bool
                                whether to duplicate the first and last token
                                to ensure the middle word is always centered


        return list of list of any

        Split a list of tokens into multiple lists of tokens of size window size,

    """
    mix_ix = middle_index(window_size)
    modified_list = lst[:]

    if pad_ends:
        for i in range(mix_ix):
            modified_list.insert(0, SENTENCE_START)
            modified_list.append(SENTENCE_END)

    length = len(modified_list)
    if length < window_size or length == 0 or window_size <= 0:
        return [modified_list]

    # create a circular buffer for the current window
    circular_buffer = [modified_list[i] for i in range(0, window_size)]
    windows = [circular_buffer[:]]
    i = window_size
    while i < length:
        circular_buffer.pop(0)
        circular_buffer.append(modified_list[i])
        windows.append(circular_buffer[:])
        i += 1

    return windows

if __name__ == "__main__":

    lst = range(10)

    windows = split_into_windows(lst, window_size=5, pad_ends=True)

    print lst
    print "Windows:"
    for win in windows:
        print win
        assert len(win) == 5, "Window not the correct length"
