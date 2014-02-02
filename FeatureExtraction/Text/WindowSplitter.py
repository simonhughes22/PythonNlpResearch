__author__ = 'simon.hughes'

def split_into_windows(lst, window_size):
    """ Computes a series of windows of size
        window_size for a list

        lst             :   list of any
                                list of tokens to split into windows
        window_size     :   int
                                size of window

        return list of list of any

        Split a list of tokens into multiple lists of tokens of size window size,

    """
    length = len(lst)
    if length < window_size or length == 0 or window_size <= 0:
        return [lst]

    queue = [lst[i] for i in range(0, window_size)]
    windows = [queue[:]]
    i = window_size
    while i < length:
        queue.pop(0)
        queue.append(lst[i])
        windows.append(queue[:])
        i += 1
    return windows