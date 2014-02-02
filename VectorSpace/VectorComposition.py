import numpy as np
import gensim
from MatrixHelper import unit_vector

def __to_np_arrays__(vectors):
    l = []
    for v in vectors:
        #some composition models can't map all words
        if len(v) > 0:
            l.append(np.array(v))
    return np.array(l)

def Add(vectors):
    arr = __to_np_arrays__(vectors)
    rslt = np.cumsum(arr, 0)[-1]
    return unit_vector(rslt)

def Multiply(vectors):
    arr = __to_np_arrays__(vectors)
    return unit_vector(np.cumprod(arr, 0)[-1])

def Mean(vectors):
    arr = __to_np_arrays__(vectors)
    return np.mean(arr, 0)

def MinMaxMean(vectors):
    arr = __to_np_arrays__(vectors)
    l_mean = np.mean(arr, 0).tolist()
    l_min = np.min(arr, 0).tolist()
    l_max = np.min(arr, 0).tolist()
    """ Note the .tolist()'s so the following is a list concatenation """
    return l_min + l_max + l_mean

def Median(vectors):
    arr = __to_np_arrays__(vectors)
    return np.median(arr, 0)

def Max(vectors):
    arr = __to_np_arrays__(vectors)
    return np.max(arr, 0)

def __circular_convolution__(a, b):
    tmp = []
    v_len = len(a)
    for i in range(v_len):
        total = 0
        for j in range(v_len):
            pair = a[j] * b[ (i - j) % v_len ]
            total += pair
        tmp.append(total)
    return tmp

def __circular_correlation__(a, b):
    tmp = []
    v_len = len(a)
    for i in range(v_len):
        total = 0
        for j in range(v_len):
            pair = a[j] * b[ (i + j) % v_len ]
            total += pair
        tmp.append(total)
    return tmp

def CircularConvolution(vectors):
    
    """ Computes a circular convolution between adjacent vectors """
    last = vectors[0]
    v_len = len(last)
    convs = []
    
    for ix in range(len(vectors) - 1):
        next = vectors[ix + 1]
        conv = __circular_convolution__(last, next)
        convs.append(conv)
        last = next
        
    return np.array(convs).mean(axis = 0)

def CircularCorrelation(vectors):
    """ Computes a circular convolution between adjacent vectors """
    last = vectors[0]
    v_len = len(last)
    convs = []
    
    for ix in range(len(vectors) - 1):
        next = vectors[ix + 1]
        conv = __circular_correlation__(last, next)
        convs.append(conv)
        last = next
        
    return np.array(convs).mean(axis = 0)

if __name__ == "__main__":
    
    a1 = [ 1, 2, 3, 4, 5]
    a2 = [ 1,-1, 4, 0, 0]
    a3 = [10,-2,-1, 4, 4]

    vectors = [a1,a2,a3]
    
    __cc = CircularConvolution(vectors)
    _add = Add(vectors)
    _mult = Multiply(vectors)
    _mean = Mean(vectors)
    _median = Median(vectors)
    _max = Max(vectors)
    _minmaxmean = MinMaxMean(vectors)
    
    pass