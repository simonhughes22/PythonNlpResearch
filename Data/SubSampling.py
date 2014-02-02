from DictionaryHelper import tally_items
import numpy as np

def over_sample(x,y):
    
    if type(y) == type(np.array([])):
        y = y.flatten().tolist()
    
    tally = tally_items(list(y))
    y_vals = sorted(set(y))
    assert len(y_vals) == 2, "sub sampling assumes a binary classification"
    
    positive_class = y_vals[-1]
    negative_class = y_vals[0]
    
    """ More positive """
    if tally[positive_class] > tally[negative_class]:
        
        alpha = round(tally[positive_class] / float(tally[negative_class]))
        to_multiply = negative_class
    else:
        alpha = round(tally[negative_class] / float(tally[positive_class]))
        to_multiply = positive_class
    alpha = int(alpha)
    new_xs, new_ys = [], []
    
    for i in range(len(x)):
        xval = x[i]
        yval = y[i]
        
        rng = 1
        if yval == to_multiply:
            rng = alpha

        for j in range(rng):
            new_xs.append(xval)
            new_ys.append(yval)
    
    new_xs = np.array(new_xs)
    new_ys = np.array(new_ys)
    
    ixs = range(len(new_xs))
    np.random.shuffle(ixs)
    
    return (new_xs[ixs], new_ys[ixs])

if __name__ == "__main__":
    
    import numpy as np
    xs = np.array(range(10))
    ys = xs % 2
    
    """ Imbalance for negatives"""
    ys[5:] = 0
    
    new_xs,new_ys = over_sample(xs, ys)
    
    print new_xs
    print new_ys
    sum(new_ys)
    assert np.sum(new_ys) == 8
    
    """ Imbalance for positives"""
    ys[5:] = 1
    ys[0]  = 1
    
    new_xs,new_ys = over_sample(xs, ys)
    
    print new_xs
    print new_ys
    sum(new_ys)
    assert np.sum(new_ys) == 8