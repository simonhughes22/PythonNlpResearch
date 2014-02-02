
from sklearn import metrics
import numpy as np

def best_threshold_for_f1(probs, num_values, ys):
    """
        Probs are a 2 X N dimensional array (a probability for each class)
    """
    positive = max(ys)
    negative = min(ys)
    
    assert len(np.unique(ys)) == 2

    def create_threshold_fn(threshold):
        def above_threshold(prob):
                if prob[1] >= threshold:
                    return positive
                return negative
        return above_threshold

    increment = 1 / float(num_values)
    best_threshold = -1
    best_f1 = -1
    
    for i in range(num_values-1): #Skip first and last
        threshold = (i + 1.0) * increment
        new_ys = map(create_threshold_fn(threshold), probs)
        score = metrics.f1_score(ys, new_ys)
        if score > best_f1:
            best_threshold = threshold
            best_f1 = score

    return (max(0.0 + increment,best_threshold), positive, negative)

def apply_threshold(classifier, xs, threshold, positive_val = 1.0, negative_val = 0.0):
    
    probs = classifier.predict_proba(xs)
    def above_threshold(prob):
        # note, this is the second entry (as being the positive class)
        # probably better illustrated as prob[-1]
        if prob[1] >= threshold:
            return positive_val
        return negative_val
    
    return map(above_threshold, probs)

if __name__ == "__main__":
    
    import numpy as np
    probs = np.array(range(10)) * 0.1
    
    probs2 = np.zeros((10,2))
    probs2[:,1] = probs
    probs2[:,0] = 1- probs
    
    ys = np.array(range(10)) % 2
    ys[:5] = 0
    ys[5:] = 1
    
    t = best_threshold_for_f1(probs2, 10, ys)