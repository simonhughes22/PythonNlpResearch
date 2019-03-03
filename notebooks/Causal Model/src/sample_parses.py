import numpy as np

def to_parse(lst):
    return tuple(sorted(lst))

def get_top_parses(crel2maxprobs, threshold=0.5):
    top_parse = [crel for crel, prob in crel2maxprobs.items() if prob >= threshold]
    if top_parse:
        return [()] + [to_parse(top_parse)]
    else:
        return [()]

def get_top_n_parses(crel2maxprobs, top_n):
    top_parses = [()]
    by_prob = sorted(crel2maxprobs.keys(), key = lambda k: -crel2maxprobs[k])
    for i in range(1, min(top_n, len(crel2maxprobs))+1):
        parse = by_prob[:i]
        top_parses.append(to_parse(parse))
    return top_parses

def sample_top_parses(crel2maxprobs, top_n):
    max_parses = 2 ** len(crel2maxprobs)  # maximum parse combinations
    assert max_parses > top_n, (max_parses, top_n)  # otherwise brute force it

    top_parses = {()}  # always seed with the empty parse
    while len(top_parses) < top_n:
        new_parse = []
        for crel, prob in crel2maxprobs.items():
            rand_val = np.random.random()  # random number >= 0 and < 1
            if rand_val < prob:
                new_parse.append(crel)
        # make hashable and enforce consistent order
        top_parses.add(to_parse(new_parse))

    return list(top_parses)

def get_top_n_parses2(crel2maxprobs, top_n):
    top_parses = [()]
    by_prob = sorted(crel2maxprobs.keys(), key=lambda k: -crel2maxprobs[k])
    num_predicted = len([crel for crel in by_prob if crel2maxprobs[crel] >= 0.5])
    for i in range(num_predicted - 1, len(by_prob) + 1):
        parse = by_prob[:i]
        top_parses.append(to_parse(parse))
        if len(top_parses) > top_n:
            break
    return top_parses