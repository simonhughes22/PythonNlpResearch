from typing import List, Dict, Tuple, Set


class FeatureExtractor(object):
    def __init__(self, extractors):
        self.extractors = extractors

    def extract(self, predicted_tag, word_seq, positive_val=1):
        """
        word: str
        word_seq: List[Word]

        returns: List[Dict{str,float}]
        """
        fts = dict()
        for ext in self.extractors:
            new_feats = ext(predicted_tag, word_seq, positive_val)
            fts.update(new_feats)
        return fts


def bag_of_word_extractor(predicted_tag, word_seq, positive_val):
    feats = {}
    for word in word_seq:
        feats["bow_" + word] = positive_val
    return feats


def bag_of_word_plus_tag_extractor(predicted_tag, word_seq, positive_val):
    feats = {}
    for word in word_seq:
        feats["bow_" + word + "_tag_" + predicted_tag] = positive_val
    return feats

def prefix_feats(prefix, feats):
    fts = dict()
    if type(feats) == dict:
        for ft,val in feats.items():
            fts[prefix + ":" + ft] = val
    elif type(feats) == list:
        for ft in feats:
            fts[prefix + ":" + ft] = 1
    else:
        raise Exception("Can't handle feats type")
    return fts

def list2feat_dict(lst, positive_val=1):
    fts = {}
    for wd in lst:
        fts[wd] = positive_val
    return fts

class NonLocalFeatureExtractor(object):
    def __init__(self, extractors):
        self.extractors = extractors

    def extract(self, stack_tags, buffer_tags, tag2word_seq, between_word_seq, positive_val=1):
        fts = dict()
        for ext in self.extractors:
            new_feats = ext(stack_tags, buffer_tags, tag2word_seq, between_word_seq, positive_val)
            fts.update(new_feats)
        return fts

def get_sequence(prefix: str, words: List[str], positive_val: int = 1)->Dict[str,int]:
    feats = {}
    for item in words:
        feats["{prefix}_{item}".format(prefix=prefix, item=str(item))] = positive_val
    return feats

def add_wp_combos(prefix:str, feats:Dict[str,int], tag: Tuple[str,int], tag2word_seq: Dict[Tuple[str,int], List[str]], positive_val:int)->None:
    str_tag = str(tag)
    feats[prefix + "p" + str_tag] = positive_val
    tag_word_seq = tag2word_seq[tag]

    feats.update(get_sequence(prefix=prefix + "w", words=tag_word_seq, positive_val=positive_val))
    feats.update(get_sequence(prefix=prefix + "wp_" + str_tag, words=tag_word_seq, positive_val=positive_val))

def get_interactions(prefix:str, fts1:Dict[str, int], fts2:Dict[str, int], positive_val:int)->Dict[str, int]:
    interactions = {}
    for fta, vala in fts1.items():
        for ftb, valb in fts2.items():
            if fta <= ftb and vala > 0 and valb > 0:
                interactions[prefix + "_" + fta + "_" + ftb] = positive_val
    return interactions

def single_words(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 left_relations: Dict[Tuple[str,int], Set[str]],
                 right_relations: Dict[Tuple[str,int], Set[str]],
                 positive_val: int)->Dict[str,int]:
    feats = {}
    if len(stack_tags) > 0:
        s0p = stack_tags[-1]
        add_wp_combos(prefix="S0", feats=feats, tag=s0p, tag2word_seq=tag2word_seq, positive_val=positive_val)

    buffer_len = len(buffer_tags)
    if buffer_len > 0:
        n0 = buffer_tags[0]
        add_wp_combos(prefix="N0", feats=feats, tag=n0, tag2word_seq=tag2word_seq, positive_val=positive_val)
        if buffer_len > 1:
            n1 = buffer_tags[1]
            add_wp_combos(prefix="N1", feats=feats, tag=n1, tag2word_seq=tag2word_seq, positive_val=positive_val)
            if buffer_len > 2:
                n2 = buffer_tags[2]
                add_wp_combos(prefix="N2", feats=feats, tag=n2, tag2word_seq=tag2word_seq, positive_val=positive_val)
    return feats

def word_pairs(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 left_relations: Dict[Tuple[str,int], Set[str]],
                 right_relations: Dict[Tuple[str,int], Set[str]],
                 positive_val: int)->Dict[str,int]:

    feats = {}
    stack_len  = len(stack_tags)
    buffer_len = len(buffer_tags)

    if buffer_len == 0 or stack_len == 0:
        return feats

    s0p = stack_tags[-1]
    str_s0p = str(s0p)

    n0p = buffer_tags[0]
    str_n0p = str(n0p)

    s0w =  get_sequence(prefix="S0w", words=tag2word_seq[s0p], positive_val=positive_val)
    s0wp = get_sequence(prefix="S0wp_" + str_s0p, words=tag2word_seq[s0p], positive_val=positive_val)

    n0w =  get_sequence(prefix="N0w", words=tag2word_seq[n0p], positive_val=positive_val)
    n0wp = get_sequence(prefix="N0wp_" + str_n0p, words=tag2word_seq[n0p], positive_val=positive_val)

    # List (see paper)
    feats.update(get_interactions("S0wp;N0wp;", s0wp, n0wp, positive_val=positive_val))
    feats.update(get_interactions("S0wp;N0w;", s0wp, n0w,   positive_val=positive_val))
    feats.update(get_interactions("S0w;N0wp;", s0w,  n0wp,  positive_val=positive_val))
    feats.update(prefix_feats(prefix="S0wp;N0p_" + str_n0p, feats=s0wp))
    feats.update(prefix_feats(prefix="S0p;N0wp_" + str_s0p, feats=n0wp))
    feats.update(get_interactions("S0w;N0w;", s0w, n0w, positive_val=positive_val))
    feats["S0p;N0p;_" + str_s0p + str_n0p] = positive_val

    if buffer_len > 1:
        n1p = buffer_tags[1]
        str_n1p = str(n1p)
        feats["N0p;N1p_" + str_n0p + str_n1p] = positive_val

    return feats

def three_words(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 left_relations: Dict[Tuple[str,int], Set[str]],
                 right_relations: Dict[Tuple[str,int], Set[str]],
                 positive_val: int)->Dict[str,int]:

    feats = {}
    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    if stack_len == 0 or buffer_len < 1:
        return feats

    s0p = stack_tags[-1]
    str_s0p = str(s0p)

    n0p = buffer_tags[0]
    str_n0p = str(n0p)

    n1p = buffer_tags[1]
    str_n1p = str(n1p)

    feats["S0p;N0p;N1p_" + str_s0p + str_n0p + str_n1p] = positive_val

    if buffer_len > 1:
        n2p = buffer_tags[2]
        str_n2p = str(n2p)

        feats["N0p;N1p;N2p_" + str_n0p + str_n1p + str_n2p] = positive_val

    return feats


def distance(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 left_relations: Dict[Tuple[str,int], Set[str]],
                 right_relations: Dict[Tuple[str,int], Set[str]],
                 positive_val: int)->Dict[str,int]:

    feats = {}
    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    str_dist = str(len(between_word_seq))

    if stack_len > 0:

        s0p = stack_tags[-1]
        str_s0p = str(s0p)

        s0w = get_sequence(prefix="S0w", words=tag2word_seq[s0p], positive_val=positive_val)
        feats['S0pd_' + str_s0p + str_dist] = positive_val
        feats.update(prefix_feats(prefix="S0wd_" + str_dist, feats=s0w))

    if buffer_len > 0:

        n0p = buffer_tags[0]
        str_n0p = str(n0p)

        n0w = get_sequence(prefix="N0w", words=tag2word_seq[n0p], positive_val=positive_val)
        feats['N0pd_' + str_n0p + str_dist] = positive_val

        n0wd = prefix_feats(prefix="N0wd_" + str_dist, feats=n0w)
        feats.update(n0wd)

    if buffer_len > 0 and stack_len > 0:
        feats.update(get_interactions("S0w;N0wd;", s0w, n0wd, positive_val=positive_val))
        feats['S0pN0pd_' + str_s0p + str_n0p + str_dist] = positive_val

    return feats

def valency(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
            tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
            left_relations: Dict[Tuple[str, int], Set[str]],
            right_relations: Dict[Tuple[str,int], Set[str]],
            positive_val: int)->Dict[str,int]:

    feats = {}
    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    if stack_len > 0:
        s0p = stack_tags[-1]
        str_s0p = str(s0p)

        str_s0vl = str(len(left_relations[s0p]))
        str_s0vr = str(len(right_relations[s0p]))

        s0w = get_sequence(prefix="S0w", words=tag2word_seq[s0p], positive_val=positive_val)
        feats.update(prefix_feats(prefix="S0wVl_" + str_s0vl, feats=s0w))
        feats.update(prefix_feats(prefix="S0wVr_" + str_s0vr, feats=s0w))

        feats['S0pVr_' + str_s0p + str_s0vr] = positive_val
        feats['S0pVl_' + str_s0p + str_s0vl] = positive_val

    if buffer_len > 0:
        n0p = buffer_tags[0]
        str_n0p = str(n0p)

        str_n0vl = str(len(left_relations[n0p]))
        n0w = get_sequence(prefix="N0w", words=tag2word_seq[n0p], positive_val=positive_val)
        feats.update(prefix_feats(prefix="N0wVl_" + str_n0vl, feats=n0w))
        feats['N0pVl_' + str_n0p + str_n0vl] = positive_val

    return feats