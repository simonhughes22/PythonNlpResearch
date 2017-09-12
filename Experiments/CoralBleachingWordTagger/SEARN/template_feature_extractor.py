from typing import List, Dict, Tuple, Set

from NgramGenerator import compute_ngrams

# NOTE: These template features are based on the list on p2 on http://www.aclweb.org/anthology/P11-2033
class NgramExtractor(object):
    def __init__(self, max_ngram_len):
        self.max_ngram_len = max_ngram_len

    def extract(self, words: List[str])->List[str]:
        ngrams = compute_ngrams(tokens=words, max_len=self.max_ngram_len, min_len=1) # type: List[List[str]]
        return [("--".join(ngram)).lower() for ngram in ngrams]

""" Template Feature Extractor """
class NonLocalTemplateFeatureExtractor(object):
    def __init__(self, extractors):
        self.extractors = extractors

    def extract(self, stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 distance: int,
                 left_relations: Dict[Tuple[str,int], Set[str]],
                 right_relations: Dict[Tuple[str,int], Set[str]],
                 positive_val: int)->Dict[str,int]:

        fts = dict()
        for ext in self.extractors:
            new_feats = ext(stack_tags, buffer_tags, tag2word_seq, between_word_seq, distance, left_relations, right_relations, positive_val)
            fts.update(new_feats.items())
        return fts

""" Template Features """

def single_words(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 distance: int,
                 left_relations: Dict[Tuple[str,int], Set[str]],
                 right_relations: Dict[Tuple[str,int], Set[str]],
                 positive_val: int)->Dict[str,int]:
    feats = {}
    if len(stack_tags) > 0:
        s0p = stack_tags[-1]
        __add_wp_combos_(prefix="S0", feats=feats, tag_pair=s0p, tag2word_seq=tag2word_seq, positive_val=positive_val)

    buffer_len = len(buffer_tags)
    if buffer_len > 0:
        n0 = buffer_tags[0]
        __add_wp_combos_(prefix="N0", feats=feats, tag_pair=n0, tag2word_seq=tag2word_seq, positive_val=positive_val)
        if buffer_len > 1:
            n1 = buffer_tags[1]
            __add_wp_combos_(prefix="N1", feats=feats, tag_pair=n1, tag2word_seq=tag2word_seq, positive_val=positive_val)
            if buffer_len > 2:
                n2 = buffer_tags[2]
                __add_wp_combos_(prefix="N2", feats=feats, tag_pair=n2, tag2word_seq=tag2word_seq, positive_val=positive_val)
    return feats

def word_pairs(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 distance: int,
                 left_relations: Dict[Tuple[str,int], Set[str]],
                 right_relations: Dict[Tuple[str,int], Set[str]],
                 positive_val: int)->Dict[str,int]:

    feats = {}
    stack_len  = len(stack_tags)
    buffer_len = len(buffer_tags)

    if buffer_len == 0 or stack_len == 0:
        return feats

    s0p = stack_tags[-1]
    str_s0p = str(s0p[0])

    n0p = buffer_tags[0]
    str_n0p = str(n0p[0])

    s0w =  __get_sequence_(prefix="S0w", words=tag2word_seq[s0p], positive_val=positive_val)
    s0wp = __get_sequence_(prefix="S0wp_" + str_s0p, words=tag2word_seq[s0p], positive_val=positive_val)

    n0w =  __get_sequence_(prefix="N0w", words=tag2word_seq[n0p], positive_val=positive_val)
    n0wp = __get_sequence_(prefix="N0wp_" + str_n0p, words=tag2word_seq[n0p], positive_val=positive_val)

    # List (see paper)
    feats.update(__get_interactions_("S0wp;N0wp;", s0wp, n0wp, positive_val=positive_val))
    feats.update(__get_interactions_("S0wp;N0w;", s0wp, n0w, positive_val=positive_val))
    feats.update(__get_interactions_("S0w;N0wp;", s0w, n0wp, positive_val=positive_val))
    feats.update(__prefix_feats_(prefix="S0wp;N0p_" + str_n0p, feats=s0wp))
    feats.update(__prefix_feats_(prefix="S0p;N0wp_" + str_s0p, feats=n0wp))
    feats.update(__get_interactions_("S0w;N0w;", s0w, n0w, positive_val=positive_val))
    feats["S0p;N0p;_" + str_s0p + "_" +  str_n0p] = positive_val

    if buffer_len > 1:
        n1p = buffer_tags[1]
        str_n1p = str(n1p[0])
        feats["N0p;N1p_" + str_n0p + "_" + str_n1p] = positive_val

    return feats

def three_words(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 distance: int,
                 left_relations: Dict[Tuple[str,int], Set[str]],
                 right_relations: Dict[Tuple[str,int], Set[str]],
                 positive_val: int)->Dict[str,int]:

    feats = {}
    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    if stack_len == 0 or buffer_len < 2:
        return feats

    s0p = stack_tags[-1]
    str_s0p = str(s0p[0])

    n0p = buffer_tags[0]
    str_n0p = str(n0p[0])

    n1p = buffer_tags[1]
    str_n1p = str(n1p[0])

    feats["S0p;N0p;N1p_" + str_s0p + "_" + str_n0p + "_" +  str_n1p] = positive_val

    if buffer_len > 2:
        n2p = buffer_tags[2]
        str_n2p = str(n2p[0])

        feats["N0p;N1p;N2p_" + str_n0p + "_" + str_n1p + "_" + str_n2p] = positive_val

    return feats


def word_distance(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                  tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                  distance: int,
                  left_relations: Dict[Tuple[str,int], Set[str]],
                  right_relations: Dict[Tuple[str,int], Set[str]],
                  positive_val: int)->Dict[str,int]:

    feats = {}
    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    str_dist = str(distance)
    feats["dist_" + str_dist] = positive_val

    s0w, n0wd, str_s0p, str_n0p = None, None, None, None
    if stack_len > 0:

        s0p = stack_tags[-1]
        str_s0p = str(s0p[0])

        s0w = __get_sequence_(prefix="S0w", words=tag2word_seq[s0p], positive_val=positive_val)
        feats['S0pd_' + str_s0p + "_" + str_dist] = positive_val
        feats.update(__prefix_feats_(prefix="S0wd_" + str_dist, feats=s0w))

    if buffer_len > 0:

        n0p = buffer_tags[0]
        str_n0p = str(n0p[0])

        n0w = __get_sequence_(prefix="N0w", words=tag2word_seq[n0p], positive_val=positive_val)
        feats['N0pd_' + str_n0p + "_" + str_dist] = positive_val

        n0wd = __prefix_feats_(prefix="N0wd_" + str_dist, feats=n0w)
        feats.update(n0wd)

    if buffer_len > 0 and stack_len > 0:
        feats.update(__get_interactions_("S0w;N0wd;", s0w, n0wd, positive_val=positive_val))
        feats['S0pN0pd_' + str_s0p + "_" + str_n0p + "_" + str_dist] = positive_val

    return feats

def valency(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
            tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
            distance: int,
            left_relations: Dict[Tuple[str, int], Set[str]],
            right_relations: Dict[Tuple[str,int], Set[str]],
            positive_val: int)->Dict[str,int]:

    feats = {}
    if len(left_relations) == 0 and len(right_relations) == 0:
        return feats

    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    if stack_len > 0:
        s0p = stack_tags[-1]
        str_s0p = str(s0p[0])

        str_s0vl = str(len(left_relations[s0p]))
        str_s0vr = str(len(right_relations[s0p]))

        s0w = __get_sequence_(prefix="S0w", words=tag2word_seq[s0p], positive_val=positive_val)
        feats.update(__prefix_feats_(prefix="S0wVl_" + str_s0vl, feats=s0w))
        feats.update(__prefix_feats_(prefix="S0wVr_" + str_s0vr, feats=s0w))

        feats['S0pVr_' + str_s0p + "_" + str_s0vr] = positive_val
        feats['S0pVl_' + str_s0p + "_" + str_s0vl] = positive_val

    if buffer_len > 0:
        n0p = buffer_tags[0]
        str_n0p = str(n0p[0])

        str_n0vl = str(len(left_relations[n0p]))
        n0w = __get_sequence_(prefix="N0w", words=tag2word_seq[n0p], positive_val=positive_val)
        feats.update(__prefix_feats_(prefix="N0wVl_" + str_n0vl, feats=n0w))
        feats['N0pVl_' + str_n0p + "_" + str_n0vl] = positive_val

    return feats

def between_word_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
            tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
            distance: int,
            left_relations: Dict[Tuple[str, int], Set[str]],
            right_relations: Dict[Tuple[str, int], Set[str]],
            positive_val: int) -> Dict[str, int]:

    feats = {}
    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    btwn_wd_fts = __prefix_feats_(prefix="btW_", feats=between_word_seq)
    feats.update(btwn_wd_fts)

    str_s0p, str_n0p = "",""
    if stack_len > 0:
        s0p = stack_tags[-1]
        str_s0p = str(s0p[0])
        feats.update(__prefix_feats_(prefix="S0p_" + str_s0p, feats=btwn_wd_fts))

    if buffer_len > 0:
        n0p = buffer_tags[0]
        str_n0p = str(n0p[0])
        feats.update(__prefix_feats_(prefix="N0p_" + str_n0p, feats=btwn_wd_fts))

    if buffer_len > 0 and stack_len > 0:
        feats.update(__prefix_feats_(prefix="S0p;N0p_" + str_s0p + "_" + str_n0p, feats=btwn_wd_fts))

    return feats

""" Template Feature Helpers"""
def __prefix_feats_(prefix, feats):
    fts = dict()
    if type(feats) == dict:
        for ft,val in feats.items():
            fts[prefix + "_" + ft] = val
    elif type(feats) == list:
        for ft in feats:
            fts[prefix + "_" + ft] = 1
    else:
        raise Exception("Can't handle feats type")
    return fts

def __get_sequence_(prefix: str, words: List[str], positive_val: int = 1)->Dict[str, int]:
    feats = {}
    for item in words:
        feats["{prefix}_{item}".format(prefix=prefix, item=str(item))] = positive_val
    return feats

def __add_wp_combos_(prefix:str, feats:Dict[str, int], tag_pair: Tuple[str, int], tag2word_seq: Dict[Tuple[str, int], List[str]], positive_val:int)->None:
    str_tag = str(tag_pair[0])
    feats[prefix + "p" + str_tag] = positive_val
    tag_word_seq = tag2word_seq[tag_pair]

    feats.update(__get_sequence_(prefix=prefix + "w", words=tag_word_seq, positive_val=positive_val))
    feats.update(__get_sequence_(prefix=prefix + "wp_" + str_tag, words=tag_word_seq, positive_val=positive_val))

def __get_interactions_(prefix:str, fts1:Dict[str, int], fts2:Dict[str, int], positive_val:int)->Dict[str, int]:
    interactions = {}
    for fta, vala in fts1.items():
        for ftb, valb in fts2.items():
            if vala > 0 and valb > 0:
                interactions[prefix + "_" + fta + "_" + ftb] = positive_val
    return interactions

