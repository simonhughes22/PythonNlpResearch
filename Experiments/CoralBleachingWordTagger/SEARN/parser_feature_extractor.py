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

def single_words(stack_tags, buffer_tags, tag2word_seq, between_word_seq, positive_val):
    feats = {}
    if len(stack_tags) > 0:
        s0 = stack_tags[-1]

    buffer_len = len(buffer_tags)
    if buffer_len > 0:
        n0 = buffer_tags[0]
        n0_wds = tag2word_seq[n0]
        n0_wdfts = list2feat_dict(n0_wds)
        feats["n0 WP"] = prefix_feats(str(n0), n0_wdfts)
        feats["n0 W"] =
        if buffer_len > 1:
            pass


