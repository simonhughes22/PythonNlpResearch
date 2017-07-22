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
