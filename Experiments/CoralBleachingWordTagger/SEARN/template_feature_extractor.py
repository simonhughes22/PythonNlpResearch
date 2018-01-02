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
        self.distinct_feats = set()

    def extract(self,
                stack_tags: List[Tuple[str,int]],
                buffer_tags: List[Tuple[str,int]],
                tag2word_seq: Dict[Tuple[str,int], List[str]],
                between_word_seq: List[str],
                distance: int,
                cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                positive_val: int)->Dict[str,int]:

        fts = dict()
        # Ensure always at least one feature as some partitions error out
        fts["BIAS"] = 1
        for ext in self.extractors:
            new_feats = ext(stack_tags, buffer_tags, tag2word_seq, between_word_seq, distance,
                            cause2effects, effect2causers, positive_val)
            fts.update(new_feats.items())
        # keep track of the number of unique features
        self.distinct_feats.update(fts.keys())
        return fts

    def num_features(self):
        return len(self.distinct_feats)

""" Template Features """

def single_words(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 distance: int,
                 cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                 effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                 positive_val: int)->Dict[str,int]:
    feats = {}
    if len(stack_tags) > 0:
        s0p = stack_tags[-1]
        feats.update(__get_wp_combos_(prefix="S0", tag_pair=s0p, tag2word_seq=tag2word_seq, positive_val=positive_val))

    buffer_len = len(buffer_tags)
    if buffer_len > 0:
        n0 = buffer_tags[0]
        feats.update(__get_wp_combos_(prefix="N0", tag_pair=n0, tag2word_seq=tag2word_seq, positive_val=positive_val))
        if buffer_len > 1:
            n1 = buffer_tags[1]
            feats.update(__get_wp_combos_(prefix="N1", tag_pair=n1, tag2word_seq=tag2word_seq, positive_val=positive_val))
            if buffer_len > 2:
                n2 = buffer_tags[2]
                feats.update(__get_wp_combos_(prefix="N2", tag_pair=n2, tag2word_seq=tag2word_seq, positive_val=positive_val))
    return feats

def word_pairs(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
                 tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
                 distance: int,
                 cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                 effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
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
    feats.update(__prefix_feats_(prefix="S0wp;N0p_" + str_n0p, feats_in=s0wp))
    feats.update(__prefix_feats_(prefix="S0p;N0wp_" + str_s0p, feats_in=n0wp))
    feats.update(__get_interactions_("S0w;N0w;", s0w, n0w, positive_val=positive_val))
    feats["S0p;N0p;_" + str_s0p + "_" +  str_n0p] = positive_val

    if buffer_len > 1:
        n1p = buffer_tags[1]
        str_n1p = str(n1p[0])
        feats["N0p;N1p_" + str_n0p + "_" + str_n1p] = positive_val

    return feats

def three_words(stack_tags:  List[Tuple[str,int]],
                buffer_tags: List[Tuple[str,int]],
                tag2word_seq: Dict[Tuple[str,int], List[str]],
                between_word_seq: List[str],
                distance: int,
                cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                positive_val: int)->Dict[str,int]:

    feats = {}
    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    if stack_len == 0 or buffer_len == 0:
        return feats

    s0p = stack_tags[-1]
    str_s0p = str(s0p[0])

    n0p = buffer_tags[0]
    str_n0p = str(n0p[0])

    s0_left_mods_causer, s0_right_mods_causer = __get_left_right_modifiers__(tag_pair=s0p,   causal_mapping=effect2causers)
    s0_left_mods_effects, s0_right_mods_effects = __get_left_right_modifiers__(tag_pair=s0p, causal_mapping=cause2effects)

    # Combine left modifiers
    s0_left_mods = s0_left_mods_causer.union(s0_left_mods_effects)
    s0_right_mods = s0_right_mods_causer.union(s0_right_mods_effects)

    if s0_left_mods:
        s0lp = min(s0_left_mods, key=lambda tpl: tpl[1])[0]
        feats["S0p;S0lp;N0p_" + str_s0p + "_" + s0lp + "_" + str_n0p] = positive_val

    if s0_right_mods:
        s0rp = max(s0_right_mods, key=lambda tpl: tpl[1])[0]
        feats["S0p;S0rp;N0p_" + str_s0p + "_" + s0rp + "_" + str_n0p] = positive_val

    # Combine causers and effects
    s0_causer_tag_pairs = s0_left_mods_causer.union(s0_right_mods_causer)
    s0_effect_tag_pairs = s0_left_mods_effects.union(s0_right_mods_effects)

    if s0_causer_tag_pairs:
        s0_causer_tags = __tag_pair_to_tags__(s0_causer_tag_pairs)
        feats.update(__prefix_feats_(prefix="s0hp_causer_;S0p;N0p_" + str_s0p + "_" + str_n0p, feats_in=s0_causer_tags))

    if s0_effect_tag_pairs:
        s0_effect_tags = __tag_pair_to_tags__(s0_effect_tag_pairs)
        feats.update(__prefix_feats_(prefix="s0hp_effect_;S0p;N0p_" + str_s0p + "_" + str_n0p, feats_in=s0_effect_tags))

    n0_left_mods_causer_tag_pairs,  _ = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=effect2causers)
    n0_left_mods_effects_tag_pairs, _ = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=cause2effects)

    if n0_left_mods_causer_tag_pairs:
        n0lp = min(n0_left_mods_causer_tag_pairs, key=lambda tpl: tpl[1])[0]
        feats["S0p;N0p;N0lp_causer_" + str_s0p + "_" + str_n0p + "_" + n0lp] = positive_val

    if n0_left_mods_effects_tag_pairs:
        n0lp = min(n0_left_mods_effects_tag_pairs, key=lambda tpl: tpl[1])[0]
        feats["S0p;N0p;N0lp_effect_" + str_s0p + "_" + str_n0p + "_" + n0lp] = positive_val

    if buffer_len > 1:
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
                  cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
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
        feats.update(__prefix_feats_(prefix="S0wd_" + str_dist, feats_in=s0w))

    if buffer_len > 0:

        n0p = buffer_tags[0]
        str_n0p = str(n0p[0])

        n0w = __get_sequence_(prefix="N0w", words=tag2word_seq[n0p], positive_val=positive_val)
        feats['N0pd_' + str_n0p + "_" + str_dist] = positive_val

        n0wd = __prefix_feats_(prefix="N0wd_" + str_dist, feats_in=n0w)
        feats.update(n0wd)

    if buffer_len > 0 and stack_len > 0:
        feats.update(__get_interactions_("S0w;N0wd;", s0w, n0wd, positive_val=positive_val))
        feats['S0pN0pd_' + str_s0p + "_" + str_n0p + "_" + str_dist] = positive_val

    return feats

def valency(stack_tags: List[Tuple[str,int]], buffer_tags: List[Tuple[str,int]],
            tag2word_seq: Dict[Tuple[str,int], List[str]], between_word_seq: List[str],
            distance: int,
            cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
            effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
            positive_val: int)->Dict[str,int]:

    feats = {}
    if len(effect2causers) == 0 and len(cause2effects) == 0:
        return feats

    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    # Compute sO features
    if stack_len > 0:
        s0p = stack_tags[-1]
        # get tag without position
        str_s0p = str(s0p[0])
        s0w = __get_sequence_(prefix="S0w", words=tag2word_seq[s0p], positive_val=positive_val)

        # comment out - include 0 length feats
        #if s0p in cause2effects:

        # get left and right modifiers of s0
        s0_left_mods, s0_right_mods =  __get_left_right_modifiers__(tag_pair=s0p, causal_mapping=cause2effects)

        str_s0vl = str(len(s0_left_mods))
        str_s0vr = str(len(s0_right_mods))

        feats.update(__prefix_feats_(prefix="S0wVlEffects_" + str_s0vl, feats_in=s0w))
        feats.update(__prefix_feats_(prefix="S0wVrEffects_" + str_s0vr, feats_in=s0w))

        feats['S0pVrEffects_' + str_s0p + "_" + str_s0vr] = positive_val
        feats['S0pVlEffects_' + str_s0p + "_" + str_s0vl] = positive_val

        # comment out - include 0 length feats
        #if s0p in effect2causers:
        # get left and right modifiers of s0
        s0_left_mods, s0_right_mods = __get_left_right_modifiers__(tag_pair=s0p, causal_mapping=effect2causers)

        str_s0vl = str(len(s0_left_mods))
        str_s0vr = str(len(s0_right_mods))

        feats.update(__prefix_feats_(prefix="S0wVlCauses_" + str_s0vl, feats_in=s0w))
        feats.update(__prefix_feats_(prefix="S0wVrCauses_" + str_s0vr, feats_in=s0w))

        feats['S0pVrCauses_' + str_s0p + "_" + str_s0vr] = positive_val
        feats['S0pVlCauses_' + str_s0p + "_" + str_s0vl] = positive_val

    if buffer_len > 0:

        n0p = buffer_tags[0]
        str_n0p = str(n0p[0])
        n0w = __get_sequence_(prefix="N0w", words=tag2word_seq[n0p], positive_val=positive_val)

        # comment out - include 0 length feats
        #if n0p in cause2effects:
        n0_left_mods, n0_right_mods = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=cause2effects)

        str_n0vl = str(len(n0_left_mods))
        str_n0vr = str(len(n0_right_mods))

        feats.update(__prefix_feats_(prefix="N0wVlEffects_" + str_n0vl, feats_in=n0w))
        feats.update(__prefix_feats_(prefix="N0wVrEffects_" + str_n0vr, feats_in=n0w))

        feats['N0pVrEffects_' + str_n0p + "_" + str_n0vr] = positive_val
        feats['N0pVlEffects_' + str_n0p + "_" + str_n0vl] = positive_val

        # comment out - include 0 length feats
        #if n0p in effect2causers:
        n0_left_mods, n0_right_mods = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=effect2causers)

        str_n0vl = str(len(n0_left_mods))
        str_n0vr = str(len(n0_right_mods))

        feats.update(__prefix_feats_(prefix="N0wVlCauses_" + str_n0vl, feats_in=n0w))
        feats.update(__prefix_feats_(prefix="N0wVrCauses_" + str_n0vr, feats_in=n0w))

        feats['N0pVrCauses_' + str_n0p + "_" + str_n0vr] = positive_val
        feats['N0pVlCauses_' + str_n0p + "_" + str_n0vl] = positive_val

    return feats

def unigrams(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
            tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
            distance: int,
            cause2effects:  Dict[Tuple[str, int], Set[Tuple[str, int]]],
            effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
            positive_val: int) -> Dict[str, int]:

    feats = {}
    if len(effect2causers) == 0 and len(cause2effects) == 0:
        return feats

    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    # Compute sO features
    if stack_len > 0:
        s0p = stack_tags[-1]
        # get tag without position
        # get left and right modifiers of s0
        # returns 2 sets of tags (str's)

        s0_left_mods_causer, s0_right_mods_causer   = __get_left_right_modifiers__(tag_pair=s0p, causal_mapping=effect2causers)
        s0_left_mods_effects, s0_right_mods_effects = __get_left_right_modifiers__(tag_pair=s0p, causal_mapping=cause2effects)

        # Combine left modifiers
        s0_left_mods = s0_left_mods_causer.union(s0_left_mods_effects)
        s0_right_mods = s0_right_mods_causer.union(s0_right_mods_effects)

        # Combine causers and effects
        s0_causer_tag_pairs = s0_left_mods_causer.union(s0_right_mods_causer)
        s0_effect_tag_pairs = s0_left_mods_effects.union(s0_right_mods_effects)

        # Unigram s0 feats
        """
        S0hw; S0hp; S0l; S0lw; S0lp; S0ll; S0rw; S0rp; S0rl;
        """
        # in place of s0_hw as we don't have a head and modifier here
        __add_word_tag_labels_for_tag_pairs_unigram__(feats=feats,
                                                      prefix="s0_causer", head_tag_pair=s0p, tag_pairs=s0_causer_tag_pairs,
                                                      cause2effects=cause2effects, effect2causers=effect2causers, tag2word_seq=tag2word_seq)

        __add_word_tag_labels_for_tag_pairs_unigram__(feats=feats,
                                                      prefix="s0_effect", head_tag_pair=s0p, tag_pairs=s0_effect_tag_pairs,
                                                      cause2effects=cause2effects, effect2causers=effect2causers, tag2word_seq=tag2word_seq)

        # lp and rp
        __add_word_tag_labels_for_tag_pairs_unigram__(feats=feats,
                                                      prefix="s0l", head_tag_pair=s0p, tag_pairs=s0_left_mods,
                                                      cause2effects=cause2effects, effect2causers=effect2causers, tag2word_seq=tag2word_seq)

        __add_word_tag_labels_for_tag_pairs_unigram__(feats=feats,
                                                      prefix="s0r", head_tag_pair=s0p, tag_pairs=s0_right_mods,
                                                      cause2effects=cause2effects, effect2causers=effect2causers, tag2word_seq=tag2word_seq)

    if buffer_len > 0:

        n0p = buffer_tags[0]

        # get left and right modifiers of n0
        # returns 2 sets of tags (str's)

        n0_left_mods_causer, _  = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=effect2causers)
        n0_left_mods_effects, _ = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=cause2effects)

        __add_word_tag_labels_for_tag_pairs_unigram__(feats=feats, prefix="n0_causer", head_tag_pair=n0p,
                                                      tag_pairs=n0_left_mods_causer,
                                                      cause2effects=cause2effects, effect2causers=effect2causers, tag2word_seq=tag2word_seq)

        __add_word_tag_labels_for_tag_pairs_unigram__(feats=feats, prefix="n0_effect", head_tag_pair=n0p,
                                                      tag_pairs=n0_left_mods_effects,
                                                      cause2effects=cause2effects, effect2causers=effect2causers, tag2word_seq=tag2word_seq)
    return feats


def third_order(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                          tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                          distance: int,
                          cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                          effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                          positive_val: int) -> Dict[str, int]:

    feats = {}
    if len(effect2causers) == 0 and len(cause2effects) == 0:
        return feats

    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    # Compute sO features
    if stack_len > 0:
        s0p = stack_tags[-1]

        s0_left_mods_causer, s0_right_mods_causer   = __get_left_right_modifiers__(tag_pair=s0p, causal_mapping=effect2causers)
        s0_left_mods_effects, s0_right_mods_effects = __get_left_right_modifiers__(tag_pair=s0p, causal_mapping=cause2effects)

        # Combine causers and effects
        s0_h_causer_tag_pairs = s0_left_mods_causer.union(s0_right_mods_causer)
        s0_h_effect_tag_pairs = s0_left_mods_effects.union(s0_right_mods_effects)

        """s0 h2"""
        if s0_h_causer_tag_pairs:

            s0_h2_causer = __get_modifiers_of_modifiers__(tag_pairs=s0_h_causer_tag_pairs, causal_mapping=effect2causers)

            #s0_h2w; s0_h2p, s0_h2l
            __add_word_tag_labels_for_tag_pairs_third_order__(feats=feats, prefix="s0_h2_causer_", head_tag_pairs=s0_h_causer_tag_pairs,
                                                          modifier_tag_pairs=s0_h2_causer,
                                                          cause2effects=cause2effects, effect2causers=effect2causers,
                                                          tag2word_seq=tag2word_seq)

            for hp, _ in s0_h_causer_tag_pairs:
                for h2p, _ in s0_h2_causer:
                    key = "s0p;s0hp;s0h2p_causer={s0p};{s0hp};{s0h2p}".format(s0p=s0p[0], s0hp=hp, s0h2p=h2p)
                    feats[key] = positive_val

        if s0_h_effect_tag_pairs:
            s0_h2_effect = __get_modifiers_of_modifiers__(tag_pairs=s0_h_effect_tag_pairs, causal_mapping=cause2effects)
            # s0_h2w; s0_h2p, s0_h2l
            __add_word_tag_labels_for_tag_pairs_third_order__(feats=feats, prefix="s0_h2_effect_", head_tag_pairs=s0_h_effect_tag_pairs,
                                                          modifier_tag_pairs=s0_h2_effect,
                                                          cause2effects=cause2effects, effect2causers=effect2causers,
                                                          tag2word_seq=tag2word_seq)

            for hp, _ in s0_h_effect_tag_pairs:
                for h2p, _ in s0_h2_effect:
                    key = "s0p;s0hp;s0h2p_effect={s0p};{s0hp};{s0h2p}".format(s0p=s0p[0], s0hp=hp, s0h2p=h2p)
                    feats[key] = positive_val


        """s0 l2"""
        if len(s0_left_mods_causer)>=2:
            slst_s0_left_mods_causer = sorted(s0_left_mods_causer, key=lambda tpl: tpl[-1])
            s0_l2p = slst_s0_left_mods_causer[1]
            # s0_l2w; s0_l2p, s0_l2l
            __add_word_tag_labels_for_tag_pairs_third_order__(feats=feats, prefix="s0_l2_causer_", head_tag_pairs={s0p},
                                                          modifier_tag_pairs={s0_l2p},
                                                          cause2effects=cause2effects, effect2causers=effect2causers,
                                                          tag2word_seq=tag2word_seq)

            key = "s0p;s0lp;s0l2p_causer={s0p};{s0lp};{s0l2p}".format(s0p=s0p[0],
                                                s0lp=slst_s0_left_mods_causer[0][0],
                                                s0l2p=slst_s0_left_mods_causer[1][0])
            feats[key] = positive_val

        if len(s0_left_mods_effects)>=2:
            slst_s0_left_mods_effects = sorted(s0_left_mods_effects, key=lambda tpl: tpl[-1])
            s0_l2p = slst_s0_left_mods_effects[1]
            # s0_l2w; s0_l2p, s0_l2l
            __add_word_tag_labels_for_tag_pairs_third_order__(feats=feats, prefix="s0_l2_effect_", head_tag_pairs={s0p},
                                                          modifier_tag_pairs={s0_l2p},
                                                          cause2effects=cause2effects, effect2causers=effect2causers,
                                                          tag2word_seq=tag2word_seq)

            key = "s0p;s0lp;s0l2p_effects={s0p};{s0lp};{s0l2p}".format(s0p=s0p[0],
                                                                      s0lp=slst_s0_left_mods_effects[0][0],
                                                                      s0l2p=slst_s0_left_mods_effects[1][0])
            feats[key] = positive_val


        """s0 r2 """
        if len(s0_right_mods_causer) >= 2:
            slst_s0_right_mods_causer = sorted(s0_right_mods_causer, key=lambda tpl: tpl[-1])
            s0_r2p = slst_s0_right_mods_causer[-2]
            # s0_r2w; s0_r2p, s0_r2l
            __add_word_tag_labels_for_tag_pairs_third_order__(feats=feats, prefix="s0_r2_causer_", head_tag_pairs={s0p},
                                                              modifier_tag_pairs={s0_r2p},
                                                              cause2effects=cause2effects,
                                                              effect2causers=effect2causers,
                                                              tag2word_seq=tag2word_seq)

            key = "s0p;s0rp;s0r2p_causer={s0p};{s0rp};{s0r2p}".format(s0p=s0p[0],
                                                                      s0rp=slst_s0_right_mods_causer[-1][0],
                                                                      s0r2p=slst_s0_right_mods_causer[-2][0])
            feats[key] = positive_val


        if len(s0_right_mods_effects) >= 2:
            slst_s0_right_mods_effects = sorted(s0_right_mods_effects, key=lambda tpl: tpl[-1])
            s0_r2p = slst_s0_right_mods_effects[-2]
            # s0_r2w; s0_r2p, s0_r2l
            __add_word_tag_labels_for_tag_pairs_third_order__(feats=feats, prefix="s0_r2_effect_", head_tag_pairs={s0p},
                                                              modifier_tag_pairs={s0_r2p},
                                                              cause2effects=cause2effects,
                                                              effect2causers=effect2causers,
                                                              tag2word_seq=tag2word_seq)

            key = "s0p;s0rp;s0r2p_effect={s0p};{s0rp};{s0r2p}".format(s0p=s0p[0],
                                                                      s0rp=slst_s0_right_mods_effects[-1][0],
                                                                      s0r2p=slst_s0_right_mods_effects[-2][0])
            feats[key] = positive_val

    if buffer_len > 0:
        n0p = buffer_tags[0]

        n0_left_mods_causer, _  = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=effect2causers)
        n0_left_mods_effects, _ = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=cause2effects)

        """ n0 l2 """
        if len(n0_left_mods_causer) >= 2:
            slst_n0_left_mods_causer = sorted(n0_left_mods_causer, key=lambda tpl: tpl[-1])
            n0_l2p = slst_n0_left_mods_causer[1]
            # n0_l2w; n0_l2p, n0_l2l
            __add_word_tag_labels_for_tag_pairs_third_order__(feats=feats, prefix="n0_l2_causer_", head_tag_pairs={n0p},
                                                              modifier_tag_pairs={n0_l2p},
                                                              cause2effects=cause2effects,
                                                              effect2causers=effect2causers,
                                                              tag2word_seq=tag2word_seq)

            key = "n0p;n0lp;n0l2p_causer={n0p};{n0lp};{n0l2p}".format(n0p=n0p[0],
                                                                      n0lp=slst_n0_left_mods_causer[0][0],
                                                                      n0l2p=slst_n0_left_mods_causer[1][0])
            feats[key] = positive_val

        if len(n0_left_mods_effects) >= 2:
            slst_n0_left_mods_effects = sorted(n0_left_mods_effects, key=lambda tpl: tpl[-1])
            n0_l2p = slst_n0_left_mods_effects[1]
            # n0_l2w; n0_l2p, n0_l2l
            __add_word_tag_labels_for_tag_pairs_third_order__(feats=feats, prefix="n0_l2_effect_", head_tag_pairs={n0p},
                                                              modifier_tag_pairs={n0_l2p},
                                                              cause2effects=cause2effects,
                                                              effect2causers=effect2causers,
                                                              tag2word_seq=tag2word_seq)

            key = "n0p;n0lp;n0l2p_effect={n0p};{n0lp};{n0l2p}".format(n0p=n0p[0],
                                                                      n0lp=slst_n0_left_mods_effects[0][0],
                                                                      n0l2p=slst_n0_left_mods_effects[1][0])
            feats[key] = positive_val

    return feats


def label_set(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                          tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                          distance: int,
                          cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                          effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                          positive_val: int) -> Dict[str, int]:

    feats = dict()


    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    def combine_features(prefix: str, feats: Dict[str,int], labels: Set[str])->Dict[str,int]:
        fts = {}
        for lbl in labels:
            for ft_wp, _ in feats.items():
                fts[prefix + "_wp_" + ft_wp + "_lbl_" + lbl] = positive_val
        return fts

    # Compute sO features
    if stack_len > 0:
        s0p = stack_tags[-1]
        # get tag without position
        # get left and right modifiers of s0
        # returns 2 sets of tags (str's)

        s0_left_mods_causer, s0_right_mods_causer = __get_left_right_modifiers__(tag_pair=s0p,
                                                                                 causal_mapping=effect2causers)
        s0_left_mods_effects, s0_right_mods_effects = __get_left_right_modifiers__(tag_pair=s0p,
                                                                                   causal_mapping=cause2effects)
        #__add_word_tag_pairs_to_set_of_tag_pairs__

        s0_wAndP = __add_word_tag_pairs_to_set_of_tag_pairs__(feats={}, prefix="s0_ls_", tag2word_seq=tag2word_seq, tag_pairs={s0p})
        lr_causer = __get_labels_for_tag_pairs__(tag_pairs=s0_left_mods_causer,
                                                 cause2effects=cause2effects,
                                                 effect2causers=effect2causers)

        feats.update(combine_features(prefix="label_set_sl_causer_", feats=s0_wAndP, labels=lr_causer))

        rr_causer = __get_labels_for_tag_pairs__(tag_pairs=s0_right_mods_causer,
                                                 cause2effects=cause2effects,
                                                 effect2causers=effect2causers)

        feats.update(combine_features(prefix="label_set_sr_causer_", feats=s0_wAndP, labels=rr_causer))

        lr_effect = __get_labels_for_tag_pairs__(tag_pairs=s0_left_mods_effects,
                                                 cause2effects=cause2effects,
                                                 effect2causers=effect2causers)

        feats.update(combine_features(prefix="label_set_sl_effect_", feats=s0_wAndP, labels=lr_effect))

        rr_effect = __get_labels_for_tag_pairs__(tag_pairs=s0_right_mods_effects,
                                                 cause2effects=cause2effects,
                                                 effect2causers=effect2causers)

        feats.update(combine_features(prefix="label_set_sr_effect_", feats=s0_wAndP, labels=rr_effect))

    if buffer_len > 0:
        n0p = buffer_tags[0]

        # get left and right modifiers of n0
        # returns 2 sets of tags (str's)

        n0_left_mods_causer, _ = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=effect2causers)
        n0_left_mods_effects, _ = __get_left_right_modifiers__(tag_pair=n0p, causal_mapping=cause2effects)

        n0_wAndP = __add_word_tag_pairs_to_set_of_tag_pairs__(feats={}, prefix="n0_ls_", tag2word_seq=tag2word_seq, tag_pairs={n0p})
        lr_causer = __get_labels_for_tag_pairs__(tag_pairs=n0_left_mods_causer,
                                                 cause2effects=cause2effects,
                                                 effect2causers=effect2causers)

        feats.update(combine_features(prefix="label_set_sl_causer_", feats=n0_wAndP, labels=lr_causer))

        lr_effect = __get_labels_for_tag_pairs__(tag_pairs=n0_left_mods_effects,
                                                 cause2effects=cause2effects,
                                                 effect2causers=effect2causers)

        feats.update(combine_features(prefix="label_set_sl_effect_", feats=n0_wAndP, labels=lr_effect))

    return feats


def between_word_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                          tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                          distance: int,
                          cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                          effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                          positive_val: int) -> Dict[str, int]:

    feats = {}
    stack_len = len(stack_tags)
    buffer_len = len(buffer_tags)

    btwn_wd_fts = __prefix_feats_(prefix="btW_", feats_in=between_word_seq)
    feats.update(btwn_wd_fts)

    str_s0p, str_n0p = "",""
    if stack_len > 0:
        s0p = stack_tags[-1]
        str_s0p = str(s0p[0])
        feats.update(__prefix_feats_(prefix="S0p_" + str_s0p, feats_in=btwn_wd_fts))

    if buffer_len > 0:
        n0p = buffer_tags[0]
        str_n0p = str(n0p[0])
        feats.update(__prefix_feats_(prefix="N0p_" + str_n0p, feats_in=btwn_wd_fts))

    if buffer_len > 0 and stack_len > 0:
        feats.update(__prefix_feats_(prefix="S0p;N0p_" + str_s0p + "_" + str_n0p, feats_in=btwn_wd_fts))

    return feats

""" Template Feature Helpers"""
def __prefix_feats_(prefix, feats_in, positive_val = 1):
    fts_out = dict()
    if type(feats_in) == dict:
        for ft,val in feats_in.items():
            fts_out[prefix + "_" + ft] = val
    elif type(feats_in) == list or type(feats_in) == set:
        for ft in feats_in:
            fts_out[prefix + "_" + ft] = positive_val
    else:
        raise Exception("Can't handle feats type")
    return fts_out

def __get_sequence_(prefix: str, words: List[str], positive_val: int = 1, fts: Dict[str,int] = None)->Dict[str, int]:
    if not fts:
        fts = {}
    for item in words:
        fts["{prefix}_{item}".format(prefix=prefix, item=str(item))] = positive_val
    return fts

def __get_wp_combos_(prefix: str,
                     tag_pair: Tuple[str, int],
                     tag2word_seq: Dict[Tuple[str, int], List[str]],
                     positive_val: int, fts: Dict[str,int] = None)-> Dict[str,int]:
    if not fts:
        fts = {}

    str_tag = str(tag_pair[0])
    tag_word_seq = tag2word_seq[tag_pair]

    fts[prefix + "p" + str_tag] = positive_val
    __get_sequence_(prefix=prefix + "w", words=tag_word_seq, positive_val=positive_val, fts=fts)
    __get_sequence_(prefix=prefix + "wp_" + str_tag, words=tag_word_seq, positive_val=positive_val, fts=fts)
    return fts

def __get_interactions_(prefix:str, fts1:Dict[str, int], fts2:Dict[str, int], positive_val:int)->Dict[str, int]:
    interactions = {}
    for fta, vala in fts1.items():
        for ftb, valb in fts2.items():
            if vala > 0 and valb > 0:
                interactions[prefix + "_" + fta + "_" + ftb] = positive_val
    return interactions

def __tag_pair_to_tags__(tag_pairs: Set[Tuple[str, int]])->Set[str]:
    tags = set()
    for tag, posn in tag_pairs:
        tags.add(tag)
    return tags

def __get_left_right_modifiers__(tag_pair: Tuple[str, int],
                                 causal_mapping: Dict[Tuple[str, int], Set[Tuple[str, int]]]
                                 )->Tuple[Set[Tuple[str,int]], Set[Tuple[str,int]]]:

    # prevent insertion of key by default dict
    if tag_pair not in causal_mapping:
        return set(), set()

    modifiers = causal_mapping[tag_pair]
    if not modifiers:
        return set(), set()

    tag, tag_posn = tag_pair
    left_mods, right_mods = set(), set()
    for modifier_tag_pair in modifiers:
        mod_tag, mod_posn = modifier_tag_pair
        # use the tags and not the tag positions
        if mod_posn < tag_posn:
            left_mods.add(modifier_tag_pair)
        elif mod_posn > tag_posn:
            right_mods.add(modifier_tag_pair)

    return left_mods, right_mods

# For third order modifiers
def __get_left_right_modifiers_of_modifiers__(
        tag_pairs: Set[Tuple[str, int]],
        causal_mapping: Dict[Tuple[str, int], Set[Tuple[str, int]]]
        )->Tuple[Set[Tuple[str,int]], Set[Tuple[str,int]]]:

    # prevent insertion of key by default dict
    if not tag_pairs:
        return set(), set()

    left_mods, right_mods = set(), set()
    for tag_pair in tag_pairs:
        if tag_pair not in causal_mapping:
            continue

        modifiers = causal_mapping[tag_pair]
        if not modifiers:
            continue

        tag, tag_posn = tag_pair
        for modifier_tag_pair in modifiers:
            mod_tag, mod_posn = modifier_tag_pair
            # use the tags and not the tag positions
            if mod_posn < tag_posn:
                left_mods.add(modifier_tag_pair)
            elif mod_posn > tag_posn:
                right_mods.add(modifier_tag_pair)

    return left_mods, right_mods

def __get_modifiers_of_modifiers__(
        tag_pairs: Set[Tuple[str, int]],
        causal_mapping: Dict[Tuple[str, int], Set[Tuple[str, int]]]
        )->Set[Tuple[str,int]]:

    # prevent insertion of key by default dict
    if not tag_pairs:
        return set()

    mods = set()
    for tag_pair in tag_pairs:
        if tag_pair in causal_mapping:
            mods.update(causal_mapping[tag_pair])
    return mods

def __get_words_from_tags__(tag_pairs: Set[Tuple[str, int]], tag2word_seq: Dict[Tuple[str, int], List[str]])->Set[str]:
    sequence = []   # type:List[str]
    for tag_pair in tag_pairs:
        tp_seq = tag2word_seq[tag_pair] # type:List[str]
        sequence.extend(tp_seq)
    return set(sequence)

def __add_word_tag_pairs_to_set_of_tag_pairs__(feats, prefix, tag2word_seq, tag_pairs):
    words = __get_words_from_tags__(tag_pairs=tag_pairs, tag2word_seq=tag2word_seq)
    feats.update(__prefix_feats_(prefix=prefix + "_w", feats_in=words))
    tags = __tag_pair_to_tags__(tag_pairs=tag_pairs)
    feats.update(__prefix_feats_(prefix=prefix + "_p", feats_in=tags))
    return feats

def __add_word_tag_labels_for_tag_pairs_unigram__(
        feats : Dict[str, int],
        prefix: str,
        head_tag_pair: Tuple[str,int],
        tag_pairs: Set[Tuple[str, int]],
        cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
        effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
        tag2word_seq: Dict[Tuple[str, int], List[str]]
        )->None:

    __add_word_tag_pairs_to_set_of_tag_pairs__(feats, prefix, tag2word_seq, tag_pairs)

    labels = set()
    for tag_pair in tag_pairs:
        if tag_pair in cause2effects:
            labels.add(format("{causer}->{effect}".format(causer=tag_pair[0], effect=head_tag_pair[0])))
        else:
            labels.add(format("{causer}->{effect}".format(causer=head_tag_pair[0], effect=tag_pair[0])))
    feats.update(__prefix_feats_(prefix=prefix+"_l", feats_in=labels))

def __add_word_tag_labels_for_tag_pairs_third_order__(
        feats : Dict[str, int],
        prefix: str,
        head_tag_pairs: Set[Tuple[str,int]],
        modifier_tag_pairs: Set[Tuple[str, int]],
        cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
        effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
        tag2word_seq: Dict[Tuple[str, int], List[str]]
        )->None:

    __add_word_tag_pairs_to_set_of_tag_pairs__(feats, prefix, tag2word_seq, modifier_tag_pairs)

    labels = set()
    for ht_pair in head_tag_pairs:
        if ht_pair in cause2effects:
            effects = cause2effects[ht_pair]
            intersectn = effects.intersection(modifier_tag_pairs)
            for effect_tag_pair in intersectn:
                labels.add(format("{causer}->{effect}".format(causer=ht_pair[0], effect=effect_tag_pair[0])))
        elif ht_pair in effect2causers:
            causers = effect2causers[ht_pair]
            intersectn = causers.intersection(modifier_tag_pairs)
            for causer_tag_pair in intersectn:
                labels.add(format("{causer}->{effect}".format(causer=causer_tag_pair [0], effect=ht_pair[0])))
    if labels:
        feats.update(__prefix_feats_(prefix=prefix+"_l", feats_in=labels))

def __get_labels_for_tag_pairs__(tag_pairs: Set[Tuple[str,int]],
                                 cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                                 effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]]
                                 )->Set[str]:
    labels = set()
    for tag_pair in tag_pairs:
        if tag_pair in cause2effects:
            effects = cause2effects[tag_pair]
            for effect_tag_pair in effects:
                labels.add(format("{causer}->{effect}".format(causer=tag_pair[0], effect=effect_tag_pair[0])))
        elif tag_pair in effect2causers:
            causers = effect2causers[tag_pair]
            for causer_tag_pair in causers:
                labels.add(format("{causer}->{effect}".format(causer=causer_tag_pair[0], effect=tag_pair[0])))
    return labels