""" GLOBAL FEATURES """
import math
from collections import defaultdict
from typing import Dict, Tuple, Set, List

from searn_essay_parser import SearnModelEssayParser

from shift_reduce_parser import ROOT

MAX_BUFFER = 999999
ARROW = "->"
ALL = "ALL"
SKIP_CODES = {ROOT, SearnModelEssayParser.SENT, None, ALL}

def gbl_concept_code_cnt_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                                  tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                                  distance: int,
                                  cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                                  effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                                  positive_val: int) -> Dict[str, int]:

    feats = {}
    buffer, ordered_tags, tos = get_tos_buffer(buffer_tags, stack_tags, tag2word_seq)

    code_tags, prev_tags, subsequent_tags = get_codes_before_after(buffer_tags, ordered_tags, stack_tags)

    greater_than_feats(feats, "num_prev_tags", value=len(prev_tags),        vals=[0, 1, 2, 3, 5, 7, 10], positive_val=positive_val)
    greater_than_feats(feats, "num_subsq",     value=len(subsequent_tags),  vals=[0, 1, 2, 3, 5, 7, 10], positive_val=positive_val)
    greater_than_feats(feats, "num_all_ptags", value=len(code_tags),        vals=[0, 1, 2, 3, 5, 7, 10], positive_val=positive_val)

    if len(code_tags) > 0:
        partition(feats, "propn_prev_tags", len(prev_tags) / len(code_tags),         num_partitions=4, positive_val=positive_val)
        partition(feats, "propn_subsq_tags", len(subsequent_tags) / len(code_tags),  num_partitions=4, positive_val=positive_val)
    return feats


def get_codes_before_after(buffer_tags, ordered_tags, stack_tags):
    if len(buffer_tags) > 0:
        top_buffer_tag, top_ix = buffer_tags[0]
    else:
        top_buffer_tag, top_ix = ordered_tags[-1]
    if len(stack_tags) > 0:
        top_stack_tag, bottom_ix = stack_tags[-1]
    else:
        top_stack_tag, bottom_ix = ordered_tags[0]
    # get all tags, sprted by index
    prev_tags, subsequent_tags, code_tags = [], [], []
    for tpl in ordered_tags:
        tag, ix = tpl
        if tag in SKIP_CODES:
            continue

        code_tags.append(tpl[0])
        if ix < bottom_ix:
            prev_tags.append(tpl[0])
        elif ix > top_ix:
            subsequent_tags.append(tpl[0])
    return code_tags, prev_tags, subsequent_tags


def gbl_adjacent_sent_code_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                                    tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                                    distance: int,
                                    cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                                    effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                                    positive_val: int) -> Dict[str, int]:

    feats = {}
    buffer, ordered_tags, tos = get_tos_buffer(buffer_tags, stack_tags, tag2word_seq)

    # get all tags, sprted by index
    tos_ix = -1
    prev_sent_ixs, next_sent_ixs = [],[]

    buffer_ix = MAX_BUFFER
    for i, tpl in enumerate(ordered_tags):
        if tpl == tos:
            tos_ix = i
        if tpl == buffer:
            buffer_ix = i
        if tpl[0] == SearnModelEssayParser.SENT:
            ix = tpl[1]
            if tos_ix == -1:
                prev_sent_ixs.append(ix)
            if buffer_ix < MAX_BUFFER:
                next_sent_ixs.append(ix)

    prev_sent_tags = []
    next_sent_tags = []

    if len(prev_sent_ixs) == 1:
        prev_sent_tags = [tpl[0] for tpl in ordered_tags
                          if tpl[1] < prev_sent_ixs[-1] and tpl[0] not in SKIP_CODES]

    elif len(prev_sent_ixs) >= 2:
        prev_sent_tags = [tpl[0] for tpl in ordered_tags
                          if tpl[1] < prev_sent_ixs[-1] and tpl[1] > prev_sent_ixs[-2] and tpl[0] not in SKIP_CODES]

    if len(next_sent_ixs) == 1:
        next_sent_tags = [tpl[0] for tpl in ordered_tags
                          if tpl[1] > next_sent_ixs[0] and tpl[0] not in SKIP_CODES]
    elif len(next_sent_ixs) >= 2:
        next_sent_tags = [tpl[0] for tpl in ordered_tags
                          if tpl[1] > next_sent_ixs[0] and tpl[1] < next_sent_ixs[1] and tpl[0] not in SKIP_CODES]

    for tag in prev_sent_tags:
        if len(stack_tags) > 0 and tos[0] not in SKIP_CODES:
            feats["prev_sent_tag_" + tag + "_TOS_" + tos[0]] = positive_val
            if tag == tos[0]:
                feats["Prev_sent_has_TOS"] = positive_val
        if len(buffer_tags) > 0 and buffer[0] not in SKIP_CODES:
            feats["prev_sent_tag_" + tag + "_BUFFER_" + buffer[0]] = positive_val
            if tag == buffer[0]:
                feats["Prev_sent_has_BUFFER"] = positive_val
        if len(stack_tags) > 0 and len(buffer_tags) > 0 and tos[0] not in SKIP_CODES and buffer[0] not in SKIP_CODES:
            feats["prev_sent_tag_" + tag + "_TOS_" + tos[0] + "_BUFFER_" + buffer[0]] = positive_val

    greater_than_feats(feats, "num_prev_sent_tags_", value=len(prev_sent_tags),
                       vals=[0, 1, 2, 3, 5, 7, 10],
                       positive_val=positive_val)

    for tag in next_sent_tags:
        if len(stack_tags) > 0 and tos[0] not in SKIP_CODES:
            feats["next_sent_tag_" + tag + "_TOS_" + tos[0]] = positive_val
        if len(buffer_tags) > 0 and buffer[0] not in SKIP_CODES:
            feats["next_sent_tag_" + tag + "_BUFFER_" + buffer[0]] = positive_val
        if len(stack_tags) > 0 and len(buffer_tags) > 0 and tos[0] not in SKIP_CODES and buffer[0] not in SKIP_CODES:
            feats["next_sent_tag_" + tag + "_TOS_" + tos[0] + "_BUFFER_" + buffer[0]] = positive_val

    greater_than_feats(feats, "num_next_sent_tags_", value=len(next_sent_tags),
                       vals=[0, 1, 2, 3, 5, 7, 10],
                       positive_val=positive_val)
    return feats

def gbl_sentence_position_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                  tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                  distance: int,
                  cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  positive_val: int) -> Dict[str, int]:

    feats = {}
    buffer, ordered_tags, tos = get_tos_buffer(buffer_tags, stack_tags, tag2word_seq)
    # get all tags, sprted by index
    num_essay_sents, sents_after, sents_before, sent_ixs = sentence_stats(buffer, ordered_tags, tos)

    sents_between = 0
    for word in between_word_seq:
        if word.lower() == SearnModelEssayParser.SENT.lower():
            sents_between += 1

    # How many sentences between the TOS and buffer?
    greater_than_feats(feats, "num_sentences_between", value=sents_between, vals=[0, 1, 2, 3, 5], positive_val=positive_val)
    greater_than_feats(feats, "num_sentences_before",  value=sents_before,  vals=[0, 1, 2, 3, 5], positive_val=positive_val)
    greater_than_feats(feats, "num_sentences_after",   value=sents_after,   vals=[0, 1, 2, 3, 5], positive_val=positive_val)

    rel_posn_tos = sents_before / num_essay_sents
    partition(feats, "sent_tos_posn", rel_posn_tos, num_partitions=3, positive_val=positive_val)

    rel_posn_buffer = (num_essay_sents-sents_after) / num_essay_sents
    partition(feats, "sent_buf_posn", rel_posn_buffer, num_partitions=3, positive_val=positive_val)

    rel_propn_btwn = (sents_between) / num_essay_sents
    partition(feats, "sent_propn_btwn", rel_propn_btwn, num_partitions=10, positive_val=positive_val)
    return feats


def gbl_causal_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                  tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                  distance: int,
                  cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  positive_val: int) -> Dict[str, int]:
    feats = {}
    buffer, ordered_tags, tos = get_tos_buffer(buffer_tags, stack_tags, tag2word_seq)
    # get all tags, sprted by index
    num_essay_sents, sents_after, sents_before, sent_ixs = sentence_stats(buffer, ordered_tags, tos)

    greater_than_feats(feats, "num_distinct_causes",  value=len(cause2effects),  vals=[0, 1, 2, 3, 5], positive_val=positive_val)
    greater_than_feats(feats, "num_distinct_effects", value=len(effect2causers), vals=[0, 1, 2, 3, 5], positive_val=positive_val)

    num_crels = 0
    crel_tally = defaultdict(int)
    num_crels_crossing_sents = 0
    num_fwd_relns = 0
    causer_tally, effect_tally = defaultdict(int), defaultdict(int)

    for cause, r_codes in cause2effects.items():
        lcode, l_ix = cause
        num_crels += len(r_codes)
        causer_tally[lcode] += 1
        for (rcode, r_ix) in r_codes: # deconstruct the tuples
            crel = lcode + ARROW + rcode
            crel_tally[crel] += 1
            effect_tally[rcode] += 1
            if r_ix > l_ix: # Is effect after causer in sentence?
                num_fwd_relns += 1
            small_ix, large_ix = sorted([l_ix, r_ix])
            for six in sent_ixs:
                # Is sentence boundary between codes
                if six > small_ix and six < large_ix:
                    num_crels_crossing_sents += 1
                    break

    # Fwd crels are ones where causer is before effect
    if num_crels > 0:
        partition(feats, "propn_fwd_crels", num_fwd_relns / num_crels, num_partitions=5, positive_val=positive_val)
        partition(feats, "propn_crossing_crels", num_crels_crossing_sents / num_crels, num_partitions=5, positive_val=positive_val)

    greater_than_feats(feats, "num_crels_crossing_sents", value=num_crels_crossing_sents, positive_val=positive_val)

    # Tally of currently parsed crels
    num_inversions = 0
    for crel, cnt in crel_tally.items():
        # feats["Num_Crel_" + crel + "_" + str(cnt)] = positive_val
        lhs, rhs = crel.split(ARROW)
        if lhs < rhs: # don't double count inversions
            inverted = rhs + ARROW + lhs
            if inverted in crel_tally:
                num_inversions += 1

    for crel in crel_tally.keys():
        if len(stack_tags) > 0 and tos[0] not in SKIP_CODES:
            feats["CREL_" + crel + "_TOS_" + tos[0]] = positive_val
        if len(buffer_tags) > 0 and buffer[0] not in SKIP_CODES:
            feats["CREL_" + crel + "_BUFFER_" + buffer[0]] = positive_val
        if len(stack_tags) > 0 and len(buffer_tags) > 0 and tos[0] not in SKIP_CODES and buffer[0] not in SKIP_CODES:
            feats["CREL_" + crel + "_TOS_" + tos[0] + "_BUFFER_" + buffer[0]] = positive_val

    possible_crels = []
    if tos[0] not in SKIP_CODES and buffer[0] not in SKIP_CODES:
        possible_crels = [
            buffer[0] + ARROW + tos[0],
            tos[0]    + ARROW + buffer[0]
        ]

    # # count of buffer and tos tags as causers amd effects
    # greater_than_feats(feats, "buffer_causer_count", value=causer_tally[buffer[0]],  positive_val=positive_val)
    # greater_than_feats(feats, "buffer_effect_count", value=effect_tally[buffer[0]],  positive_val=positive_val)
    #
    # greater_than_feats(feats, "tos_causer_count",    value=causer_tally[tos[0]], positive_val=positive_val)
    # greater_than_feats(feats, "tos_effect_count",    value=effect_tally[tos[0]], positive_val=positive_val)

    for i, crel in enumerate(possible_crels):
        if crel in crel_tally:
            feats["CREL_already_exists"] = positive_val
            greater_than_feats(feats, "existing_crel_count", value=crel_tally[crel],
                               vals=[0, 1, 2, 3], positive_val=positive_val)

    greater_than_feats(feats, "num_crels",      value=num_crels, vals=[0,1,2,3,5,7,10], positive_val=positive_val)
    greater_than_feats(feats, "num_inversions", value=num_inversions, vals=[0,1,2,3], positive_val=positive_val)
    if num_crels > 0:
        partition(feats, "propn_inv", num_inversions/num_crels, num_partitions=4, positive_val=positive_val)

    if num_essay_sents > 0:
        partition(feats, "propn_crel_sents", num_crels / num_essay_sents, num_partitions=10, positive_val=positive_val)
    return feats

def gbl_ratio_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                  tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                  distance: int,
                  cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  positive_val: int) -> Dict[str, int]:

    feats = {}
    buffer, ordered_tags, tos = get_tos_buffer(buffer_tags, stack_tags, tag2word_seq)
    code_tags, prev_tags, subsequent_tags = get_codes_before_after(buffer_tags, ordered_tags, stack_tags)

    # get all tags, sprted by index
    num_essay_sents, sents_after, sents_before, sent_ixs = sentence_stats(buffer, ordered_tags, tos)
    all_words = tag2word_seq[(ALL, -1)]
    num_words = len(all_words)

    num_crels = 0
    for cause, r_codes in cause2effects.items():
        num_crels += len(r_codes)

    # feats["propn_crel_all_sents"] = num_crels / num_essay_sents
    if sents_before > 0:
        feats["crel_tos_sents_ratio"] = num_crels / sents_before
        feats["prev_codes_tos_sents_ratio"] = len(prev_tags) / sents_before

    if sents_after > 0:
        feats["crel_buf_sents_ratio"] = num_crels / sents_after
        feats["next_codes_tos_sents_ratio"] = len(subsequent_tags) / sents_after

    feats["crel_all_sents_ratio"] = num_crels / num_essay_sents
    feats["codes_all_sents_ratio"] = len(code_tags) / num_essay_sents

    tos_words = tos[-1]
    buf_words = buffer[-1]
    if tos_words > 0:
        feats["crel_tos_word_ratio"] = num_crels / tos_words
        feats["prev_code_tos_word_ratio"] = len(prev_tags) / tos_words

    if buf_words > 0:
        feats["crel_buf_word_ratio"] = num_crels / buf_words
        feats["next_code_buf_word_ratio"] = len(subsequent_tags) / buf_words

    feats["crel_all_word_ratio"] = num_crels / num_words
    feats["codes_all_word_ratio"] = len(code_tags) / num_words

    if len(code_tags) > 0:
        feats["crel_2_codes_ratio"] = num_crels / len(code_tags)
        if len(prev_tags) > 0:
            feats["crel_2_prev_codes_ratio"] = num_crels / len(prev_tags)
        if len(subsequent_tags) > 0:
            feats["crel_2_next_codes_ratio"] = num_crels / len(subsequent_tags)
    return feats


def get_tos_buffer(buffer_tags, stack_tags, tag2word_seq):

    keys = (tpl for tpl in tag2word_seq.keys() if tpl[0] != ALL)
    ordered_tags = sorted(keys, key=lambda tpl: tpl[1])

    if len(stack_tags) > 0:
        tos = stack_tags[-1]
    else:
        tos = None
    if len(buffer_tags) > 0:
        buffer = buffer_tags[0]
    else:
        buffer = None
    return buffer, ordered_tags, tos

def sentence_stats(buffer, ordered_tags, tos):
    num_essay_sents = 0
    sents_before = -1
    buffer_sents_before = -1
    sent_ixs = set()
    for i, tpl in enumerate(ordered_tags):
        if tpl == tos:
            sents_before = num_essay_sents
        if tpl == buffer:
            buffer_sents_before = num_essay_sents
        if tpl[0] == SearnModelEssayParser.SENT:
            num_essay_sents += 1
            sent_ixs.add(tpl[-1])
    sents_after = num_essay_sents - buffer_sents_before -1
    return num_essay_sents, sents_after, sents_before, sent_ixs

def partition(fts, ft_name, propn, num_partitions=3, positive_val=1):
    if propn >= 1.0:
        propn = 0.999 # ensure if 1 that it does not go into a different bucket
    partition = str(math.floor(propn * num_partitions))
    fts[ft_name + "_" + partition] = positive_val

def greater_than_feats(fts, ft_name, value, vals=list(range(5)), positive_val=1):
    for i in vals:
        if value > i:
            fts[ft_name + " gtr " + str(i)] = positive_val
        else:
            fts[ft_name + " lte " + str(i)] = positive_val