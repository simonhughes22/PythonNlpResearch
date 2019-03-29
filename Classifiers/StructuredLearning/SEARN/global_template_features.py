""" GLOBAL FEATURES """
import math
from collections import defaultdict
from typing import Dict, Tuple, Set, List

from searn_essay_parser import SearnModelEssayParser


def gbl_concept_code_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                  tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                  distance: int,
                  cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  positive_val: int) -> Dict[str, int]:

    feats = {}
    ordered_tags = sorted(tag2word_seq.keys(), key=lambda tpl: tpl[1])
    if len(buffer_tags) > 0:
        top_buffer_tag, top_ix = buffer_tags[0]
    else:
        top_buffer_tag, top_ix = ordered_tags[-1]

    if len(stack_tags) > 0:
        top_stack_tag, bottom_ix = stack_tags[-1]
    else:
        top_stack_tag, bottom_ix = ordered_tags[0]

    # get all tags, sprted by index
    prev_tags, subsequent_tags = [], []
    for tpl in ordered_tags:
        tag, ix = tpl
        if ix < bottom_ix:
            prev_tags.append(tpl[0])
        elif ix > top_ix:
            subsequent_tags.append(tpl[0])

    for i in [0, 1, 2, 3, 5, 8]:
        i_str = str(i)
        if len(prev_tags) > i:
            feats["num_prev_tags gtr " + i_str] = positive_val
        else:
            feats["num_prev_tags lte " + i_str] = positive_val

        if len(subsequent_tags) > i:
            feats["num_subsq_tag gtr" + i_str] = positive_val
        else:
            feats["num_subsq_tag lte" + i_str] = positive_val

        if len(ordered_tags) > i:
            feats["num_all_ptags gtr " + i_str] = positive_val
        else:
            feats["num_all_ptags lte " + i_str] = positive_val

    for tag in prev_tags:
        feats["prev_tag_" + tag] = positive_val

    for tag in subsequent_tags:
        feats["subsq_tag_" + tag] = positive_val

    return feats

def gbl_sentence_code_features(stack_tags: List[Tuple[str, int]], buffer_tags: List[Tuple[str, int]],
                  tag2word_seq: Dict[Tuple[str, int], List[str]], between_word_seq: List[str],
                  distance: int,
                  cause2effects: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  effect2causers: Dict[Tuple[str, int], Set[Tuple[str, int]]],
                  positive_val: int) -> Dict[str, int]:

    feats = {}
    buffer, ordered_tags, tos = get_tos_buffer(buffer_tags, stack_tags, tag2word_seq)

    # get all tags, sprted by index
    tos_ix = -1
    buffer_ix = len(ordered_tags)
    for i, tpl in enumerate(ordered_tags):
        if tpl == tos:
            tos_ix = i
        if tpl == buffer:
            buffer_ix = i

    prev_sent_tags = []
    next_sent_tags = []

    i = tos_ix
    current_tag = ""
    # Keep going backwards until we hit the next sentence
    while i > 0 and current_tag.upper() != SearnModelEssayParser.SENT:
        i -= 1
        tpl = ordered_tags[i]
        current_tag = tpl[0]
        prev_sent_tags.append(current_tag)

    i = buffer_ix
    current_tag = ""
    # Keep going forwards until we hit the next sentence
    while i > (len(ordered_tags) - 1) and current_tag.upper() != SearnModelEssayParser.SENT:
        i += 1
        tpl = ordered_tags[i]
        current_tag = tpl[0]
        next_sent_tags.append(current_tag)

    for tag in prev_sent_tags:
        feats["prev_sent_tag_" + tag] = positive_val
    feats["num_prev_sent_tags_" + str(len(prev_sent_tags))] = positive_val

    for tag in next_sent_tags:
        feats["next_sent_tag_" + tag] = positive_val
    feats["num_next_sent_tags_" + str(len(next_sent_tags))] = positive_val

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

    rel_posn = sents_before / num_essay_sents
    partition(feats, "sent_posn", rel_posn, num_partitions=3, positive_val=positive_val)
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

    greater_than_feats(feats, "num_causes", value=len(cause2effects), vals=[0, 1, 2, 3, 5], positive_val=positive_val)
    greater_than_feats(feats, "num_effects", value=len(effect2causers), vals=[0, 1, 2, 3, 5], positive_val=positive_val)

    num_crels = 0
    crel_tally = defaultdict(int)
    num_crels_crossing_sents = 0
    num_fwd_relns = 0
    for cause, r_codes in cause2effects.items():
        lcode, l_ix = cause
        num_crels += len(r_codes)
        for (rcode, r_ix) in r_codes: # deconstruct the tuples
            crel = lcode + "->" + rcode
            crel_tally[crel] += 1
            if r_ix > l_ix: # Is effect after causer in sentence?
                num_fwd_relns += 1
            small_ix, large_ix = sorted([l_ix, r_ix])
            for six in sent_ixs:
                # Is sentence boundary between codes
                if six > small_ix and six < large_ix:
                    num_crels_crossing_sents += 1
                    break

    partition(feats, "propn_fwd_crels", num_fwd_relns / num_crels, num_partitions=5, positive_val=positive_val)
    greater_than_feats(feats, "num_crels_crossing_sents", value=num_crels_crossing_sents, positive_val=positive_val)
    partition(feats, "propn_crossing_crels", num_crels_crossing_sents / num_crels, num_partitions=5, positive_val=positive_val)

    # Tally of currently parsed crels
    num_inversions = 0
    for crel, cnt in crel_tally.items():
        feats["Num_Crel_" + crel + "_" + str(cnt)] = positive_val
        lhs, rhs = crel.split("->")
        if lhs < rhs: # don't double count inversions
            inverted = rhs + "->" + lhs
            if inverted in crel_tally:
                num_inversions += 1

    for crel in crel_tally.keys():
        feats["CREL_" + crel] = positive_val

    feats["Max_Dupe_Crels_" + str(max(crel_tally.values()))] = positive_val
    greater_than_feats(feats, "num_inversions", value=num_inversions, vals=[0,1,2,3], positive_val=positive_val)
    partition(feats, "propn_inv", num_inversions/num_crels, num_partitions=4, positive_val=positive_val)
    greater_than_feats(feats, "num_crels", value=num_crels, vals=[0,1,2,3,5,7,10], positive_val=positive_val)
    partition(feats, "propn_crel_sents", num_crels / num_essay_sents, num_partitions=10, positive_val=positive_val)

    return feats


def get_tos_buffer(buffer_tags, stack_tags, tag2word_seq):
    ordered_tags = sorted(tag2word_seq.keys(), key=lambda tpl: tpl[1])
    if len(buffer_tags) > 0:
        tos = buffer_tags[0]
    else:
        tos = ordered_tags[-1]
    if len(stack_tags) > 0:
        buffer = stack_tags[-1]
    else:
        buffer = ordered_tags[0]
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
    sents_after = num_essay_sents - buffer_sents_before
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