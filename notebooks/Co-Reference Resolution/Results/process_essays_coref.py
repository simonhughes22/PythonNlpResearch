from collections import defaultdict
from processessays import Essay
from CoRefHelper import EMPTY
from BrattEssay import ANAPHORA

import warnings

def build_chain(e):
    """ Takes an essay object, and creats a map of Dict[str, List[Tuple{int,int}]]
        which maps a coref id (essay scope) to a list of (sent_ix,wd_ix) pairs
    """
    corefid_2_chain = defaultdict(list)
    for sent_ix in range(len(e.sentences)):
        sent = e.sentences[sent_ix]
        coref_ids = e.pred_corefids[sent_ix]
        for wd_ix in range(len(sent)):
            wd_coref_ids = coref_ids[wd_ix]  # Set[str]
            for cr_id in wd_coref_ids:
                if cr_id.strip() == "":
                    continue
                pair = (sent_ix, wd_ix)
                corefid_2_chain[cr_id].append(pair)
    return corefid_2_chain


def build_segmented_chain(e):
    """ Takes an essay object, and creats a map of Dict[str, List[List[Tuple{int,int}]]
        which maps a coref id (essay scope) to a nested list of (sent_ix,wd_ix) pairs.
        The nested list has a separate inner list for every distinct coreference seq/phrase
    """

    corefid_2_chain = build_chain(e)
    corefid_2_segmented_chain = dict()
    for cref, pairs in corefid_2_chain.items():
        segmented = [[pairs[0]]]
        corefid_2_segmented_chain[cref] = segmented
        last_sent_ix, last_wd_ix = pairs[0]
        for pair in pairs[1:]:
            sent_ix, wd_ix = pair
            if sent_ix != last_sent_ix or (wd_ix - last_wd_ix) > 1:
                # create a new nested list
                segmented.append([])
            # append pair to last list item
            segmented[-1].append(pair)
            last_sent_ix, last_wd_ix = pair
    return corefid_2_segmented_chain


def find_coref_length(chain, sent_ix, word_ix):
    if len(chain) == 0:
        return 0
    length = 0
    for phrase in chain:
        first_pair = phrase[0]
        last_pair = phrase[-1]
        first_sent, first_word = first_pair
        last_sent, last_word = last_pair

        # within same sentence
        if sent_ix >= first_sent and sent_ix <= last_sent:
            assert first_sent == last_sent, "Phrase spans different sentences"
            if word_ix >= first_word and word_ix <= last_word:
                length = max(length, (last_word - first_word) + 1)
    assert length > 0, "can't find matching coref"
    return length

#def find_nearest_reference(chain, sent_ix, word_ix):
def find_reference_ix(chain, sent_ix, word_ix):
    if len(chain) == 0:
        return -1
    
    ana_ix = -1
    for ix,phrase in enumerate(chain):
        first_pair = phrase[0]
        last_pair = phrase[-1]
        first_sent, first_word = first_pair
        last_sent, last_word = last_pair

        # within same sentence
        if sent_ix >= first_sent and sent_ix <= last_sent:
            assert first_sent == last_sent, "Phrase spans different sentences"
            if word_ix >= first_word and word_ix <= last_word:
                ana_ix = ix
                break
    return ana_ix

# some of the parsed data has empty string id's in it
def fix_coref_ids(essay):
    corefids = essay.pred_corefids
    for sent in corefids:
        for set_ids in sent:
            if "" in set_ids:
                set_ids.remove("")


def get_coref_processed_essays(essays, format_ana_tags=True, filter_to_predicted_tags=True, look_back_only=True,
                               nearest_ref_only=False,
                               max_ana_phrase_len=None, max_cref_phrase_len=None,
                               ner_ch_filter=None, pos_ana_filter=None, pos_ch_filter=None
                               ):
    """
    Create a copy of essays, augmenting the pred_tagged_sentences object with additional anaphora tags

    essays:                   List[Essay] objects - merged tagged essays
    format_ana_tags:          bool - Add ana tags as Anaphor[xyz] or as just the regular concept codes
    filter_to_predicted_tags: bool - Filter to just the predicted anaphor tags
    look_back_only:           bool - Only look to coreferences occuring earlier in the essay
    nearest_ref_only:         bool - Only look at the nearest reference (last preceding entry in the chain)

    max_ana_phrase_len:       Union(int,None) - if specified, maximum  length to consider
    max_cref_phrase_len:      Union(int,None) - if specified, maximum coreference length to consider
    ner_ch_filter:            Union(Set[str],None) - if specified, filters to words in the cref chain
                                with one of those NER tags
    pos_ana_filter:               Union(Set[str],None) - if specified, filters crefs to words with one of those POS tags
    pos_ch_filter:            Union(Set[str],None) - if specified, filters to words in the cref chain
                                with one of those POS tags
    """
    if ner_ch_filter and EMPTY in ner_ch_filter:
        warnings.warn("EMPTY tag in NER filter ", UserWarning)
    if pos_ana_filter and EMPTY in pos_ana_filter:
        warnings.warn("EMPTY tag in POS filter ", UserWarning)
    if pos_ch_filter and EMPTY in pos_ch_filter:
        warnings.warn("EMPTY tag in POS chain filter ", UserWarning)

    ana_tagged_essays = []
    for eix, e in enumerate(essays):

        fix_coref_ids(e)

        ana_tagged_e = Essay(e.name, e.sentences)
        ana_tagged_e.pred_tagged_sentences = []
        ana_tagged_e.pred_pos_tags_sentences = list(e.pred_pos_tags_sentences)
        ana_tagged_e.pred_ner_tags_sentences = list(e.pred_pos_tags_sentences)
        ana_tagged_e.ana_tagged_sentences    = list(e.ana_tagged_sentences)
        ana_tagged_e.pred_corefids           = list(e.pred_corefids)
        ana_tagged_essays.append(ana_tagged_e)

        # map coref ids to sent_ix, wd_ix tuples
        corefid_2_chain = build_segmented_chain(e)

        # now look for ana tags that are also corefs, and cross reference
        for sent_ix in range(len(e.sentences)):
            ana_tagged_sent = []
            ana_tagged_e.pred_tagged_sentences.append(ana_tagged_sent)

            sent = e.sentences[sent_ix]

            # SENTENCE LEVEL TAGS / PREDICTIONS
            ana_tags = e.ana_tagged_sentences[sent_ix]
            coref_ids = e.pred_corefids[sent_ix]
            # ner_tags = e.pred_ner_tags_sentences[sent_ix]
            pos_tags = e.pred_pos_tags_sentences[sent_ix]
            ptags = e.pred_tagged_sentences[sent_ix]

            for wd_ix in range(len(sent)):
                pos_tag = pos_tags[wd_ix]  # POS tag

                word, _ = sent[wd_ix]  # ignore actual tags
                pred_cc_tag = ptags[wd_ix]  # predict cc tag

                is_ana_tag = ana_tags[wd_ix] == ANAPHORA
                wd_coref_ids = coref_ids[wd_ix]  # Set[str]

                # note we are changing this to a set rather than a single string
                wd_ptags = set()
                # add predicted concept code tag (filtered out by evaluation code, which filters to specific tags)
                if pred_cc_tag != EMPTY:
                    wd_ptags.add(pred_cc_tag)

                # initialize predicted tags, inc. cc tag
                # DON'T run continue until after this point
                ana_tagged_sent.append(wd_ptags)

                if len(wd_coref_ids) == 0:
                    continue

                # POS FILTER - for cref words and NOT words in the cref chain
                if pos_ana_filter and pos_tag not in pos_ana_filter:
                    continue

                if filter_to_predicted_tags and not is_ana_tag:
                    continue

                # Get codes for corresponding co-ref chain entries
                for cr_id in wd_coref_ids:

                    segmented_chain = corefid_2_chain[cr_id]

                    if max_ana_phrase_len:
                        anaphor_length = find_coref_length(chain=segmented_chain, sent_ix=sent_ix, word_ix=wd_ix)
                        if anaphor_length > max_ana_phrase_len:
                            continue
                            
                    ana_ix = find_reference_ix(chain=segmented_chain, sent_ix=sent_ix, word_ix=wd_ix)
                    assert ana_ix > -1, "Could not find matching reference"

                    if nearest_ref_only:
                        if ana_ix > 0: # must be >= 1 as there is no prev ref to the first phrase
                            nearest_phrase  = segmented_chain[ana_ix-1]
                            segmented_chain = [nearest_phrase]
                        else:
                            continue
                                                
                    for cix, cref_phrase in enumerate(segmented_chain):  # iterate thru the list of sent_ix,wd_ix's
                        # phrase contains current word
                        if cix == ana_ix:
                            continue

                        if look_back_only is True and cix > ana_ix:
                            continue

                        # LENGTH FILTER
                        if max_cref_phrase_len and len(cref_phrase) > max_cref_phrase_len:
                            continue

                        for ch_sent_ix, ch_wd_ix in cref_phrase:
                            # if it's the current word, skip
                            if ch_sent_ix == sent_ix and ch_wd_ix == wd_ix:
                                continue
                            # for anaphors only - only look at chain ixs before the current word
                            # if's it's after the current word in the essay, skip
                            if look_back_only:
                                # sentence later in the essay, or same sentence but word is after current word
                                if ch_sent_ix > sent_ix or \
                                        (ch_sent_ix == sent_ix and ch_wd_ix >= wd_ix):
                                    continue

                            chain_ptag = e.pred_tagged_sentences[ch_sent_ix][ch_wd_ix]
                            ch_ner_tag = e.pred_ner_tags_sentences[ch_sent_ix][ch_wd_ix]
                            ch_pos_tag = e.pred_pos_tags_sentences[ch_sent_ix][ch_wd_ix]

                            # CHAIN WORD TYPE FILTERS
                            # NER TAG TYPE FILTER - on chain
                            if ner_ch_filter and ch_ner_tag not in ner_ch_filter:
                                continue
                            # POS TAG TYPE FILTER - on chain
                            if pos_ch_filter and ch_pos_tag not in pos_ch_filter:
                                continue

                            if chain_ptag != EMPTY:
                                code = chain_ptag
                                if format_ana_tags:
                                    code = "{anaphora}:[{code}]".format(
                                        anaphora=ANAPHORA, code=chain_ptag)
                                wd_ptags.add(code)
    # validation check
    #   check essay and sent lengths align
    for e in ana_tagged_essays:
        assert len(e.sentences) == len(e.pred_tagged_sentences)
        for ix in range(len(e.sentences)):
            assert len(e.sentences[ix]) == len(e.pred_tagged_sentences[ix])

    return ana_tagged_essays
    
def processed_essays_replace_ana_tags_with_regular(essays):
    """
        Replaces  Anaphor[xyz] tags with the original tag
        
        essays:                   List[Essay] objects - merged tagged essays
    """

    ana_tagged_essays = []
    for eix, e in enumerate(essays):

        fix_coref_ids(e)
        seq_pred_tags = [] # all predicted tags
        
        new_sentences = []
        ana_tagged_e = Essay(e.name, new_sentences)
        ana_tagged_e.pred_tagged_sentences   = list(e.pred_tagged_sentences)
        ana_tagged_e.pred_pos_tags_sentences = list(e.pred_pos_tags_sentences)
        ana_tagged_e.pred_ner_tags_sentences = list(e.pred_pos_tags_sentences)
        ana_tagged_e.ana_tagged_sentences    = list(e.ana_tagged_sentences)
        ana_tagged_e.pred_corefids           = list(e.pred_corefids)
        ana_tagged_essays.append(ana_tagged_e)
    
        # now look for ana tags that are also corefs, and cross reference
        for sent_ix in range(len(e.sentences)):
            new_sent = []
            new_sentences.append(new_sent)
            
            sent = e.sentences[sent_ix]
            for wd, tags in sent:
                new_tags = set()
                for t in tags:
                    if t.startswith("Anaphor:["):
                        t_old = t
                        t = t.replace("Anaphor:[","").replace("]","")
#                         print(t_old, t)
                    new_tags.add(t)
                new_sent.append((wd, new_tags))
            assert len(new_sent) == len(sent)
        assert len(new_sentences) == len(e.sentences)
    
    # validation check
    #   check essay and sent lengths align
    for e in ana_tagged_essays:
        assert len(e.sentences) == len(e.pred_tagged_sentences), (e.name, len(e.sentences),len(e.pred_tagged_sentences))
        for ix in range(len(e.sentences)):
            assert len(e.sentences[ix]) == len(e.pred_tagged_sentences[ix]), (len(e.sentences[ix]), len(e.pred_tagged_sentences[ix]))

    return ana_tagged_essays

    
def find_previous_predicted_tag(ix, seq_ptags, seq_is_ana_tag):
    """
        Given a sequence of predicted tags and booleans indicating whether or not a tag is 
        a (predicted) anaphora tag, finds the previously predicted tag (before current)
        anaphora sequence.
    
    """
    assert len(seq_ptags) == len(seq_is_ana_tag)
    if len(seq_ptags) == 0:
        return None
    
    current_ix = ix
    current_is_ana = seq_is_ana_tag[current_ix]
    
    # Find first non-anaphora tag
    while current_is_ana:
        current_ix -= 1
        if current_ix < 0:
            return None
        current_is_ana = seq_is_ana_tag[current_ix]
    
    current_ptag = seq_ptags[current_ix]
    while current_ptag == EMPTY:
        current_ix -= 1
        if current_ix < 0:
            return None
        current_ptag = seq_ptags[current_ix]
    return current_ptag

def processed_essays_predict_most_recent_tag(essays, format_ana_tags=True):

    """
    Uses the most recently predicted concept code as the predicted tag
    
            essays:                   List[Essay] objects - merged tagged essays
    """

    ana_tagged_essays = []
    for eix, e in enumerate(essays):

        fix_coref_ids(e)
        
        # following are flattened so they span sentences
        seq_pred_tags  = [] # all predicted tags
        seq_is_ana_tag = [] # is ana tag
        seq_ix = -1
        
        ana_tagged_e = Essay(e.name, e.sentences)
        ana_tagged_e.pred_tagged_sentences = []
        ana_tagged_e.pred_pos_tags_sentences = list(e.pred_pos_tags_sentences)
        ana_tagged_e.pred_ner_tags_sentences = list(e.pred_pos_tags_sentences)
        ana_tagged_e.ana_tagged_sentences    = list(e.ana_tagged_sentences)
        ana_tagged_e.pred_corefids           = list(e.pred_corefids)
        ana_tagged_essays.append(ana_tagged_e)

        # now look for ana tags that are also corefs, and cross reference
        for sent_ix in range(len(e.sentences)):
            ana_tagged_sent = []
            ana_tagged_e.pred_tagged_sentences.append(ana_tagged_sent)

            sent = e.sentences[sent_ix]

            # SENTENCE LEVEL TAGS / PREDICTIONS
            ana_tags = e.ana_tagged_sentences[sent_ix]
            coref_ids = e.pred_corefids[sent_ix]
            # ner_tags = e.pred_ner_tags_sentences[sent_ix]
            pos_tags = e.pred_pos_tags_sentences[sent_ix]
            ptags = e.pred_tagged_sentences[sent_ix]

            for wd_ix in range(len(sent)):
                seq_ix +=1
                
                pos_tag = pos_tags[wd_ix]  # POS tag

                word, _ = sent[wd_ix]  # ignore actual tags
                pred_cc_tag = ptags[wd_ix]  # predict cc tag
                seq_pred_tags.append(pred_cc_tag)

                is_ana_tag = ana_tags[wd_ix] == ANAPHORA
                seq_is_ana_tag.append(is_ana_tag)
                
                wd_coref_ids = coref_ids[wd_ix]  # Set[str]

                # note we are changing this to a set rather than a single string
                wd_ptags = set()
                # initialize predicted tags, inc. cc tag
                # DON'T run continue until after this point
                ana_tagged_sent.append(wd_ptags)

                # add predicted concept code tag (filtered out by evaluation code, which filters to specific tags)
                if pred_cc_tag != EMPTY:
                    wd_ptags.add(pred_cc_tag)
                # else here because we don't want to assign additional cc tags if there are already ones
                elif is_ana_tag and pred_cc_tag == EMPTY: # and current tag is EMPTY
                    code = find_previous_predicted_tag(seq_ix, seq_pred_tags, seq_is_ana_tag)  
                    if code is None:
                    	code = EMPTY              
                    if format_ana_tags:
                        code = "{anaphora}:[{code}]".format(anaphora=ANAPHORA, code=code)
                    wd_ptags.add(code)

    # validation check
    #   check essay and sent lengths align
    for e in ana_tagged_essays:
        assert len(e.sentences) == len(e.pred_tagged_sentences)
        for ix in range(len(e.sentences)):
            assert len(e.sentences[ix]) == len(e.pred_tagged_sentences[ix])

    return ana_tagged_essays