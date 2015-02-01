from BrattEssay import load_bratt_essays
from processessays import process_essays
from featureextractortransformer import FeatureExtractorTransformer

DFLT_MIN_SENTENCE_FREQ   = 2        # i.e. df. Note this is calculated BEFORE creating windows
DFLT_REMOVE_INFREQUENT   = False    # if false, infrequent words are replaced with "INFREQUENT"
DFLT_SPELLING_CORRECT    = True
DFLT_STEM                = False     # note this tends to improve matters, but is needed to be on for pos tagging and dep parsing
                               # makes tagging model better but causal model worse
DFLT_REPLACE_NUMS        = True     # 1989 -> 0000, 10 -> 00
DFLT_MIN_SENTENCE_LENGTH = 3
DFLT_REMOVE_STOP_WORDS   = False
DFLT_REMOVE_PUNCTUATION  = True
DFLT_LOWER_CASE          = False

DFLT_INCLUDE_VAGUE       = True
DFLT_INCLUDE_NORMAL      = True

""" FEATURE PARAMS """
DFLT_WINDOW_SIZE         = 7
DFLT_POS_WINDOW_SIZE     = 1
DFLT_MIN_FEAT_FREQ       = 5        # 5 best so far
DFLT_CV_FOLDS            = 5

def load_process_essays(folder, min_df=DFLT_MIN_SENTENCE_FREQ, remove_infrequent=DFLT_REMOVE_INFREQUENT,
                       spelling_correct=DFLT_SPELLING_CORRECT,
                       replace_nums=DFLT_REPLACE_NUMS, stem=DFLT_STEM, remove_stop_words=DFLT_REMOVE_STOP_WORDS,
                       remove_punctuation=DFLT_REMOVE_PUNCTUATION, lower_case=DFLT_LOWER_CASE,
                       include_vague=DFLT_INCLUDE_VAGUE, include_normal=DFLT_INCLUDE_NORMAL):

    essays = load_bratt_essays(directory=folder, include_vague=include_vague, include_normal=include_normal)
    return process_essays(essays, min_df=min_df, remove_infrequent=remove_infrequent, spelling_correct=spelling_correct,
                          replace_nums=replace_nums, stem=stem, remove_stop_words=remove_stop_words,
                          remove_punctuation=remove_punctuation, lower_case=lower_case)

def extract_features(tagged_essays, folder, extractors,
                         min_df=DFLT_MIN_SENTENCE_FREQ,
                         rem_infreq=DFLT_REMOVE_INFREQUENT, sp_crrct=DFLT_SPELLING_CORRECT,
                         replace_nos=DFLT_REPLACE_NUMS, stem=DFLT_STEM, rem_stop_wds=DFLT_REMOVE_STOP_WORDS,
                         rem_punc=DFLT_REMOVE_PUNCTUATION, lcase=DFLT_LOWER_CASE,
                         win_size=DFLT_WINDOW_SIZE, pos_win_size=DFLT_POS_WINDOW_SIZE, min_ft_freq=DFLT_MIN_FEAT_FREQ,
                         inc_vague=DFLT_INCLUDE_VAGUE, inc_normal=DFLT_INCLUDE_NORMAL):
    feature_extractor = FeatureExtractorTransformer(extractors)
    return feature_extractor.transform(tagged_essays)
