__author__ = 'simon.hughes'

# Settings for loading essays
INCLUDE_VAGUE       = True
INCLUDE_NORMAL      = False

# Settings for essay pre-processing
MIN_SENTENCE_FREQ   = 2        # i.e. df. Note this is calculated BEFORE creating windows
REMOVE_INFREQUENT   = False    # if false, infrequent words are replaced with "INFREQUENT"
SPELLING_CORRECT    = True
STEM                = False    # note this tends to improve matters, but is needed to be on for pos tagging and dep parsing
                               # makes tagging model better but causal model worse
REPLACE_NUMS        = True     # 1989 -> 0000, 10 -> 00
MIN_SENTENCE_LENGTH = 3
REMOVE_STOP_WORDS   = False
REMOVE_PUNCTUATION  = True
LOWER_CASE          = False

# FEATURE SETTINGS
WINDOW_SIZE         = 7
# END FEATURE SETTINGS

def get_config(folder):
    config = {
        "folder"            : folder,

        "include_vague"     : INCLUDE_VAGUE,
        "include_normal"    : INCLUDE_NORMAL,

        "min_df"            : MIN_SENTENCE_FREQ,
        "remove_infrequent" : REMOVE_INFREQUENT,
        "spelling_correct"  : SPELLING_CORRECT,
        "stem"              : STEM,

        "replace_nums"      : REPLACE_NUMS,
        "min_sentence_length": MIN_SENTENCE_LENGTH,
        "remove_stop_words" : REMOVE_STOP_WORDS,
        "remove_punctuation": REMOVE_PUNCTUATION,
        "lower_case"        : LOWER_CASE,

        "window_size"       : WINDOW_SIZE
    }
    return config