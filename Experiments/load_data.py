from BrattEssay import load_bratt_essays
from processessays import process_essays
from featureextractortransformer import FeatureExtractorTransformer

def load_process_essays(window_size, min_sentence_length, folder, min_df, remove_infrequent,
                       spelling_correct,
                       replace_nums, stem, remove_stop_words,
                       remove_punctuation, lower_case,
                       include_vague, include_normal):

    essays = load_bratt_essays(directory=folder, include_vague=include_vague, include_normal=include_normal)
    return process_essays(essays, min_df=min_df, remove_infrequent=remove_infrequent, spelling_correct=spelling_correct,
                          replace_nums=replace_nums, stem=stem, remove_stop_words=remove_stop_words,
                          remove_punctuation=remove_punctuation, lower_case=lower_case, spelling_corrector=None)

def load_process_essays_without_annotations(window_size, min_sentence_length, folder, min_df, remove_infrequent,
                            spelling_correct,
                            replace_nums, stem, remove_stop_words,
                            remove_punctuation, lower_case,
                            include_vague, include_normal):

    essays = load_bratt_essays(directory=folder, include_vague=include_vague, include_normal=include_normal, load_annotations=False)
    return process_essays(essays, min_df=min_df, remove_infrequent=remove_infrequent, spelling_correct=spelling_correct,
                          replace_nums=replace_nums, stem=stem, remove_stop_words=remove_stop_words,
                          remove_punctuation=remove_punctuation, lower_case=lower_case, spelling_corrector=None)

# parameters below are not used in this function, but are used to force memoization to take into account the parameters used
# to derive the data
def extract_features(tagged_essays, extractors=None, window_size=None, min_sentence_length=None, folder=None,
                     min_df=None, remove_infrequent=None,
                     spelling_correct=None,
                     replace_nums=None, stem=None, remove_stop_words=None,
                     remove_punctuation=None, lower_case=None,
                     include_vague=None, include_normal=None):
    feature_extractor = FeatureExtractorTransformer(extractors)
    return feature_extractor.transform(tagged_essays)
