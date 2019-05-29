from collections import defaultdict

from cost_functions import *
from function_helpers import get_functions_by_name
from searn_parser_breadth_first import SearnModelBreadthFirst
from template_feature_extractor import *

def create_extractor_functions():
    base_extractors = [
        single_words,
        word_pairs,
        three_words,
        between_word_features
    ]
    all_extractor_fns = base_extractors + [
        word_distance,
        valency,
        unigrams,
        third_order,
        label_set,
        size_features
    ]
    all_cost_functions = [
        micro_f1_cost,
        micro_f1_cost_squared,
        micro_f1_cost_plusone,
        micro_f1_cost_plusepsilon,
        binary_cost,
        inverse_micro_f1_cost,
        uniform_cost
    ]
    return base_extractors, all_extractor_fns, all_cost_functions

def add_labels(observed_tags, ys_bytag_sent, set_cr_tags):
    for tag in set_cr_tags:
        if tag in observed_tags:
            ys_bytag_sent[tag].append(1)
        else:
            ys_bytag_sent[tag].append(0)


def get_label_data_essay_level(tagged_essays, set_cr_tags):
    # outputs
    ys_bytag_essay = defaultdict(list)

    for essay in tagged_essays:
        unique_cr_tags = set()
        for sentence in essay.sentences:
            for word, tags in sentence:
                unique_cr_tags.update(set_cr_tags.intersection(tags))
        add_labels(unique_cr_tags, ys_bytag_essay, set_cr_tags)
    return dict(ys_bytag_essay)  # convert to dict so no issue when iterating over if additional keys are present

def essay_to_crels(tagged_essays, set_cr_tags):
    # outputs
    name2crels = defaultdict(set)
    for essay in tagged_essays:
        unique_cr_tags = set()
        for sentence in essay.sentences:
            for word, tags in sentence:
                unique_cr_tags.update(set_cr_tags.intersection(tags))
        name2crels[essay.name] = unique_cr_tags
    return dict(name2crels)


def predict_essay_level(parser, essays, set_cr_tags):
    pred_ys_by_sent = defaultdict(list)
    for essay_ix, essay in enumerate(essays):
        unq_pre_relations = set()
        for sent_ix, taggged_sentence in enumerate(essay.sentences):
            predicted_tags = essay.pred_tagged_sentences[sent_ix]
            pred_relations = parser.predict_sentence(taggged_sentence, predicted_tags)
            unq_pre_relations.update(pred_relations)
        # Store predictions for evaluation
        add_labels(unq_pre_relations, pred_ys_by_sent, set_cr_tags)
    return pred_ys_by_sent

def train_sr_parser(essays_TD, essays_VD, extractor_names, all_extractor_fns, ngrams, stemmed, beta, max_epochs, set_cr_tags,
                    min_feat_freq, cr_tags, base_learner_fact, model):
    extractors = get_functions_by_name(extractor_names, all_extractor_fns)
    # get single cost function
    cost_fn = micro_f1_cost_plusepsilon
    # Ensure all extractors located
    assert len(extractors) == len(extractor_names), "number of extractor functions does not match the number of names"

    template_feature_extractor = NonLocalTemplateFeatureExtractor(extractors=extractors)
    if stemmed:
        ngram_extractor = NgramExtractorStemmed(max_ngram_len=ngrams)
    else:
        ngram_extractor = NgramExtractor(max_ngram_len=ngrams)
    parse_model = model(feature_extractor=template_feature_extractor,
                                         cost_function=cost_fn,
                                         min_feature_freq=min_feat_freq,
                                         ngram_extractor=ngram_extractor, cr_tags=cr_tags,
                                         base_learner_fact=base_learner_fact,
                                         beta=beta,
                                         # log_fn=lambda s: print(s))
                                         log_fn=lambda s: None)

    parse_model.train(essays_TD, max_epochs=max_epochs)

    num_feats = template_feature_extractor.num_features()

    sent_td_ys_bycode = get_label_data_essay_level(essays_TD, set_cr_tags)
    sent_vd_ys_bycode = get_label_data_essay_level(essays_VD, set_cr_tags)

    sent_td_pred_ys_bycode = predict_essay_level(parse_model, essays_TD, cr_tags)
    sent_vd_pred_ys_bycode = predict_essay_level(parse_model, essays_VD, cr_tags)

    return parse_model, num_feats, sent_td_ys_bycode, sent_vd_ys_bycode, sent_td_pred_ys_bycode, sent_vd_pred_ys_bycode