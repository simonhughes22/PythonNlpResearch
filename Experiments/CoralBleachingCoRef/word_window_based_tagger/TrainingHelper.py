from sklearn.linear_model import LogisticRegression

from featurevectorizer import FeatureVectorizer
from wordtagginghelper import flatten_to_wordlevel_feat_tags, get_wordlevel_mostfrequent_ys, get_wordlevel_ys_by_code, \
    get_by_code_from_powerset_predictions

SPARSE_WD_FEATS     = True
MIN_FEAT_FREQ       = 5        # 5 best so far

def train_tagger(fold, essays_TD, essays_VD, wd_test_tags, wd_train_tags,
                 dual, C, penalty, fit_intercept, multi_class, tag_freq):

    # TD and VD are lists of Essay objects. The sentences are lists
    # of featureextractortransformer.Word objects
    """ Data Partitioning and Training """
    td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)
    feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)
    td_X, vd_X = feature_transformer.fit_transform(td_feats), feature_transformer.transform(vd_feats)

    """ compute most common tags per word for training only (but not for evaluation) """
    wd_td_ys = get_wordlevel_mostfrequent_ys(td_tags, wd_train_tags, tag_freq)

    """ TRAIN Tagger """
    solver = 'liblinear'
    if multi_class == 'multinomial':
        solver = "lbfgs"
    model = LogisticRegression(dual=dual, C=C,
                               penalty=penalty, fit_intercept=fit_intercept, multi_class=multi_class, solver=solver,
                               n_jobs=1)
    if fold == 0:
        print(model)

    model.fit(td_X, wd_td_ys)

    wd_td_pred = model.predict(td_X)
    wd_vd_pred = model.predict(vd_X)

    """ TEST Tagger """
    td_wd_predictions_by_code = get_by_code_from_powerset_predictions(wd_td_pred, wd_test_tags)
    vd_wd_predictions_by_code = get_by_code_from_powerset_predictions(wd_vd_pred, wd_test_tags)

    """ Get Actual Ys by code (dict of label to predictions """
    wd_td_ys_by_code = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    wd_vd_ys_by_code = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)
    return td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_by_code, wd_vd_ys_by_code
