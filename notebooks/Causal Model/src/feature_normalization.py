from collections import defaultdict
import numpy as np

def accumulate_feat_vals(xs_train):
    def merge_feats(feats):
        for ft, val in feats.items():
            fts_vals[ft].append(val)

    fts_vals = defaultdict(list)
    cnt = 0
    for parser_input in xs_train:
        cnt += 1
        merge_feats(parser_input.opt_features)
        for x in parser_input.other_features_array:
            cnt += 1
            merge_feats(x)
    return fts_vals, cnt


def z_score_normalize_feats(xs_train, xs_test):
    fts_vals, cnt = accumulate_feat_vals(xs_train)

    fts_mean, fts_std = dict(), dict()
    for ft, vals in fts_vals.items():
        v_with_zeros = vals + ([0] * (cnt - len(vals)))
        std = np.std(v_with_zeros)
        if std == 0.0:
            fts_mean[ft] = 0
            fts_std[ft] = vals[0]
        else:
            fts_mean[ft] = np.mean(v_with_zeros)
            fts_std[ft] = np.std(v_with_zeros)

    def to_z_score(fts):
        new_fts = defaultdict(fts.default_factory)
        for ft, val in fts.items():
            if ft in fts_mean:
                new_val = (val - fts_mean[ft]) / fts_std[ft]
                if new_val:
                    new_fts[ft] = new_val
        return new_fts

    def z_score_normalize(parser_input):
        clone = parser_input.clone_without_feats()
        clone.opt_features = to_z_score(parser_input.opt_features)
        clone.all_feats_array = [to_z_score(x) for x in parser_input.all_feats_array]
        clone.other_features_array = [to_z_score(x) for x in parser_input.other_features_array]
        return clone

    new_xs_train = [z_score_normalize(x) for x in xs_train]
    new_xs_test = [z_score_normalize(x) for x in xs_test]
    return new_xs_train, new_xs_test


def min_max_normalize_feats(xs_train, xs_test):
    fts_vals, cnt = accumulate_feat_vals(xs_train)

    fts_min, fts_range = dict(), dict()
    for ft, vals in fts_vals.items():
        v_with_zeros = vals + ([0] * (cnt - len(vals)))
        min_val = np.min(v_with_zeros)
        range_val = np.max(v_with_zeros) - min_val
        fts_min[ft] = min_val
        fts_range[ft] = range_val

    def to_min_max_score(fts):
        new_fts = defaultdict(fts.default_factory)
        for ft, val in fts.items():
            if ft in fts_min and fts_range[ft] != 0:
                new_val = (val - fts_min[ft]) / fts_range[ft]
                if new_val:
                    new_fts[ft] = new_val
        return new_fts

    def min_max_normalize(parser_input):
        clone = parser_input.clone_without_feats()
        clone.opt_features = to_min_max_score(parser_input.opt_features)
        clone.all_feats_array = [to_min_max_score(x) for x in parser_input.all_feats_array]
        clone.other_features_array = [to_min_max_score(x) for x in parser_input.other_features_array]
        return clone

    new_xs_train = [min_max_normalize(x) for x in xs_train]
    new_xs_test = [min_max_normalize(x) for x in xs_test]
    return new_xs_train, new_xs_test