from collections import defaultdict

def filter_feats(xs_train, xs_test, prefixes):

    def to_filtered(fts):
        new_fts = defaultdict(fts.default_factory)
        for ft, val in fts.items():
            for prefix in prefixes:
                if ft.startswith(prefix):
                    new_fts[ft] = val
                    break
        return new_fts

    def do_filter(parser_input):
        clone = parser_input.clone_without_feats()
        clone.opt_features = to_filtered(parser_input.opt_features)
        clone.all_feats_array = [to_filtered(x) for x in parser_input.all_feats_array]
        clone.other_features_array = [to_filtered(x) for x in parser_input.other_features_array]
        return clone

    new_xs_train = [do_filter(x) for x in xs_train]
    new_xs_test  = [do_filter(x) for x in xs_test]
    return new_xs_train, new_xs_test