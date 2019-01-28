from collections import defaultdict

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from Data.featurevectorizer import FeatureVectorizer


class DependencyFeatureInputs(object):
    def __init__(self, essay_name, lsent_ix, rsent_ix, causer_tag, result_tag, causer_words, between_words,
                 result_words,
                 causer_first, between_codes, num_sentences_between):
        self.essay_name = essay_name
        self.lsent_ix = lsent_ix
        self.rsent_ix = rsent_ix
        self.num_sentences_between = num_sentences_between
        self.between_codes = between_codes
        self.causer_first = causer_first
        self.result_words = result_words
        self.between_words = between_words
        self.causer_words = causer_words
        self.result_tag = result_tag
        self.causer_tag = causer_tag
        self.crel = "Causer:{a}->Result:{b}".format(a=causer_tag, b=result_tag)


class DependencyClassifier(object):
    def __init__(self, classifier_fn=LogisticRegression, negative_label=0, sentence_span=2,
                 min_feat_freq=10, log_fn=lambda s: print(s), ):
        self.log = log_fn
        self.epoch = 0
        self.negative_label = negative_label
        self.sentence_span = sentence_span
        self.min_feat_freq = min_feat_freq
        self.vectorizer = FeatureVectorizer(min_feature_freq=min_feat_freq)
        self.clf = classifier_fn()
        self.fit_vectorizer = False

    def __fill_in_gaps__(self, tag_seq):
        new_tag_seq = []
        for i, tag in enumerate(tag_seq):
            if tag == EMPTY \
                    and i > 0 \
                    and tag_seq[i - 1] != EMPTY \
                    and i < len(tag_seq) - 1 \
                    and tag_seq[i - 1] == tag_seq[i + 1]:
                tag = tag_seq[i - 1]

            new_tag_seq.append(tag)
        return new_tag_seq

    def __compute_tag_2_spans__(self, essay):
        sent_tag2spans = []
        wd_ix = -1
        essay_words = []
        essay_ptags = []
        for sent_ix in range(len(essay.sentences)):
            words, tag_seq = zip(*essay.sentences[sent_ix])

            tag2spans = []  # maps to list of start and end spans for each tag
            sent_tag2spans.append(tag2spans)

            last_tag = EMPTY
            tag_start = None
            ptags_sent = self.__fill_in_gaps__(essay.pred_tagged_sentences[sent_ix])
            current_crel_tags = set()
            for i, ptag in enumerate(ptags_sent):
                wd_ix += 1
                essay_words.append(words[i])
                essay_ptags.append(ptag)
                # Tag changed
                if ptag != last_tag:
                    if last_tag != EMPTY:
                        tag2spans.append((last_tag, tag_start, wd_ix - 1, sent_ix, current_crel_tags))
                    tag_start = wd_ix
                    current_crel_tags = set()
                current_crel_tags.update(to_is_valid_crel(tag_seq[i]))
                last_tag = ptag
            if last_tag != EMPTY:
                tag2spans.append((last_tag, tag_start, wd_ix, len(essay.sentences) - 1, current_crel_tags))
        assert len(essay_words) == len(essay_ptags)
        return sent_tag2spans, essay_words, essay_ptags

    def __combine_feats__(self, ftsa, ftsb):
        fts = {}
        for a, aval in ftsa.items():
            for b, bval in ftsb.items():
                fts[a + "|" + b] = aval * bval
        return fts

    def create_features(self, feat_inp):
        feats = {}
        feats[feat_inp.crel] = 1
        feats["Causer:{tag}".format(tag=feat_inp.causer_tag)] = 1
        feats["Result:{tag}".format(tag=feat_inp.result_tag)] = 1
        cs_fts, res_fts = {}, {}
        for wd in feat_inp.causer_words:
            cs_fts["Causer:{wd}".format(wd=wd)] = 1
        feats.update(cs_fts)
        for wd in feat_inp.result_words:
            res_fts["Result:{wd}".format(wd=wd)] = 1
        feats.update(res_fts)
        feats.update(self.__combine_feats__(cs_fts, res_fts))
        btwn_fts = {}
        for wd in feat_inp.between_words:
            btwn_fts["Between:{wd}".format(wd=wd)] = 1
        feats.update(btwn_fts)
        #         feats.update(self.__combine_feats__(cs_fts, btwn_fts))
        #         feats.update(self.__combine_feats__(res_fts, btwn_fts))
        if feat_inp.causer_first:
            feats["Left2Right"] = 1
        else:
            feats["Right2Left"] = 1

        if feat_inp.num_sentences_between == 0:
            feats["SameSentence"] = 1
        feats["SentBetween"] = feat_inp.num_sentences_between
        if feat_inp.num_sentences_between <= 1:
            feats["SentBetween<=1"] = 1
        if feat_inp.num_sentences_between <= 2:
            feats["SentBetween<=2"] = 1
        else:
            feats["SentBetween>2"] = 1

        num_codes_between = len(feat_inp.between_codes)
        feats["CodesBetween"] = num_codes_between
        if num_codes_between <= 1:
            feats["CodesBetween<=1"] = 1
        if num_codes_between <= 2:
            feats["CodesBetween<=2"] = 1
        else:
            feats["CodesBetween>2"] = 1
        return feats

    def __generate_training_data__(self, essays):
        xs, ys, essay_sent_feat_inpts = [], [], []
        for essay_ix, essay in enumerate(essays):
            sent_tag2spans, essay_words, essay_ptags = self.__compute_tag_2_spans__(essay)
            for sent_ix in range(len(sent_tag2spans)):
                # tag 2 spans for sentence
                next_tag2spans = []
                # grab next few sentences' predicted tags
                for offset in range(0, self.sentence_span + 1):
                    if (sent_ix + offset) < len(sent_tag2spans):
                        next_tag2spans.extend(sent_tag2spans[sent_ix + offset])

                for ltag_ix, (ltag, lstart_ix, lend_ix, lsent_ix, lcrels) in enumerate(sent_tag2spans[sent_ix]):
                    for rtag, rstart_ix, rend_ix, rsent_ix, rcrels in next_tag2spans[ltag_ix + 1:]:
                        num_sent_between = rsent_ix - lsent_ix

                        ltag_words = essay_words[lstart_ix:lend_ix + 1]
                        between_words = essay_words[lend_ix + 1:rstart_ix]
                        rtag_words = essay_words[rstart_ix:rend_ix + 1]
                        between_codes = essay_ptags[lend_ix + 1:rstart_ix]

                        lbls = set(lcrels).union(rcrels)

                        feat_ext_inp = DependencyFeatureInputs(essay_name=essay.name, lsent_ix=lsent_ix,
                                                               rsent_ix=rsent_ix,
                                                               causer_tag=ltag, result_tag=rtag,
                                                               causer_words=ltag_words, between_words=between_words,
                                                               result_words=rtag_words, causer_first=True,
                                                               between_codes=between_codes,
                                                               num_sentences_between=num_sent_between)
                        x = self.create_features(feat_ext_inp)
                        xs.append(x)
                        ys.append(1 if feat_ext_inp.crel in lbls else self.negative_label)
                        essay_sent_feat_inpts.append(feat_ext_inp)

                        feat_ext_inp = DependencyFeatureInputs(essay_name=essay.name, lsent_ix=lsent_ix,
                                                               rsent_ix=rsent_ix,
                                                               causer_tag=rtag, result_tag=ltag,
                                                               causer_words=rtag_words, between_words=between_words,
                                                               result_words=ltag_words, causer_first=False,
                                                               between_codes=between_codes,
                                                               num_sentences_between=num_sent_between)
                        x = self.create_features(feat_ext_inp)
                        xs.append(x)
                        ys.append(1 if feat_ext_inp.crel in lbls else self.negative_label)
                        essay_sent_feat_inpts.append(feat_ext_inp)

        if not self.fit_vectorizer:
            xs_array = self.vectorizer.fit_transform(xs)
            self.fit_vectorizer = True
        else:
            xs_array = self.vectorizer.transform(xs)
        return xs_array, ys, essay_sent_feat_inpts

    def train(self, train_essays):
        # Note that there are a small number of crels that span 2 sentences
        xs, ys, essay_sent_crel = self.__generate_training_data__(essays=train_essays)
        self.clf.fit(X=xs, y=ys)

    def __group_predictions_by_essay__(self, essay_sent_feat_inpts, preds, threshold):
        name2pred = defaultdict(set)
        for feat_inputs, pred in zip(essay_sent_feat_inpts, preds):
            if pred >= threshold:
                name2pred[feat_inputs.essay_name].add(feat_inputs.crel)
        return name2pred

    def __group_predictions_by_sentence__(self, essay_sent_feat_inpts, preds, threshold):
        namesent2pred = defaultdict(set)
        for feat_inputs, pred in zip(essay_sent_feat_inpts, preds):
            if pred >= threshold:
                namesent2pred[(feat_inputs.essay_name, feat_inputs.lsent_ix)].add(feat_inputs.crel)
                namesent2pred[(feat_inputs.essay_name, feat_inputs.rsent_ix)].add(feat_inputs.crel)
        return namesent2pred

    def predict_probability(self, tagged_essays, min_prob=0.1):
        # Get predicted probabilities
        xs, _, essay_sent_feat_inpts = self.__generate_training_data__(essays=tagged_essays)
        probs = self.clf.predict_proba(xs)[:, 1]
        return self.__group_predictions_by_essay__(essay_sent_feat_inpts=essay_sent_feat_inpts, preds=probs,
                                                   threshold=min_prob)

    def evaluate(self, tagged_essays, print_classification_report=True):
        # Note that there are a small number of crels that span 2 sentences
        xs, ys, essay_sent_feat_inpts = self.__generate_training_data__(essays=tagged_essays)
        preds = self.clf.predict(xs)
        if print_classification_report:
            print(classification_report(y_true=ys, y_pred=preds))

        namesent2pred = self.__group_predictions_by_sentence__(
            essay_sent_feat_inpts=essay_sent_feat_inpts, preds=preds, threshold=1.0)

        pred_ys_bytag_sent = defaultdict(list)
        for essay in tagged_essays:
            for sent_ix, sentence in enumerate(essay.sentences):
                unique_cr_tags = namesent2pred[(essay.name, sent_ix)]
                add_cr_labels(unique_cr_tags, pred_ys_bytag_sent)
        return pred_ys_bytag_sent

    def evaluate_essay_level(self, tagged_essays, print_classification_report=True):
        # Note that there are a small number of crels that span 2 sentences
        xs, ys, essay_sent_feat_inpts = self.__generate_training_data__(essays=tagged_essays)
        preds = self.clf.predict(xs)
        if print_classification_report:
            print(classification_report(y_true=ys, y_pred=preds))

        namesent2pred = self.__group_predictions_by_essay__(
            essay_sent_feat_inpts=essay_sent_feat_inpts, preds=preds, threshold=1.0)

        pred_ys_bytag_essay = defaultdict(list)
        for essay in tagged_essays:
            unique_cr_tags = namesent2pred[essay.name]
            add_cr_labels(unique_cr_tags, pred_ys_bytag_essay)
        return pred_ys_bytag_essay
