from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer

from NgramGenerator import compute_ngrams
from Rpfa import micro_rpfa
from oracle import Oracle
from parser import Parser
from results_procesor import ResultsProcessor
from shift_reduce_helper import *
from stack import Stack
from weighted_examples import WeightedExamples
import numpy as np

PARSE_ACTIONS = [
    SHIFT,
    REDUCE,
    LARC,
    RARC,
    SKIP
]

CAUSE_EFFECT = "CAUSE_EFFECT"
EFFECT_CAUSE = "EFFECT_CAUSE"
CAUSE_AND_EFFECT = "CAUSE_AND_EFFECT"
REJECT = "REJECT"  # Not a CREL

CREL_ACTIONS = [
    CAUSE_EFFECT,
    EFFECT_CAUSE,
    CAUSE_AND_EFFECT,
    REJECT
]

class SearnModel(object):
    def __init__(self, feature_extractor, cr_tags, base_learner_fact, beta_decay_fn=lambda b: b - 0.1, positive_val=1):
        # init checks
        # assert CAUSER in tags, "%s must be in tags" % CAUSER
        # assert RESULT in tags, "%s must be in tags" % RESULT
        # assert EXPLICIT in tags, "%s must be in tags" % EXPLICIT

        self.feat_extractor = feature_extractor  # feature extractor (for use later)
        self.positive_val = positive_val
        self.base_learner_fact = base_learner_fact  # Sklearn classifier

        self.cr_tags = set(cr_tags)  # causal relation tags
        self.epoch = -1
        self.beta = 1.0  # probability of using oracle for each parsing decision, initialize to 1 as we don't use until second epoch
        self.beta_decay_fn = beta_decay_fn
        self.stack = []

        self.parser_models = []
        self.current_parser_models = None
        self.current_parser_dict_vectorizer = None
        self.crel_models = []
        self.current_crel_model = None
        self.current_crel_dict_vectorizer = None

        self.training_datasets_parsing = {}
        self.training_datasets_crel = {}
        self.current_model = None

    def add_labels(self, observed_tags, ys_bytag_sent):
        for tag in self.cr_tags:
            if tag in observed_tags:
                ys_bytag_sent[tag].append(1)
            else:
                ys_bytag_sent[tag].append(0)

    def get_label_data(self, tagged_essays):

        # outputs
        ys_bytag_sent = defaultdict(list)

        # cut texts after this number of words (among top max_features most common words)
        tag_freq = defaultdict(int)
        for essay in tagged_essays:
            for sentence in essay.sentences:
                unique_tags = set()
                for word, tags in sentence:
                    unique_tags.update(self.cr_tags.intersection(tags))
                    for tag in tags:
                        tag_freq[tag] += 1
                self.add_labels(unique_tags, ys_bytag_sent)
        return ys_bytag_sent, tag_freq

    def train(self, tagged_essays, max_epochs):

        trained_with_beta0 = False
        ys_by_sent, tag_freq = self.get_label_data(tagged_essays)

        for i in range(0, max_epochs):
            if self.beta < 0:
                trained_with_beta0 = True

            self.epoch += 1
            print("Epoch: {epoch}".format(epoch=self.epoch))
            print("Beta:  {beta}".format(beta=self.beta))

            # TODO - provide option for different model types here?
            parse_examples = WeightedExamples(labels=PARSE_ACTIONS, positive_value=self.positive_val)
            crel_examples  = WeightedExamples(labels=None,          positive_value=self.positive_val)

            pred_ys_by_sent = defaultdict(list)
            for essay_ix, essay in enumerate(tagged_essays):
                for sent_ix, taggged_sentence in enumerate(essay.sentences):
                    predicted_tags = essay.pred_tagged_sentences[sent_ix]
                    pred_relations = self.parse_sentence(taggged_sentence, predicted_tags, tag_freq, parse_examples, crel_examples)
                    # Store predictions for evaluation
                    self.add_labels(pred_relations, pred_ys_by_sent)

            class2metrics = ResultsProcessor.compute_metrics(ys_by_sent, pred_ys_by_sent)
            micro_metrics = micro_rpfa(class2metrics.values()) # type: rpfa
            print("Training Metrics: {metrics}".format(metrics=micro_metrics))

            # TODO, dictionary vectorize examples, train a weighted binary classifier for each separate parsing action
            self.train_parse_models(parse_examples)
            self.train_crel_models(crel_examples)

            self.training_datasets_parsing[self.epoch] = parse_examples
            self.training_datasets_crel[self.epoch] = crel_examples

            # Decay beta
            self.beta = self.beta_decay_fn(self.beta)
            if self.beta < 0 and trained_with_beta0:
                print("beta decayed below 0 - beta:'{beta}', stopping".format(beta=self.beta))
                break
        # end [for each epoch]
        if not trained_with_beta0:
            print("Algorithm hit max epochs without training with beta <= 0 - final_beta:{beta}".format(beta=self.beta))

    def train_parse_models(self, examples):
        models = {}
        self.current_parser_dict_vectorizer = DictVectorizer(sparse=True)
        xs = self.current_parser_dict_vectorizer.fit_transform(examples.xs)

        for action in PARSE_ACTIONS:
            mdl = self.base_learner_fact()
            ys = examples.get_labels_for(action)
            weights = examples.get_weights_for(action)
            # TODO - train cost sensitive classifier
            mdl.fit(xs, ys)
            models[action] = mdl
        self.current_parser_models = models
        self.parser_models.append(models)

    def train_crel_models(self, examples):

        self.current_crel_dict_vectorizer = DictVectorizer(sparse=True)

        model = self.base_learner_fact()
        xs = self.current_crel_dict_vectorizer.fit_transform(examples.xs)
        ys = examples.get_labels()
        model.fit(xs, ys)

        self.current_crel_model = model
        self.crel_models.append(model)

    def add_relation(self, action, tos, buffer, ground_truth, relations):
        if action in [LARC, RARC]:
            if (tos, buffer) in ground_truth:
                relations.add((tos, buffer))
            elif (buffer, tos) in ground_truth:
                relations.add((buffer, tos))

    def relations_for_action(self, forced_action, ground_truth, remaining_buffer, oracle):
        relns = set()
        oracle = oracle.clone()
        first_action = True

        for buffer in remaining_buffer:
            while True:
                tos = oracle.tos()
                if not first_action:  # need to force first action
                    action = oracle.consult(tos, buffer)
                else:
                    action = forced_action
                    first_action = False
                self.add_relation(action, tos, buffer, ground_truth, relns)
                if not oracle.execute(action, tos, buffer):
                    break
                if oracle.is_stack_empty():
                    break
        return relns

    def allowed_action(self, action, tos):
        return not(tos == ROOT and action in (REDUCE, LARC, RARC))

    def compute_cost(self, ground_truth, remaining_buffer, oracle):

        tos = oracle.tos()
        gold_action = oracle.consult(tos, remaining_buffer[0])
        gold_parse = self.relations_for_action(gold_action, ground_truth, remaining_buffer, oracle)

        action_costs = {}
        for action in PARSE_ACTIONS:
            if action == gold_action:
                continue

            # Prevent invalid parse actions
            #TODO is the best option?
            if not self.allowed_action(action, tos):
                # cost is number of relations that will be missed or at least 1
                action_costs[action] = max(1, len(gold_parse))
                continue

            parse = self.relations_for_action(action, ground_truth, remaining_buffer, oracle)
            num_matches = len(gold_parse.intersection(parse))
            # recall
            false_negatives = len(gold_parse) - num_matches
            # precision
            false_positives = len(parse) - num_matches
            # Cost is the total of the false positives + false negatives
            cost = false_positives + false_negatives
            action_costs[action] = cost

        # Cost of the gold action is the mean of all of the wrong choices
        action_costs[gold_action] = np.mean(list(action_costs.values()))
        return action_costs

    def predict_parse_action(self, feats, tos):
        xs = self.current_parser_dict_vectorizer.transform(feats)
        prob_by_label = {}
        for action in PARSE_ACTIONS:
            if not self.allowed_action(action, tos):
                continue

            prob_by_label[action] = self.current_parser_models[action].predict_proba(xs)[0][-1]

        max_act, max_prob = max(prob_by_label.items(), key=lambda tpl: tpl[1])
        return max_act

    def predict_crel_action(self, feats):
        xs = self.current_crel_dict_vectorizer.transform(feats)
        return self.current_crel_model.predict(xs)

    def get_tags_relations_for(self, tagged_sentence, predicted_tags, tag_freq, cr_tags):

        most_common_tag = [None]  # seed with None
        most_common_crel = [None]
        pos_tag_seq = []
        crel_seq = []

        pos_crel_child_tags = defaultdict(set)
        # non positional
        sent_reg_predicted_tags = set()
        sent_act_cr_tags = set()
        tag2words = defaultdict(list)
        for i, (wd, actual_tags) in enumerate(tagged_sentence):
            # just a single predicted tag
            pred_rtag = predicted_tags[i]
            tag = None
            # Get tag seq
            if pred_rtag != EMPTY_TAG:
                # only use explicit tag if it's the only tag (prefer concept code tags if both present)
                tag = pred_rtag
                sent_reg_predicted_tags.add(tag)
                # if no prev tag and the current matches -2 (a gap of one), skip over
                if tag != most_common_tag[-1] and \
                        not (most_common_tag[-1] is None and (len(most_common_tag) > 2) and tag == most_common_tag[-2]):
                    pos_tag_seq.append((tag, i))

            if len(pos_tag_seq) > 0:
                tag2words[pos_tag_seq[-1]].append(wd)
            most_common_tag.append(tag)

            pos_crels = actual_tags.intersection(cr_tags)
            sent_act_cr_tags.update(pos_crels)
            crel = None
            if pos_crels:
                crel = max(pos_crels, key=lambda cr: tag_freq[cr])
                # skip over gaps of one crel
                if crel != most_common_crel[-1] \
                    and not (most_common_crel[-1] is None and (len(most_common_crel) > 2) and crel == most_common_crel[-2]):
                    crel_seq.append((crel, i))
            most_common_crel.append(crel)

            # to have child tags, need a tag sequence and a current valid regular tag
            if not tag or len(pos_tag_seq) == 0 or not crel or len(crel_seq) == 0:
                continue

            if tag != pos_tag_seq[-1][0]:
                raise Exception("Tags don't match % s" % str((i, tag, pos_tag_seq[-1])))
            if crel != crel_seq[-1][0]:
                raise Exception("Crels don't match % s" % str((i, crel, crel_seq[-1])))

            l, r = normalize_cr(crel)
            if tag in (l, r):
                pos_crel_child_tags[crel_seq[-1]].add(pos_tag_seq[-1])

        pos_crels = []
        for _, tag_pairs in pos_crel_child_tags.items():
            tag2pairs = defaultdict(set)
            for tag, ix in tag_pairs:
                tag2pairs[tag].add((tag, ix))
            for taga, pairsa in tag2pairs.items():
                for tagb, pairsb in tag2pairs.items():
                    if pairsa != pairsb:
                        for pa in pairsa:
                            for pb in pairsb:
                                pos_crels.append((pa, pb))

        return pos_tag_seq, pos_crels, tag2words, sent_reg_predicted_tags, sent_act_cr_tags

    def parse_sentence(self, tagged_sentence, predicted_tags, tag_freq, parse_examples, crel_examples):

        action_history = []
        action_tag_pair_history = []

        pos_ptag_seq, pos_ground_truth, tag2words, all_predicted_rtags, all_actual_crels = self.get_tags_relations_for(tagged_sentence, predicted_tags, tag_freq, self.cr_tags)

        # Need at least 2 tags for a causal relation to be detected
        # WRONG - could be an A=>A relation
        #if len(all_predicted_rtags) < 2:
        #    return []

        if len(all_predicted_rtags) == 0:
            return []

        # Initialize stack, basic parser and oracle
        stack = Stack(verbose=False)
        # needs to be a tuple
        stack.push((ROOT,0))
        parser = Parser(stack)
        oracle = Oracle(pos_ground_truth, parser)

        predicted_relations = set()

        # tags without positional info
        tag_seq = [t for t,i in pos_ptag_seq]

        # Oracle parsing logic
        for tag_ix, buffer in enumerate(pos_ptag_seq):
            buffer_tag = buffer[0]
            word_seq = tag2words[buffer]
            buffer_feats = self.feat_extractor.extract(buffer_tag, word_seq, self.positive_val)

            while True:
                tos = oracle.tos()
                tos_tag = tos[0]
                tos_word_seq = tag2words[tos]
                tos_feats = self.feat_extractor.extract(tos_tag, tos_word_seq, self.positive_val)

                feats = self.get_conditional_feats(action_history, action_tag_pair_history, tos_tag, buffer_tag,
                                                   tag_seq[:tag_ix], tag_seq [tag_ix + 1:])
                interaction_feats = self.get_interaction_feats(tos_feats, buffer_feats)
                feats.update(buffer_feats)
                feats.update(tos_feats)
                feats.update(interaction_feats)

                gold_action = oracle.consult(tos, buffer)

                # Consult Oracle or Model based on coin toss
                rand_float = np.random.random_sample()  # between [0,1) (half-open interval, includes 0 but not 1)
                # If no trained models, always use Oracle
                if rand_float >= self.beta and len(self.parser_models) > 0:
                    action = self.predict_parse_action(feats, tos)
                else:
                    action = gold_action

                action_history.append(action)
                action_tag_pair_history.append((action, tos_tag, buffer_tag))

                cost_per_action = self.compute_cost(pos_ground_truth, pos_ptag_seq[tag_ix:], oracle)
                # make a copy as changing later
                parse_examples.add(dict(feats), gold_action, cost_per_action)

                # Decide the direction of the causal relation
                if action in [LARC, RARC]:

                    cause_effect = denormalize_cr((tos_tag,    buffer_tag))
                    effect_cause = denormalize_cr((buffer_tag, tos_tag))

                    if cause_effect in all_actual_crels and effect_cause in all_actual_crels:
                        gold_lr_action = CAUSE_AND_EFFECT
                    elif cause_effect in all_actual_crels:
                        gold_lr_action = CAUSE_EFFECT
                    elif effect_cause in all_actual_crels:
                        gold_lr_action = EFFECT_CAUSE
                    else:
                        gold_lr_action = REJECT

                    # Add additional features
                    # needs to be before predict below
                    feats.update(self.crel_features(action, tos_tag, buffer_tag))
                    rand_float = np.random.random_sample()
                    if rand_float >= self.beta and len(self.crel_models) > 0:
                        lr_action = self.predict_crel_action(feats)
                    else:
                        lr_action = gold_lr_action

                    if lr_action == CAUSE_AND_EFFECT:
                        predicted_relations.add(cause_effect)
                        predicted_relations.add(effect_cause)
                    elif lr_action == CAUSE_EFFECT:
                        predicted_relations.add(cause_effect)
                    elif lr_action == EFFECT_CAUSE:
                        predicted_relations.add(effect_cause)
                    elif lr_action == REJECT:
                        pass
                    else:
                        raise Exception("Invalid CREL type")

                    # cost is always 1 for this action (cost of 1 for getting it wrong)
                    #  because getting the wrong direction won't screw up the parse as it doesn't modify the stack
                    crel_examples.add(dict(feats), gold_lr_action)
                    # Not sure we want to condition on the actions of this crel model
                    # action_history.append(lr_action)
                    # action_tag_pair_history.append((lr_action, tos, buffer))

                # end if action in [LARC,RARC]
                if not oracle.execute(action, tos, buffer):
                    break
                if oracle.is_stack_empty():
                    break

        return predicted_relations

    def crel_features(self, action, tos, buffer):
        feats = {}
        feats["ARC_action:" + action]                   = self.positive_val
        feats["ARC_tos_buffer:" + tos + "->" + buffer]  = self.positive_val
        feats["ARC_tos:" + tos]                         = self.positive_val
        feats["ARC_buffer:" + buffer]                   = self.positive_val
        feats["ARC_buffer_equals_tos:" + str(tos == buffer)] = self.positive_val
        feats["ARC_combo:" + ",".join(sorted([tos,buffer]))] = self.positive_val
        return feats

    def get_interaction_feats(self, fts1, fts2):
        interactions = {}
        for fta, vala in fts1.items():
            for ftb, valb in fts2.items():
                if vala > 0 and valb > 0:
                    interactions["inter: " + fta + "|" + ftb] = self.positive_val
        return interactions

    def get_conditional_feats(self, action_history, action_tag_pair_history, tos, buffer, previous_tags,
                              subsequent_tags):
        feats = {}
        if len(action_history) == 0:
            feats["first_action"] = self.positive_val
        if len(subsequent_tags) == 0:
            feats["last_tag"] = 1

        feats["num_actions"] = len(action_history)
        feats["num_prev_tags"] = len(previous_tags)
        feats["num_subsequent_tags"] = len(subsequent_tags)

        feats["num_tags"] = 1 + len(previous_tags) + len(subsequent_tags)

        feats["tos:" + tos] = self.positive_val
        feats["buffer:" + buffer] = self.positive_val
        feats["tos_buffer:" + tos + "|" + buffer] = self.positive_val
        feats["tos_buffer_combo:" + ",".join(sorted([tos, buffer]))] = self.positive_val

        ### PREVIOUS TAGS
        for i, tag in enumerate(previous_tags[::-1]):
            feats["prev_tag-{i}:{tag}".format(i=i, tag=tag)] = self.positive_val
            feats["prev_tag:{tag}".format(tag=tag)] = self.positive_val

        if len(previous_tags) > 0:
            feats["prev-tag-tos-buffer:{tag}_{tos}_{buffer}".format(tag=previous_tags[-1], tos=tos,
                                                                    buffer=buffer)] = self.positive_val
            feats["prev-tag-buffer:{tag}_{buffer}".format(tag=previous_tags[-1], buffer=buffer)] = self.positive_val
            feats["prev-tag-tos:{tag}_{tos}".format(tag=previous_tags[-1], tos=tos)] = self.positive_val
            bigrams = compute_ngrams(previous_tags, 2, 2)
            for i, bigram in enumerate(bigrams[::-1]):
                feats["prev_bigram-tag-{i}:{tag}".format(i=i, tag=str(bigram))] = self.positive_val
                feats["prev_bigram-tag:{tag}".format(tag=str(bigram))] = self.positive_val

        ### REMAINING TAGS
        for i, tag in enumerate(subsequent_tags):
            feats["subseq_tag-{i}:{tag}".format(i=i, tag=tag)] = self.positive_val
            feats["subseq_tag:{tag}".format(i=i, tag=tag)] = self.positive_val

        if len(subsequent_tags) > 0:
            feats["subseq-tag-tos-buffer:{tag}_{buffer}".format(tag=subsequent_tags[0], tos=tos,
                                                                buffer=buffer)] = self.positive_val
            feats["subseq-tag-buffer:{tag}_{buffer}".format(tag=subsequent_tags[0], buffer=buffer)] = self.positive_val
            feats["subseq-tag-tos:{tag}_{tos}".format(tag=subsequent_tags[0], tos=tos)] = self.positive_val
            bigrams = compute_ngrams(subsequent_tags, 2, 2)
            for i, bigram in enumerate(bigrams):
                feats["subseq_bigram-tag-{i}:{tag}".format(i=i, tag=str(bigram))] = self.positive_val
                feats["subseq_bigram-tag:{tag}".format(tag=str(bigram))] = self.positive_val

        # features for each previous action
        action_tally = defaultdict(int)
        for i, action in enumerate(action_history[::-1]):
            feats["action-{i}:{action}".format(i=i, action=action)] = self.positive_val
            feats["action:{action}".format(action=action)] = self.positive_val
            action_tally[action] += 1

            # Features for the number of times each action has been performed
        for action, count in action_tally.items():
            feats["action-tally:{action}_{count}".format(action=action, count=count)] = self.positive_val

        if len(action_history) > 0:
            feats["prev_action-tos-buffer:{action}_{tos}_{buffer}".format(action=action_history[-1], tos=tos,
                                                                          buffer=buffer)] = self.positive_val
            feats["prev_action-buffer:{action}_{buffer}".format(action=action_history[-1],
                                                                buffer=buffer)] = self.positive_val
            feats["prev_action-tos:{action}_{tos}".format(action=action_history[-1], tos=tos)] = self.positive_val
            bigrams = compute_ngrams(action_history, 2, 2)
            for i, bigram in enumerate(bigrams[::-1]):
                feats["prev_bigram_action-{i}:{tag}".format(i=i, tag=str(bigram))] = self.positive_val
                feats["prev_bigram_action:{tag}".format(tag=str(bigram))] = self.positive_val

        for i, (action, prev_tos, prev_buffer) in enumerate(action_tag_pair_history[::-1]):
            feats["actiontag-{i}:{action}_{tos}_{buffer}".format(i=i, action=action, tos=prev_tos,
                                                                 buffer=prev_buffer)] = self.positive_val
            feats["actiontag:{action}_{tos}_{buffer}".format(action=action, tos=prev_tos,
                                                             buffer=prev_buffer)] = self.positive_val

            feats["actiontos-{i}:{action}_{tos}".format(i=i, action=action, tos=prev_tos)] = self.positive_val
            feats["actiontos:{action}_{tos}".format(action=action, tos=prev_tos)] = self.positive_val

            feats[
                "actionbuffer-{i}:{action}_{buffer}".format(i=i, action=action, buffer=prev_buffer)] = self.positive_val
            feats["actionbuffer:{action}_{buffer}".format(action=action, buffer=prev_buffer)] = self.positive_val

        if len(action_tag_pair_history) > 0:
            action, prev_tos, prev_buffer = action_tag_pair_history[-1]
            feats[
                "prev_actiontag_tos_buffer_currnet_tos_current_buffer:{action}_{prev_tos}_{prev_buffer}_{tos}_{buffer}".format(
                    action=action, prev_tos=prev_tos, prev_buffer=prev_buffer, tos=tos,
                    buffer=buffer)] = self.positive_val
            feats["prev_actiontag_tos_buffer_current_buffer:{action}_{prev_tos}_{prev_buffer}_{buffer}".format(
                action=action, prev_tos=prev_tos, prev_buffer=prev_buffer, buffer=buffer)] = self.positive_val
            feats["prev_actiontag_tos_buffer_current_tos:{action}_{prev_tos}_{prev_buffer}_{tos}".format(action=action,
                                                                                                         prev_tos=prev_tos,
                                                                                                         prev_buffer=prev_buffer,
                                                                                                         tos=tos)] = self.positive_val

        return feats
