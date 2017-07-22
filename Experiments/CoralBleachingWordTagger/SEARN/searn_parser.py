from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer

from NgramGenerator import compute_ngrams
from oracle import Oracle
from parser import Parser
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
REJECT = "REJECT"  # Not a CREL

CREL_ACTIONS = [
    CAUSE_EFFECT,
    EFFECT_CAUSE,
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

    def train(self, tagged_essays, max_epochs):

        trained_with_beta0 = False
        for i in range(0, max_epochs):
            if self.beta < 0:
                trained_with_beta0 = True

            self.epoch += 1
            print("Epoch: {epoch}".format(epoch=self.epoch))
            print("Beta:  {beta}".format(beta=self.beta))

            # TODO - provide option for different model types here?
            parse_examples = WeightedExamples(labels=PARSE_ACTIONS, positive_value=self.positive_val)
            crel_examples  = WeightedExamples(labels=None,          positive_value=self.positive_val)

            for essay_ix, essay in enumerate(tagged_essays):
                for sent_ix, taggged_sentence in enumerate(essay.sentences):
                    predicted_tags = essay.pred_tagged_sentences[sent_ix]
                    relations = self.parse_sentence(taggged_sentence, predicted_tags, parse_examples, crel_examples)

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
        self.parser_models.append(models)

    def train_crel_models(self, examples):

        self.current_crel_dict_vectorizer = DictVectorizer(sparse=True)

        model = self.base_learner_fact()
        xs = self.current_crel_dict_vectorizer.fit_transform(examples.xs)
        ys = examples.get_labels()
        model.fit(xs, ys)

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
            if action in (REDUCE, LARC) and oracle.tos() == ROOT:
                action_costs[action] = 10
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

    def predict_parse_action(self, feats):
        xs = self.current_parser_dict_vectorizer.transform(feats)
        prob_by_label = {}
        for action in PARSE_ACTIONS:
            prob_by_label[action] = self.current_parser_models[action].predict_proba(xs)[0][-1]

        # TODO - prevent predicting invalid actions here (in case of TOS == ROOT)
        max_act, max_prob = max(prob_by_label.items(), key=lambda tpl: tpl[1])
        return max_act

    def predict_crel_action(self, feats):
        xs = self.current_crel_dict_vectorizer.transform(feats)
        return self.current_crel_model.predict(xs)

    def parse_sentence(self, tagged_sentence, predicted_tags, parse_examples, crel_examples):

        action_history = []
        action_tag_pair_history = []

        all_tags = set()
        all_predicted_tags = set()

        min_ixs, max_ixs = defaultdict(lambda: len(tagged_sentence) + 1), defaultdict(lambda: -1)
        ptag_seq = []
        words = []
        for i, (wd, tags) in enumerate(tagged_sentence):
            words.append(wd)
            all_tags.update(tags)
            ptag = predicted_tags[i]
            if ptag == EMPTY_TAG:
                continue
            if not ptag in all_predicted_tags:
                ptag_seq.append(ptag)
            all_predicted_tags.add(ptag)
            # determine span of each predicted tag
            min_ixs[ptag] = min(min_ixs[ptag], i)
            max_ixs[ptag] = max(max_ixs[ptag], i)

        ground_truth = all_tags.intersection(self.cr_tags)
        ground_truth = set([normalize_cr(crel) for crel in ground_truth])
        # Filter to only those crels that have support in the predicted tags
        supported_crels = set()
        for crel in ground_truth:
            l, r = crel
            if l in all_predicted_tags and r in all_predicted_tags:
                supported_crels.add(crel)
        ground_truth = supported_crels

        # Initialize stack, basic parser and oracle
        stack = Stack(False)
        stack.push(ROOT)
        parser = Parser(stack)
        oracle = Oracle(ground_truth, parser)

        predicted_relations = []

        # Oracle parsing logic
        for tag_ix, buffer in enumerate(ptag_seq):
            buffer = buffer
            word_seq = words[min_ixs[buffer]:max_ixs[buffer] + 1]
            buffer_feats = self.feat_extractor.extract(buffer, word_seq, self.positive_val)
            while True:
                tos = oracle.tos()
                tos_word_seq = words[min_ixs[tos]:max_ixs[tos] + 1]
                tos_feats = self.feat_extractor.extract(tos, tos_word_seq, self.positive_val)

                feats = self.get_conditional_feats(action_history, action_tag_pair_history, tos, buffer,
                                                   ptag_seq[:tag_ix], ptag_seq[tag_ix + 1:])
                interaction_feats = self.get_interaction_feats(tos_feats, buffer_feats)
                feats.update(buffer_feats)
                feats.update(tos_feats)
                feats.update(interaction_feats)

                gold_action = oracle.consult(tos, buffer)

                # Consult Oracle or Model based on coin toss
                rand_float = np.random.random_sample()  # between [0,1) (half-open interval, includes 0 but not 1)
                # If no trained models, always use Oracle
                if rand_float >= self.beta and len(self.self.parser_models) > 0:
                    action = self.predict_parse_action(feats)
                else:
                    action = gold_action

                action_history.append(action)
                action_tag_pair_history.append((action, tos, buffer))

                cost_per_action = self.compute_cost(ground_truth, ptag_seq[tag_ix:], oracle)
                # make a copy as changing later
                parse_examples.add(dict(feats), gold_action, cost_per_action)

                # Prevent invalid action
                if action in (REDUCE, LARC) and oracle.tos() == ROOT:
                    action = gold_action

                # Decide the direction of the causal relation
                if action in [LARC, RARC]:
                    if (tos, buffer) in ground_truth:
                        gold_lr_action = CAUSE_EFFECT
                    elif (buffer, tos) in ground_truth:
                        gold_lr_action = EFFECT_CAUSE
                    else:
                        gold_lr_action = REJECT

                    # Add arc to features
                    feats["ARC:" + action] = self.positive_val
                    rand_float = np.random.random_sample()
                    if rand_float >= self.beta and len(self.crel_models) > 0:
                        # TODO - we need separate models for different decisions
                        lr_action = self.predict_crel_action(feats)
                    else:
                        lr_action = gold_lr_action

                    if lr_action == CAUSE_EFFECT:
                        predicted_relations.append((tos, buffer))
                    elif lr_action == EFFECT_CAUSE:
                        predicted_relations.append((buffer, tos))

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

        pred_relns = [denormalize_cr(cr) for cr in predicted_relations]
        return pred_relns

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
        feats["tos:" + tos] = self.positive_val
        feats["buffer:" + buffer] = self.positive_val
        feats["tos_buffer:" + tos + "|" + buffer] = self.positive_val

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