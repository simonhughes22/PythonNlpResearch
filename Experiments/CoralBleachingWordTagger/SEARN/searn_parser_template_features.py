from collections import defaultdict
from typing import Set, List

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
import string

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

class SearnModelTemplateFeaturesCostSensitive(object):
    def __init__(self, ngram_extractor, feature_extractor, cr_tags, base_learner_fact,
                 beta_decay_fn=lambda b: b - 0.1, positive_val=1, sparse=True, log_fn=lambda s: print(s)):

        # init checks
        # assert CAUSER in tags, "%s must be in tags" % CAUSER
        # assert RESULT in tags, "%s must be in tags" % RESULT
        # assert EXPLICIT in tags, "%s must be in tags" % EXPLICIT

        self.log = log_fn

        self.ngram_extractor = ngram_extractor
        self.feat_extractor = feature_extractor  # feature extractor (for use later)
        self.positive_val = positive_val
        self.base_learner_fact = base_learner_fact  # Sklearn classifier
        self.sparse = sparse

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

    def add_cr_labels(self, observed_tags, ys_bytag_sent):
        for tag in self.cr_tags:
            if tag in observed_tags:
                ys_bytag_sent[tag].append(1)
            else:
                ys_bytag_sent[tag].append(0)

    def get_label_data(self, tagged_essays):

        # outputs
        ys_bytag_sent = defaultdict(list)

        for essay in tagged_essays:
            for sentence in essay.sentences:
                unique_cr_tags = set()
                for word, tags in sentence:
                    unique_cr_tags.update(self.cr_tags.intersection(tags))
                self.add_cr_labels(unique_cr_tags, ys_bytag_sent)
        return ys_bytag_sent

    def predict(self, tagged_essays):

        pred_ys_by_sent = defaultdict(list)
        for essay_ix, essay in enumerate(tagged_essays):
            for sent_ix, taggged_sentence in enumerate(essay.sentences):
                predicted_tags = essay.pred_tagged_sentences[sent_ix]
                pred_relations = self.predict_sentence(taggged_sentence, predicted_tags)
                # Store predictions for evaluation
                self.add_cr_labels(pred_relations, pred_ys_by_sent)
        return pred_ys_by_sent

    def train(self, tagged_essays, max_epochs):

        trained_with_beta0 = False
        ys_by_sent = self.get_label_data(tagged_essays)

        for i in range(0, max_epochs):
            if self.beta < 0:
                trained_with_beta0 = True

            self.epoch += 1
            self.log("Epoch: {epoch}".format(epoch=self.epoch))
            self.log("Beta:  {beta}".format(beta=self.beta))

            # TODO - provide option for different model types here?
            parse_examples = WeightedExamples(labels=PARSE_ACTIONS, positive_value=self.positive_val)
            crel_examples  = WeightedExamples(labels=None,          positive_value=self.positive_val)

            pred_ys_by_sent = defaultdict(list)
            for essay_ix, essay in enumerate(tagged_essays):
                for sent_ix, taggged_sentence in enumerate(essay.sentences):
                    predicted_tags = essay.pred_tagged_sentences[sent_ix]
                    pred_relations = self.generate_training_data(taggged_sentence, predicted_tags, parse_examples, crel_examples)
                    # Store predictions for evaluation
                    self.add_cr_labels(pred_relations, pred_ys_by_sent)

            class2metrics = ResultsProcessor.compute_metrics(ys_by_sent, pred_ys_by_sent)
            micro_metrics = micro_rpfa(class2metrics.values()) # type: rpfa
            self.log("Training Metrics: {metrics}".format(metrics=micro_metrics))

            # TODO, dictionary vectorize examples, train a weighted binary classifier for each separate parsing action
            self.train_parse_models(parse_examples)
            self.train_crel_models(crel_examples)

            self.training_datasets_parsing[self.epoch] = parse_examples
            self.training_datasets_crel[self.epoch] = crel_examples

            # Decay beta
            self.beta = self.beta_decay_fn(self.beta)
            if self.beta < 0 and trained_with_beta0:
                self.log("beta decayed below 0 - beta:'{beta}', stopping".format(beta=self.beta))
                break
        # end [for each epoch]
        if not trained_with_beta0:
            self.log("Algorithm hit max epochs without training with beta <= 0 - final_beta:{beta}".format(beta=self.beta))

    def train_parse_models(self, examples):
        models = {}
        self.current_parser_dict_vectorizer = DictVectorizer(sparse=self.sparse)
        xs = self.current_parser_dict_vectorizer.fit_transform(examples.xs)

        for action in PARSE_ACTIONS:

            ys = [1 if i > 0 else 0 for i in examples.get_labels_for(action)]
            weights = examples.get_weights_for(action)

            mdl = self.base_learner_fact()
            mdl.fit(xs, ys, sample_weight=weights)

            models[action] = mdl

        self.current_parser_models = models
        self.parser_models.append(models)

    def predict_parse_action(self, feats, tos):
        xs = self.current_parser_dict_vectorizer.transform(feats)
        prob_by_label = {}
        for action in PARSE_ACTIONS:
            if not self.allowed_action(action, tos):
                continue

            prob_by_label[action] = self.current_parser_models[action].predict_proba(xs)[0][-1]

        max_act, max_prob = max(prob_by_label.items(), key=lambda tpl: tpl[1])
        return max_act

    def train_crel_models(self, examples):

        self.current_crel_dict_vectorizer = DictVectorizer(sparse=self.sparse)

        model = self.base_learner_fact()
        xs = self.current_crel_dict_vectorizer.fit_transform(examples.xs)
        ys = examples.get_labels()
        model.fit(xs, ys)

        self.current_crel_model = model
        self.crel_models.append(model)

    def predict_crel_action(self, feats):
        xs = self.current_crel_dict_vectorizer.transform(feats)
        return self.current_crel_model.predict(xs)[0]

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

    def get_tags_relations_for(self, tagged_sentence, predicted_tags, cr_tags):

        sent_reg_predicted_tags = set()
        sent_act_cr_tags = set()
        tag2ixs = defaultdict(list)

        tag_seq = [None]  # seed with None
        crel_set_seq = [set()]

        pos_tag_seq = []
        latest_tag_posns = {}
        crel_child_tags = defaultdict(set)
        for i, (wd, tags) in enumerate(tagged_sentence):
            if wd in string.punctuation:
                continue

            active_tag = None
            rtag = predicted_tags[i]
            if rtag != EMPTY_TAG:
                active_tag = rtag
                sent_reg_predicted_tags.add(active_tag)
                # if no prev tag and the current matches -2 (a gap of one), skip over
                if active_tag != tag_seq[-1] and \
                        not (tag_seq[-1] is None and (len(tag_seq) > 2) and active_tag == tag_seq[-2]):
                    latest_tag_posns[active_tag] = (active_tag, i)
                    pos_tag_seq.append((active_tag, i))
                # need to be after we update the latest tag position
                tag2ixs[latest_tag_posns[active_tag]].append(i)
            tag_seq.append(active_tag)

            active_crels = tags.intersection(cr_tags)
            for cr in sorted(active_crels):
                sent_act_cr_tags.add(cr)
                if cr not in crel_set_seq[-1] \
                        and not (cr not in crel_set_seq[-1] and (len(crel_set_seq) > 2) and cr in crel_set_seq[-2]):
                    latest_tag_posns[cr] = (cr, i)
            crel_set_seq.append(active_crels)

            # to have child tags, need a tag sequence and a current valid regular tag
            if not active_tag or len(active_crels) == 0:
                continue

            for crel in active_crels:
                l, r = normalize_cr(crel)
                if active_tag in (l, r):
                    crel_child_tags[latest_tag_posns[crel]].add(latest_tag_posns[active_tag])

        pos_crels = []
        for (crelation, crix), tag_pairs in crel_child_tags.items():
            l, r = normalize_cr(crelation)
            # unsupported relation
            if l not in sent_reg_predicted_tags or r not in sent_reg_predicted_tags:
                continue
            tag2pair = defaultdict(list)
            for taga, ixa in tag_pairs:
                tag2pair[taga].append((taga, ixa))
            # un-supported relation
            if l not in tag2pair or r not in tag2pair:
                continue

            l_pairs = tag2pair[l]
            r_pairs = tag2pair[r]
            for pairsa in l_pairs:
                for pairsb in r_pairs:
                    if pairsa != pairsb:
                        pos_crels.append((pairsa, pairsb))

        tag2span = dict()
        for tagpos, ixs in tag2ixs.items():
            tag2span[tagpos] = (min(ixs), max(ixs))

        return pos_tag_seq, pos_crels, tag2span, sent_reg_predicted_tags, sent_act_cr_tags

    def generate_training_data(self, tagged_sentence, predicted_tags, parse_examples, crel_examples, predict_only=False):

        pos_ptag_seq, pos_ground_truth, tag2span, all_predicted_rtags, all_actual_crels = self.get_tags_relations_for(tagged_sentence, predicted_tags, self.cr_tags)
        if predict_only:
            # clear labels
            pos_ground_truth = []
            all_actual_crels = set()

        if len(all_predicted_rtags) == 0:
            return []

        words = [wd for wd, tags in tagged_sentence]

        # Initialize stack, basic parser and oracle
        stack = Stack(verbose=False)
        # needs to be a tuple
        stack.push((ROOT,0))
        parser = Parser(stack)
        oracle = Oracle(pos_ground_truth, parser)

        predicted_relations = set() # type: Set[str]

        # instead of head and modifiers, we will map causers to effects, and vice versa
        effect2causers = defaultdict(set)
        # heads can have multiple modifiers
        cause2effects = defaultdict(set)
        # TODO - get labels

        # tags without positional info
        rtag_seq = [t for t,i in pos_ptag_seq if t[0].isdigit()]
        # if not at least 2 concept codes, then can't parse
        if len(rtag_seq) < 2:
            return []

        tag2words = defaultdict(list)
        for ix, tag_pair in enumerate(pos_ptag_seq):
            bstart, bstop = tag2span[tag_pair]
            word_seq = words[bstart:bstop + 1]
            tag2words[tag_pair] = self.ngram_extractor.extract(word_seq)  # type: List[str]

        # Oracle parsing logic
        # consume the buffer
        for tag_ix, buffer_tag_pair in enumerate(pos_ptag_seq):
            buffer_tag = buffer_tag_pair[0]
            bstart, bstop = tag2span[buffer_tag_pair]

            remaining_buffer_tags = pos_ptag_seq[tag_ix:]
            # Consume the stack
            while True:
                tos_tag_pair = oracle.tos()
                tos_tag = tos_tag_pair[0]

                # Returns -1,-1 if TOS is ROOT
                if tos_tag == ROOT:
                    tstart, tstop = -1, -1
                else:
                    tstart, tstop = tag2span[tos_tag_pair]

                # Note that the end ix in tag2span is always the last index, not the last + 1
                btwn_start, btwn_stop = min(tstop+1, len(words)),  max(0, bstart)

                btwn_word_seq = words[btwn_start:btwn_stop]
                distance = len(btwn_word_seq)
                btwn_word_ngrams = self.ngram_extractor.extract(btwn_word_seq)  # type: List[str]

                feats = self.feat_extractor.extract(stack_tags=stack.contents(), buffer_tags=remaining_buffer_tags,
                                                    tag2word_seq=tag2words,
                                                    between_word_seq=btwn_word_ngrams, distance=distance,
                                                    cause2effects=cause2effects, effect2causers=effect2causers,
                                                    positive_val=self.positive_val)

                # Consult Oracle or Model based on coin toss
                if predict_only:
                    action = self.predict_parse_action(feats, tos_tag)
                else: # if training
                    gold_action = oracle.consult(tos_tag_pair, buffer_tag_pair)
                    rand_float = np.random.random_sample()  # between [0,1) (half-open interval, includes 0 but not 1)
                    # If no trained models, always use Oracle
                    if rand_float >= self.beta and len(self.parser_models) > 0:
                        action = self.predict_parse_action(feats, tos_tag)
                    else:
                        action = gold_action
                    cost_per_action = self.compute_cost(pos_ground_truth, remaining_buffer_tags, oracle)
                    # make a copy as changing later
                    parse_examples.add(dict(feats), gold_action, cost_per_action)

                # Decide the direction of the causal relation
                if action in [LARC, RARC]:

                    c_e_pair = (tos_tag, buffer_tag)
                    # Convert to a string Causer:{l}->Result:{r}
                    cause_effect = denormalize_cr(c_e_pair)

                    e_c_pair = (buffer_tag, tos_tag)
                    # Convert to a string Causer:{l}->Result:{r}
                    effect_cause = denormalize_cr(e_c_pair)

                    if predict_only:
                        gold_lr_action = None
                    else:
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
                    if predict_only or (rand_float >= self.beta and len(self.crel_models) > 0):
                        lr_action = self.predict_crel_action(feats)
                    else:
                        lr_action = gold_lr_action

                    if lr_action == CAUSE_AND_EFFECT:
                        predicted_relations.add(cause_effect)
                        predicted_relations.add(effect_cause)

                        cause2effects[tos_tag_pair].add(buffer_tag_pair)
                        effect2causers[buffer_tag_pair].add(tos_tag_pair)

                        cause2effects[buffer_tag_pair].add(tos_tag_pair)
                        effect2causers[tos_tag_pair].add(buffer_tag_pair)

                    elif lr_action == CAUSE_EFFECT:
                        predicted_relations.add(cause_effect)

                        cause2effects[tos_tag_pair].add(buffer_tag_pair)
                        effect2causers[buffer_tag_pair].add(tos_tag_pair)

                    elif lr_action == EFFECT_CAUSE:
                        predicted_relations.add(effect_cause)

                        cause2effects[buffer_tag_pair].add(tos_tag_pair)
                        effect2causers[tos_tag_pair].add(buffer_tag_pair)

                    elif lr_action == REJECT:
                        pass
                    else:
                        raise Exception("Invalid CREL type")

                    # cost is always 1 for this action (cost of 1 for getting it wrong)
                    #  because getting the wrong direction won't screw up the parse as it doesn't modify the stack
                    if not predict_only:
                        crel_examples.add(dict(feats), gold_lr_action)
                    # Not sure we want to condition on the actions of this crel model
                    # action_history.append(lr_action)
                    # action_tag_pair_history.append((lr_action, tos, buffer))

                # end if action in [LARC,RARC]
                if not oracle.execute(action, tos_tag_pair, buffer_tag_pair):
                    break
                if oracle.is_stack_empty():
                    break

        # Validation logic. Break on pass as relations that should be parsed
        # for pcr in all_actual_crels:
        #     l,r = normalize_cr(pcr)
        #     if l in rtag_seq and r in rtag_seq and pcr not in predicted_relations:
        #         pass

        return predicted_relations

    def predict_sentence(self, tagged_sentence, predicted_tags):
        return self.generate_training_data(tagged_sentence=tagged_sentence, predicted_tags=predicted_tags,
                                           parse_examples=set(), crel_examples=set(), predict_only=True)

    def crel_features(self, action, tos_tag, buffer_tag):
        feats = {}
        feats["ARC_action:" + action]                                               = self.positive_val
        feats["ARC_tos_buffer:" + action + "_:" + tos_tag + "->" + buffer_tag]      = self.positive_val
        feats["ARC_tos:"    + action + "_" + tos_tag]                               = self.positive_val
        feats["ARC_buffer:" + action + "_" + buffer_tag]                            = self.positive_val
        feats["ARC_buffer_equals_tos:" + action + "_" + str(tos_tag == buffer_tag)] = self.positive_val
        feats["ARC_combo:" + action + "_" ",".join(sorted([tos_tag, buffer_tag]))]  = self.positive_val
        return feats

