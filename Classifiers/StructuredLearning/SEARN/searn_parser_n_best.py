import string
from collections import defaultdict

import numpy as np
from oracle import Oracle
from shift_reduce_helper import *
from typing import Set, List

from shift_reduce_parser import ShiftReduceParser
from weighted_examples import WeightedExamples

from Rpfa import micro_rpfa
from StructuredLearning.SEARN.stack import Stack
from featurevectorizer import FeatureVectorizer
from results_procesor import ResultsProcessor

from Classifiers.StructuredLearning.SEARN.searn_parser import SearnModelTemplateFeatures


class SearnModelNBest(SearnModelTemplateFeatures):
    def __init__(self, *args, **kwargs):
        super(SearnModelNBest, self).__init__(*args, **kwargs)

    def predict(self, tagged_essays):

        pred_ys_by_sent = defaultdict(list)
        for essay_ix, essay in enumerate(tagged_essays):
            for sent_ix, taggged_sentence in enumerate(essay.sentences):
                predicted_tags = essay.pred_tagged_sentences[sent_ix]
                pred_relations = self.predict_sentence(taggged_sentence, predicted_tags)
                # Store predictions for evaluation
                self.add_cr_labels(pred_relations, pred_ys_by_sent)
        return pred_ys_by_sent

    def predict_parse_action(self, feats, tos, models, vectorizer):

        xs = vectorizer.transform(feats)
        prob_by_label = {}
        for action in self.randomize_actions():
            if not allowed_action(action, tos):
                continue

            prob_by_label[action] = models[action].predict_proba(xs)[0][-1]

        max_act, max_prob = max(prob_by_label.items(), key=lambda tpl: tpl[1])
        return max_act

    def generate_training_data(self, tagged_sentence, predicted_tags):

        pos_ptag_seq, _, tag2span, all_predicted_rtags, _ = self.get_tags_relations_for(
            tagged_sentence, predicted_tags, self.cr_tags)

        if len(all_predicted_rtags) == 0:
            return set()

        words = [wd for wd, tags in tagged_sentence]

        # Initialize stack, basic parser and oracle
        parser = ShiftReduceParser(Stack(verbose=False))
        parser.stack.push((ROOT, 0))
        # needs to be a tuple
        oracle = Oracle([], parser)

        predicted_relations = set()  # type: Set[str]

        # instead of head and modifiers, we will map causers to effects, and vice versa
        effect2causers = defaultdict(set)
        # heads can have multiple modifiers
        cause2effects = defaultdict(set)

        # tags without positional info
        rtag_seq = [t for t, i in pos_ptag_seq if t[0].isdigit()]
        # if not at least 2 concept codes, then can't parse
        if len(rtag_seq) < 2:
            return set()

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
                btwn_start, btwn_stop = min(tstop + 1, len(words)), max(0, bstart)

                btwn_word_seq = words[btwn_start:btwn_stop]
                distance = len(btwn_word_seq)
                btwn_word_ngrams = self.ngram_extractor.extract(btwn_word_seq)  # type: List[str]

                feats = self.feat_extractor.extract(stack_tags=parser.stack.contents(), buffer_tags=remaining_buffer_tags,
                                                    tag2word_seq=tag2words,
                                                    between_word_seq=btwn_word_ngrams, distance=distance,
                                                    cause2effects=cause2effects, effect2causers=effect2causers,
                                                    positive_val=self.positive_val)

                action = self.predict_parse_action(feats=feats,
                                                   tos=tos_tag,
                                                   models=self.parser_models[-1],
                                                   vectorizer=self.parser_feature_vectorizers[-1])

                # Decide the direction of the causal relation
                if action in [LARC, RARC]:

                    cause_effect, effect_cause = self.update_feats_with_action(action, buffer_tag, feats, tos_tag)

                    lr_action = self.predict_crel_action(feats=feats,
                                                         model=self.crel_models[-1],
                                                         vectorizer=self.crel_feat_vectorizers[-1])

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

                # end if action in [LARC,RARC]
                if not oracle.execute(action, tos_tag_pair, buffer_tag_pair):
                    break
                if oracle.is_stack_empty():
                    break

        return predicted_relations

    def update_feats_with_action(self, action, buffer_tag, feats, tos_tag):
        c_e_pair = (tos_tag, buffer_tag)
        # Convert to a string Causer:{l}->Result:{r}
        cause_effect = denormalize_cr(c_e_pair)
        e_c_pair = (buffer_tag, tos_tag)
        # Convert to a string Causer:{l}->Result:{r}
        effect_cause = denormalize_cr(e_c_pair)
        # Add additional features
        # needs to be before predict below
        crel_feats = self.crel_features(action, tos_tag, buffer_tag)
        feats.update(crel_feats)
        return cause_effect, effect_cause

    def predict_sentence(self, tagged_sentence, predicted_tags):
        return self.generate_training_data(tagged_sentence=tagged_sentence, predicted_tags=predicted_tags)
