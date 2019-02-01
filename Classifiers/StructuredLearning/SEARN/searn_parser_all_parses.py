from collections import defaultdict
from typing import Set, List

from StructuredLearning.SEARN.stack import Stack
from oracle import Oracle
from shift_reduce_helper import *
from shift_reduce_parser import ShiftReduceParser

from Classifiers.StructuredLearning.SEARN.searn_parser import SearnModelTemplateFeatures

class ParseActionResult(object):
    def __init__(self, action, relations, prob, cause2effects, effect2causers):
        self.action = action
        self.relations = relations
        self.prob = prob
        self.cause2effects = cause2effects
        self.effect2causers = effect2causers

class SearnModelAllParses(SearnModelTemplateFeatures):
    def __init__(self, *args, **kwargs):
        super(SearnModelAllParses, self).__init__(*args, **kwargs)

    def generate_all_potential_parses_for_sentence(self, tagged_sentence, predicted_tags, min_probability=0.1):

        pos_ptag_seq, _, tag2span, all_predicted_rtags, _ = self.get_tags_relations_for(
            tagged_sentence, predicted_tags, self.cr_tags)

        if len(all_predicted_rtags) == 0:
            return []

        # tags without positional info
        rtag_seq = [t for t, i in pos_ptag_seq if t[0].isdigit()]
        # if not at least 2 concept codes, then can't parse
        if len(rtag_seq) < 2:
            return []

        words = [wd for wd, tags in tagged_sentence]

        # Initialize stack, basic parser and oracle
        parser = ShiftReduceParser(Stack(verbose=False))
        parser.stack.push((ROOT, 0))
        # needs to be a tuple
        oracle = Oracle([], parser)

        tag2words = defaultdict(list)
        for ix, tag_pair in enumerate(pos_ptag_seq):
            bstart, bstop = tag2span[tag_pair]
            tag2words[tag_pair] = self.ngram_extractor.extract(words[bstart:bstop + 1])  # type: List[str]

        all_parses = self.recursively_parse(defaultdict(set), defaultdict(set),
                                            oracle, pos_ptag_seq, tag2span, tag2words, 0, words, defaultdict(list), min_probability)
        return all_parses

    def clone_default_dict(self, d):
        new_dd = defaultdict(d.default_factory)
        new_dd.update(d)
        return new_dd

    def recursively_parse(self, cause2effects, effect2causers, oracle, pos_ptag_seq,
                          tag2span, tag2words, tag_ix, words, current_parse_probs, min_prob):

        if tag_ix >= len(pos_ptag_seq):
            if len(current_parse_probs) == 0:
                return []
            else:
                return [current_parse_probs]

        full_parses = []

        buffer_tag_pair = pos_ptag_seq[tag_ix]
        buffer_tag = buffer_tag_pair[0]
        bstart, bstop = tag2span[buffer_tag_pair]
        remaining_buffer_tags = pos_ptag_seq[tag_ix:]
        # Consume the stack
        tos_tag_pair = oracle.tos()
        parse_action_results = self.get_parse_action_results(bstart, buffer_tag, buffer_tag_pair, cause2effects,
                                                             effect2causers, tos_tag_pair, oracle.parser,
                                                             remaining_buffer_tags,
                                                             tag2span, tag2words, words)

        for pa_result in sorted(parse_action_results, key = lambda par: -par.prob):
            if pa_result.prob < min_prob:
                continue

            new_current_parse_probs = self.clone_default_dict(current_parse_probs)
            new_oracle = oracle.clone()
            if pa_result.relations:
                for reln in pa_result.relations:
                    new_current_parse_probs[reln].append(pa_result.prob)

            if not new_oracle.execute(pa_result.action, tos_tag_pair, buffer_tag_pair) or new_oracle.is_stack_empty():
                # increment tag_ix
                full_parses.extend(self.recursively_parse(pa_result.cause2effects, pa_result.effect2causers,
                                    new_oracle, pos_ptag_seq, tag2span, tag2words,
                                    tag_ix+1,
                                    words, new_current_parse_probs, min_prob))
            else:
                # advance parse state
                # don't increment tag index'
                full_parses.extend(self.recursively_parse(pa_result.cause2effects, pa_result.effect2causers,
                                    new_oracle, pos_ptag_seq, tag2span, tag2words,
                                    tag_ix,
                                    words, new_current_parse_probs, min_prob))
        return full_parses

    def get_parse_action_results(self, bstart, buffer_tag, buffer_tag_pair, cause2effects, effect2causers, tos_tag_pair,
                                 parser, remaining_buffer_tags, tag2span, tag2words, words):


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

        action_probabilities = self.predict_parse_action_probabilities(feats=feats,
                                           tos=tos_tag,
                                           models=self.parser_models[-1],
                                           vectorizer=self.parser_feature_vectorizers[-1])

        parse_action_results = []
        for action, prob in action_probabilities.items():
            # Decide the direction of the causal relation
            new_relations = set()
            new_cause2effects = self.clone_default_dict(cause2effects)
            new_effect2causers = self.clone_default_dict(effect2causers)

            if action in [LARC, RARC]:
                feats_copy = dict(feats)  # don't modify feats as we iterate through possibilities
                cause_effect, effect_cause = self.update_feats_with_action(action, buffer_tag, feats_copy, tos_tag)
                lr_action = self.predict_crel_action(feats=feats_copy,
                                                     model=self.crel_models[-1],
                                                     vectorizer=self.crel_feat_vectorizers[-1])

                new_relations = self.update_cause_effects(buffer_tag_pair,
                                                          new_cause2effects, cause_effect,
                                                          new_effect2causers, effect_cause,
                                                          lr_action, tos_tag_pair)

            parse_action_result = ParseActionResult(action, new_relations, prob, new_cause2effects, new_effect2causers)
            parse_action_results.append(parse_action_result)
        return parse_action_results

    def update_cause_effects(self, buffer_tag_pair, cause2effects, cause_effect, effect2causers, effect_cause,
                             lr_action, tos_tag_pair):
        new_relations = set()
        if lr_action == CAUSE_AND_EFFECT:
            new_relations.add(cause_effect)
            new_relations.add(effect_cause)

            cause2effects[tos_tag_pair].add(buffer_tag_pair)
            effect2causers[buffer_tag_pair].add(tos_tag_pair)

            cause2effects[buffer_tag_pair].add(tos_tag_pair)
            effect2causers[tos_tag_pair].add(buffer_tag_pair)

        elif lr_action == CAUSE_EFFECT:
            new_relations.add(cause_effect)

            cause2effects[tos_tag_pair].add(buffer_tag_pair)
            effect2causers[buffer_tag_pair].add(tos_tag_pair)

        elif lr_action == EFFECT_CAUSE:
            new_relations.add(effect_cause)

            cause2effects[buffer_tag_pair].add(tos_tag_pair)
            effect2causers[tos_tag_pair].add(buffer_tag_pair)

        elif lr_action == REJECT:
            pass
        else:
            raise Exception("Invalid CREL type")
        return new_relations

    def predict_parse_action_probabilities(self, feats, tos, models, vectorizer):

        xs = vectorizer.transform(feats)
        prob_by_label = {}
        for action in self.randomize_actions():
            if not allowed_action(action, tos):
                continue

            prob_by_label[action] = models[action].predict_proba(xs)[0][-1]
        return prob_by_label

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

