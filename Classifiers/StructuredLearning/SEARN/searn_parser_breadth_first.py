from collections import defaultdict
from typing import Set, List

from StructuredLearning.SEARN.stack import Stack
from oracle import Oracle
from shift_reduce_helper import *
from shift_reduce_parser import ShiftReduceParser
import numpy as np

from Classifiers.StructuredLearning.SEARN.searn_parser import SearnModelTemplateFeatures

def geo_mean(vals):
    return np.product(vals)**(1/len(vals))

class ParseActionResult(object):
    def __init__(self, action, relations, prob, cause2effects, effect2causers, oracle, tag_ix, ctx, parent_action, lr_action):
        self.action = action
        self.relations = relations
        self.prob = prob
        self.cause2effects = cause2effects
        self.effect2causers = effect2causers
        self.oracle = oracle
        self.current_tag_ix = tag_ix # store for reference / debugging
        self.tag_ix = tag_ix
        self.ctx = ctx
        self.parent_action = parent_action
        self.lr_action = lr_action

        self.probs = [self.prob]
        if parent_action is not None:
            self.probs = parent_action.probs + self.probs
        # use the geometric mean here so we don't penalize longer parses
        self.cum_prob = geo_mean(self.probs)
        self.__execute__()

    def __execute__(self):
        buffer_tag_pair = self.ctx.pos_ptag_seq[self.tag_ix]
        if not self.oracle.execute(self.action, self.oracle.tos(), buffer_tag_pair) or self.oracle.is_stack_empty():
            # increment tag_ix
            self.tag_ix += 1

    def is_terminal(self):
        return self.tag_ix >= len(self.ctx.pos_ptag_seq)

class ParseContext(object):
    def __init__(self, pos_ptag_seq, tag2span, tag2words, words):
        self.pos_ptag_seq = pos_ptag_seq
        self.tag2span = tag2span
        self.tag2words = tag2words
        self.words = words

class SearnModelBreadthFirst(SearnModelTemplateFeatures):
    def __init__(self, *args, **kwargs):
        super(SearnModelBreadthFirst, self).__init__(*args, **kwargs)

    def build_parse_context(self, tagged_sentence, predicted_tags):
        pos_ptag_seq, _, tag2span, all_predicted_rtags, _ = self.get_tags_relations_for(
            tagged_sentence, predicted_tags, self.cr_tags)

        if len(all_predicted_rtags) == 0:
            return None

        # tags without positional info
        rtag_seq = [t for t, i in pos_ptag_seq if t[0].isdigit()]
        # if not at least 2 concept codes, then can't parse
        if len(rtag_seq) < 2:
            return None

        words = [wd for wd, tags in tagged_sentence]

        tag2words = defaultdict(list)
        for ix, tag_pair in enumerate(pos_ptag_seq):
            bstart, bstop = tag2span[tag_pair]
            tag2words[tag_pair] = self.ngram_extractor.extract(words[bstart:bstop + 1])  # type: List[str]

        ctx = ParseContext(pos_ptag_seq=pos_ptag_seq, tag2span=tag2span, tag2words=tag2words, words=words)
        return ctx

    def generate_all_potential_parses_for_sentence(self, tagged_sentence, predicted_tags, top_n):

        ctx = self.build_parse_context(tagged_sentence, predicted_tags)
        if not ctx:
            return []

        terminal_actions = []
        actions_queue = [None]
        while True:
            current_actions_queue = list(actions_queue)
            actions_queue = []
            for act in current_actions_queue:
                if act and act.is_terminal():
                    terminal_actions.append(act)
                actions_queue.extend(self.get_next_actions(act, ctx))

            if len(actions_queue) == 0:
                break
            # trim to top_n
            actions_queue = sorted(actions_queue,   key=lambda act: -act.cum_prob)[:top_n]

        terminal_actions = sorted(terminal_actions, key=lambda act: -act.cum_prob)
        return terminal_actions[:top_n]

    def get_next_actions(self, parse_action, ctx):
        next_actions = []
        if parse_action is None:
            # Initialize stack, basic parser and oracle
            oracle = self.create_oracle()
            tag_ix = 0
            cause2effects, effect2causers = defaultdict(set), defaultdict(set)
        else:
            if parse_action.is_terminal():
                return []
            oracle = parse_action.oracle
            tag_ix = parse_action.tag_ix
            cause2effects, effect2causers = parse_action.cause2effects, parse_action.effect2causers

            if tag_ix >= len(ctx.pos_ptag_seq):
                return next_actions

        return self.get_parse_action_results(cause2effects, effect2causers, oracle, tag_ix, ctx, parse_action)

    def get_parse_action_results(self, cause2effects, effect2causers, oracle, tag_ix, ctx, parent_action):
        # Get Buffer Info
        buffer_tag_pair = ctx.pos_ptag_seq[tag_ix]
        buffer_tag = buffer_tag_pair[0]
        bstart, bstop = ctx.tag2span[buffer_tag_pair]
        remaining_buffer_tags = ctx.pos_ptag_seq[tag_ix:]

        # Get Stack Info
        tos_tag_pair = oracle.tos()
        tos_tag = tos_tag_pair[0]
        # Returns -1,-1 if TOS is ROOT
        if tos_tag == ROOT:
            tstart, tstop = -1, -1
        else:
            tstart, tstop = ctx.tag2span[tos_tag_pair]

        # Get Between features
        # Note that the end ix in tag2span is always the last index, not the last + 1
        btwn_start, btwn_stop = min(tstop + 1, len(ctx.words)), max(0, bstart)
        btwn_word_seq = ctx.words[btwn_start:btwn_stop]
        distance = len(btwn_word_seq)
        btwn_word_ngrams = self.ngram_extractor.extract(btwn_word_seq)  # type: List[str]
        feats = self.feat_extractor.extract(stack_tags=oracle.parser.stack.contents(), buffer_tags=remaining_buffer_tags,
                                            tag2word_seq=ctx.tag2words,
                                            between_word_seq=btwn_word_ngrams, distance=distance,
                                            cause2effects=cause2effects, effect2causers=effect2causers,
                                            positive_val=self.positive_val)

        action_probabilities = self.predict_parse_action_probabilities(feats=feats,
                                           tos=tos_tag,
                                           models=self.parser_models[-1],
                                           vectorizer=self.parser_feature_vectorizers[-1])

        parse_action_results = []
        for action, parse_action_prob in action_probabilities.items():
            # Decide the direction of the causal relation
            if action in [LARC, RARC]:
                feats_copy = dict(feats)  # don't modify feats as we iterate through possibilities
                cause_effect, effect_cause = self.update_feats_with_action(action, buffer_tag, feats_copy, tos_tag)

                lr_action_probs = self.predict_crel_action_probs(feats=feats_copy,
                                                     model=self.crel_models[-1],
                                                     vectorizer=self.crel_feat_vectorizers[-1])

                for lr_action, lra_prob in lr_action_probs.items():
                    new_cause2effects = self.clone_default_dict(cause2effects)
                    new_effect2causers = self.clone_default_dict(effect2causers)
                    new_relations = self.update_cause_effects(buffer_tag_pair,
                                                              new_cause2effects, cause_effect,
                                                              new_effect2causers, effect_cause,
                                                              lr_action, tos_tag_pair)

                    parse_action_result = ParseActionResult(
                        action, new_relations, parse_action_prob * lra_prob, new_cause2effects, new_effect2causers, oracle.clone(), tag_ix, ctx,
                        parent_action, lr_action)
                    parse_action_results.append(parse_action_result)
            else:
                parse_action_result = ParseActionResult(
                    action, set(), parse_action_prob, self.clone_default_dict(cause2effects), self.clone_default_dict(effect2causers),
                    oracle.clone(), tag_ix, ctx, parent_action, None)
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

    def clone_default_dict(self, d):
        new_dd = defaultdict(d.default_factory)
        new_dd.update(d)
        return new_dd

    def create_oracle(self):
        parser = ShiftReduceParser(Stack(verbose=False))
        parser.stack.push((ROOT, 0))
        # needs to be a tuple
        return Oracle([], parser)

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

