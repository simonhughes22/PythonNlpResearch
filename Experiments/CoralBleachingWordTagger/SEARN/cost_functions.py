from searn_parser import PARSE_ACTIONS
from shift_reduce_helper import allowed_action, LARC, RARC

def __get_relations_for_action__(forced_action, ground_truth, remaining_buffer, oracle):
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

            if action in [LARC, RARC]:
                if (tos, buffer) in ground_truth:
                    relns.add((tos, buffer))
                elif (buffer, tos) in ground_truth:
                    relns.add((buffer, tos))

            if not oracle.execute(action, tos, buffer):
                break
            if oracle.is_stack_empty():
                break
    return relns

def micro_f1_cost(ground_truth, remaining_buffer, oracle):

    tos = oracle.tos()
    gold_action = oracle.consult(tos, remaining_buffer[0])
    # Get the best possible parse given the current state (stored in the oracle instance)
    #   a lot of times the gold parse will have 0 relations, but then we still need to penalize the algorithm
    #   for a set of false positives
    gold_parse = __get_relations_for_action__(gold_action, ground_truth, remaining_buffer, oracle)

    action_costs = {}
    for action in PARSE_ACTIONS:
        if action == gold_action:
            continue

        # Prevent invalid parse actions
        if not allowed_action(action, tos):
            # cost is number of relations that will be missed or at least 1
            # - alternatively we return a -1 here and don't include these as training data points
            # action_costs[action] = max(1, len(gold_parse))
            action_costs[action] = 0
            # action_costs[action] = 1
            continue

        parse = __get_relations_for_action__(action, ground_truth, remaining_buffer, oracle)
        num_matches = len(gold_parse.intersection(parse))
        # in both cases below, num_matches can't be bigger than the gold_parse or the parse as it's the intersection
        # recall (fn = misses)
        false_negatives = len(gold_parse) - num_matches
        # precision (fp = false alarms)
        false_positives = len(parse) - num_matches
        # Cost is the total of the false positives + false negatives
        cost = false_positives + false_negatives
        assert cost >= 0.0, "Cost should always be non-negative"
        action_costs[action] = cost

    # Weight of the gold action is size of the gold_parse
    # NOTE - these aren't really costs, more example weights
    action_costs[gold_action] = len(gold_parse)
    return action_costs

def micro_f1_cost_squared(ground_truth, remaining_buffer, oracle):

    # Weight of the gold action is size of the gold_parse
    # NOTE - these aren't really costs, more example weights
    tmp_action_costs = micro_f1_cost(ground_truth=ground_truth, remaining_buffer=remaining_buffer, oracle=oracle)
    action_costs = {}
    for action, cost in tmp_action_costs.items():
        action_costs[action] = cost*cost
    return action_costs

def binary_cost(ground_truth, remaining_buffer, oracle):

    # Weight of the gold action is size of the gold_parse
    # NOTE - these aren't really costs, more example weights
    tmp_action_costs = micro_f1_cost(ground_truth=ground_truth, remaining_buffer=remaining_buffer, oracle=oracle)
    action_costs = {}
    for action, cost in tmp_action_costs.items():
        action_costs[action] = 0 if cost == 0 else 1
    return action_costs

def inverse_micro_f1_cost(ground_truth, remaining_buffer, oracle):

    gold_action = oracle.consult(oracle.tos(), remaining_buffer[0])
    tmp_action_costs = micro_f1_cost(ground_truth=ground_truth, remaining_buffer=remaining_buffer, oracle=oracle)
    # invert the costs
    max_cost = max(tmp_action_costs.values())
    action_costs = {}
    for action, cost in tmp_action_costs.items():
        if action == gold_action:
            action_costs[action] = cost
        else:
            action_costs[action] = max_cost - cost
    return action_costs

def uniform_cost(ground_truth, remaining_buffer, oracle):
    action_costs = {}
    for action in PARSE_ACTIONS:
        action_costs[action] = 1
    return action_costs

"""

2018-01-06 11:18:02,196 : INFO : Started at: 2018-01-06 11:18:02.196569
2018-01-06 11:18:02,196 : INFO : Number of pred tagged essays 902
2018-01-06 11:18:02,293 : INFO : ********************************************************************************
2018-01-06 11:18:02,293 : INFO : NGRAM SIZE: 3
2018-01-06 11:18:02,293 : INFO : ********************************************************************************
2018-01-06 11:18:02,293 : INFO : COST FN: micro_f1_cost
2018-01-06 11:18:02,293 : INFO : --------------------------------------------------------------------------------
2018-01-06 11:18:02,293 : INFO : Evaluating 3 features, with ngram size: 3 and beta decay: 0.250001, current feature extractors: between_word_features,three_words
2018-01-06 11:18:02,293 : INFO : 	Extractors: between_word_features,single_words,three_words
2018-01-06 11:19:33,985 : INFO : 		Mean num feats: 287702.40
2018-01-06 11:19:34,683 : INFO : 		Micro F1: 0.7081424039478322 NEW BEST ******************************
2018-01-06 11:19:34,683 : INFO : ********************************************************************************
2018-01-06 11:19:34,683 : INFO : COST FN: micro_f1_cost_squared
2018-01-06 11:19:34,683 : INFO : --------------------------------------------------------------------------------
2018-01-06 11:19:34,683 : INFO : Evaluating 3 features, with ngram size: 3 and beta decay: 0.250001, current feature extractors: between_word_features,three_words
2018-01-06 11:19:34,683 : INFO : 	Extractors: between_word_features,single_words,three_words
2018-01-06 11:21:08,236 : INFO : 		Mean num feats: 288957.20
2018-01-06 11:21:08,943 : INFO : 		Micro F1: 0.7070849488896722 NEW BEST ******************************
2018-01-06 11:21:08,943 : INFO : ********************************************************************************
2018-01-06 11:21:08,943 : INFO : COST FN: inverse_micro_f1_cost
2018-01-06 11:21:08,943 : INFO : --------------------------------------------------------------------------------
2018-01-06 11:21:08,943 : INFO : Evaluating 3 features, with ngram size: 3 and beta decay: 0.250001, current feature extractors: between_word_features,three_words
2018-01-06 11:21:08,943 : INFO : 	Extractors: between_word_features,single_words,three_words
2018-01-06 11:22:40,334 : INFO : 		Mean num feats: 265378.00
2018-01-06 11:22:40,983 : INFO : 		Micro F1: 0.6953210010881393 NEW BEST ******************************
2018-01-06 11:22:40,984 : INFO : ********************************************************************************
2018-01-06 11:22:40,984 : INFO : COST FN: uniform_cost
2018-01-06 11:22:40,984 : INFO : --------------------------------------------------------------------------------
2018-01-06 11:22:40,984 : INFO : Evaluating 3 features, with ngram size: 3 and beta decay: 0.250001, current feature extractors: between_word_features,three_words
2018-01-06 11:22:40,984 : INFO : 	Extractors: between_word_features,single_words,three_words
2018-01-06 11:23:57,349 : INFO : 		Mean num feats: 244596.00
2018-01-06 11:23:58,005 : INFO : 		Micro F1: 0.6526674233825198 NEW BEST ******************************
2018-01-06 11:27:11,511 : INFO : ********************************************************************************
2018-01-06 11:27:11,511 : INFO : COST FN: binary_cost
2018-01-06 11:27:11,511 : INFO : --------------------------------------------------------------------------------
2018-01-06 11:27:11,511 : INFO : Evaluating 3 features, with ngram size: 3 and beta decay: 0.250001, current feature extractors: between_word_features,three_words
2018-01-06 11:27:11,511 : INFO : 	Extractors: between_word_features,single_words,three_words
2018-01-06 11:28:46,093 : INFO : 		Mean num feats: 286991.80
2018-01-06 11:28:46,747 : INFO : 		Micro F1: 0.7083112758073054 NEW BEST ******************************

"""