from collections import defaultdict
from itertools import izip
from DictionaryHelper import sort_by_value
from IterableFP import flatten
from Entropy import entropy
from Metrics import accuracy

from Tree import Node, Leaf

import numpy as np

"""
    An attempt to learn an ordered decision tree, where sub-ordinate branches
    find rules that form bi-grams with the parent node values.
"""

class __ObjectLiteral__(object):
    def __init__(self):
        pass

def compute_weighted_entropy(attribute_vals, ys):
    """ Pivot by match value """
    ys_by_match = defaultdict(list)
    for is_match, y in izip(attribute_vals, ys):
        ys_by_match[is_match].append(y)

    """ Compute entropy, weighted by size of attribute """
    wt_entropy = 0.0
    num_items = float(len(ys))
    for ismatch, lst in ys_by_match.items():
        ent = entropy(lst)
        wt = (len(lst) / num_items)
        wt_entropy += wt * ent
    return wt_entropy


def information_gain_attribute_evaluator(attribute_values, ys):
    return 1.0 - compute_weighted_entropy(attribute_values, ys)


class DecisionTreeBase(object):

    """
        Learns a decision tree, built from Node and Leaf objects
    """
    def __init__(self, attributes, fn_attribute_val_extractor, fn_attribute_evaluator, max_depth = 1000000):
        """
            attributes  :                   list or array like
                                                list of attributes to eval using fn_attribute_evaluator

            fn_attribute_val_extractor  :   fn - (xs, attribute_name, parent_attribute_name, parent_attribute_val) => attribute_values
                                                A function that takes a list of examples,
                                                an attribute name, parent_attribute name and value,
                                                 and extracts a list of values

            fn_attribute_evaluator  :       fn - (attribute_values, ys => score)
                                                A function for evaluating a set of attribute values and the predicted class,
                                                and outputs a score indicating the attribute quality.
                                                The higher the score, the better the attribute
        """
        self.attributes = attributes
        self.fn_attribute_val_extractor = fn_attribute_val_extractor
        self.fn_attribute_evaluator = fn_attribute_evaluator
        self.tree = None
        self.max_depth = max_depth
        pass

    """ Private Methods """

    def __get_most_frequent_value__(self, ys):
        tally = defaultdict(int)
        for i in ys:
            tally[i] += 1
        val = sort_by_value(tally, reverse=True)[0][0]
        return val

    def __create_node__(self, parent_node, parent_attribute_val, xs, ys, attributes, level):

        if level >= self.max_depth:
            val = self.__get_most_frequent_value__(ys)
            return Leaf(parent_attribute_val, val)

        distinct_ys = set(ys)
        if len(distinct_ys) == 1:
            return Leaf(parent_attribute_val, ys[0])

        best_attribute_name = self.__find_best_attribute__(attributes, parent_node.attribute_name, parent_attribute_val, xs, ys)
        new_node = Node(best_attribute_name, parent_attribute_val, parent_node, self.fn_attribute_val_extractor)

        # Pivot by value to segment dataset for children
        xs_by_val, ys_by_val = self.__partition_by_attribute__(best_attribute_name, parent_node.attribute_name, parent_attribute_val,  xs, ys)

        # If only one grouping, no further attributes are useful, take majority vote for class
        if len(ys_by_val) == 1:
            val = self.__get_most_frequent_value__(ys)
            return Leaf(parent_attribute_val, val)

        # Remove matched attribute
        remaining_attributes = [a for a in attributes if a != best_attribute_name]
        for attr_val in xs_by_val.keys():
            xs4val = xs_by_val[attr_val]
            ys4val = ys_by_val[attr_val]

            child_node = self.__create_node__(new_node, attr_val, xs4val, ys4val, remaining_attributes, level + 1)
            new_node.add_child(child_node)
        return new_node

    def __partition_by_attribute__(self, best_attribute_name, parent_attribute_name, parent_attribute_val,  xs, ys):
        xs_by_val = defaultdict(list)
        ys_by_val = defaultdict(list)
        attribute_vals = self.fn_attribute_val_extractor(xs,
                                                         best_attribute_name,
                                                         parent_attribute_name,
                                                         parent_attribute_val)
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            val = attribute_vals[i]

            xs_by_val[val].append(x)
            ys_by_val[val].append(y)
        return xs_by_val, ys_by_val

    def __find_best_attribute__(self, attributes, parent_attribute_name, parent_attribute_val, xs, ys):
        best_score = -1.0
        best_attribute_name = None
        for i, attribute_name in enumerate(attributes):
            values = self.fn_attribute_val_extractor(xs, attribute_name, parent_attribute_name, parent_attribute_val)
            score = self.fn_attribute_evaluator(values, ys)
            if score > best_score:
                best_score = score
                best_attribute_name = attribute_name
        return best_attribute_name

    """ End Private Methods """

    """ Public Methods """
    """ Public Methods """

    """ sklearn protocol """
    def fit(self, xs, ys):

        top_node = __ObjectLiteral__()
        top_node.attribute_name = None
        self.tree = self.__create_node__(top_node, None, xs, ys, self.attributes, 1)

    def predict(self, xs):
        """
        xs  :   list or array like
                    examples

        Predict labels for xs using trained tree
        """
        if self.tree is None:
            raise Exception("fit must be called before predict to construct a tree")

        predictions = []
        for x in xs:
            val = self.tree.eval(x)
            predictions.append(val)

        return np.array(predictions)

    """ END sklearn protocol """

    def __repr__(self):
        return str(self.tree)

