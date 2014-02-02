from DecisionTreeBase import DecisionTreeBase, information_gain_attribute_evaluator, compute_weighted_entropy
from SkipGramGenerator import skip_gram_matches
from Metrics import accuracy
from IterableFP import flatten

def not_a_followed_by_b(a, b, example):
    if b not in example:
        return False
    ix_b = example.index(b)
    b4_b = example[:ix_b]
    if not a in b4_b:
        return True
    else:
        return False

def a_followed_by_not_b(a, b, example):
    if a not in example:
        return False

    ix_a = example.index(a)
    after_a = example[ix_a+1:]
    if not b in after_a:
        return True
    else:
        return False

def compute_attribute_values_forward_only(xs, attribute_pair, parent_attribute_pair, parent_attribute_val):
    """
        Only allows child attributes to follow parent in sentence
    """

    child_attribute_name, is_child_b4 = attribute_pair
    if parent_attribute_pair is None:
        attrib_values = map(lambda x: child_attribute_name in x, xs)
    else:
        parent_attribute_name, _ = parent_attribute_pair
        if parent_attribute_val == True:
            attrib_values = map(lambda sentence: skip_gram_matches([parent_attribute_name, child_attribute_name], sentence), xs)
        else:
            attrib_values = map(lambda sentence: not_a_followed_by_b(parent_attribute_name, child_attribute_name, sentence), xs)
    return attrib_values

def compute_attribute_values_both_ways(xs, attribute_pair, parent_attribute_pair, parent_attribute_val):
    """
        Allows child attribute to be before or after parent attribute
    """

    child_attribute_name, is_child_b4 = attribute_pair
    if parent_attribute_pair is None:
        attrib_values = map(lambda x: child_attribute_name in x, xs)
    else:
        parent_attribute_name, _ = parent_attribute_pair
        if parent_attribute_val == True:
            if is_child_b4:
                attrib_values = map(lambda sentence: skip_gram_matches([child_attribute_name, parent_attribute_name], sentence), xs)
            else:
                attrib_values = map(lambda sentence: skip_gram_matches([parent_attribute_name, child_attribute_name], sentence), xs)
        else:
            if is_child_b4:
                attrib_values = map(lambda sentence: a_followed_by_not_b(child_attribute_name, parent_attribute_name, sentence), xs)
            else:
                attrib_values = map(lambda sentence: not_a_followed_by_b(parent_attribute_name, child_attribute_name, sentence), xs)
    return attrib_values

class OrderedDecisionTree(DecisionTreeBase):

    def __init__(self, max_depth = 1000000):
        DecisionTreeBase.__init__(self, [], compute_attribute_values_both_ways, information_gain_attribute_evaluator, max_depth)
        pass

    def fit(self, xs, ys):
        attribute_names = list(set(flatten(xs)))
        # Compute pairs of attributes : (attribute_name, is_child_before)
        self.attributes = [(a, True) for a in attribute_names] + [(a, False) for a in attribute_names]
        self.attributes.sort()
        DecisionTreeBase.fit(self, xs, ys)

if __name__ == "__main__":

    "*7,8,9* => 0 # even if contains 1,2,3"
    "*1,2,3* => 1"
    "other => 0"

    complex_ordered_dataset = [
        ([1, 2, 3], 1 ),
        ([5, 1, 2, 3], 1 ),
        ([1, 2, 3, 4], 1 ),
        ([1, 4, 2, 5, 3], 1 ),

        ([1, 2, 5, 8, 9 ], 0 ),
        ([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 0),
        ([1, 2], 0),
        ([2, 3], 0),
        ([2, 3, 1], 0),
        ([3, 2, 1], 0),
        ([1, 3], 0),

    ]
    "1,2,3 => 1"
    "else => 0"
    simple_ordered_dataset = [
        ([1, 2, 3], 1 ),
        ([1, 2, 3, 4, 5], 1 ),
        ([7, 8, 1, 2, 3, 4, 5], 1 ),

        ([1, 2], 0 ),
        ([2, 3], 0 ),
        ([3, 2, 1], 0 ),
        ([2, 3, 1], 0 ),
        ([3, 1, 2], 0 ),
        ([1, 2, 4, 5, 6, 7, 8], 0 ),
    ]

    simple_ordered_dataset2 = [
        (["big", "fat", "hairy", "cat"],  1 ),
        (["big", "fat", "hairy", "troublesome", "cat"],  1 ),
        (["big", "fat", "hirsute", "hairy", "troublesome", "cat"],  1 ),

        (["big", "hairy", "fat", "cat"],  0),
        (["big", "hairy", "cat"],  0),
        (["big", "fat", "cat"],  0),
        (["big", "fat"],  0),
        (["fat"], 0 ),
        (["hairy", "troublesome"], 0 ),
        (["a", "cat"], 0 ),
        (["a", "big"], 0 ),
        (["fat", "cat"], 0 ),
        (["big", "cat"], 0 ),
    ]

    xs, ys = zip(*complex_ordered_dataset)

    dt = OrderedDecisionTree(4)
    dt.fit(xs, ys)
    predictions = dt.predict(xs)
    acc = accuracy(ys, predictions, class_value=1)
    print "\nAccuracy: " + str(acc)
    print ""
    print str(dt.tree)
    pass