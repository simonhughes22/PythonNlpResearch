from IterableFP import flatten
from DecisionTreeBase import DecisionTreeBase, information_gain_attribute_evaluator
from Metrics import accuracy

def compute_attribute_values(xs, attribute_name, parent_attribute_name, parent_attribute_val):
    return map(lambda sentence: attribute_name in sentence, xs)

class DecisionTree(DecisionTreeBase):

    def __init__(self, attributes, max_depth = 100000):
        DecisionTreeBase.__init__(self, attributes, compute_attribute_values, information_gain_attribute_evaluator, max_depth)
        pass

if __name__ == "__main__":

    "1,2,3 => 1"
    "else => 0"
    simple_dataset = [
        ([1, 2, 3, 7, 8], 1 ),
        ([1, 2, 3, 4, 5], 1 ),
        ([7, 8, 1, 2, 3, 4, 5], 1 ),

        ([1, 2], 0 ),
        ([2, 3], 0 ),
        ([1, 3], 0 ),
        ([5, 6, 7], 0 ),
        ([4, 1, 2], 0 ),
        ([3, 7, 9], 0 ),
        ([1, 2, 4, 5, 6, 7, 8], 0 ),
    ]

    xs, ys = zip(*simple_dataset)
    attributes = list(set(flatten(xs)))

    dt = DecisionTree(attributes)
    dt.fit(xs, ys)
    predictions = dt.predict(xs)

    acc = accuracy(ys, predictions, class_value=1)
    print "\nAccuracy: " + str(acc)
    print ""
    print str(dt.tree)
    pass