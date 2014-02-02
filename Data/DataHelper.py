__author__ = 'simon.hughes'

import numpy as np

""" Contains routines for manipulating datasets, supporting the following manipulations:

        1. Partitioning according to some binary split (e.g. for training \ validation data)
        2. Shuffling datasets (e.g. re-order the data on each NNet iteration)

"""

def partition(td_percentage, xs):
    """
        td_percentage   :   float
                                0 - 1.0 - proportion of test data
        xs              :   list \ array like
                                examples
        returns         :   (xs, xs)
                                a tuple of 2 partitions of xs, in a td_percentage : 1-td_percentage split

        Partitions the input data in two partitions according to the td_percentage
    """
    if td_percentage < 0.0:
        raise Exception("td_percentage: [%d] must be >= 0.0" % td_percentage)
    elif td_percentage > 1.0:
        raise Exception("td_percentage: [%d] must be <= 1.0" % td_percentage)

    size = len(xs)
    td_size = int(td_percentage * size)
    td, vd = xs[:td_size], xs[td_size:]

    assert len(td) + len(vd) == len(xs), "Total size of partitions does not equal size of original dataset"
    return (td, vd)

def partition_multiple(td_percentage, *xs):
    """
        td_percentage   :   float
                                0 - 1.0 - proportion of test data
        xs              :   list \ array like
                                datasets to split (atleast one needed).
                                Note these are passed in as separate params, not as a collection
        returns         :   list of pairs of tuples : (xs, xs)
                                a tuple of 2 partitions of xs and 2 partitions of ys,
                                in a td_percentage : 1-td_percentage split

        Partitions the two input data sets in partitions according to the td_percentage

    """
    assert len(xs) > 0, "No iterables specified"

    output = []
    for dataset in xs:
        td_xs, vd_xs = partition(td_percentage, dataset)
        output.append((td_xs, vd_xs))
    return output

def shuffle_examples(*examples):
    """
        examples    :   list \ array like
                            Dataset to re-order.
                            Note these need to be passed in as separate parameters (not a collection)
        returns     :   list of shuffled examples

        Randomly re-oders the examples. The same re-ordering is applied to each
        dataset. The size of each dataset needs to be the same.
    """
    assert len(examples) > 0, "No examples passed in"

    size = len(examples[0])

    shuffled_ixs = np.array(range(size))
    np.random.shuffle(shuffled_ixs)

    outputs = []
    for i, dataset in enumerate(examples):
        dataset_size = len(dataset)
        assert dataset_size == size, "The size of all the datasets must be equal. " \
                                     "Dataset %d differs in size (%d) " \
                                     "from the first dataset (%d)" % (i, dataset_size, size)
        np_dataset = np.array(dataset)
        outputs.append( np_dataset[shuffled_ixs] )

    return np.array(outputs)

if __name__ == "__main__":

    print "shuffle_examples"
    ex1 = [0,1,2,3,4,5]
    ex2 = ["A", "B", "C", "D", "E", "F"]

    __shuffled_exs__ = shuffle_examples(ex1, ex2)
    for ex in __shuffled_exs__:
        print str(ex)

    print "\npartition :            partition(0.9, range(10))"
    print(partition(0.9, range(10)))

    print "\npartition :            partition(0.85, range(10))"
    print(partition(0.85, range(10)))

    print "\npartition_multiple:    partition_multiple(0.8, range(10), range(10)[::-1])"
    ptnd = partition_multiple(0.8, range(10), range(10)[::-1])
    for p in ptnd:
        print str(p)