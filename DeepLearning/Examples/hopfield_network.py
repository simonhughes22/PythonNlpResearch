#!/usr/bin/env python3
# A translation of http://www.cleveralgorithms.com/nature-inspired/neural/hopfield_network.html
# SH: from http://onehourhacks.blogspot.com/2012/12/a-python-hopfield-network.html

import random

def random_vector(minmax):
    return [row[0] + ((row[1] - row[0]) * random.random()) for row in minmax]

def initialize_weights(problem_size):
    return random_vector([[-0.5,0.5] for i in range(problem_size)])

def create_neuron(num_inputs):
    return {'weights' : initialize_weights(num_inputs)}

def propagate_was_change(neurons):
    i = random.randrange(len(neurons))
    activation = 0

    for j, other in enumerate(neurons):
        activation += other['weights'][i] * other['output'] if i != j else 0

    output = 1 if activation >= 0 else -1
    change = output != neurons[i]['output']
    neurons[i]['output'] = output
    return change

def flatten(nested):
    try:
        return [item for sublist in nested for item in sublist]
    except TypeError:
        return nested

def get_output(neurons, pattern, evals=100):
    vector = flatten(pattern)

    for i, neuron in enumerate(neurons):
        neuron['output'] = vector[i]

    for j in range(evals):
        propagate_was_change(neurons)

    return [neuron['output'] for neuron in neurons]

def train_network(neurons, patters):
    for i, neuron in enumerate(neurons):
        for j in range((i+1), len(neurons)):
            if i == j:
                continue

            wij = 0.0

            for pattern in patters:
                vector = flatten(pattern)
                wij += vector[i] * vector[j]

            neurons[i]['weights'][j] = wij
            neurons[j]['weights'][i] = wij

def to_binary(vector):
    return [0 if i == -1 else 1 for i in vector]

def print_patterns(provided, expected, actual):
    p, e, a = to_binary(provided), to_binary(expected), to_binary(actual)
    p1, p2, p3 = ', '.join(map(str, p[0:2])), ', '.join(map(str, p[3:5])), ', '.join(map(str, p[6:8]))
    e1, e2, e3 = ', '.join(map(str, e[0:2])), ', '.join(map(str, e[3:5])), ', '.join(map(str, e[6:8]))
    a1, a2, a3 = ', '.join(map(str, a[0:2])), ', '.join(map(str, a[3:5])), ', '.join(map(str, a[6:8]))
    print( "Provided\tExpected\tGot")
    print( "%s\t\t%s\t\t%s" % (p1, e1, a1))
    print( "%s\t\t%s\t\t%s" % (p2, e2, a2))
    print( "%s\t\t%s\t\t%s" % (p3, e3, a3))


def calculate_error(expected, actual):
    return sum([1 for i in range(len(expected)) if expected[i] != actual[i]])

def perturb_pattern(vector, num_errors=1):
    perturbed = [v for v in vector]
    indicies = [0 for i in range(random.randrange(len(perturbed)))]

    while len(indicies) < num_errors:
        index = random.randrange(len(perturbed))
        if not index in indicies:
            indicies.append(index)

    for i in indicies:
        perturbed[i] = -1 if perturbed[i] == 1 else 1

    return perturbed


def test_network(neurons, patterns):
    error = 0.0

    for pattern in patterns:
        vector = flatten(pattern)
        perturbed = perturb_pattern(vector)
        output = get_output(neurons, perturbed)
        error += calculate_error(vector, output)
        print_patterns(perturbed, vector, output)

    error = error / float(len(patterns))
    print("Final Result: avg pattern error=%s" % (error))

    return error

def execute(patters, num_inputs):
    neurons = [create_neuron(num_inputs) for i in range(num_inputs)]
    train_network(neurons, patters)
    test_network(neurons, patters)
    return neurons

if __name__ == "__main__":

    print "Neurolab (installed) has a better implementation"

    def simple_test():
        # problem configuration
        num_inputs = 9
        p1 = [[1,1,1],[-1,1,-1],[-1,1,-1]] # T
        p2 = [[1,-1,1],[1,-1,1],[1,1,1]] # U
        patters = [p1, p2]
        # execute the algorithm
        execute(patters, num_inputs)

    def test_on_data():
        import GwData
        import WordTokenizer
        import TfIdf

        import Converter
        import MatrixHelper

        data = GwData.GwData()
        tokenized = WordTokenizer.tokenize(data.documents)
        tfidf = TfIdf.TfIdf(data.documents)

