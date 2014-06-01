'''
Created on Aug 18, 2013
@author: simon.hughes

Auto-encoder implementation. Can be used to implement a denoising auto-encoder, sparse or contractive auto-encoder
'''

import numpy as np
import gnumpy as gp
from numpy import matlib

USE_GPU = False


def get_array(a):
    if USE_GPU:
        if type(a) == gp.garray:
            return a
        return gp.garray(a)

    #ELSE NP
    if type(a) == np.array:
        return a
    return np.array(a)

class Layer(object):
    def __init__(self, num_inputs, num_outputs, activation_fn="tanh", initial_wt_max=0.01, weights=None, bias=None):
        self.activation_fn = activation_fn
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs

        if weights is None:
            weights = get_array(matlib.rand((num_outputs, num_inputs)) * initial_wt_max)

        if bias is None:
            bias = get_array(matlib.rand((1, num_outputs)) * initial_wt_max)

        self.initial_wt_max = initial_wt_max
        self.weights        = weights
        self.bias           = bias
        self.save_state()

        assert self.num_inputs == self.weights.shape[1]
        assert self.num_outputs == self.weights.shape[0]
        assert self.num_outputs == self.bias.shape[1]

    def save_state(self):
        self.best_weights   = self.weights.copy()
        self.best_bias      = self.bias.copy()

    def revert_state(self):
        self.weights        = self.best_weights.copy()
        self.bias           = self.best_bias.copy()

    # for prediction (for case like dropout where we need to do something different
    # here by overriding this function
    def feed_forward(self, inputs_T):
        return self.prop_up(inputs_T)

    # for training
    def prop_up(self, inputs_T):

        """ Compute activations """
        z = self.__compute_z__(inputs_T, self.weights, self.bias)
        a = self.__activate__(z, self.activation_fn)
        return (z, a)

    def derivative(self, activations):

        if self.activation_fn == "sigmoid":
            """ f(z)(1 - f(z)) """
            return np.multiply(activations, (1 - activations))
        elif self.activation_fn == "tanh":
            """ 1 - f(z)^2 """
            return 1 - np.square(activations)
        elif self.activation_fn == "linear":
            return activations
        elif self.activation_fn == "relu":
            copy = activations.copy()
            copy[copy < 0] = 0
            return copy
        else:
            raise NotImplementedError("Only sigmoid, tanh, linear and relu currently implemented")

    def update(self, wtdiffs, biasdiff):
        self.weights -= wtdiffs
        self.bias    -= biasdiff

    def __compute_z__(self, inputs, weights, bias):
        #Can we speed this up by making the bias a column vector?
        return np.dot(weights, inputs) + bias.T

    def __activate__(self, z, activation_fn):

        if activation_fn == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif activation_fn == "tanh":
            return np.tanh(z)
        elif activation_fn == "linear":
            return z
        elif activation_fn == "relu":
            copy = z.copy()
            copy[copy < 0] = 0
            return copy
        else:
            raise NotImplementedError("Only sigmoid, tanh, linear and relu currently implemented")


class MLP(object):
    '''
    classdocs
    '''

    def __init__(self, layers, learning_rate=0.1, weight_decay=0.0, epochs = 50, batch_size = 32,
                 lr_increase_multiplier = 1.0, lr_decrease_multiplier = 1.0):
        '''
        learning_rate           = the learning rate
        weight_decay            = a regularization term to stop over-fitting. Only turn on if network converges too fast or overfits the data
        epochs                  = number of epochs to train for. Can be overridden when calling fit
        batch_size              = mini batch size. Can be overridden when calling fit
        lr_increase_multiplier  = factor used to multiply the learning rate by if error decreeases
        lr_decrease_multiplier  = factor used to multiply the learning rate by if error increases
        '''

        """ Properties """
        self.layers = layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_increase_multiplier = lr_increase_multiplier
        self.lr_decrease_multiplier = lr_decrease_multiplier
        """ END Properties """

    def predict(self, inputs, layer_ix = np.inf):
        a = self.__ensure_vector_format__(inputs).T
        for i, layer in enumerate(self.layers):
            z, a = layer.feed_forward(a)
            if i == layer_ix:
                break
        return a.T

    def fit(self, xs, ys, epochs = None, batch_size = None):

        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size

        inputs  = self.__ensure_vector_format__(xs)
        outputs = self.__ensure_vector_format__(ys)

        num_rows = inputs.shape[0]

        """ Number of rows in inputs should match those in outputs """
        assert inputs.shape[0] == outputs.shape[0], "Xs and Ys do not have the same row count"

        assert inputs.shape[1]  == self.layers[0].weights.shape[1],  "The input layer does not match the Xs column count"
        assert outputs.shape[1] == self.layers[-1].weights.shape[0], "The output layer does not match the Ys column count"

        """ Check outputs match the range for the activation function for the layer """
        self.__validate__(outputs, self.layers[-1])

        num_batches = num_rows / batch_size
        if num_rows % batch_size > 0:
            num_batches += 1

        mse = -1.0
        mae = -1.0

        lst_mse = []
        lst_mae = []
        for epoch in range(epochs):

            """ Note that the error may start increasing exponentially at some point
                if so, halt training
            """

            errors = []
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                mini_batch_in = inputs[start:end]
                mini_batch_out = outputs[start:end]
                if len(mini_batch_in) == 0:
                    continue

                mini_batch_errors = self.__train_mini_batch__(mini_batch_in, mini_batch_out)
                errors.append(mini_batch_errors)

            errors = get_array(errors)
            mse = np.mean(np.square(errors))
            mae = np.mean(np.abs(errors))

            DIGITS = 6
            print "MSE for epoch {0} is {1}".format(epoch, np.round(mse,DIGITS)),
            print "\tMAE for epoch {0} is {1}".format(epoch, np.round(mae,DIGITS)),
            print "\tlearning rate is {0}".format(self.learning_rate)

            if epoch > 0:
                self.__adjust_learning_rate__(lst_mae[-1], mae)
            lst_mse.append(mse)
            lst_mae.append(mae)
        return (mse, mae)

    def __train_mini_batch__(self, input_vectors, outputs):
        rows = input_vectors.shape[0]
        inputs_T = input_vectors.T
        outputs_T = outputs.T

        aas = []
        zzs = []
        derivatives = []

        a = inputs_T
        for layer in self.layers:
            z, a = layer.prop_up(a)
            deriv = layer.derivative(a)
            aas.append(a)
            zzs.append(z)
            derivatives.append(deriv)

        top_layer_output = aas[-1]
        """ errors = mean( 0.5 sum squared error)  """
        assert outputs_T.shape == top_layer_output.shape
        errors = (outputs_T - top_layer_output)

        # Compute weight updates
        delta = -(errors) * derivatives[-1]

        deltas = [delta.copy()]
        for i in range(len(self.layers) -1):
            ix = -(i + 1)
            layer = self.layers[ix]
            """ THIS IS BACK PROP OF ERRORS TO HIDDEN LAYERS"""
            delta = np.dot(layer.weights.T, delta) * derivatives[ix-1]
            deltas.insert(0, delta.copy())

        #TODO Sparsity
        frows = float(rows)
        for i, layer in enumerate(self.layers):
            delta = deltas[i]
            activation = input_vectors if i == 0 else aas[i-1].T

            """ Delta for weights is the dot product of the layer delta (error deltas for output)
                and activations for that layer

                For each weight in the weight matrix, update it using the input activation * output delta.
                Compute a mean over all examples in the batch.

                The dot product is used here in a very clever  way to compute the activation * the delta
                for each input and hidden layer node (taking the dot product of each weight over all input_vectors
                (adding up the weight deltas) and then dividing this by num rows to get the mean
             """
            wtdelta   = (np.dot(delta, activation)) / frows
            biasdelta = (np.sum(delta, 1) / frows).T

            if self.weight_decay > 0.0:
                wds = self.learning_rate * (wtdelta + self.weight_decay * layer.weights)
            else:
                wds = self.learning_rate *  wtdelta

            bds = self.learning_rate * biasdelta
            layer.update(wds, bds)

        """ return a list of errors (one item per row in mini batch) """
        errors = np.nan_to_num(errors)
        return errors.T

    def __adjust_learning_rate__(self, previous_mae, mae):
        # error improved on the training data?
        if mae <= previous_mae:
            self.learning_rate *= self.lr_increase_multiplier
            for layer in self.layers:
                layer.save_state()
        else:
            #print "MAE increased from %s to %s. Decreasing learning rate from %s to %s" % \
            #      (str(previous_mae), str(mae),
            #       str(self.learning_rate), str(self.learning_rate * self.lr_decrease_multiplier))
            self.learning_rate *=  self.lr_decrease_multiplier
            self.learning_rate = max(0.001, self.learning_rate)
            for layer in self.layers:
                layer.revert_state()

    def __ensure_vector_format__(self, a):
        return get_array(a)

    def __validate__(self, inputs, layer):

        min_inp = np.min(inputs)
        max_inp = np.max(inputs)

        if layer.activation_fn == "sigmoid":
            self.__in_range__(min_inp, max_inp, 0.0, 1.0)
        elif layer.activation_fn == "tanh":
            self.__in_range__(min_inp, max_inp, -1.0, 1.0)
        elif layer.activation_fn == "relu":
            self.__in_range__(min_inp, max_inp, 0.0, np.inf)
        elif layer.activation_fn == "linear":
            pass
        else:
            raise Exception("Unknown activation function %s" % layer.activation_fn)

    def __in_range__(self, actual_min, actual_max, exp_min, exp_max):
        assert actual_min >= exp_min
        assert actual_max <= exp_max

if __name__ == "__main__":

    """
    # Test Sum
    xs = [
          [1,    0,     0.5,    0.1],
          [0,    1,     1.0,    0.5],
          [1,    0.5,   1,      0  ],
          [0,    0.9,   0,      1  ],
          [0.25, 0,     0.5,    0.1],
          [0.1,  1,     1.0,    0.5],
          [1,    0.5,   0.65,   0  ],
          [0.7,  0.9,   0,      1  ]
    ]


    # Identity - can memorize inputs ?
    xs = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ]
    """

    xs = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 1]
    ]
    xs = np.array(xs)

    input_activation_fn  = "relu"
    output_activation_fn = "sigmoid"

    if input_activation_fn == "tanh":
        xs = (xs - 0.5) * 2.0

    ys = np.sum(xs, axis=1, keepdims=True) * 1.0
    ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys))
    ys = get_array(ys)
    """ Test as an Auto Encoder """
    #ys = xs

    if output_activation_fn == "tanh" and np.min(ys.flatten()) == 0.0:
        ys = (ys - 0.5) * 2.0

    num_hidden = int(round(np.log2(xs.shape[1])))

    layers = [
        Layer(xs.shape[1], num_hidden,  activation_fn = input_activation_fn),
        #Layer(num_hidden, num_hidden,  activation_fn = input_activation_fn),
        Layer(num_hidden,  ys.shape[1], activation_fn = output_activation_fn),
    ]

    """ Note that the range of inputs for tanh is 2* sigmoid, and so the MAE should be 2* """
    nn = MLP(layers,
             learning_rate=0.5, weight_decay=0.0, epochs=10000, batch_size=4,
             lr_increase_multiplier=1.05, lr_decrease_multiplier=0.95)

    nn.fit(xs, ys)
    hidden_activations = nn.predict(xs, 0)
    predictions = nn.predict(xs)

    print "ys"
    print np.round(ys, 1)
    print "predictions"
    #print np.round(ae.prop_up(xs, xs)[0] * 3.0) * 0.3
    print predictions
    print np.round(predictions, 1)
    print np.round(predictions, 0)
    pass

    """ TODO

    *** Use finite gradients method to verify gradient descent calc. Bake into code as a flag ***

    allow different activation functions per layer. Normally hidden layer uses RELU and dropout (http://fastml.com/deep-learning-these-days/)
       don't use RELU for output layer as you cannot correct for erors (i.e. gradient is 0 for negative updates!)
    implement momentum (refer to early parts of this https://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf)
    implement adaptive learning rate adjustments (see link above)
    implement DROPOUT

    """
