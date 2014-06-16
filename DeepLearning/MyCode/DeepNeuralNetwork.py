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

def reverse_layers(layers):
    """ For flipping auto encoders and other deep networks,
        makes a copy of the layers but in reverse, sharing the weights and bias'

        layers = a list of Layer

        returns a list of Layer
    """
    rev_layers = []
    for i in range(len(layers)):
        layer = layers[i]
        bias = None if i == 0 else layers[i - 1].bias
        l = Layer(layer.num_outputs, layer.num_inputs, activation_fn=layer.activation_fn, weights=layer.weights.T,
                  bias=bias)
        rev_layers.insert(0, l)
    return rev_layers

class Layer(object):
    def __init__(self, num_inputs, num_outputs, activation_fn="tanh", initial_wt_max=0.01, weights=None, bias=None):
        self.activation_fn = activation_fn
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs

        if weights is None:
            weights = get_array(matlib.rand((num_outputs, num_inputs)) * initial_wt_max)

        if bias is None:
            bias = get_array(matlib.rand((num_outputs, 1)) * initial_wt_max)

        self.initial_wt_max = initial_wt_max
        self.weights        = weights
        self.bias           = bias

        #Force creation of best weights\bias
        self.save_state()

        assert self.num_inputs == self.weights.shape[1]
        assert self.num_outputs == self.weights.shape[0]
        assert self.num_outputs == self.bias.shape[0]

    def clone(self):
        return Layer( self.num_inputs, self.num_outputs, self.activation_fn, self.initial_wt_max, self.best_weights.copy(), self.best_bias.copy())

    def save_state(self):
        self.best_weights   = self.weights.copy()
        self.best_bias      = self.bias.copy()

    def revert_state(self):
        self.weights        = self.best_weights.copy()
        self.bias           = self.best_bias.copy()

    # for prediction (for case like dropout where we need to do something different
    # here by overriding this function
    def feed_forward(self, inputs_T):
        z = self.__compute_z__(inputs_T, self.best_weights, self.best_bias)
        a = self.__activate__(z, self.activation_fn)
        return (z, a)

    # for training
    def prop_up(self, inputs_T):

        """ Compute activations """
        z = self.__compute_z__(inputs_T, self.weights, self.bias)
        a = self.__activate__(z, self.activation_fn)
        return (z, a)

    def derivative(self, activations):

        if self.activation_fn == "sigmoid":
            """ f(z)(1 - f(z)) """
            return np.multiply(activations, (1.0 - activations))
        elif self.activation_fn == "tanh":
            """ 1 - f(z)^2 """
            return 1.0 - np.square(activations)
        elif self.activation_fn == "linear":
            return 1.0
        elif self.activation_fn == "relu":
            copy = activations.copy() # don't modify vector
            copy[copy < 0] = 0
            copy[copy > 0] = 1.0
            return copy
        else:
            raise NotImplementedError("Only sigmoid, tanh, linear and relu currently implemented")

    def update(self, wtdiffs, biasdiff):
        self.weights -= wtdiffs
        self.bias    -= biasdiff

    def __compute_z__(self, inputs_T, weights, bias):
        #Can we speed this up by making the bias a column vector?
        return np.dot(weights, inputs_T) + bias

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
        self.lst_mse = []
        self.lst_mae = []

    def predict(self, inputs, layer_ix = np.inf, layers = None):
        if layers is None:
            layers = self.layers
        a = self.__ensure_vector_format__(inputs).T
        for i, layer in enumerate(layers):
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

        ixs = range(len(xs))
        for epoch in range(epochs):

            """ Shuffle the dataset on each epoch """
            np.random.shuffle(ixs)
            inputs = inputs[ixs]
            outputs = outputs[ixs]
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

                mini_batch_errors, gradients = self.__compute_gradient__(mini_batch_in, mini_batch_out, len(xs),
                                                                         self.layers, self.learning_rate, self.weight_decay)
                if np.any(np.isnan(mini_batch_errors)):
                    print "Nans in errors. Stopping"
                    return (mse, mae)
                errors.extend(mini_batch_errors)

                # apply weight updates
                for layer, gradient in zip(self.layers, gradients):
                    wds, bds = gradient
                    layer.update(wds, bds)

            errors = get_array(errors)
            mse = np.mean(np.square(errors))
            mae = np.mean(np.abs(errors))

            DIGITS = 6
            print "MSE for epoch {0} is {1}".format(epoch, np.round(mse,DIGITS)),
            print "\tMAE for epoch {0} is {1}".format(epoch, np.round(mae,DIGITS)),
            print "\tlearning rate is {0}".format(self.learning_rate)

            if len(self.lst_mae) > 0:
                self.__adjust_learning_rate__(self.lst_mae[-1], mae)
            self.lst_mse.append(mse)
            self.lst_mae.append(mae)
        return (mse, mae)

    """ Gradient Checking """
    def estimate_gradient(self, xs, ys, layers = None, epsilon = 0.0001):

        if layers is None:
            layers = self.layers[::]

        layer_gradient = []
        for ix,l in enumerate(layers):
            wgrad = np.zeros(l.best_weights.shape)
            bgrad = np.zeros(l.best_bias.shape)

            layer_gradient.append((wgrad, bgrad))

            for i in range(l.best_weights.shape[0]):
                for j in range(l.best_weights.shape[1]):

                    p_clone = l.clone()
                    n_clone = l.clone()

                    p_layers = layers[::]
                    p_layers[ix] = p_clone

                    n_layers = layers[::]
                    n_layers[ix] = n_clone

                    p_clone.best_weights[i,j] += epsilon
                    n_clone.best_weights[i,j] -= epsilon

                    p_loss = self.loss(xs, ys, p_layers )
                    n_loss = self.loss(xs, ys, n_layers )
                    wgrad[i,j] = ((p_loss - n_loss) / (2*epsilon)).sum()

            for i in range(len(l.best_bias)):
                p_clone = l.clone()
                n_clone = l.clone()

                p_layers = layers[::]
                p_layers[ix] = p_clone

                n_layers = layers[::]
                n_layers[ix] = n_clone

                p_clone.best_bias[i] += epsilon
                n_clone.best_bias[i] -= epsilon

                p_loss = self.loss(xs, ys, p_layers)
                n_loss = self.loss(xs, ys, n_layers)
                bgrad[i] = ((p_loss - n_loss) / (2 * epsilon)).sum()

        return layer_gradient

    def loss(self, input_vectors, outputs, layers = None):

        if layers is None:
            layers = self.layers

        predictions = self.predict(input_vectors, layers=layers)

        errors = predictions - outputs

        # weight decay loss
        sum_wts = 0.0
        for l in layers:
            sum_wts += (l.best_weights ** 2.0).sum()

        mean_squared_loss = (0.5 * ((errors) ** 2.0)).mean(axis=0)
        weight_decay_loss = (self.weight_decay / 2.0) * sum_wts

        return mean_squared_loss + weight_decay_loss

    def verify_gradient(self, xs, ys):

        epsilon = 0.0001
        errors, grad = self.__compute_gradient__(xs, ys, len(xs), self.layers, 1.0, self.weight_decay)
        grad_est = self.estimate_gradient(xs, ys, self.layers, epsilon)

        for i in range(len(grad)):
            wdelta, bdelta = grad[i]
            est_wdelta, est_bdelta = grad_est[i]

            assert np.max(np.abs( wdelta - est_wdelta )) <= epsilon, "Significant Difference in estimated versus computed weights gradient"
            assert np.max(np.abs( bdelta - est_bdelta )) <= epsilon, "Significant Difference in estimated versus computed bias gradient"

        print "Gradient is correct"

    """ END Gradient Checking """
    def __compute_gradient__(self, input_vectors, outputs, total_rows, layers, learning_rate, weight_decay):

        rows = input_vectors.shape[0]
        inputs_T = input_vectors.T
        outputs_T = outputs.T

        activations = [] # f(Wt.x)
        zs = []          # Wt.x
        derivatives = [] #f'(Wt.x)

        a = inputs_T
        for layer in layers:
            z, a = layer.prop_up(a)
            deriv = layer.derivative(a)
            activations.append(a)
            zs.append(z)
            derivatives.append(deriv)

        top_layer_output = activations[-1]
        """ errors = mean( 0.5 sum squared error)  """
        assert outputs_T.shape == top_layer_output.shape
        errors = (outputs_T - top_layer_output)

        # Compute weight updates
        delta = -(errors) * derivatives[-1]

        deltas = [delta]
        for i in range(len(layers) -1):
            ix = -(i + 1)
            layer = layers[ix]
            """ THIS IS BACK PROP OF ERRORS TO HIDDEN LAYERS"""
            delta = np.dot(layer.weights.T, delta) * derivatives[ix-1]
            deltas.insert(0, delta)

        #TODO Sparsity
        frows = float(rows)
        batch_proportion = frows / float(total_rows)

        gradients = []
        for i, layer in enumerate(layers):
            delta = deltas[i]
            activation_T = input_vectors if i == 0 else activations[i-1].T

            """ Delta for weights is the dot product of the layer delta (error deltas for output)
                and activations for that layer

                For each weight in the weight matrix, update it using the input activation * output delta.
                Compute a mean over all examples in the batch.

                The dot product is used here in a very clever  way to compute the activation * the delta
                for each input and hidden layer node (taking the dot product of each weight over all input_vectors
                (adding up the weight deltas) and then dividing this by num rows to get the mean
             """
            wtdelta   = ((np.dot(delta, activation_T))         / (frows))

            """ As the inputs are always 1 then the activations are omitted for the bias """
            biasdelta = ((np.sum(delta, axis=1, keepdims=True) / (frows)))

            if weight_decay > 0.0:
                """ Weight decay is typically not done for the bias as this is known
                    to have marginal effect.
                """
                wds = learning_rate * batch_proportion * (wtdelta + weight_decay * layer.weights)
            else:
                wds = learning_rate * batch_proportion * wtdelta

            bds = learning_rate * biasdelta
            gradients.append((wds, bds))

        """ return a list of errors (one item per row in mini batch) """
        return (errors.T, gradients)

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
            for layer in self.layers:
                layer.revert_state()

        # restrict learning rate to sensible bounds
        self.learning_rate = max(0.001, self.learning_rate)
        self.learning_rate = min(1.000, self.learning_rate)

    def __ensure_vector_format__(self, a):
        return get_array(a)

    def __validate__(self, outputs, layer):

        min_outp = np.min(outputs)
        max_outp = np.max(outputs)

        if layer.activation_fn == "sigmoid":
            self.__in_range__(min_outp, max_outp, 0.0, 1.0)
        elif layer.activation_fn == "tanh":
            self.__in_range__(min_outp, max_outp, -1.0, 1.0)
        elif layer.activation_fn == "relu":
            self.__in_range__(min_outp, max_outp, 0.0, np.inf)
        elif layer.activation_fn == "linear":
            pass
        else:
            raise Exception("Unknown activation function %s" % layer.activation_fn)

    def __in_range__(self, actual_min, actual_max, exp_min, exp_max):
        assert actual_max <= exp_max
        assert actual_min >= exp_min


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
        #[0, 1, 0, 0, 1, 0, 1, 0],
        #[1, 0, 1, 0, 1, 0, 0, 1]
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
    ys = xs

    if output_activation_fn == "tanh" and np.min(ys.flatten()) == 0.0:
        ys = (ys - 0.5) * 2.0

    num_hidden = int(round(np.log2(xs.shape[1]))) + 1
    #num_hidden = int(round((xs.shape[1] / 2.0)))

    layers = [
        Layer(xs.shape[1], num_hidden,  activation_fn = input_activation_fn,  initial_wt_max=0.01),
        #Layer(num_hidden,  num_hidden,  activation_fn = input_activation_fn,  initial_wt_max=0.01),
        #Layer(num_hidden,  num_hidden,  activation_fn = input_activation_fn,  initial_wt_max=0.01),
        Layer(num_hidden,  ys.shape[1], activation_fn = output_activation_fn, initial_wt_max=0.01),
    ]

    """ Note that the range of inputs for tanh is 2* sigmoid, and so the MAE should be 2* """
    nn = MLP(layers,
             learning_rate=0.5, weight_decay=0.0, epochs=100, batch_size=4,
             lr_increase_multiplier=1.1, lr_decrease_multiplier=0.9)

    nn.fit(     xs, ys, epochs=1000,)

    """ Verift Gradient Calculation """
    errors, grad = nn.__compute_gradient__(xs, ys, xs.shape[0], nn.layers, 1.0, nn.weight_decay)
    grad_est = nn.estimate_gradient(xs, ys)
    nn.verify_gradient(xs, ys)

    hidden_activations = nn.predict(xs, 0)
    predictions = nn.predict(xs)

    if output_activation_fn == "tanh":
        ys = ys / 2.0 + 0.5
        predictions = predictions / 2.0 + 0.5

    print "ys"
    print np.round(ys, 1) * 1.0
    print "predictions"
    #print np.round(ae.prop_up(xs, xs)[0] * 3.0) * 0.3
    print np.round(predictions, 1)
    print predictions
    print "Weights"
    print nn.layers[0].weights
    print ""
    print nn.layers[1].weights
    pass

    """ TODO
    implement momentum (refer to early parts of this https://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf)
    implement DROPOUT
    use LBFGS or conjugate gradient descent to optimize the parameters instead as supposedly faster

    >>>> DONE allow different activation functions per layer. Normally hidden layer uses RELU and dropout (http://fastml.com/deep-learning-these-days/)
          don't use RELU for output layer as you cannot correct for errors (i.e. gradient is 0 for negative updates!)
    >>>> DONE implement adaptive learning rate adjustments (see link above)
    >>>> DONE Use finite gradients method to verify gradient descent calc. Bake into code as a flag ***
    """
