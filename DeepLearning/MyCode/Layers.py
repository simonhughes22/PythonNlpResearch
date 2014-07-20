import numpy as np

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
    def __init__(self, num_inputs, num_outputs, activation_fn="tanh", momentum=0.5, weight_decay=0.0, weights=None, bias=None):
        self.activation_fn = activation_fn
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.weight_decay = weight_decay
        self.momentum = momentum

        init_min, init_max = self.__initial_weights_min_max__()

        if weights is None:
            weights = np.random.uniform(low=init_min, high=init_max, size=(self.num_outputs, self.num_inputs))

        if bias is None:
            if self.activation_fn == "relu":
                # enforce positive
                bias = np.ones((num_outputs, 1))
            else:
                bias = np.zeros((num_outputs, 1))

        self.weights        = weights
        self.bias           = bias

        #Force creation of best weights\bias
        self.save_state()
        #Moving Average of weights update
        self.ma_weights = None
        self.ma_bias    = None

        assert self.num_inputs == self.weights.shape[1]
        assert self.num_outputs == self.weights.shape[0]
        assert self.num_outputs == self.bias.shape[0]

    def __initial_weights_min_max__(self):
        if self.activation_fn == "tanh":
            val = np.sqrt(6.0 / (self.num_inputs + self.num_inputs))
            return (-val, val)
        elif self.activation_fn == "sigmoid" or self.activation_fn == "softmax":
            val = np.sqrt(4 * (6.0 / (self.num_inputs + self.num_inputs)))
            return (-val, val)
        elif self.activation_fn == "linear" or self.activation_fn == "relu":
            #for relu we set the bias' to 1, ensuring activations on positive inputs
            return (-0.001, 0.001)

    def clone(self):
        return Layer( self.num_inputs, self.num_outputs, self.activation_fn, self.momentum, self.weight_decay, self.best_weights.copy(), self.best_bias.copy())

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

    def update(self, wtdiffs, biasdiff):

        if self.ma_weights is None:
            momentum_wt_update   = wtdiffs
            momentum_bias_update = biasdiff
        else:
            momentum_wt_update   = self.momentum * self.ma_weights + (1.0 - self.momentum) * wtdiffs
            momentum_bias_update = self.momentum * self.ma_bias    + (1.0 - self.momentum) * biasdiff

        self.weights -= momentum_wt_update
        self.bias    -= momentum_bias_update

        self.ma_weights = momentum_wt_update
        self.ma_bias    = momentum_bias_update

    def derivative(self, activations):

        if self.activation_fn == "sigmoid":
            """ f(z)(1 - f(z)) """
            return np.multiply(activations, (1.0 - activations))

        elif self.activation_fn == "softmax":
            # So long as we correctly compute the soft max output, derivative is linear
            return np.ones(activations.shape)

        elif self.activation_fn == "tanh":
            """ 1 - f(z)^2 """
            return 1.0 - np.square(activations)

        elif self.activation_fn == "linear":
            return np.ones(activations.shape)

        elif self.activation_fn == "relu":
            copy = activations.copy() # don't modify vector
            copy[copy < 0] = 0
            copy[copy > 0] = 1.0
            return copy
        else:
            raise NotImplementedError("Only sigmoid, tanh, linear and relu currently implemented")

    def backprop_deltas(self, delta):
        return np.dot(self.weights.T, delta)

    def gradients(self, delta, inputs):
        frows = float(inputs.shape[1])
        wtdelta = ((np.dot(delta, inputs.T)) / (frows))

        """ For each weight, update it using the input activation * output delta. Compute a mean over all examples in the batch.

            The dot product is used here in a very clever  way to compute the activation * the delta
            for each input and hidden layer node and then dividing this by num rows to get the mean

            As the inputs are always 1, the activations are omitted for the bias
        """
        biasdelta = ((np.sum(delta, axis=1, keepdims=True) / (frows)))

        if self.weight_decay > 0.0:
            """ Weight decay is typically not done for the bias as has marginal effect."""
            wtdelta += (self.weight_decay * self.weights)
        return (wtdelta, biasdelta)


    def __compute_z__(self, inputs_T, weights, bias):
        #Can we speed this up by making the bias a column vector?
        return np.dot(weights, inputs_T) + bias

    def __activate__(self, zs, activation_fn):

        if activation_fn == "sigmoid":
            return 1 / (1 + np.exp(-zs))

        elif activation_fn == "softmax":
            exponents = np.exp(zs)
            totals = exponents.sum(axis=0)
            if len(totals.shape) == 1:
                totals = totals.reshape((1, len(totals)))
            return exponents / totals

        elif activation_fn == "tanh":
            return np.tanh(zs)
        elif activation_fn == "linear":
            return zs
        elif activation_fn == "relu":
            copy = zs.copy()
            copy[copy < 0] = 0
            return copy
        else:
            raise NotImplementedError("Only sigmoid, tanh, softmax, linear and relu currently implemented")

    def __repr__(self):
        return str(self.weights.shape[1]) + "->" + str(self.weights.shape[0]) + " : " + self.activation_fn

class ConvolutionalLayer(Layer):
    def __init__(self, num_inputs, num_outputs, convolutions, activation_fn="tanh", momentum=0.5, weight_decay=0.0, weights=None, bias=None):
        Layer.__init__(self, num_inputs / convolutions, num_outputs / convolutions, activation_fn, momentum, weight_decay, weights, bias)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.convolutions = convolutions

    def clone(self):
        return ConvolutionalLayer(self.num_inputs, self.num_outputs, self.convolutions, self.activation_fn, self.momentum, self.weight_decay,
                     self.best_weights.copy(), self.best_bias.copy())

    def __compute_z__(self, inputs_T, weights, bias):
        in_cols  = weights.shape[1]
        zs = []
        for c in range(self.convolutions):
            con_inputs_T = inputs_T[c * in_cols: (c+1)*in_cols, :]
            conv_z = np.dot(weights, con_inputs_T) + bias
            #Stack results
            zs.append(conv_z)
        return np.vstack(zs)

    def backprop_deltas(self, delta):
        in_rows = self.weights.shape[0]
        dotprods = []
        for c in range(self.convolutions):
            con_delta = delta[c * in_rows: (c + 1) * in_rows, :]
            conv_z = np.dot(self.weights.T, con_delta)
            # Stack results
            dotprods.append(conv_z)
        return np.vstack(dotprods)


    def __repr__(self):
        conv = str(self.convolutions) + " * ["
        return conv + str(self.weights.shape[1]) + "->" + str(self.weights.shape[0]) + "] : " + self.activation_fn


"""
    def update(self, wtdiffs, biasdiff):

        in_cols  = self.weights.shape[1]
        out_cols = self.weights.shape[0]
        con_wt_diffs =  wtdiffs[:,  0: in_cols]
        con_bias     = biasdiff[0: out_cols,:]

        for c in range(1, self.convolutions):
            con_wt_diffs += wtdiffs[:,  c * in_cols: (c + 1) * in_cols]
            con_bias     += biasdiff[c * out_cols: (c + 1) * out_cols,:]

        if self.ma_weights is None:
            momentum_update = con_wt_diffs
        else:
            momentum_update = self.momentum * self.ma_weights + (1.0 - self.momentum) * con_wt_diffs
        self.weights -= momentum_update
        self.bias -= biasdiff
        self.ma_weights = momentum_update
"""

def dropout_mask(inputs_T, drop_out_prob):
    mask = np.matlib.rand(inputs_T.shape)
    mask[mask >= drop_out_prob] = 1.0
    mask[mask <  drop_out_prob] = 0.0
    return mask

class DropOutLayer(Layer):

    def __init__(self, num_inputs, num_outputs, activation_fn="tanh", drop_out_prob = 0.5, weight_decay=0.0, weights=None, bias=None):
        Layer.__init__(self, num_inputs, num_outputs, activation_fn, weight_decay, weights, bias)
        self.drop_out_prob = drop_out_prob

    def feed_forward(self, inputs_T):
        wts = self.best_weights * (1.0 - self.drop_out_prob)
        z = self.__compute_z__(inputs_T, wts, self.best_bias)
        a = self.__activate__(z, self.activation_fn)
        return (z, a)

    def clone(self):
        return DropOutLayer(self.num_inputs, self.num_outputs,
                            self.activation_fn, self.drop_out_prob,
                            self.best_weights.copy(), self.best_bias.copy())

    def revert_state(self):
        pass