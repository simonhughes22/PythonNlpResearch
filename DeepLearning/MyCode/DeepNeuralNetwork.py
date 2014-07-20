'''
Created on Aug 18, 2013
@author: simon.hughes

Auto-encoder implementation. Can be used to implement a denoising auto-encoder, sparse or contractive auto-encoder
'''

import numpy as np
import gnumpy as gp

from Layers import Layer, ConvolutionalLayer, DropOutLayer, dropout_mask
from GradientChecking import GradientChecker

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

    def fit(self, xs, ys, min_error=0.000001, epochs = None, batch_size = None):

        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size

        inputs  = self.__ensure_vector_format__(xs)
        outputs = self.__ensure_vector_format__(ys)

        num_rows = inputs.shape[0]

        """ Number of rows in inputs should match those in outputs """
        assert inputs.shape[0] == outputs.shape[0], "Xs and Ys do not have the same row count"

        assert inputs.shape[1]  == self.layers[0].num_inputs,   "The input layer does not match the Xs column count"
        assert outputs.shape[1] == self.layers[-1].num_outputs, "The output layer does not match the Ys column count"

        """ Check outputs match the range for the activation function for the layer """
        self.__validate_outputs__(outputs, self.layers[-1])

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

                mini_batch_errors, gradients = self.__compute_gradient__(mini_batch_in, mini_batch_out,
                                                                         self.layers, self.learning_rate)
                if np.any(np.isnan(mini_batch_errors)):
                    print "Nans in errors. Stopping"
                    self.__reset_layers__()
                    return (mse, mae)
                errors.extend(mini_batch_errors)
                # apply weight updates
                for layer, gradient in zip(self.layers, gradients):
                    wds, bds = gradient
                    layer.update(wds, bds)

            errors = get_array(errors)
            mse = np.mean(np.square(errors))
            mae = np.mean(np.abs(errors))

            DIGITS = 8
            print "MSE for epoch {0} is {1}".format(epoch, np.round(mse,DIGITS)),
            print "\tMAE for epoch {0} is {1}".format(epoch, np.round(mae,DIGITS)),
            print "\tlearning rate is {0}".format(self.learning_rate)
            if len(self.lst_mae) > 0:
                self.__adjust_learning_rate__(self.lst_mae[-1], mae)
            if mse <= min_error:
                print "MSE is %s. Stopping" % str(mse)
                return (mse, mae)
            self.lst_mse.append(mse)
            self.lst_mae.append(mae)
        return (mse, mae)

    def __compute_gradient__(self, input_vectors, outputs, layers, learning_rate, input_masks=None):

        inputs_T = input_vectors.T
        outputs_T = outputs.T

        layer_inputs =  []

        masks = []
        a = inputs_T
        outputs = []
        for ix, layer in enumerate(layers):
            layer_input = a
            mask, layer_input = self.__get_masked_input__(layer_input, input_masks, layer, ix)
            z, a = layer.prop_up(layer_input)

            masks.append(mask)
            layer_inputs.append(layer_input)
            outputs.append(a)

        top_layer_output = a
        activations = layer_inputs[1:] + [top_layer_output]

        """ errors = mean( 0.5 sum squared error) (but gradient w.r.t. weights is sum(errors) """
        assert outputs_T.shape == top_layer_output.shape
        errors = (outputs_T - top_layer_output)

        # Compute weight updates
        delta = np.multiply( -(errors), layers[-1].derivative(activations[-1]))
        deltas = [delta]
        for i in range(len(layers) - 1):
            ix = -(i + 1)
            layer = layers[ix - 1]
            layer_deriv = layer.derivative(activations[ix - 1])

            """ THIS IS BACK PROP OF ERRORS TO HIDDEN LAYERS"""
            upper_layer = layers[ix]
            delta = np.multiply(upper_layer.backprop_deltas(delta), layer_deriv)
            if masks[ix] is not None:
                delta = np.multiply(delta, masks[ix])
            deltas.insert(0, delta)

        # TODO Sparsity
        gradients = []
        for i in range(len(layers)):
            wtdelta, biasdelta = layers[i].gradients(deltas[i], layer_inputs[i])
            gradients.append( (learning_rate * wtdelta, learning_rate * biasdelta) )
        """ return a list of errors (one item per row in mini batch) """
        return (errors.T, gradients)


    def __get_masked_input__(self, layer_input, input_masks, layer, ix):
        if type(layer) == DropOutLayer:
            if input_masks is None:
                mask = dropout_mask(layer_input, layer.drop_out_prob)
            else:
                mask = input_masks[ix]
            masked_input = np.multiply(mask, layer_input)
            return (mask, masked_input)
        else:
            return (None, layer_input)

    def __reset_layers__(self):
        for layer in self.layers:
            layer.revert_state()

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
            self.__reset_layers__()
        # restrict learning rate to sensible bounds
        self.learning_rate = max(0.001, self.learning_rate)
        self.learning_rate = min(1.000, self.learning_rate)

    def __ensure_vector_format__(self, a):
        return get_array(a)

    def __validate_outputs__(self, outputs, layer):

        min_outp = np.min(outputs)
        max_outp = np.max(outputs)

        if layer.activation_fn == "sigmoid":
            self.__in_range__(min_outp, max_outp, 0.0, 1.0)
        elif layer.activation_fn == "softmax":
            unique = set(outputs.flatten())
            assert len(unique) == 2,                                "Wrong number of outputs. Outputs for softmax must be 0's and 1's"
            assert min(unique) == 0 and max(unique) ==1,            "Outputs for softmax must be 0's and 1's only"
            assert np.all(outputs.sum(axis=1) == 1.0), "Outputs for a softmax layer must sum to 1."

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

    xs = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0],
        #[1, 0, 1, 0, 1, 0, 0, 1]
    ]
    xs = np.array(xs)

    input_activation_fn  = "sigmoid"

    # Having a linear output layer seems to work REALLY well
    output_activation_fn = "tanh"

    if input_activation_fn == "tanh":
        xs = (xs - 0.5) * 2.0

    ys = np.sum(xs, axis=1, keepdims=True) * 1.0
    ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys))
    ys = get_array(ys)
    """ Test as an Auto Encoder """
    soft_max_ys = []
    for x in xs:
        l = [0 for i in x]
        l[int(sum(x)) - 1] = 1
        soft_max_ys.append(l)
    soft_max_ys = get_array(soft_max_ys)

    ys = xs
    if output_activation_fn == "softmax":
        ys = soft_max_ys

    if output_activation_fn == "tanh" and np.min(ys.flatten()) == 0.0:
        ys = (ys - 0.5) * 2.0

    #num_hidden = int(round(np.log2(xs.shape[1]))) + 1
    num_hidden = 4
    #num_hidden = int(round((xs.shape[1])) * 1.1)

    layers = [
        Layer(xs.shape[1], num_hidden-1,  activation_fn = input_activation_fn,  momentum=0.5),
        Layer(num_hidden-1, num_hidden,  activation_fn = input_activation_fn,  momentum=0.5),

        #ConvolutionalLayer(6, num_hidden, convolutions=2, activation_fn = input_activation_fn,  momentum=0.5),
        #Layer(xs.shape[1], num_hidden,  activation_fn = input_activation_fn,  momentum=0.5),
        #Layer(num_hidden,  num_hidden,  activation_fn = input_activation_fn,  momentum=0.5),
        Layer(num_hidden,  ys.shape[1], activation_fn = output_activation_fn, momentum=0.5),
    ]


    """ Note that the range of inputs for tanh is 2* sigmoid, and so the MAE should be 2* """
    nn = MLP(layers,
             learning_rate=0.5, weight_decay=0.0, epochs=100, batch_size=8,
             lr_increase_multiplier=1.1, lr_decrease_multiplier=0.9)

    nn.fit(     xs, ys, epochs=10,)

    """ Verify Gradient Calculation """
    grad_checker = GradientChecker()
    grad_checker.verify_gradient(xs, ys, nn)

    hidden_activations = nn.predict(xs, 0)
    predictions = nn.predict(xs)

    if np.min(ys) == -1 and np.max(ys) == 1:
        ys = ys / 2.0 + 0.5
        predictions = predictions / 2.0 + 0.5

    print "ys"
    print np.round(ys, 1) * 1.0
    print "predictions"
    #print np.round(ae.prop_up(xs, xs)[0] * 3.0) * 0.3
    print np.round(predictions, 1)
    print predictions

    """
    print "Weights"
    print nn.layers[0].weights
    print ""
    print nn.layers[1].weights
    pass
    """

    """ TODO
    can we use a clustering algorithm to initialize the weights for hidden layer neurons? Normalize the data.
        learn k clusters. Implement a 3 layer NN with k hidden neurons, whose weights are initialized to the values for
        the cluster centroid (as that maximizes cosine similarity). Adjust weights for a non linearity such that centroid
        input values would cause it to fire (which means just scaling up the weight vector, probably 6x for sigmoid). Train
        network as normal (top layer has random initial weights). cf regular initialization of nnet.
    use LBFGS or conjugate gradient descent to optimize the parameters instead as supposedly faster

    >>>> DONE allow different activation functions per layer. Normally hidden layer uses RELU and dropout (http://fastml.com/deep-learning-these-days/)
          don't use RELU for output layer as you cannot correct for errors (i.e. gradient is 0 for negative updates!)
    >>>> DONE  Implement adaptive learning rate adjustments (see link above)
    >>>> DONE  Use finite gradients method to verify gradient descent calc. Bake into code as a flag ***
    >>>> DONE  Implement momentum (refer to early parts of this https://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf)
    >>>> ~DONE implement DROPOUT
    """
