import numpy as np
import gnumpy as gp

from Layers import Layer, DropOutLayer, dropout_mask

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

class GradientChecker(object):
    """ Gradient Checking """

    def estimate_gradient(self, xs, ys, layers, epsilon=0.0001, input_masks=None):

        loss_type = "mse"
        if layers[-1].activation_fn == "softmax":
            loss_type = "crossentropy"

        layer_gradient = []
        for ix, l in enumerate(layers):
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

                    p_clone.best_weights[i, j] += epsilon
                    n_clone.best_weights[i, j] -= epsilon

                    p_loss = self.loss(xs, ys, p_layers, loss_type=loss_type, input_masks=input_masks)
                    n_loss = self.loss(xs, ys, n_layers, loss_type=loss_type, input_masks=input_masks)
                    wgrad[i, j] = ((p_loss - n_loss) / (2 * epsilon)).sum()

            for i in range(len(l.best_bias)):
                p_clone = l.clone()
                n_clone = l.clone()

                p_layers = layers[::]
                p_layers[ix] = p_clone

                n_layers = layers[::]
                n_layers[ix] = n_clone

                p_clone.best_bias[i] += epsilon
                n_clone.best_bias[i] -= epsilon

                p_loss = self.loss(xs, ys, p_layers, loss_type=loss_type, input_masks=input_masks)
                n_loss = self.loss(xs, ys, n_layers, loss_type=loss_type, input_masks=input_masks)
                bgrad[i] = ((p_loss - n_loss) / (2 * epsilon)).sum()

        return layer_gradient

    def loss(self, input_vectors, outputs, layers, loss_type="mse", input_masks=None):

        # Note that this function does not transpose the inputs or outputs
        # Each row is a separate example \ label (i.e. row not column vectors)

        # Predict
        a = self.__ensure_vector_format__(input_vectors).T
        for i, layer in enumerate(layers):
            if input_masks is not None \
                    and input_masks[i] is not None:
                a = np.multiply(a, input_masks[i])

            # Ensure method called on layer not drop out layer
            if type(layer) == DropOutLayer:
                z, a = Layer.feed_forward(layer, a)
            else:
                z, a = layer.feed_forward(a)
        predictions = a.T

        # error loss
        if loss_type == "mse":
            errors = predictions - outputs
            error_loss = (0.5 * ( np.multiply(errors, errors))  ).mean(axis=0)
        elif loss_type == "crossentropy":
            error_loss = -((np.multiply(outputs, np.log(predictions)).sum(axis=1).mean()))
        else:
            raise Exception("Unknown loss type: " + loss_type)

        # weight decay loss
        weight_decay_loss = 0.0
        for l in layers:
            if l.weight_decay > 0.0:
                weight_decay_loss += (l.weight_decay / 2.0) * (l.best_weights ** 2.0).sum()

        # return combined loss function
        return error_loss + weight_decay_loss

    def verify_gradient(self, xs, ys, nnet):

        epsilon = 0.00001
        input_masks = []
        rows = xs.shape[0]

        momentums = [l.momentum for l in nnet.layers]

        for l in nnet.layers:
            l.momentum = 0.0
            if type(l) == DropOutLayer:
                input_masks.append(dropout_mask(np.ones((l.weights.shape[1], rows)), l.drop_out_prob))
            else:
                input_masks.append(None)

        grad_est = self.estimate_gradient(xs, ys, nnet.layers, epsilon, input_masks)
        errors, grad = nnet.__compute_gradient__(xs, ys, nnet.layers, 1.0, input_masks)

        for i in range(len(grad)):
            wdelta, bdelta = grad[i]
            est_wdelta, est_bdelta = grad_est[i]

            max_wts_diff = np.max(np.abs(wdelta - est_wdelta))
            max_bias_diff = np.max(np.abs(bdelta - est_bdelta))
            if max_wts_diff > epsilon:
                print "Estimated"
                print est_wdelta
                print "Actual"
                print wdelta
                assert 1 == 2, "Significant Difference in estimated versus computed weights gradient: " + str(
                    max_wts_diff)

            if max_bias_diff > epsilon:
                print "Estimated"
                print est_bdelta
                print "Actual"
                print bdelta
                assert 1 == 2, "Significant Difference in estimated versus computed bias gradient :" + str(
                    max_bias_diff)

            print i, "Max Wt Diff:", str(max_wts_diff), "Max Bias Diff:", max_bias_diff

        for i, l in enumerate(nnet.layers):
            l.momentum = momentums[i]
        print "Gradient is correct"

    """ END Gradient Checking """

    def __ensure_vector_format__(self, a):
        return get_array(a)