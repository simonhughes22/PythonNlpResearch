import numpy as np
from sklearn import svm, metrics

import Metrics
import NeuralNetwork as nnet

class RecursiveAutoEncoder(object):
    
    def __init__(self, projector, learning_rate = 0.1, activation_fn = "sigmoid", supervised_wt = 0.2,
                 initial_wt_max = 0.01, weight_decay = 0.0, desired_sparsity = 0.05, sparsity_wt = 0.01):
        
        self.projector = projector
        """ We initialize the first time train is called """
        self.init = False
        
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.supervised_wt = supervised_wt
        self.autoencoder_wt = 1.0 - supervised_wt
        
        """ An auto-encoder """
        self.initial_wt_max = initial_wt_max
        self.weight_decay = weight_decay
        self.desired_sparsity = desired_sparsity
        self.sparsity_wt = sparsity_wt
        """ END Properties """
        
        self.a2 = None
        self.nnet = None
        self.parent_vector_size = None
    
    def __init_learners__(self, xs, ys):
    
        self.init = True
        
        row1 = self.projector.project(xs[0])
        
        """ Hidden layer outputs vector of dimensionality of word vector """
        parent_vector_size = len(row1[0])
        """ Input layer size is two word vectors """
        input_vector_size = parent_vector_size * 2
        
        self.parent_vector_size = parent_vector_size
        num_classes = len(ys[0])
        
        self.ae = nnet.NeuralNetwork(input_vector_size, parent_vector_size, input_vector_size, learning_rate = self.learning_rate, activation_fns= self.activation_fn,
                                initial_wt_max = self.initial_wt_max, weight_decay = self.weight_decay, desired_sparsity = self.desired_sparsity,
                                sparsity_wt = self.sparsity_wt, w1_b1 = None, w2_b2 = None)
        
        """ Share the first set of params """
        w1_b1 = (self.ae.w1, self.ae.b1)
        """ DON'T share second set of weights """
        
        """ Initialize with a different learning rate, based on relative weighting """
        self.nnet = nnet.NeuralNetwork(input_vector_size, parent_vector_size, num_classes, learning_rate = self.learning_rate, activation_fns= self.activation_fn,
                                initial_wt_max = self.initial_wt_max, weight_decay = self.weight_decay, desired_sparsity = self.desired_sparsity,
                                sparsity_wt = self.sparsity_wt, w1_b1 = w1_b1, w2_b2 = None)
        
    def train(self, tokenized_docs, ys, epochs, batch_size = 500):
        
        if self.activation_fn == "tanh":
            def to_tanh_val(y):
                if y > 0:
                    return 1
                else:
                    return -1
            ys = [map(to_tanh_val, y) for y in ys]
        
        elif self.activation_fn == "sigmoid":
            def to_sigmoid_val(y):
                if y > 0:
                    return 1
                else:
                    return 0
            ys = [map(to_sigmoid_val, y) for y in ys]
        
        if not self.init:
            self.__init_learners__(tokenized_docs, ys)
    
        outputs = np.array(ys)
        
        num_rows = len(tokenized_docs)
        assert num_rows == outputs.shape[0]
        
        num_batches = num_rows / batch_size
        if num_rows % batch_size > 0:
            num_batches += 1
        
        batch_leaf_nodes = {}
       
        for epoch in range(epochs):
            top_level_inputs = []
            recon_errors = None
            cls_errors = None
            
            print ""
            print "EPOCH: ", epoch
            
            for batch in range(num_batches):
                print batch,
                
                start = batch * batch_size
                end = start + batch_size
                mini_batch_in = tokenized_docs[start:end]
                mini_batch_out = outputs[start:end]
                
                """ Leaf level input data will NOT change thru learning """
                if batch not in batch_leaf_nodes:
                    leaf_nodes, word_pairs, indices = self.__construct_leaf_nodes__(mini_batch_in)
                    batch_leaf_nodes[batch] = (leaf_nodes, word_pairs, indices)
                else:
                    leaf_nodes, word_pairs, indices = batch_leaf_nodes[batch]
                    
                reconstruction_errors, classification_errors, top_nts = self.__train_mini_batch__(leaf_nodes, word_pairs, indices, mini_batch_out)
                top_level_inputs.extend(top_nts)
                
                if recon_errors == None:
                    recon_errors = reconstruction_errors
                    cls_errors = classification_errors
                else:
                    recon_errors = np.append(recon_errors, reconstruction_errors, 0)
                    cls_errors   = np.append(cls_errors, classification_errors, 0)
                
            recon_mse = np.mean(np.square(recon_errors))
            cls_mse = np.mean(np.square(classification_errors))
            
            recon_mae = np.mean(np.abs(recon_errors))
            cls_mae = np.mean(np.abs(classification_errors))
            
            print ""
            print "[AE]   MSE for EPOCH: " + str(recon_mse)
            print "[AE]   MAE for EPOCH: " + str(recon_mae)
            print ""
            print "[NNet] MSE for EPOCH: " + str(cls_mse)
            print "[NNet] MAE for EPOCH: " + str(cls_mae)
            print ""
            
            a3, a2, err = self.nnet.prop_up(top_level_inputs, outputs)
            a3sorted = np.argsort(a3, 1)
            if self.activation_fn == "tanh":
                """ If tanh h, adjust to be [-1,1] """
                a3sorted = ((2 * a3sorted) -1)
            
            expected = outputs[:,1]
            actual = a3sorted[:,1].flatten().tolist()[0]
            
            r,p,f1 = Metrics.rpf1(expected, actual, class_value = 1)
            mse = np.mean(np.square(err))
            mae = np.mean(np.abs(err))
            print "Top-Level Classification Results:"
            print "\tMSE for EPOCH: " + str(mse)
            print "\tMAE for EPOCH: " + str(mae)
            print ""
            print "\tRecall:        " + str(r)
            print "\tPrecision:     " + str(p)
            print "\tF1:            " + str(f1)
            
            if epoch > 0 and epoch % 5 == 0:
                self.__run_classifier__(top_level_inputs, expected)
    
    def __run_classifier__(self, top_nodes, ys):
        print ""
        print "Training SVM Classifier"
        
        classifier = svm.SVC(C = 1.0, probability=True)
        classifier.fit(top_nodes, ys)
        probs = classifier.predict_proba(top_nodes)            
        
        best_threshold = -1
        best_f1 = -1
        
        def create_threshold_fn(threshold):
            def above_threshold(prob):
                    if prob[1] >= threshold:
                        return 1
                    return -1
            return above_threshold
        
        for i in range(9):
            threshold = (i + 1.0) / 10.0
            new_ys = map(create_threshold_fn(threshold), probs)
            score = metrics.f1_score(ys, new_ys)
            if score > best_f1:
                best_threshold = threshold
                best_f1 = score

        below = best_threshold - 0.1
        for i in range(21):
            threshold = below + (i / 100.0)
            new_ys = map(create_threshold_fn(threshold), probs)
            score = metrics.f1_score(ys, new_ys)
            if score > best_f1:
                best_threshold = threshold
                best_f1 = score
        
        new_ys = map(create_threshold_fn(best_threshold), probs)
        r = metrics.recall_score(ys, new_ys)
        p = metrics.precision_score(ys, new_ys)
        
        print "SVM Classification Results"
        print "\tThreshold:     " + str(best_threshold)
        print "\tRecall:        " + str(r)
        print "\tPrecision:     " + str(p)
        print "\tF1:            " + str(best_f1)
        
    def __construct_leaf_nodes__(self, tokenized_docs):
        leaf_nodes = []
        word_pairs = []
        indices = []
        
        tmp_input_vectors = [ self.__normalize__(self.projector.project(doc) ) for doc in tokenized_docs ]
        
        for i, input_vector in enumerate(tmp_input_vectors):
            
            doc = tokenized_docs[i]
            """ Handle single words for now (HACK)"""
            if len(input_vector) == 1:
                input_vector = [input_vector[0], input_vector[0]]
                doc = [doc[0], doc[0]]
                
            stop = len(input_vector) -1
            assert stop > 0
            for j in range(stop):
                leaf_nodes.append(np.append(input_vector[j] , input_vector[j+1], 0))
                word_pairs.append((doc[j],          doc[j+1]))
                indices.append(i)
        return (leaf_nodes, word_pairs, indices)
    
    def __train_mini_batch__(self, leaf_nodes, word_pairs, indices,  outputs):
        
        """ Xs start with leaf_nodes """
        xs = []
        ys = []
        
        top_level_inputs = []
        
        a2, err_left, err_right = self.__prop_up__(leaf_nodes)
        
        current_ix = indices[0]
        
        sent_leaf_nodes, sent_pairs, sent_errors_left, sent_errors_right, sent_parent_nodes = [], [], None, None, []
        prev_y = outputs[0]
        
        for i,ix in enumerate(indices):
            if ix != current_ix:
                current_ix = ix
                
                xs.extend(sent_leaf_nodes[:])
                new_inputs, output, top_nt  = self.__parse_sentence__(sent_leaf_nodes, sent_pairs, sent_parent_nodes, sent_errors_left, sent_errors_right)
                top_level_inputs.append(top_nt)
                
                xs.extend(new_inputs)
                ys.extend([prev_y for i in range(len(sent_leaf_nodes) + len(new_inputs) )])
                assert len(xs) == len(ys)
                """ Reset vars, store last y """
                sent_leaf_nodes, sent_pairs, sent_errors_left, sent_errors_right, sent_parent_nodes = [], [], None, None, []
                prev_y = outputs[ix]
            
            pass #End if
            sent_parent_nodes.append(a2[i])
            
            if sent_errors_left == None:
                sent_errors_left  = err_left[i]
                sent_errors_right = err_right[i]
            else:
                sent_errors_left  = np.append(sent_errors_left,  err_left[i],  0)
                sent_errors_right = np.append(sent_errors_right, err_right[i], 0)
                
            sent_leaf_nodes.append(leaf_nodes[i])
            sent_pairs.append(word_pairs[i])
        pass
        
        """ Process Last Sentence """
        new_inputs, output, top_nt = self.__parse_sentence__(sent_leaf_nodes, sent_pairs, sent_parent_nodes, sent_errors_left, sent_errors_right)
        top_level_inputs.append(top_nt)
        
        xs.extend(sent_leaf_nodes)
        xs.extend(new_inputs)
        ys.extend([prev_y for i in range(len(sent_leaf_nodes) + len(new_inputs) )])
       
        """ Train on data for the bath """
        
        aeW1ds,   aeW2ds,  aeB1ds,  aeB2ds,  aeErrs =   self.ae.get_training_errors(xs, xs)
        clsW1ds, clsW2ds, clsB1ds, clsB2ds, clsErrs = self.nnet.get_training_errors(xs, ys)
        
        """ Update the joint weights """
        
        self.ae.w1 -= (self.autoencoder_wt * aeW1ds + self.supervised_wt * clsW1ds)
        self.ae.b1 -= (self.autoencoder_wt * aeB1ds + self.supervised_wt * clsB1ds)
        
        """ Update separate second layers """
        self.ae.w2 -= aeW2ds
        self.ae.b2 -= aeB2ds
        
        self.nnet.w2 -= clsW2ds
        self.nnet.b2 -= clsB2ds
            
        """ Return Errors """
        return (aeErrs, clsErrs, top_level_inputs)
    
    def __normalize__(self, vectors):
        vectors = np.asarray(vectors, dtype=float)
        vectors /= np.sqrt((vectors ** 2).sum(-1))[..., np.newaxis]
        if self.activation_fn == "sigmoid":
            vectors = (vectors + 1.0) / 2.0
        return vectors #Ensure in the right output range for the ae
    
    def __prop_up__(self, vectors):
        a3, a2, err = self.ae.prop_up(vectors, vectors)
        
        err_left = np.square(err[:,:self.parent_vector_size]).mean(1)
        err_right = np.square(err[:,self.parent_vector_size:]).mean(1)
        return (self.__normalize__(a2), err_left, err_right)
    
    def __lowest_error_index__(self, err_l, err_r, left_node_count, right_node_count, node_totals):
        
        wtd_left =  np.multiply( (left_node_count  / node_totals), err_l)
        wtd_right = np.multiply( (right_node_count / node_totals), err_r)
        
        return np.argsort(wtd_left + wtd_right, 0)[0,0]
    
    def __parse_sentence__(self, leaf_nodes, word_pairs, parent_nodes, errors_left, errors_right):
        
        new_inputs = []
        nodes = leaf_nodes
        left_node_counts  = np.ones((len(leaf_nodes),1),dtype=float)
        right_node_counts = np.ones((len(leaf_nodes),1),dtype=float)
        total_node_counts = np.ones((len(leaf_nodes),1),dtype=float) * 2.0 #node pairs
        
        while len(nodes) > 1:
            to_replace = []
            ix = self.__lowest_error_index__(errors_left, errors_right, left_node_counts, right_node_counts, total_node_counts)
            new_nodes = []
            new_word_pairs = []
            
            new_node_size = left_node_counts[ix,0] + right_node_counts[ix,0]
            """ Construct new input vectors from words either side of lowest error """
            if ix > 0:
                to_replace.append(ix - 1)
                new_lhs = nodes[ix - 1][  :self.parent_vector_size]
                new_nodes.append( np.append(   new_lhs,  parent_nodes[ix])    )  
                              
                """ Construct a new word pair from the lhs of the left pair, and the new NT node """
                new_word_pairs.append( (word_pairs[ix - 1][0], word_pairs[ix]) )
                right_node_counts[ix - 1,0] = new_node_size
                total_node_counts[ix -1] = left_node_counts[ix - 1] + new_node_size
                
            if ix < len(errors_left) - 1:
                to_replace.append(ix + 1)
                new_rhs = nodes[ix + 1][self.parent_vector_size:  ]
                new_nodes.append(   np.append(  parent_nodes[ix], new_rhs ))
                
                """ Construct a new word pair from the rhs of the right pair, and the new NT node """
                new_word_pairs.append( (word_pairs[ix], word_pairs[ix + 1][1]) )
                left_node_counts[ix + 1,0] = new_node_size
                total_node_counts[ix + 1] = new_node_size + right_node_counts[ix + 1] 
            
            """ Run Auto-encoder once more """
            a2, err_left, err_right = self.__prop_up__(new_nodes)
            """ TODO Compute weighted errors_left """
                
            """ Replace with new inputs """
            for j, ix_rep in enumerate(to_replace):
                nodes[ix_rep] = new_nodes[j]
                word_pairs[ix_rep] = new_word_pairs[j]
                
                parent_nodes[ix_rep] = a2[j]
                errors_left[ix_rep] = err_left[j]
                errors_right[ix_rep] = err_right[j]
                
            b4 = len(nodes)
            """ Remove combined word pair """
            nodes = nodes[:ix] + nodes[ix + 1:]
            word_pairs = word_pairs[:ix] + word_pairs[ix + 1:]
            parent_nodes = parent_nodes[:ix] + parent_nodes[ix + 1:]
            
            errors_left = np.append(errors_left[:ix], errors_left[ix + 1:], 0)
            errors_right = np.append(errors_right[:ix], errors_right[ix + 1:], 0)
            
            left_node_counts = np.append(left_node_counts[:ix], left_node_counts[ix + 1:], 0)
            right_node_counts = np.append(right_node_counts[:ix], right_node_counts[ix + 1:], 0)
            total_node_counts = np.append(total_node_counts[:ix], total_node_counts[ix + 1:], 0)
            
            assert len(nodes) == b4 - 1
            assert len(errors_left) == b4 -1
            new_inputs.extend(new_nodes)
        
        a2, err_left, err_right = self.__prop_up__(nodes)
        """ Return new inputs, top_level activation, top-level input) """
        return (new_inputs, a2[0], nodes[0]) 
        
    
if __name__ == "__main__":
    
    import GwData
    import WordTokenizer
    from LatentWordVectors import LatentWordVectors
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    vector_size = 50
    target_code = "50"
    min_sentence_length = 3
    cut_off = 99999999
    activation_fn = "tanh"
    
    data = GwData.GwData()
    docs = data.documents[:cut_off]
    
    tokenized_docs = WordTokenizer.tokenize(docs, min_word_count=5, stem = True, remove_stop_words=True, spelling_correct=True)
    lsa_v = LatentWordVectors.LsaSpace(tokenized_docs, vector_size)
    
    
    def get_y(y):
        if activation_fn == "tanh":
            if y == 0:
                return [1.0,  -1.0]
            elif y == 1:
                return [-1.0,  1.0]
            else:
                raise Exception("Invalid y val: " + str(y))
        elif activation_fn == "sigmoid":
            if y == 0:
                return [1.0,  0.0]
            elif y == 1:
                return [0.0,  1.0]
            else:
                raise Exception("Invalid y val: " + str(y))
        else:
            raise Exception("Invalid activation fn: " + activation_fn)
    
    """ RAE inputs """
    ys = [get_y(y) for y in data.labels_for(target_code)[:cut_off]]
    
    """ Remove empty docs """
    tmp_d = []
    tmp_ys = []
    for i in range(len(tokenized_docs)):
        if len(tokenized_docs[i]) >= min_sentence_length:
            tmp_d.append(tokenized_docs[i])
            tmp_ys.append(ys[i])
    
    tokenized_docs = tmp_d
    ys = tmp_ys
    
    """ Init and Run """
    rae = RecursiveAutoEncoder(lsa_v, learning_rate = 0.1, activation_fn = activation_fn, supervised_wt = 0.8)
    """ Learning rate of 0.1, tanh and supervised wt of .08 seems best so far """
    
    print "Vector Size:         " + str(vector_size)
    print "Activation Function: " + activation_fn
    print "Learning Rate:       " + str(rae.learning_rate)
    print "Supervised Weight:   " + str(rae.supervised_wt)
    
    rae.train(tokenized_docs, ys, epochs = 200, batch_size = 100)
    pass

    """ TODO
            Check they are updating the same matrix
            1. Incorporate relative sizing information in the error calc
            2. Randomize the order of the training data on each epoch 
                - that means not storing leaf nodes for a mini-batch

            Ideas:
                Try training a new classifier (svm or RF) on the output as the NN is trained on all parses
                Try training the classifier portion against all codes not just one
                Add n grams and skip grams to LSA training
                Weight the NT nodes higher than the leaf nodes when training 
                    both networks
                Build a true deep RAE. Train one RAE (100 -> 75) on leaf nodes only
                Train a second layer (75 -> 50) to compute NT's also
                Place a higher weight on errors for positive examples
                Try different levels of contraction (e.g. 200 => 150 (4 words to 3) in one RAE)
                Try Dropout
                When creating a parse for the sentence, look at the predicted values at each step
                    in the tree construction. We could take a max over all NT nodes (for classification)
                    to determine the class, or pick the one with the biggest diff in the output nodes
                Try an initial prime (training iteration) on the leaf nodes, 
                    before starting the algorithm proper
                Try this semi-supervised approach to a BOW input (without all the parsing)
                Train a classifier on the joint output of this model and the vector compositional representation
                When normalizing the leaf nodes, divide by the log(doc freq) so frequent words have smaller vectors
                Train a multi-layered AE, training all but the top layer on the leaf-nodes initially, then learn in 
                    a layer-wise manner until the top layer is trained. Then continue trainin in a recursive manner
        
    """ 