import rbm
import numpy as np

class DBN(object):
    
    def __init__(self, layers, learning_rate):
        assert(len(layers) == len(learning_rate) + 1)
        
        self.rbms = []
        for i,_ in enumerate(layers[:-1]):
            r = rbm.RBM(layers[i], layers[i+1], learning_rate[i] )
            self.rbms.append(r)
    
        def arr_to_binary(num):
            if num >= 0.5:
                return 1.0
            return 0.0
        
        v = np.vectorize(arr_to_binary) 
        def to_b(self, data):
            return v(data)
        
        self.to_binary = v
    
    def train(self, data, epochs):
        assert(len(epochs) == len(self.rbms))
        
        output = data
        for i,r in enumerate(self.rbms):
            print "Training Layer: " + str(i + 1)
            r.train(output, max_epochs = epochs[i])
            output = self.run_visible(output, r, 10)
            print ("-" * 20)
    
    def run_visible(self, data, r, iterations = 10):
        output = data
        
        activations = np.asarray(r.run_visible(output))

        for j in range(iterations - 1):
            activations += np.asarray(r.run_visible(data))
    
        mean_activations = activations / iterations
        output = self.to_binary(mean_activations)
        return output
    
    def run_hidden(self, data, r, iterations = 10):
        output = data
        
        activations = np.asarray(r.run_hidden(output))

        for j in range(iterations - 1):
            activations += np.asarray(r.run_hidden(data))
    
        mean_activations = activations / iterations
        output = self.to_binary(mean_activations)
        return output
    
    def get_activations(self, data, iterations = 1):
        
        output = data
        for r in self.rbms:
            output  = self.run_visible(output, r, iterations)            
        return output

    def echo_inputs(self, data, iterations = 1):
        
        output = self.get_activations(data, iterations)
        
        copy = self.rbms[:]
        copy.reverse()
        
        for r in copy:
            output  = self.run_hidden(output, r, iterations)            
        return output

if __name__ == '__main__':
    
    import GwData
    data = GwData.GwData.as_binary()
   
    layers = [data.shape[1], 600, 300]
    learning_rate = [1, 1]
    epochs = [2000, 2000]
    
    net = DBN(layers, learning_rate)
    net.train(data, epochs)
    
    echoes = net.echo_inputs(data, 10)
    errs = data - echoes
    
    num_e = []
    for err in errs:
        c = len([e for e in err if e != 0])
        num_e.append(c)
    
    avg_err = ((sum(num_e) / (1.0 * len(num_e))) / data.shape[1]) * 100
    
    print "Mean error: " + str(avg_err) + "%"    
    pass