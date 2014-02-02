from functools import partial
import sparse_autoencoder
import scipy.optimize

if __name__ == '__main__':
    
    import GwData
    data = GwData.GwData.as_binary()
    # Network Architecture 
    visible_size = data.shape[1]
    hidden_size = 300
    
    # Training params
    weight_decay = 3e-3
    sparsity_param = 0.1
    beta = 3
    max_iter = 500            # Maximum number of iterations of L-BFGS to run 

    # Get the data
    num_samples = data.shape[0]
    

    # set up L-BFGS args
    theta = sparse_autoencoder.initialize_params(hidden_size, visible_size)
    sae_cost = partial(sparse_autoencoder.cost,
                        visible_size=visible_size, 
                        hidden_size=hidden_size,
                        weight_decay = weight_decay,
                        beta=beta,
                        sparsity_param=sparsity_param,
                        data=data.T)

    # Train!
    trained, cost, d = scipy.optimize.lbfgsb.fmin_l_bfgs_b(sae_cost, 
                                                           theta, 
                                                           maxfun=max_iter, 
                                                           m=100,
                                                           factr=1.0,
                                                           pgtol=1e-100,
                                                           iprint=1)
    # Save the trained weights
    W1, W2, b1, b2 = sparse_autoencoder.unflatten_params(trained, hidden_size, visible_size)
    