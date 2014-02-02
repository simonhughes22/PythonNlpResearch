import neurolab as nl
from neurolab import train
import GwData

if __name__ == '__main__':

    data = GwData.GwData.as_binary()
    
    cols = data.shape[1]
    layers = [300, cols]
    minmax = [[0,1] for i in range(0,cols)]
    
    n = nl.net.newff(minmax, layers, transf = [nl.trans.LogSig()] * len(layers) )
    n.trainf = train.train_cg
    #n.errorf = nl.error.MSE()
    
    error = n.train(data, data, epochs=500, show=1, goal=0.02)
    
    pass