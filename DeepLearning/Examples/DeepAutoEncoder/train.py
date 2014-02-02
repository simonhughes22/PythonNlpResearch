from SDA_Layers import *
import numpy as np

if __name__ == "__main__":
    import GwData
    data = GwData.GwData.as_binary()
    fullData = GwData.GwData()
    y = np.asarray([[l] for l in fullData.labels_for("50")])
    
    autoencoder = StackedDA([300], alpha=0.1)
    autoencoder.pre_train(data, 50)
    autoencoder.finalLayer(y, 10, 1)
    autoencoder.fine_tune(data, y, 50)
    pass