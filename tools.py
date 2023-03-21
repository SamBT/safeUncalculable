import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
from sklearn.metrics import roc_auc_score, roc_curve
from multiprocessing import Process

def get_data(file):
    with h5py.File(file,"r") as f:
        z = f['jet1_constit_z'][()]
        eta = f['jet1_constit_eta'][()]
        phi = f['jet1_constit_phi'][()]

        pz = f['pjet1_constit_z'][()]
        peta = f['pjet1_constit_eta'][()]
        pphi = f['pjet1_constit_phi'][()]
        
    X1 = np.concatenate((z[:,:,np.newaxis],eta[:,:,np.newaxis],phi[:,:,np.newaxis]),axis=-1)
    Y1 = np.ones(z.shape[0])

    X2 = np.concatenate((pz[:,:,np.newaxis],peta[:,:,np.newaxis],pphi[:,:,np.newaxis]),axis=-1)
    Y2 = np.zeros(pz.shape[0])

    X = np.concatenate((X1,X2),axis=0)
    Y = np.concatenate((Y1,Y2),axis=0)
    
    del X1,Y1,X2,Y2
    
    return X,Y

def get_data_softDrop(file):
    with h5py.File(file,"r") as f:
        z = f['jet1_sd_constit_z'][()]
        eta = f['jet1_sd_constit_eta'][()]
        phi = f['jet1_sd_constit_phi'][()]

        pz = f['pjet1_sd_constit_z'][()]
        peta = f['pjet1_sd_constit_eta'][()]
        pphi = f['pjet1_sd_constit_phi'][()]
        
    X1 = np.concatenate((z[:,:,np.newaxis],eta[:,:,np.newaxis],phi[:,:,np.newaxis]),axis=-1)
    Y1 = np.ones(z.shape[0])

    X2 = np.concatenate((pz[:,:,np.newaxis],peta[:,:,np.newaxis],pphi[:,:,np.newaxis]),axis=-1)
    Y2 = np.zeros(pz.shape[0])

    X = np.concatenate((X1,X2),axis=0)
    Y = np.concatenate((Y1,Y2),axis=0)
    
    del X1,Y1,X2,Y2
    
    return X,Y