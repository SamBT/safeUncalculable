import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
from sklearn.metrics import roc_auc_score, roc_curve
import energyflow as ef
from energyflow.archs import EFN, PFN
from energyflow.utils import data_split, to_categorical
import tensorflow as tf
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

def train_efn(file,efn_kwargs,train_kwargs,plot=True,sd=False):
    if sd:
        X,Y = get_data_softDrop(file)
    else:
        X,Y = get_data(file)

    (z_train, z_val, z_test, 
     p_train, p_val, p_test,
     Y_train, Y_val, Y_test) = data_split(X[:,:,0], X[:,:,1:], Y, val=0.1, test=0.15, shuffle=True)

    efn = EFN(**efn_kwargs)
    
    hist = efn.fit([z_train, p_train], Y_train,
            validation_data=([z_val, p_val], Y_val),
           **train_kwargs)
    
    plt.figure(figsize=(8,6))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    preds = efn.predict([z_test, p_test], batch_size=1000)
    efn_fp, efn_tp, threshs = roc_curve(Y_test, preds[:,0])
    auc = roc_auc_score(Y_test, preds[:,0])
    print('EFN AUC:', auc)
    if plot:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['figure.autolayout'] = True

        plt.figure(figsize=(10,10))
        plt.plot(efn_tp, 1-efn_fp, '-', color='black', label='EFN')
    
    del X, Y, z_train, z_val, z_test, p_train, p_val, p_test, Y_train, Y_val, Y_test, preds
    
    return efn, auc, efn_fp, efn_tp, threshs

def train_pfn(file,pfn_kwargs,train_kwargs,plot=True,sd=False):
    if sd:
        X,Y = get_data_softDrop(file)
    else:
        X,Y = get_data(file)

    (X_train, X_val, X_test, Y_train, Y_val, Y_test) = data_split(X, Y, val=0.1, test=0.15, shuffle=True)

    pfn = PFN(**pfn_kwargs)
    
    hist = pfn.fit(X_train, Y_train, validation_data=(X_val, Y_val), **train_kwargs)
    
    plt.figure(figsize=(8,6))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    preds = pfn.predict(X_test, batch_size=1000)
    pfn_fp, pfn_tp, threshs = roc_curve(Y_test, preds[:,0])
    auc = roc_auc_score(Y_test, preds[:,0])
    print('PFN AUC:', auc)
    if plot:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['figure.autolayout'] = True

        plt.figure(figsize=(10,10))
        plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label='PFN')
    
    del X, X_train, X_val, X_test, Y, Y_train, Y_val, Y_test, preds
    
    return pfn, auc, pfn_fp, pfn_tp, threshs

def arch_scan(file,Phi_depth=3,F_depth=3):
    phis = [20,40,60,80,100,120,140]
    Fs = [40,60,80,100,120,140,160]
    aucs = np.zeros((len(phis),len(Fs)))
    for i,phi in enumerate(phis):
        for j,f in enumerate(Fs):
            efn, auc, fp, tp, threshs = train_efn(file,
                                             Phi_sizes=tuple(phi for _ in range(Phi_depth)),
                                             F_sizes=tuple(f for _ in range(F_depth)),
                                             verbose=0,
                                             summary=False,
                                             plot=False)
            aucs[i,j] = auc
    return aucs, phis, Fs