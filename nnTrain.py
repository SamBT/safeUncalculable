import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
from sklearn.metrics import roc_auc_score, roc_curve
import energyflow as ef
from energyflow.archs import EFN, PFN
from energyflow.utils import data_split, to_categorical
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from multiprocessing import Process
import os

def get_data(jtype,hlevel=True,efrac=False):
    base = "/uscms/home/sbrightt/nobackup/jets-ml/datasets/safeIncalculable/"
    gjets = [base+f for f in os.listdir(base) if "h2gg_set" in f and ".h5" in f]
    qjets = [base+f for f in os.listdir(base) if "h2qq_set" in f and ".h5" in f]
    files = qjets if jtype=='q' else gjets
    
    z = []
    eta = []
    phi = []
    
    for file in files:
        with h5py.File(file,"r") as f:
            if hlevel:
                z.append(f['jet1_constit_z'][()])
                eta.append(f['jet1_constit_eta'][()])
                phi.append(f['jet1_constit_phi'][()])
            else:
                z.append(f['pjet1_constit_z'][()])
                eta.append(f['pjet1_constit_eta'][()])
                phi.append(f['pjet1_constit_phi'][()])
    z = np.concatenate(z,axis=0)
    eta = np.concatenate(eta,axis=0)
    phi = np.concatenate(phi,axis=0)
        
    output = np.concatenate((z[:,:,np.newaxis],eta[:,:,np.newaxis],phi[:,:,np.newaxis]),axis=-1)
    del z,eta,phi
    return output

def get_vars(jtype,vars,hlevel=True,efrac=False):
    base = "/uscms/home/sbrightt/nobackup/jets-ml/datasets/safeIncalculable/"
    gjets = [base+f for f in os.listdir(base) if "h2gg_set" in f and ".h5" in f]
    qjets = [base+f for f in os.listdir(base) if "h2qq_set" in f and ".h5" in f]
    files = qjets if jtype=='q' else gjets
    
    output = [[] for v in vars]
    for file in files:
        with h5py.File(file,"r") as f:
            if hlevel:
                for i,v in enumerate(vars):
                    output[i].append(f[f"jet1_{v}"][()])
            else:
                for i,v in enumerate(vars):
                    output[i].append(f[f"pjet1_{v}"][()])
    output = np.concatenate([np.concatenate(k,axis=0).reshape(-1,1) for k in output],axis=-1)
        
    return output

def get_constit_vars(jtype,vars,hlevel=True,efrac=False):
    base = "/uscms/home/sbrightt/nobackup/jets-ml/datasets/safeIncalculable/"
    gjets = [base+f for f in os.listdir(base) if "h2gg_set" in f and ".h5" in f]
    qjets = [base+f for f in os.listdir(base) if "h2qq_set" in f and ".h5" in f]
    files = qjets if jtype=='q' else gjets
    
    output = [[] for v in vars]
    for file in files:
        with h5py.File(file,"r") as f:
            if hlevel:
                for i,v in enumerate(vars):
                    output[i].append(f[f"jet1_constit_{v}"][()])
            else:
                for i,v in enumerate(vars):
                    output[i].append(f[f"pjet1_constit_{v}"][()])
    output = np.concatenate([np.concatenate(k,axis=0)[:,:,np.newaxis] for k in output],axis=-1)
    return output

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

def train_efn(train,test,val,efn_kwargs,train_kwargs,plot=True):
    X_train, Y_train = train
    X_test, Y_test = test
    X_val, Y_val = val
    
    z_train, z_test, z_val = X_train[:,:,0], X_test[:,:,0], X_val[:,:,0]
    p_train, p_test, p_val = X_train[:,:,1:], X_test[:,:,1:], X_val[:,:,1:]

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
    bce_loss = BinaryCrossentropy(from_logits=False)
    test_loss = bce_loss(Y_test,preds).numpy()
    print(f"Test Loss : {test_loss:.5f}")
    efn_fp, efn_tp, threshs = roc_curve(Y_test, preds[:,0])
    auc = roc_auc_score(Y_test, preds[:,0])
    print('EFN AUC:', auc)
    if plot:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['figure.autolayout'] = True

        plt.figure(figsize=(10,10))
        plt.plot(efn_tp, 1-efn_fp, '-', color='black', label='EFN')
    
    del X_train,X_test,X_val,Y_train,Y_test,Y_val
    del z_train, z_val, z_test, p_train, p_val, p_test, preds
    del train, test, val
    
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