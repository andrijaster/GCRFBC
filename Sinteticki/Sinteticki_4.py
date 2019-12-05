# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:29:06 2019

@author: Andrija Master
"""

import numpy as np
import time
from sklearn.metrics import roc_auc_score
from GCRFCNB import GCRFCNB
from GCRFC import GCRFC
from GCRFC_fast import GCRFC_fast
import matplotlib.pyplot as plt



NoModelUN = 2
NoGraph = 2
NodeNo = np.array([2, 5, 10, 20, 50])
Nopoint = 10000
duzina = len(NodeNo)


time_1 = np.zeros([duzina])
time_21 = np.zeros([duzina])
time_22 = np.zeros([duzina])
time_23 = np.zeros([duzina])
time_3 = np.zeros([duzina])
time_4 = np.zeros([duzina])
time_5 = np.zeros([duzina])
time_6 = np.zeros([duzina])


k=0
for i in NodeNo:

    R = np.random.rand(Nopoint,NoModelUN)*2-1
    
    
    Nopoint = R.shape[0]
    Noinst = np.round(Nopoint/i).astype(int)
    i1 = np.arange(i)
    i2 = np.arange(Noinst)
    
    a = np.random.rand(Nopoint,2)
    
    
    Se = np.random.rand(Noinst,NoGraph,i,i)
    
    Notrain = (Noinst*0.8).astype(int)
    Notest = (Noinst*0.2).astype(int)
    
    Se_train = Se[:Notrain,:,:,:]
    R_train = R[:Notrain*i,:]
    R_test = R[Notrain*i:Noinst*i,:]
    Se_test = Se[Notrain:Noinst,:,:,:]
    
    mod1 = GCRFC()
    mod21 = GCRFC_fast()
    mod22 = GCRFC_fast()
    mod23 = GCRFC_fast()
    mod3 = GCRFC_fast()
    mod4 = GCRFC_fast()
    mod5 = GCRFC_fast()
    mod6 = GCRFCNB()
    
    
    mod1.alfa = np.array([3,4])
    mod1.beta = np.array([4,5])
    
    prob, Y, _ = mod1.predict(R,Se)
    
    Y_test = Y[Notrain:Noinst,:]
    Y_train = Y[:Notrain,:]
    
    
    start_time = time.time()
    mod1.fit(R_train, Se_train, Y_train, learn = 'TNC',maxiter = 500)  
    time_1[k] = time.time() - start_time 
    
    start_time = time.time()
    mod21.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 2)
    time_21[k] = time.time() - start_time        
    
    start_time = time.time()
    mod23.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 5)
    time_23[k] = time.time() - start_time 
    
    start_time = time.time()
    mod3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 50)
    time_3[k] = time.time() - start_time   
    
    start_time = time.time()
    mod4.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 100)
    time_4[k] = time.time() - start_time 
    
    start_time = time.time()
    mod5.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 250)
    time_5[k] = time.time() - start_time 
    
    start_time = time.time()
    mod6.fit(R_train, Se_train, Y_train, learn = 'TNC', maxiter = 500)  
    time_6[k] = time.time() - start_time 
    
    
label = ['GCRFBCb', 'GCRFBCnb', 'GCRFBCb-fast 2', 'GCRFBCb-fast 5', 'GCRFBCb-fast 50', 'GCRFBCb-fast 100', 'GCRFBCb-fast 250']
    
plt.plot(NodeNo, time_1)
plt.plot(NodeNo, time_6)
plt.plot(NodeNo, time_21)
plt.plot(NodeNo, time_23)
plt.plot(NodeNo, time_3)
plt.plot(NodeNo, time_4)
plt.plot(NodeNo, time_5)

plt.xlabel('Node No.')
plt.ylabel('Time [sec]')
plt.legend(label, loc='upper left')
plt.grid(b=True)

plt.savefig('Synthetic_4.pdf')
plt.savefig('Synthetic_4.png')


