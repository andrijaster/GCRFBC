# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:29:06 2019

@author: Andrija Master
"""

import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from GCRFCNB import GCRFCNB
from GCRFC import GCRFC
from GCRFC_fast import GCRFC_fast



NoModelUN = 1
NodeNo = 10
NoGraph = 1
rang = np.arange(1,31,3)
duzina = len(rang)

Var = np.zeros([duzina,duzina])
Var_3 = np.zeros([duzina,duzina])
time_1 = np.zeros([duzina,duzina])
time_2 = np.zeros([duzina,duzina])
time_3 = np.zeros([duzina,duzina])
AUC_1 = np.zeros([duzina,duzina])
AUC_2 = np.zeros([duzina,duzina])
AUC_3 = np.zeros([duzina,duzina])
LogPRob_1 = np.zeros([duzina,duzina])
LogPRob_2 = np.zeros([duzina,duzina])
LogPRob_3 = np.zeros([duzina,duzina])



R = np.random.rand(1000,NoModelUN)*2-1

Nopoint = R.shape[0]
Noinst = np.round(Nopoint/NodeNo).astype(int)
i1 = np.arange(NodeNo)
i2 = np.arange(Noinst)

a = np.random.rand(Nopoint,2)


Se = np.random.rand(Noinst,NoGraph,NodeNo,NodeNo)

Notrain = (Noinst*0.8).astype(int)
Notest = (Noinst*0.2).astype(int)

Se_train = Se[:Notrain,:,:,:]
R_train = R[:Notrain*NodeNo,:]
R_test = R[Notrain*NodeNo:Noinst*NodeNo,:]
Se_test = Se[Notrain:Noinst,:,:,:]

mod1 = GCRFC()
mod2 = GCRFCNB()
mod3 = GCRFC_fast()
k1=0

for i in rang:
    k2=0
    for j in rang:
        mod1.alfa = np.array([i])
        mod1.beta = np.array([j])
        
        prob, Y, _ = mod1.predict(R,Se)
        
        Y_test = Y[Notrain:Noinst,:]
        Y_train = Y[:Notrain,:]
        
        
        start_time = time.time()
        mod1.fit(R_train, Se_train, Y_train, learn = 'TNC',maxiter = 500)  
        prob_1, Y_1, Var[k1,k2] = mod1.predict(R_test,Se_test)
        time_1[k1,k2] = start_time - time.time()
        
        mod2.fit(R_train, Se_train, Y_train, learn = 'TNC',maxiter = 500)  
        prob_2, Y_2 = mod2.predict(R_test,Se_test)
        time_2[k1,k2] = start_time - time.time()
        
        mod3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 5)
        prob_3, Y_3, Var_3[k1,k2] = mod3.predict(R_test,Se_test)
        time_3[k1,k2] = start_time - time.time()       
        
        Prob_1 = prob_1.copy()
        Prob_1[Y_1==0] = 1 - Prob_1[Y_1==0]  
        Prob_2 = prob_2.copy()
        Prob_2[Y_2==0] = 1 - Prob_2[Y_2==0]  
        Prob_3 = prob_3.copy()
        Prob_3[Y_3==0] = 1 - Prob_3[Y_3==0]  
        
        
        Y_test1 = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
        
        probr_1 = prob_1.reshape([prob_1.shape[0]*prob_1.shape[1]])
        Probr_1 = Prob_1.reshape([Prob_1.shape[0]*Prob_1.shape[1]])
        probr_2 = prob_2.reshape([prob_2.shape[0]*prob_2.shape[1]])
        Probr_2 = Prob_2.reshape([Prob_2.shape[0]*Prob_2.shape[1]])
        probr_3 = prob_3.reshape([prob_3.shape[0]*prob_3.shape[1]])
        Probr_3 = Prob_3.reshape([Prob_3.shape[0]*Prob_3.shape[1]])
        
        AUC_1[k1,k2] = roc_auc_score(Y_test1,probr_1)
        AUC_2[k1,k2] = roc_auc_score(Y_test1,probr_2)
        AUC_3[k1,k2] = roc_auc_score(Y_test1,probr_3)
        LogPRob_1[k1,k2] = np.sum(np.log(Probr_1))
        LogPRob_2[k1,k2] = np.sum(np.log(Probr_2))
        LogPRob_3[k1,k2] = np.sum(np.log(Probr_3))
        
        
        k2+=1
    k1+=1

x_axis_labels = rang
y_axis_labels = rang

hmap_10 = np.abs(AUC_1-AUC_2)
hmap_11 = np.abs(LogPRob_1-LogPRob_2)
hmap_20 = np.abs(AUC_1-AUC_3)
hmap_21 = np.abs(LogPRob_1-LogPRob_3)
    
figure1 = plt.figure(figsize=(8, 15))
ax1 = plt.subplot(311)
sns.heatmap(hmap_10, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax1.set_ylabel(r'$\alpha$')
#ax1.set_xlabel(r'$\beta$')
ax1.set_title('a)')

ax2 = plt.subplot(312)
sns.heatmap(hmap_11, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax2.set_ylabel(r'$\alpha$')
#ax2.set_xlabel('$\beta$')
ax2.set_title('b)')

ax3 = plt.subplot(313)
sns.heatmap(Var, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax3.set_ylabel(r'$\alpha$')
#ax3.set_xlabel(r'$\beta$')
ax3.set_title('c)')

figure2 = plt.figure(figsize=(8, 8))
ax11 = plt.subplot(211) 
sns.heatmap(hmap_20, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax11.set_ylabel(r'$\alpha$')
#ax11.set_xlabel('$\beta$')
ax11.set_title("a)")
    
ax12 = plt.subplot(212) 
sns.heatmap(hmap_21, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax12.set_ylabel(r'$\alpha$')
ax12.set_xlabel(r'$\beta$')
ax12.set_title('b)')


