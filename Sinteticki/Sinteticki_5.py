# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:22:36 2019

@author: Andri
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
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
Var_2 = np.zeros([duzina,duzina]) 
Var_3 = np.zeros([duzina,duzina])
Var_4 = np.zeros([duzina,duzina])
Var_5 = np.zeros([duzina,duzina])

AUC_1 = np.zeros([duzina,duzina])
AUC_2 = np.zeros([duzina,duzina])
AUC_3 = np.zeros([duzina,duzina])
AUC_4 = np.zeros([duzina,duzina])
AUC_5 = np.zeros([duzina,duzina])

LogPRob_1 = np.zeros([duzina,duzina])
LogPRob_2 = np.zeros([duzina,duzina])
LogPRob_3 = np.zeros([duzina,duzina])
LogPRob_4 = np.zeros([duzina,duzina])
LogPRob_5 = np.zeros([duzina,duzina])


R = np.random.rand(1200,NoModelUN)*2-1

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
mod2 = GCRFC_fast()
mod3 = GCRFC_fast()
mod4 = GCRFC_fast()
mod5 = GCRFC_fast()
k1=0

for i in rang:
    k2=0
    for j in rang:
        mod1.alfa = np.array([i])
        mod1.beta = np.array([j])
        
        prob, Y, _ = mod1.predict(R,Se)
        
        Y_test = Y[Notrain:Noinst,:]
        Y_train = Y[:Notrain,:]
        
        
        mod1.fit(R_train, Se_train, Y_train, learn = 'TNC',maxiter = 500)  
        prob_1, Y_1, Var[k1,k2] = mod1.predict(R_test,Se_test)
        
        mod2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'KMeans', clus_no = 5)
        prob_2, Y_2, Var_2[k1,k2] = mod2.predict(R_test,Se_test)
        
        mod3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 5)
        prob_3, Y_3, Var_3[k1,k2] = mod3.predict(R_test,Se_test)
        
        mod4.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'GaussianMixture', clus_no = 5)
        prob_4, Y_4, Var_4[k1,k2] = mod4.predict(R_test,Se_test)
        
        mod5.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'GaussianMixtureProb', clus_no = 5)
        prob_5, Y_5, Var_5[k1,k2] = mod5.predict(R_test,Se_test)
        
        Prob_1 = prob_1.copy()
        Prob_1[Y_1==0] = 1 - Prob_1[Y_1==0]  
        Prob_2 = prob_2.copy()
        Prob_2[Y_2==0] = 1 - Prob_2[Y_2==0]  
        Prob_3 = prob_3.copy()
        Prob_3[Y_3==0] = 1 - Prob_3[Y_3==0]  
        Prob_4 = prob_4.copy()
        Prob_4[Y_4==0] = 1 - Prob_4[Y_4==0]  
        Prob_5 = prob_5.copy()
        Prob_5[Y_5==0] = 1 - Prob_5[Y_5==0]  
        
        
        Y_test1 = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
        
        probr_1 = prob_1.reshape([prob_1.shape[0]*prob_1.shape[1]])
        Probr_1 = Prob_1.reshape([Prob_1.shape[0]*Prob_1.shape[1]])
        probr_2 = prob_2.reshape([prob_2.shape[0]*prob_2.shape[1]])
        Probr_2 = Prob_2.reshape([Prob_2.shape[0]*Prob_2.shape[1]])
        probr_3 = prob_3.reshape([prob_3.shape[0]*prob_3.shape[1]])
        Probr_3 = Prob_3.reshape([Prob_3.shape[0]*Prob_3.shape[1]])
        probr_4 = prob_4.reshape([prob_4.shape[0]*prob_4.shape[1]])
        Probr_4 = Prob_4.reshape([Prob_4.shape[0]*Prob_4.shape[1]])
        probr_5 = prob_5.reshape([prob_5.shape[0]*prob_5.shape[1]])
        Probr_5 = Prob_5.reshape([Prob_5.shape[0]*Prob_5.shape[1]])
        
        AUC_1[k1,k2] = roc_auc_score(Y_test1,probr_1)
        AUC_2[k1,k2] = roc_auc_score(Y_test1,probr_2)
        AUC_3[k1,k2] = roc_auc_score(Y_test1,probr_3)
        AUC_4[k1,k2] = roc_auc_score(Y_test1,probr_4)
        AUC_5[k1,k2] = roc_auc_score(Y_test1,probr_5)
        LogPRob_1[k1,k2] = np.sum(np.log(Probr_1))
        LogPRob_2[k1,k2] = np.sum(np.log(Probr_2))
        LogPRob_3[k1,k2] = np.sum(np.log(Probr_3))
        LogPRob_4[k1,k2] = np.sum(np.log(Probr_4))
        LogPRob_5[k1,k2] = np.sum(np.log(Probr_5))
        
        
        k2+=1
    k1+=1
    
x_axis_labels = rang
y_axis_labels = rang

heat_map_1 = (AUC_1 - AUC_2)
heat_map_4 = (AUC_1 - AUC_3)
heat_map_5 = (AUC_1 - AUC_4)
heat_map_6 = (AUC_1 - AUC_5)

heat_map_11 = (LogPRob_1 - LogPRob_2)
heat_map_41 = (LogPRob_1 - LogPRob_3)
heat_map_51 = (LogPRob_1 - LogPRob_4)
heat_map_61 = (LogPRob_1 - LogPRob_5)

figure1 = plt.figure(figsize=(8, 8))
ax11 = plt.subplot(211) 
sns.heatmap(heat_map_1, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax11.set_ylabel(r'$\alpha$')
ax11.set_title("a)")
    
ax12 = plt.subplot(212) 
sns.heatmap(heat_map_11, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax12.set_ylabel(r'$\alpha$')
ax12.set_xlabel(r'$\beta$')
ax12.set_title('b)')
    
figure4 = plt.figure(figsize=(8, 8))
ax11 = plt.subplot(211) 
sns.heatmap(heat_map_4, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax11.set_ylabel(r'$\alpha$')
ax11.set_title("a)")
    
ax12 = plt.subplot(212) 
sns.heatmap(heat_map_41, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax12.set_ylabel(r'$\alpha$')
ax12.set_xlabel(r'$\beta$')
ax12.set_title('b)')

figure5 = plt.figure(figsize=(8, 8))
ax11 = plt.subplot(211) 
sns.heatmap(heat_map_5, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax11.set_ylabel(r'$\alpha$')
ax11.set_title("a)")
    
ax12 = plt.subplot(212) 
sns.heatmap(heat_map_51, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax12.set_ylabel(r'$\alpha$')
ax12.set_xlabel(r'$\beta$')
ax12.set_title('b)')
    
figure6 = plt.figure(figsize=(8, 8))
ax11 = plt.subplot(211) 
sns.heatmap(heat_map_6, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax11.set_ylabel(r'$\alpha$')
ax11.set_title("a)")
    
ax12 = plt.subplot(212) 
sns.heatmap(heat_map_61, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax12.set_ylabel(r'$\alpha$')
ax12.set_xlabel(r'$\beta$')
ax12.set_title('b)')