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
Var_21 = np.zeros([duzina,duzina])
Var_22 = np.zeros([duzina,duzina])
Var_23 = np.zeros([duzina,duzina])
Var_3 = np.zeros([duzina,duzina])
Var_4 = np.zeros([duzina,duzina])
Var_5 = np.zeros([duzina,duzina])
time_1 = np.zeros([duzina,duzina])
time_21 = np.zeros([duzina,duzina])
time_22 = np.zeros([duzina,duzina])
time_23 = np.zeros([duzina,duzina])
time_3 = np.zeros([duzina,duzina])
time_4 = np.zeros([duzina,duzina])
time_5 = np.zeros([duzina,duzina])
AUC_1 = np.zeros([duzina,duzina])
AUC_21 = np.zeros([duzina,duzina])
AUC_22 = np.zeros([duzina,duzina])
AUC_23 = np.zeros([duzina,duzina])
AUC_3 = np.zeros([duzina,duzina])
AUC_4 = np.zeros([duzina,duzina])
AUC_5 = np.zeros([duzina,duzina])
LogPRob_1 = np.zeros([duzina,duzina])
LogPRob_21 = np.zeros([duzina,duzina])
LogPRob_22 = np.zeros([duzina,duzina])
LogPRob_23 = np.zeros([duzina,duzina])
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
mod21 = GCRFC_fast()
mod22 = GCRFC_fast()
mod23 = GCRFC_fast()
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
        
        
        start_time = time.time()
        mod1.fit(R_train, Se_train, Y_train, learn = 'TNC',maxiter = 500)  
        prob_1, Y_1, Var[k1,k2] = mod1.predict(R_test,Se_test)
        time_1[k1,k2] = start_time - time.time()
        
        mod21.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 2)
        prob_21, Y_21, Var_21[k1,k2] = mod21.predict(R_test,Se_test)
        time_21[k1,k2] = start_time - time.time()       
        
        mod22.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 3)
        prob_22, Y_22, Var_22[k1,k2] = mod22.predict(R_test,Se_test)
        time_22[k1,k2] = start_time - time.time()  
        
        mod23.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 4)
        prob_23, Y_23, Var_23[k1,k2] = mod23.predict(R_test,Se_test)
        time_23[k1,k2] = start_time - time.time()     
        
        mod3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 50)
        prob_3, Y_3, Var_3[k1,k2] = mod3.predict(R_test,Se_test)
        time_3[k1,k2] = start_time - time.time()       
        
        mod4.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 100)
        prob_4, Y_4, Var_4[k1,k2] = mod4.predict(R_test,Se_test)
        time_4[k1,k2] = start_time - time.time()    
        
        mod5.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = 500,method_clus = 'MiniBatchKMeans', clus_no = 250)
        prob_5, Y_5, Var_5[k1,k2] = mod5.predict(R_test,Se_test)
        time_5[k1,k2] = start_time - time.time()    
        
        Prob_1 = prob_1.copy()
        Prob_1[Y_1==0] = 1 - Prob_1[Y_1==0]  
        Prob_21 = prob_21.copy()
        Prob_21[Y_21==0] = 1 - Prob_21[Y_21==0]  
        Prob_22 = prob_22.copy()
        Prob_22[Y_22==0] = 1 - Prob_22[Y_22==0]  
        Prob_23 = prob_23.copy()
        Prob_23[Y_23==0] = 1 - Prob_23[Y_23==0]  
        Prob_3 = prob_3.copy()
        Prob_3[Y_3==0] = 1 - Prob_3[Y_3==0]  
        Prob_4 = prob_4.copy()
        Prob_4[Y_4==0] = 1 - Prob_4[Y_4==0]  
        Prob_5 = prob_5.copy()
        Prob_5[Y_5==0] = 1 - Prob_5[Y_5==0]  
        
        
        Y_test1 = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
        
        probr_1 = prob_1.reshape([prob_1.shape[0]*prob_1.shape[1]])
        Probr_1 = Prob_1.reshape([Prob_1.shape[0]*Prob_1.shape[1]])
        probr_21 = prob_21.reshape([prob_21.shape[0]*prob_21.shape[1]])
        Probr_21 = Prob_21.reshape([Prob_21.shape[0]*Prob_21.shape[1]])
        probr_22 = prob_22.reshape([prob_22.shape[0]*prob_22.shape[1]])
        Probr_22 = Prob_22.reshape([Prob_22.shape[0]*Prob_22.shape[1]])
        probr_23 = prob_23.reshape([prob_23.shape[0]*prob_23.shape[1]])
        Probr_23 = Prob_23.reshape([Prob_23.shape[0]*Prob_23.shape[1]])
        probr_3 = prob_3.reshape([prob_3.shape[0]*prob_3.shape[1]])
        Probr_3 = Prob_3.reshape([Prob_3.shape[0]*Prob_3.shape[1]])
        probr_4 = prob_4.reshape([prob_4.shape[0]*prob_4.shape[1]])
        Probr_4 = Prob_4.reshape([Prob_4.shape[0]*Prob_4.shape[1]])
        probr_5 = prob_5.reshape([prob_5.shape[0]*prob_5.shape[1]])
        Probr_5 = Prob_5.reshape([Prob_5.shape[0]*Prob_5.shape[1]])
        
        AUC_1[k1,k2] = roc_auc_score(Y_test1,probr_1)
        AUC_21[k1,k2] = roc_auc_score(Y_test1,probr_21)
        AUC_22[k1,k2] = roc_auc_score(Y_test1,probr_22)
        AUC_23[k1,k2] = roc_auc_score(Y_test1,probr_23)
        AUC_3[k1,k2] = roc_auc_score(Y_test1,probr_3)
        AUC_4[k1,k2] = roc_auc_score(Y_test1,probr_4)
        AUC_5[k1,k2] = roc_auc_score(Y_test1,probr_5)
        LogPRob_1[k1,k2] = np.sum(np.log(Probr_1))
        LogPRob_21[k1,k2] = np.sum(np.log(Probr_21))
        LogPRob_22[k1,k2] = np.sum(np.log(Probr_22))
        LogPRob_23[k1,k2] = np.sum(np.log(Probr_23))
        LogPRob_3[k1,k2] = np.sum(np.log(Probr_3))
        LogPRob_4[k1,k2] = np.sum(np.log(Probr_4))
        LogPRob_5[k1,k2] = np.sum(np.log(Probr_5))
        
        
        k2+=1
    k1+=1

x_axis_labels = rang
y_axis_labels = rang

heat_map_1 = (AUC_1 - AUC_21)
heat_map_2 = (AUC_1 - AUC_22)
heat_map_3 = (AUC_1 - AUC_23)
heat_map_4 = (AUC_1 - AUC_3)
heat_map_5 = (AUC_1 - AUC_4)
heat_map_6 = (AUC_1 - AUC_5)

heat_map_11 = (LogPRob_1 - LogPRob_21)
heat_map_21 = (LogPRob_1 - LogPRob_22)
heat_map_31 = (LogPRob_1 - LogPRob_23)
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

figure2 = plt.figure(figsize=(8, 8))
ax11 = plt.subplot(211) 
sns.heatmap(heat_map_2, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax11.set_ylabel(r'$\alpha$')
ax11.set_title("a)")
    
ax12 = plt.subplot(212) 
sns.heatmap(heat_map_21, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax12.set_ylabel(r'$\alpha$')
ax12.set_xlabel(r'$\beta$')
ax12.set_title('b)')

figure3 = plt.figure(figsize=(8, 8))
ax11 = plt.subplot(211) 
sns.heatmap(heat_map_3, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
ax11.set_ylabel(r'$\alpha$')
ax11.set_title("a)")
    
ax12 = plt.subplot(212) 
sns.heatmap(heat_map_31, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
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