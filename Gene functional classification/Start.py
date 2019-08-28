# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:20:07 2018

@author: Andrija Master
"""

""" Packages"""
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from Nestrukturni import Nestrukturni_fun
from Struktura import Struktura_fun 
from GCRFCNB import GCRFCNB
from GCRFC import GCRFC
from GCRFC_fast import GCRFC_fast
from Yeast_dataset import output, atribute



""" Initialization """
No_class = 14
NoGraph = 4
ModelUNNo = 4
testsize2 = 0.2
broj_fold = 10
iteracija = 400

output = output.iloc[:,:No_class]

AUCNB = np.zeros(broj_fold)
AUCB = np.zeros(broj_fold)
AUCBF = np.zeros(broj_fold)
AUCBF2 = np.zeros(broj_fold)
AUCBF21 = np.zeros(broj_fold)
AUCBF22 = np.zeros(broj_fold)
AUCBF3 = np.zeros(broj_fold)
AUCBF4 = np.zeros(broj_fold)
AUCBF41 = np.zeros(broj_fold)
AUCBF42 = np.zeros(broj_fold)
AUCBF5 = np.zeros(broj_fold)
AUCBF6 = np.zeros(broj_fold)
AUCBF61 = np.zeros(broj_fold)
AUCBF62 = np.zeros(broj_fold)
AUCBF7 = np.zeros(broj_fold)
AUCBF8 = np.zeros(broj_fold)
AUCBF81 = np.zeros(broj_fold)
AUCBF82 = np.zeros(broj_fold)

logProbNB = np.zeros(broj_fold)
logProbB = np.zeros(broj_fold)
logProbBF = np.zeros(broj_fold)
logProbBF2 = np.zeros(broj_fold)
logProbBF21 = np.zeros(broj_fold)
logProbBF22 = np.zeros(broj_fold)
logProbBF3 = np.zeros(broj_fold)
logProbBF4 = np.zeros(broj_fold)
logProbBF41 = np.zeros(broj_fold)
logProbBF42 = np.zeros(broj_fold)
logProbBF5 = np.zeros(broj_fold)
logProbBF6 = np.zeros(broj_fold)
logProbBF61 = np.zeros(broj_fold)
logProbBF62 = np.zeros(broj_fold)
logProbBF7 = np.zeros(broj_fold)
logProbBF8 = np.zeros(broj_fold)
logProbBF81 = np.zeros(broj_fold)
logProbBF82 = np.zeros(broj_fold)

timeNB = np.zeros(broj_fold)
timeB = np.zeros(broj_fold)
timeBF = np.zeros(broj_fold)
timeBF2 = np.zeros(broj_fold)
timeBF21 = np.zeros(broj_fold)
timeBF22 = np.zeros(broj_fold)
timeBF3 = np.zeros(broj_fold)
timeBF4 = np.zeros(broj_fold)
timeBF41 = np.zeros(broj_fold)
timeBF42 = np.zeros(broj_fold)
timeBF5 = np.zeros(broj_fold)
timeBF6 = np.zeros(broj_fold)
timeBF61 = np.zeros(broj_fold)
timeBF62 = np.zeros(broj_fold)
timeBF7 = np.zeros(broj_fold)
timeBF8 = np.zeros(broj_fold)
timeBF81 = np.zeros(broj_fold)
timeBF82 = np.zeros(broj_fold)

Skor_com_AUC = np.zeros([broj_fold,ModelUNNo])
Skor_com_AUC2 = np.zeros([broj_fold,ModelUNNo])

skf = KFold(n_splits = broj_fold)
skf.get_n_splits(atribute, output)
i = 0

for train_index,test_index in skf.split(atribute, output):
    x_train_com, x_test = atribute.iloc[train_index,:], atribute.iloc[test_index,:]
    y_train_com, Y_test = output.iloc[train_index,:], output.iloc[test_index,:]
    provera = Y_test[Y_test==1].any().all()
    print(provera)

file = open("rezultatiYEAST.txt","w")

for train_index,test_index in skf.split(atribute, output):
    
    x_train_com, x_test = atribute.iloc[train_index,:], atribute.iloc[test_index,:]
    y_train_com, Y_test = output.iloc[train_index,:], output.iloc[test_index,:] 
    x_train_un, x_train_st, y_train_un, Y_train = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)

    Skor_com_AUC[i,:], Skor_com_AUC2[i,:], R_train, R_test, R2, Noinst_train, Noinst_test = Nestrukturni_fun(x_train_un, y_train_un, x_train_st, Y_train, x_test, Y_test, No_class)
    Se_train, Se_test = Struktura_fun(No_class,NoGraph, R2 , y_train_com, Noinst_train, Noinst_test)
    
    
    """ Model GCRFC """
    Y_train = Y_train.values
    Y_test = Y_test.values 
    
    start_time = time.time()
    mod1 = GCRFCNB()
    mod1.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 6e-4, maxiter = iteracija)  
    probNB, YNB = mod1.predict(R_test,Se_test)
    timeNB[i] = time.time() - start_time
    
    
    start_time = time.time()
    mod2 = GCRFC()
    mod2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija)  
    probB, YB, VarB = mod2.predict(R_test,Se_test)
    timeB[i] = time.time() - start_time 
    
    
    start_time = time.time()
    mod3 = GCRFC_fast()
    mod3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'KMeans', clus_no = 5)  
    probBF, YBF, VarBF = mod3.predict(R_test,Se_test)  
    timeBF[i] = time.time() - start_time
    
    start_time = time.time()
    mod4 = GCRFC_fast()
    mod4.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'KMeans', clus_no = 50)  
    probBF2, YBF2, VarBF2 = mod4.predict(R_test,Se_test)  
    timeBF2[i] = time.time() - start_time
    
    start_time = time.time()
    mod41 = GCRFC_fast()
    mod41.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'KMeans', clus_no = 150)  
    probBF21, YBF21, VarBF21 = mod41.predict(R_test,Se_test)  
    timeBF21[i] = time.time() - start_time
    
    start_time = time.time()
    mod42 = GCRFC_fast()
    mod42.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'KMeans', clus_no = 250)  
    probBF22, YBF22, VarBF22 = mod42.predict(R_test,Se_test)  
    timeBF22[i] = time.time() - start_time
    
    start_time = time.time()
    mod5 = GCRFC_fast()
    mod5.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'MiniBatchKMeans', clus_no = 5)  
    probBF3, YBF3, VarBF3 = mod5.predict(R_test,Se_test)  
    timeBF3[i] = time.time() - start_time
    
    start_time = time.time()
    mod6 = GCRFC_fast()
    mod6.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'MiniBatchKMeans', clus_no = 50)  
    probBF4, YBF4, VarBF4 = mod6.predict(R_test,Se_test)  
    timeBF4[i] = time.time() - start_time
    
    start_time = time.time()
    mod61 = GCRFC_fast()
    mod61.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'MiniBatchKMeans', clus_no = 150)  
    probBF41, YBF41, VarBF41 = mod61.predict(R_test,Se_test)  
    timeBF41[i] = time.time() - start_time
    
    start_time = time.time()
    mod62 = GCRFC_fast()
    mod62.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'MiniBatchKMeans', clus_no = 250)  
    probBF42, YBF42, VarBF42 = mod62.predict(R_test,Se_test)  
    timeBF42[i] = time.time() - start_time
    
    start_time = time.time()
    mod7 = GCRFC_fast()
    mod7.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixture', clus_no = 5)  
    probBF5, YBF5, VarBF5 = mod7.predict(R_test,Se_test)  
    timeBF5[i] = time.time() - start_time
    
    start_time = time.time()
    mod8 = GCRFC_fast()
    mod8.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixture', clus_no = 50)  
    probBF6, YBF6, VarBF6 = mod8.predict(R_test,Se_test)  
    timeBF6[i] = time.time() - start_time
    
    start_time = time.time()
    mod81 = GCRFC_fast()
    mod81.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixture', clus_no = 150)  
    probBF61, YBF61, VarBF61 = mod81.predict(R_test,Se_test)  
    timeBF61[i] = time.time() - start_time
    
    start_time = time.time()
    mod82 = GCRFC_fast()
    mod82.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixture', clus_no = 250)  
    probBF62, YBF62, VarBF62 = mod82.predict(R_test,Se_test)  
    timeBF62[i] = time.time() - start_time
    
    start_time = time.time()
    mod9 = GCRFC_fast()
    mod9.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixtureProb', clus_no = 5)  
    probBF7, YBF7, VarBF7 = mod9.predict(R_test,Se_test)  
    timeBF7[i] = time.time() - start_time
    
    start_time = time.time()
    mod10 = GCRFC_fast()
    mod10.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixtureProb', clus_no = 50)  
    probBF8, YBF8, VarBF8 = mod10.predict(R_test,Se_test)  
    timeBF8[i] = time.time() - start_time
    
    start_time = time.time()
    mod101 = GCRFC_fast()
    mod101.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixtureProb', clus_no = 150)  
    probBF81, YBF81, VarBF81 = mod101.predict(R_test,Se_test)  
    timeBF81[i] = time.time() - start_time
    
    start_time = time.time()
    mod102 = GCRFC_fast()
    mod102.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixtureProb', clus_no = 250)  
    probBF82, YBF82, VarBF82 = mod102.predict(R_test,Se_test)  
    timeBF82[i] = time.time() - start_time
    
    
    Y_test = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
    
    YNB =  YNB.reshape([YNB.shape[0]*YNB.shape[1]])
    probNB = probNB.reshape([probNB.shape[0]*probNB.shape[1]])
    YB =  YB.reshape([YB.shape[0]*YB.shape[1]])
    probB = probB.reshape([probB.shape[0]*probB.shape[1]])
    YBF =  YBF.reshape([YBF.shape[0]*YBF.shape[1]])
    probBF = probBF.reshape([probBF.shape[0]*probBF.shape[1]])
    YBF2 =  YBF2.reshape([YBF2.shape[0]*YBF2.shape[1]])
    probBF2 = probBF2.reshape([probBF2.shape[0]*probBF2.shape[1]])
    YBF21 =  YBF21.reshape([YBF21.shape[0]*YBF21.shape[1]])
    probBF21 = probBF21.reshape([probBF21.shape[0]*probBF21.shape[1]])
    YBF22 =  YBF22.reshape([YBF22.shape[0]*YBF22.shape[1]])
    probBF22 = probBF22.reshape([probBF22.shape[0]*probBF22.shape[1]])
    YBF3 =  YBF3.reshape([YBF3.shape[0]*YBF3.shape[1]])
    probBF3 = probBF2.reshape([probBF3.shape[0]*probBF3.shape[1]])
    YBF4 =  YBF4.reshape([YBF4.shape[0]*YBF4.shape[1]])
    probBF4 = probBF4.reshape([probBF4.shape[0]*probBF4.shape[1]])
    YBF41 =  YBF41.reshape([YBF41.shape[0]*YBF41.shape[1]])
    probBF41 = probBF41.reshape([probBF41.shape[0]*probBF41.shape[1]])
    YBF42 =  YBF42.reshape([YBF42.shape[0]*YBF42.shape[1]])
    probBF42 = probBF42.reshape([probBF42.shape[0]*probBF42.shape[1]])
    YBF5 =  YBF5.reshape([YBF5.shape[0]*YBF5.shape[1]])
    probBF5 = probBF5.reshape([probBF5.shape[0]*probBF5.shape[1]])
    YBF6 =  YBF6.reshape([YBF6.shape[0]*YBF6.shape[1]])
    probBF6 = probBF6.reshape([probBF6.shape[0]*probBF6.shape[1]])
    YBF61 =  YBF61.reshape([YBF61.shape[0]*YBF61.shape[1]])
    probBF61 = probBF61.reshape([probBF61.shape[0]*probBF61.shape[1]])
    YBF62 =  YBF62.reshape([YBF62.shape[0]*YBF62.shape[1]])
    probBF62 = probBF62.reshape([probBF62.shape[0]*probBF62.shape[1]])
    YBF7 =  YBF7.reshape([YBF7.shape[0]*YBF7.shape[1]])
    probBF7 = probBF7.reshape([probBF7.shape[0]*probBF7.shape[1]])
    YBF8 =  YBF8.reshape([YBF8.shape[0]*YBF8.shape[1]])
    probBF8 = probBF8.reshape([probBF8.shape[0]*probBF8.shape[1]])
    YBF81 =  YBF81.reshape([YBF81.shape[0]*YBF81.shape[1]])
    probBF81 = probBF81.reshape([probBF81.shape[0]*probBF81.shape[1]])
    YBF82 =  YBF82.reshape([YBF82.shape[0]*YBF82.shape[1]])
    probBF82 = probBF82.reshape([probBF82.shape[0]*probBF82.shape[1]])
    
    AUCNB[i] = roc_auc_score(Y_test,probNB)
    AUCB[i] = roc_auc_score(Y_test,probB)
    AUCBF[i] = roc_auc_score(Y_test,probBF)
    AUCBF2[i] = roc_auc_score(Y_test,probBF2)
    AUCBF21[i] = roc_auc_score(Y_test,probBF21)
    AUCBF22[i] = roc_auc_score(Y_test,probBF22)
    AUCBF3[i] = roc_auc_score(Y_test,probBF3)
    AUCBF4[i] = roc_auc_score(Y_test,probBF4)
    AUCBF41[i] = roc_auc_score(Y_test,probBF41)
    AUCBF42[i] = roc_auc_score(Y_test,probBF42)
    AUCBF5[i] = roc_auc_score(Y_test,probBF5)
    AUCBF6[i] = roc_auc_score(Y_test,probBF6)
    AUCBF61[i] = roc_auc_score(Y_test,probBF61)
    AUCBF62[i] = roc_auc_score(Y_test,probBF62)
    AUCBF7[i] = roc_auc_score(Y_test,probBF7)
    AUCBF8[i] = roc_auc_score(Y_test,probBF8)
    AUCBF81[i] = roc_auc_score(Y_test,probBF81)
    AUCBF82[i] = roc_auc_score(Y_test,probBF82)
    
    probNB[Y_test==0] = 1 - probNB[Y_test==0]
    probB[Y_test==0] = 1 - probNB[Y_test==0]
    probBF[Y_test==0] = 1 - probBF[Y_test==0]
    probBF2[Y_test==0] = 1 - probBF2[Y_test==0]
    probBF21[Y_test==0] = 1 - probBF21[Y_test==0]
    probBF22[Y_test==0] = 1 - probBF22[Y_test==0]
    probBF3[Y_test==0] = 1 - probBF3[Y_test==0]
    probBF4[Y_test==0] = 1 - probBF4[Y_test==0]
    probBF41[Y_test==0] = 1 - probBF41[Y_test==0]
    probBF42[Y_test==0] = 1 - probBF42[Y_test==0]
    probBF5[Y_test==0] = 1 - probBF5[Y_test==0]
    probBF6[Y_test==0] = 1 - probBF6[Y_test==0]
    probBF61[Y_test==0] = 1 - probBF61[Y_test==0]
    probBF62[Y_test==0] = 1 - probBF62[Y_test==0]
    probBF7[Y_test==0] = 1 - probBF7[Y_test==0]
    probBF8[Y_test==0] = 1 - probBF8[Y_test==0]
    probBF81[Y_test==0] = 1 - probBF81[Y_test==0]
    probBF82[Y_test==0] = 1 - probBF82[Y_test==0]
    
    logProbNB[i] = np.sum(np.log(probNB))
    logProbB[i] = np.sum(np.log(probB))
    logProbBF[i] = np.sum(np.log(probBF))
    logProbBF2[i] = np.sum(np.log(probBF2))
    logProbBF21[i] = np.sum(np.log(probBF21))
    logProbBF22[i] = np.sum(np.log(probBF22))
    logProbBF3[i] = np.sum(np.log(probBF3))
    logProbBF4[i] = np.sum(np.log(probBF4))
    logProbBF41[i] = np.sum(np.log(probBF41))
    logProbBF42[i] = np.sum(np.log(probBF42))
    logProbBF5[i] = np.sum(np.log(probBF5))
    logProbBF6[i] = np.sum(np.log(probBF6))
    logProbBF61[i] = np.sum(np.log(probBF61))
    logProbBF62[i] = np.sum(np.log(probBF62))
    logProbBF7[i] = np.sum(np.log(probBF7))
    logProbBF8[i] = np.sum(np.log(probBF8))
    logProbBF81[i] = np.sum(np.log(probBF81))
    logProbBF82[i] = np.sum(np.log(probBF82))
    
    file.write('AUC GCRFCNB prediktora je {}'.format(AUCNB[i]) + "\n")
    file.write('AUC GCRFCB prediktora je {}'.format(AUCB[i]) + "\n")
    file.write('AUC GCRFCB_fast prediktora je {}'.format(AUCBF[i]) + "\n")
    file.write('AUC GCRFCB2_fast prediktora je {}'.format(AUCBF2[i]) + "\n")
    file.write('AUC GCRFCB21_fast prediktora je {}'.format(AUCBF21[i]) + "\n")
    file.write('AUC GCRFCB22_fast prediktora je {}'.format(AUCBF22[i]) + "\n")
    file.write('AUC GCRFCB3_fast prediktora je {}'.format(AUCBF3[i]) + "\n")
    file.write('AUC GCRFCB4_fast prediktora je {}'.format(AUCBF4[i]) + "\n")
    file.write('AUC GCRFCB41_fast prediktora je {}'.format(AUCBF41[i]) + "\n")
    file.write('AUC GCRFCB42_fast prediktora je {}'.format(AUCBF42[i]) + "\n")
    file.write('AUC GCRFCB5_fast prediktora je {}'.format(AUCBF5[i]) + "\n")
    file.write('AUC GCRFCB6_fast prediktora je {}'.format(AUCBF6[i]) + "\n")
    file.write('AUC GCRFCB61_fast prediktora je {}'.format(AUCBF61[i]) + "\n")
    file.write('AUC GCRFCB62_fast prediktora je {}'.format(AUCBF62[i]) + "\n")
    file.write('AUC GCRFCB7_fast prediktora je {}'.format(AUCBF7[i]) + "\n")
    file.write('AUC GCRFCB8_fast prediktora je {}'.format(AUCBF8[i]) + "\n")
    file.write('AUC GCRFCB41_fast prediktora je {}'.format(AUCBF81[i]))
    file.write('AUC GCRFCB42_fast prediktora je {}'.format(AUCBF82[i]) + "\n")
    
    file.write('AUC nestruktuiranih prediktora je {}'.format(Skor_com_AUC[i,:]) + "\n")
    file.write('AUC2 nestruktuiranih prediktora je {}'.format(Skor_com_AUC2[i,:]) + "\n")
    
    file.write('Logprob GCRFCNB je {}'.format(logProbNB[i]) + "\n")
    file.write('Logprob GCRFCB je {}'.format(logProbB[i]) + "\n")
    file.write('Logprob GCRFCB_fast je {}'.format(logProbBF[i]) + "\n")
    file.write('Logprob GCRFCB2_fast je {}'.format(logProbBF2[i]) + "\n")
    file.write('Logprob GCRFCB21_fast je {}'.format(logProbBF21[i]) + "\n")
    file.write('Logprob GCRFCB22_fast je {}'.format(logProbBF22[i]) + "\n")
    file.write('Logprob GCRFCB3_fast je {}'.format(logProbBF3[i]) + "\n")
    file.write('Logprob GCRFCB4_fast je {}'.format(logProbBF4[i]) + "\n")
    file.write('Logprob GCRFCB41_fast je {}'.format(logProbBF41[i]) + "\n")
    file.write('Logprob GCRFCB42_fast je {}'.format(logProbBF42[i]) + "\n")
    file.write('Logprob GCRFCB5_fast je {}'.format(logProbBF5[i]) + "\n")
    file.write('Logprob GCRFCB6_fast je {}'.format(logProbBF6[i]) + "\n")
    file.write('Logprob GCRFCB61_fast je {}'.format(logProbBF61[i]) + "\n")
    file.write('Logprob GCRFCB62_fast je {}'.format(logProbBF62[i]) + "\n")
    file.write('Logprob GCRFCB7_fast je {}'.format(logProbBF7[i]) + "\n")
    file.write('Logprob GCRFCB8_fast je {}'.format(logProbBF8[i]) + "\n")
    file.write('Logprob GCRFCB81_fast je {}'.format(logProbBF81[i]) + "\n")
    file.write('Logprob GCRFCB82_fast je {}'.format(logProbBF82[i]) + "\n")
    
    file.write("--- %s seconds --- GCRFCNB" % (timeNB[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB" % (timeB[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB_fast" % (timeBF[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB2_fast" % (timeBF2[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB21_fast" % (timeBF21[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB22_fast" % (timeBF22[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB3_fast" % (timeBF3[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB4_fast" % (timeBF4[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB41_fast" % (timeBF41[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB42_fast" % (timeBF42[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB5_fast" % (timeBF5[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB6_fast" % (timeBF6[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB61_fast" % (timeBF61[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB62_fast" % (timeBF62[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB7_fast" % (timeBF7[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB8_fast" % (timeBF8[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB61_fast" % (timeBF81[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB62_fast" % (timeBF82[i]) + "\n")
    
    i= i + 1

file.write('CROSS AUC GCRFCNB prediktora je {}'.format(np.mean(AUCNB)) + "\n")
file.write('CROSS AUC GCRFCB prediktora je {}'.format(np.mean(AUCB)) + "\n")
file.write('CROSS AUC GCRFCB_fast prediktora je {}'.format(np.mean(AUCBF)) + "\n")
file.write('CROSS AUC GCRFCB2_fast prediktora je {}'.format(np.mean(AUCBF2)) + "\n")
file.write('CROSS AUC GCRFCB21_fast prediktora je {}'.format(np.mean(AUCBF21)) + "\n")
file.write('CROSS AUC GCRFCB22_fast prediktora je {}'.format(np.mean(AUCBF22)) + "\n")
file.write('CROSS AUC GCRFCB3_fast prediktora je {}'.format(np.mean(AUCBF3)) + "\n")
file.write('CROSS AUC GCRFCB4_fast prediktora je {}'.format(np.mean(AUCBF4)) + "\n")
file.write('CROSS AUC GCRFCB41_fast prediktora je {}'.format(np.mean(AUCBF41)) + "\n")
file.write('CROSS AUC GCRFCB42_fast prediktora je {}'.format(np.mean(AUCBF42)) + "\n")
file.write('CROSS AUC GCRFCB5_fast prediktora je {}'.format(np.mean(AUCBF5)) + "\n")
file.write('CROSS AUC GCRFCB6_fast prediktora je {}'.format(np.mean(AUCBF6)) + "\n")
file.write('CROSS AUC GCRFCB61_fast prediktora je {}'.format(np.mean(AUCBF61)) + "\n")
file.write('CROSS AUC GCRFCB62_fast prediktora je {}'.format(np.mean(AUCBF62)) + "\n")
file.write('CROSS AUC GCRFCB7_fast prediktora je {}'.format(np.mean(AUCBF7)) + "\n")
file.write('CROSS AUC GCRFCB8_fast prediktora je {}'.format(np.mean(AUCBF8)) + "\n")
file.write('CROSS AUC GCRFCB81_fast prediktora je {}'.format(np.mean(AUCBF81)) + "\n")
file.write('CROSS AUC GCRFCB82_fast prediktora je {}'.format(np.mean(AUCBF82)) + "\n")

file.write('CROSS AUC nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_AUC,axis=0)) + "\n")
file.write('CROSS AUC2 nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_AUC2,axis=0)) + "\n")

file.write('CROSS Logprob GCRFCNB je {}'.format(np.mean(logProbNB)) + "\n")
file.write('CROSS Logprob GCRFCB je {}'.format(np.mean(logProbB)) + "\n")
file.write('CROSS Logprob GCRFCB_fast je {}'.format(np.mean(logProbBF)) + "\n")
file.write('CROSS Logprob GCRFCB2_fast je {}'.format(np.mean(logProbBF2)) + "\n")
file.write('CROSS Logprob GCRFCB21_fast je {}'.format(np.mean(logProbBF21)) + "\n")
file.write('CROSS Logprob GCRFCB22_fast je {}'.format(np.mean(logProbBF22)) + "\n")
file.write('CROSS Logprob GCRFCB3_fast je {}'.format(np.mean(logProbBF3)) + "\n")
file.write('CROSS Logprob GCRFCB4_fast je {}'.format(np.mean(logProbBF4)) + "\n")
file.write('CROSS Logprob GCRFCB41_fast je {}'.format(np.mean(logProbBF41)) + "\n")
file.write('CROSS Logprob GCRFCB42_fast je {}'.format(np.mean(logProbBF42)) + "\n")
file.write('CROSS Logprob GCRFCB5_fast je {}'.format(np.mean(logProbBF5)) + "\n")
file.write('CROSS Logprob GCRFCB6_fast je {}'.format(np.mean(logProbBF6)) + "\n")
file.write('CROSS Logprob GCRFCB61_fast je {}'.format(np.mean(logProbBF61)) + "\n")
file.write('CROSS Logprob GCRFCB62_fast je {}'.format(np.mean(logProbBF62)) + "\n")
file.write('CROSS Logprob GCRFCB7_fast je {}'.format(np.mean(logProbBF7)) + "\n")
file.write('CROSS Logprob GCRFCB8_fast je {}'.format(np.mean(logProbBF8)) + "\n")
file.write('CROSS Logprob GCRFCB81_fast je {}'.format(np.mean(logProbBF81)) + "\n")
file.write('CROSS Logprob GCRFCB82_fast je {}'.format(np.mean(logProbBF82)) + "\n")

file.write("--- %s seconds mean --- GCRFCNB" % (np.mean(timeNB)) + "\n")
file.write("--- %s seconds mean --- GCRFCB" % (np.mean(timeB)) + "\n")
file.write("--- %s seconds mean --- GCRFCB_fast" % (np.mean(timeBF)) + "\n")
file.write("--- %s seconds mean --- GCRFCB2_fast" % (np.mean(timeBF2)) + "\n")
file.write("--- %s seconds mean --- GCRFCB21_fast" % (np.mean(timeBF21)) + "\n")
file.write("--- %s seconds mean --- GCRFCB22_fast" % (np.mean(timeBF22)) + "\n")
file.write("--- %s seconds mean --- GCRFCB3_fast" % (np.mean(timeBF3)) + "\n")
file.write("--- %s seconds mean --- GCRFCB4_fast" % (np.mean(timeBF4)) + "\n")
file.write("--- %s seconds mean --- GCRFCB41_fast" % (np.mean(timeBF41)) + "\n")
file.write("--- %s seconds mean --- GCRFCB42_fast" % (np.mean(timeBF42)) + "\n")
file.write("--- %s seconds mean --- GCRFCB5_fast" % (np.mean(timeBF5)) + "\n")
file.write("--- %s seconds mean --- GCRFCB6_fast" % (np.mean(timeBF6)) + "\n")
file.write("--- %s seconds mean --- GCRFCB61_fast" % (np.mean(timeBF61)) + "\n")
file.write("--- %s seconds mean --- GCRFCB62_fast" % (np.mean(timeBF62)) + "\n")
file.write("--- %s seconds mean --- GCRFCB7_fast" % (np.mean(timeBF7)) + "\n")
file.write("--- %s seconds mean --- GCRFCB8_fast" % (np.mean(timeBF8)) + "\n")
file.write("--- %s seconds mean --- GCRFCB81_fast" % (np.mean(timeBF81)) + "\n")
file.write("--- %s seconds mean --- GCRFCB82_fast" % (np.mean(timeBF82)) + "\n")
    

file.close()

    