# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:25:05 2018

@author: Andrija Master
"""

""" LOGISTICKA REGRESIJA """

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score

plt.close('all')

def evZ(x):
    return -np.log(1/x-1)


staze = np.load('staze.npy')
staze  = staze.astype(int)

testsize = 0.2 #0.25
testsize2 = 0.5 #0.2
#testsize3 = 0.5

skijasi = pd.read_csv(str(staze[1]),index_col='date1')
    
skijasi.target = skijasi.label
skijasi.data = skijasi.drop(['label','vreme_pros'],axis=1)
#skijasi.data = skijasi.drop(['label'],axis=1)
   
x_train_com, x_test, y_train_com, y_test = train_test_split(skijasi.data, skijasi.target, test_size=testsize, random_state=31)
x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)            
#x_train_un, x_train_un2, y_train_un, y_train_un2 = train_test_split(x_train_un, y_train_un, test_size=testsize3, random_state=31)

no_train = x_train_st.shape[0]
no_test = x_test.shape[0]
no_train_un = x_train_un.shape[0]

Z_train = np.zeros([no_train,len(staze)])
Z_test = np.zeros([no_test,len(staze)])
Z1_train = np.zeros([no_train,len(staze)])
Z1_test = np.zeros([no_test,len(staze)])
labels = np.zeros([no_test,len(staze)])
labels1 = np.zeros([no_test,len(staze)]) 
k=0
std_scl = StandardScaler()
skor = np.zeros([1,2])
skorAUC = np.zeros([1,2])
Y_train = np.zeros([no_train,len(staze)])
Y_test = np.zeros([no_test, len(staze)])
predictions_test = np.zeros([no_test,len(staze)])
predictions1_test = np.zeros([no_test,len(staze)])


for i in staze:
    
    skijasi = pd.read_csv(str(i),index_col='date1')
    
    skijasi.target = skijasi.label
#    skijasi.data = skijasi.drop(['label','broj','broj_unique','hour-minute','windSpeed','pressure','brzinaMEDIAN',\
#                                 'brzinaPERCENTILE10','brzinaPERCENTILE25',\
#                                 'vreme_pros','cloudCover','brzinaPERCENTILE100','temperature'],axis=1)
    skijasi.data = skijasi.drop(['label','vreme_pros'],axis=1)
    
    x_train_com, x_test, y_train_com, y_test = train_test_split(skijasi.data, skijasi.target, test_size=testsize, random_state=31)
    x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)            
    std_scl.fit(x_train_un)
    x_train_un = std_scl.transform(x_train_un)
    x_train_st = std_scl.transform(x_train_st)
    x_test = std_scl.transform(x_test)
    
    logRegression = LogisticRegressionCV(Cs = 10)
#    logRegression = LogisticRegression(C=0.001)
    logRegression1 = LogisticRegressionCV(Cs = 10,penalty = 'l1', solver = 'saga')
    logRegression.fit(x_train_un, y_train_un)
    logRegression1.fit(x_train_un, y_train_un)
    print(logRegression.coef_)
    
    skijasi.data = std_scl.transform(skijasi.data)
    
    predictions_train_un = logRegression.predict_proba(x_train_un)
    predictions_test[:,k] = logRegression.predict_proba(x_test)[:,1]
    predictions1_train_un = logRegression1.predict_proba(x_train_un)
    predictions1_test[:,k] = logRegression1.predict_proba(x_test)[:,1]
    
    labels[:,k] = logRegression.predict(x_test)
    labels1[:,k] = logRegression1.predict(x_test)
    
    Z_train[:,k] = logRegression.decision_function(x_train_st)
    Z_test[:,k] = logRegression.decision_function(x_test)
    Z1_train[:,k] = logRegression1.decision_function(x_train_st)
    Z1_test[:,k] = logRegression1.decision_function(x_test)
    
    Y_train[:,k] = y_train_st.values
    Y_test[:,k] = y_test.values
    
#    skor[k,0] = accuracy_score(y_test,labels)
#    skor[k,1] = accuracy_score(y_test,labels1)
#    skorAUC[k,0] = roc_auc_score(y_test,predictions_test[:,1])
#    skorAUC[k,1] = roc_auc_score(y_test,predictions1_test[:,1])
#    print('l2 score je {} '.format(skor[k,0]))
#    print('l1 score je {} '.format(skor[k,1]))
    k+=1
#    score = logRegression.score(x_test, y_test)
#    print(score)

skor[:,0] = accuracy_score(Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]]),labels.reshape([labels.shape[0]*labels.shape[1]]))
skor[:,1] = accuracy_score(Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]]),labels1.reshape([labels1.shape[0]*labels1.shape[1]]))
skorAUC[:,0] = roc_auc_score(Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]]),predictions_test.reshape([Y_test.shape[0]*Y_test.shape[1]]))
skorAUC[:,1] = roc_auc_score(Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]]),predictions1_test.reshape([Y_test.shape[0]*Y_test.shape[1]]))
Z_train_fin = np.concatenate((Z_train.reshape([Z_train.shape[0]*Z_train.shape[1],1]), \
                    Z1_train.reshape([Z1_train.shape[0]*Z1_train.shape[1],1])),axis=1)
Z_test_fin = np.concatenate((Z_test.reshape([Z_test.shape[0]*Z_test.shape[1],1]), \
                    Z1_test.reshape([Z1_test.shape[0]*Z1_test.shape[1],1])),axis=1)
np.save('SkorlogAUC',skorAUC)
np.save('Skorlog', skor)
np.save('Z_train', Z_train_fin)
np.save('Z_test', Z_test_fin)
np.save('Y_train', Y_train)
np.save('Y_test', Y_test)