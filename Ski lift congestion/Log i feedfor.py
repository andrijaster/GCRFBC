# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:21:11 2018

@author: Andrija Master
"""

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
from keras.models import Sequential
from keras.layers import Dense

plt.close('all')

def evZ(x):
    return -np.log(1/x-1)


staze = np.load('staze.npy')
staze  = staze.astype(int)


testsize2 = 0.1

skijasi = pd.read_csv(str(staze[1]),index_col='date1')
    
skijasi.target = skijasi.label
#skijasi.data = skijasi.drop(['label','broj','broj_unique','hour-minute','windSpeed'],axis=1)
skijasi.data = skijasi.drop(['label'],axis=1)
   
x_train_com, x_test, y_train_com, y_test = train_test_split(skijasi.data, skijasi.target, test_size=0.25, random_state=31)
x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)            

no_train = x_train_st.shape[0]
no_test = x_test.shape[0]
no_train_un = x_train_un.shape[0]
X_TRAIN_UN = []
X_TRAIN_ST = []
X_TEST = []
Y_TRAIN_UN = []
Y_TRAIN_ST = []
Y_TEST = []
for i in staze:
    
    skijasi = pd.read_csv(str(i),index_col='date1')
    
    skijasi.target = skijasi.label
#    skijasi.data = skijasi.drop(['label','broj','broj_unique','hour-minute','windSpeed','pressure','brzinaMEDIAN', 'brzinaPERCENTILE10'],axis=1)
    skijasi.data = skijasi.drop(['label'],axis=1)
    
    x_train_com, x_test, y_train_com, y_test = train_test_split(skijasi.data, skijasi.target, test_size=0.25, random_state=31)
    x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)
    
    X_TRAIN_UN.append(x_train_un)
    X_TRAIN_ST.append(x_train_st)
    X_TEST.append(x_test)
    Y_TRAIN_UN.append(y_train_un)
    Y_TRAIN_ST.append(y_train_st)
    Y_TEST.append(y_test)
    
X_TRAIN_UN = np.concatenate(X_TRAIN_UN, axis = 0)
X_TRAIN_ST = np.concatenate(X_TRAIN_ST, axis = 0)
X_TEST = np.concatenate(X_TEST, axis = 0)
Y_TRAIN_UN = np.concatenate(Y_TRAIN_UN, axis = 0)
Y_TEST = np.concatenate(Y_TEST, axis = 0)
Y_TRAIN_ST = np.concatenate(Y_TRAIN_ST, axis = 0)

std_scl = StandardScaler()
std_scl.fit(X_TRAIN_UN)
X_TRAIN_UN = std_scl.transform(X_TRAIN_UN)
X_TRAIN_ST = std_scl.transform(X_TRAIN_ST)
X_TEST = std_scl.transform(X_TEST)

logRegression = LogisticRegressionCV(Cs = 10)
#    logRegression = LogisticRegression(C=0.001)
logRegression1 = LogisticRegressionCV(Cs = 10,penalty = 'l1', solver = 'saga')
logRegression.fit(X_TRAIN_UN, Y_TRAIN_UN)
logRegression1.fit(X_TRAIN_UN, Y_TRAIN_UN)


predictions_test = logRegression.predict_proba(X_TEST)[:,1]
predictions1_test = logRegression1.predict_proba(X_TEST)[:,1]

Z_train = logRegression.decision_function(X_TRAIN_ST)
Z_train = Z_train.reshape([len(staze),no_train]).T
Z_test = logRegression.decision_function(X_TEST)
Z_test = Z_test.reshape([len(staze),no_test]).T

Z1_train = logRegression1.decision_function(X_TRAIN_ST)
Z1_train = Z1_train.reshape([len(staze),no_train]).T
Z1_test = logRegression1.decision_function(X_TEST)    
Z1_test = Z1_test.reshape([len(staze),no_test]).T

predictions_test = logRegression.predict_proba(X_TEST)[:,1]
Skor1 = roc_auc_score(Y_TEST,predictions_test)
predictions1_test = logRegression.predict_proba(X_TEST)[:,1]
Skor2 = roc_auc_score(Y_TEST,predictions1_test) 

model = Sequential()
model.add(Dense(30, input_dim=skijasi.data.shape[1], activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.fit(X_TRAIN_UN, Y_TRAIN_UN, epochs=1200, batch_size=100,validation_data=(x_test, y_test))

model2 = Sequential()
model2.add(Dense(30, input_dim=skijasi.data.shape[1], weights = model.layers[0].get_weights() ,activation='relu'))
model2.add(Dense(20,weights = model.layers[1].get_weights() , activation='relu'))
model2.add(Dense(14,weights = model.layers[2].get_weights() , activation='relu'))
model2.add(Dense(1 , weights = model.layers[3].get_weights(), activation='linear'))
    
Z2_train = model2.predict(X_TRAIN_ST)
Z2_train_un = model2.predict(X_TRAIN_UN)
Z2_test = model2.predict(X_TEST)   
Skor3 = roc_auc_score(Y_TEST,Z2_test) 

Z2_train = Z2_train.reshape([len(staze),no_train]).T  
Z2_test = Z2_test.reshape([len(staze),no_test]).T   
Z2_train_un = Z2_train_un.reshape([len(staze),no_train_un]).T        

Z_train_fin = np.concatenate((Z_train.reshape([Z_train.shape[0]*Z_train.shape[1],1]), \
                    Z1_train.reshape([Z1_train.shape[0]*Z1_train.shape[1],1])),axis=1)
Z_test_fin = np.concatenate((Z_test.reshape([Z_test.shape[0]*Z_test.shape[1],1]), \
                    Z1_test.reshape([Z1_test.shape[0]*Z1_test.shape[1],1])),axis=1) 

Z_train_com = np.concatenate((Z_train_fin, Z2_train.reshape([Z2_train.shape[0]*Z2_train.shape[1],1])),axis = 1)
Z_test_com = np.concatenate((Z_test_fin, Z2_test.reshape([Z2_test.shape[0]*Z2_test.shape[1],1])), axis = 1)

Skor_com_AUC = np.array((Skor1,Skor2,Skor3))
np.save('Skor_com_AUC.npy', Skor_com_AUC)
np.save('Z_train_com', Z_train_com)
np.save('Z_test_com.npy', Z_test_com)
np.save('Z_train_un.npy',Z2_train_un)   