# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:05:05 2018

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
from sklearn.metrics import accuracy_score 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score




plt.close('all')

staze = np.load('staze.npy')
Y_test = np.load('Y_test.npy')
Y_test = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
staze  = staze.astype(int)

testsize = 0.2 #0.25
testsize2 = 0.5 #0.2
#testsize3 = 0.5

skijasi = pd.read_csv(str(staze[1]),index_col='date1')
    
skijasi.target = skijasi.label
skijasi.data = skijasi.drop(['label','vreme_pros'],axis=1)
    
x_train_com, x_test, y_train_com, y_test = train_test_split(skijasi.data, skijasi.target, test_size=testsize, random_state=31)
x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)            
#x_train_un1, x_train_un, y_train_un1, y_train_un = train_test_split(x_train_un, y_train_un, test_size=testsize3, random_state=31)


no_train = x_train_st.shape[0]
no_test = x_test.shape[0]
no_train_un = x_train_un.shape[0]

Z_train = np.zeros([no_train,len(staze)])
Z_test = np.zeros([no_test,len(staze)])
labels = np.zeros([no_test,len(staze)])
Z_train_un = np.zeros([no_train_un,len(staze)])
k=0
std_scl = StandardScaler()
skor = np.zeros([7,1])
skorAUC = np.zeros([7,1])
for i in staze:
    
    skijasi = pd.read_csv(str(i),index_col='date1')
    
    skijasi.target = skijasi.label
#    skijasi.data = skijasi.drop(['label','broj','broj_unique','hour-minute','windSpeed','pressure','brzinaMEDIAN',\
#                                 'brzinaPERCENTILE75','brzinaPERCENTILE90',\
#                                 'vreme_pros','cloudCover','brzinaPERCENTILE100','temperature','brzinaMEAN','brzinaKURT'],axis=1)
    skijasi.data = skijasi.drop(['label','vreme_pros'],axis=1)
    
    x_train_com, x_test, y_train_com, y_test = train_test_split(skijasi.data, skijasi.target, test_size=testsize, random_state=31)
    x_train_un, x_train_st, y_train_un, y_train_st = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)      
#    x_train_un1, x_train_un, y_train_un1, y_train_un = train_test_split(x_train_un, y_train_un, test_size=testsize3, random_state=31)
 
    std_scl.fit(x_train_un)
    x_train_un = std_scl.transform(x_train_un)
    x_train_st = std_scl.transform(x_train_st)
    x_test = std_scl.transform(x_test)
    
    model = Sequential()
    model.add(Dense(30, input_dim=skijasi.data.shape[1], activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    model.fit(x_train_un, y_train_un, epochs=1200, batch_size=100,validation_data=(x_test, y_test))
    labelsx = model.predict_classes(x_test)
    labels[:,k] = labelsx.reshape([len(labelsx)])
    y_test = y_test.values

    
    model2 = Sequential()
    model2.add(Dense(30, input_dim=skijasi.data.shape[1], weights = model.layers[0].get_weights() ,activation='relu'))
    model2.add(Dense(20,weights = model.layers[1].get_weights() , activation='relu'))
    model2.add(Dense(14,weights = model.layers[2].get_weights() , activation='relu'))
    model2.add(Dense(1 , weights = model.layers[3].get_weights(), activation='linear'))
    
    skijasi.data = std_scl.transform(skijasi.data)
    Z_train[:,k] = model2.predict(x_train_st).reshape(no_train)
    Z_train_un[:,k] = model2.predict(x_train_un).reshape(no_train_un)
    Z_test[:,k] = model2.predict(x_test).reshape(no_test)
    skorAUC[k,0] = roc_auc_score(y_test,Z_test[:,k])
    k += 1


skorAUC = roc_auc_score(Y_test,Z_test.reshape([Z_test.shape[0]*Z_test.shape[1]]))
skor = accuracy_score(Y_test, labels.reshape([labels.shape[0]*labels.shape[1]]))
Z_trainlog = np.load('Z_train.npy')
Z_testlog = np.load('Z_test.npy')
Z_train_com = np.concatenate((Z_trainlog, Z_train.reshape([Z_train.shape[0]*Z_train.shape[1],1])),axis = 1)
Z_test_com = np.concatenate((Z_testlog, Z_test.reshape([Z_test.shape[0]*Z_test.shape[1],1])), axis = 1)

SkorlogAUC = np.load('SkorlogAUC.npy')
Skorlog = np.load('Skorlog.npy')
SkorlogAUC = SkorlogAUC.reshape([SkorlogAUC.shape[0]*SkorlogAUC.shape[1]])
Skor_com_AUC = np.append(SkorlogAUC,skorAUC)
Skorlog = Skorlog.reshape([Skorlog.shape[0]*Skorlog.shape[1]])
Skor_com = np.append(Skorlog,skor)

np.save('Skor_com_AUC.npy', Skor_com_AUC)
np.save('Skor_com.npy', Skor_com)
np.save('Z_train_com', Z_train_com)
np.save('Z_test_com.npy', Z_test_com)
np.save('Z_train_un.npy',Z_train_un)



    