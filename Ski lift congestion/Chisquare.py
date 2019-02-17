# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:54:44 2018

@author: Andrija Master
"""


import pandas as pd
import scipy.stats as sp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split

staze = np.load('staze.npy')
staze  = staze.astype(int)
koef1 = 2e-4
koef2 = 5e-7
NoGraph = 5
NodeNo = 7
Se = np.zeros([NoGraph,NodeNo,NodeNo])
i1 = 0
k=0
for i in staze:
    j1 = i1
    for j in staze[i1+1:]:
        j1 += 1
        skijasi1 = pd.read_csv(str(i),index_col='date1')
        skijasi2 = pd.read_csv(str(j),index_col='date1')
        Mut_info = normalized_mutual_info_score(skijasi1.label.values,skijasi2.label.values)
        Mat = pd.crosstab(skijasi1.label.values,skijasi2.label.values)
        chi2, pvalue, dof, ex = sp.chi2_contingency(Mat)
        Se[0,i1,j1] = chi2
        print([chi2,pvalue])
        Se[0,j1,i1] = Se[0,i1,j1]
        Se[1,i1,j1] = Mut_info
        Se[1,j1,i1] = Se[1,i1,j1]  
        Se[3,i1,j1] = np.exp(-koef1*np.sum(np.abs(skijasi1.label.values-skijasi2.label.values)))
        Se[3,j1,i1] = Se[3,i1,j1]
        Se[4,i1,j1] = np.exp(-koef2*np.sum(np.abs(skijasi1.vreme_pros.values-skijasi2.vreme_pros.values)))
        Se[4,j1,i1] = Se[4,i1,j1]
    i1+=1
Se[0,:,:] = np.exp(-3/Se[0,:,:])
Se[1,:,:] = np.exp(-1/Se[1,:,:])  
Se[3,:,:] = np.exp(-1/Se[3,:,:])  
Se[4,:,:] = np.exp(-3/Se[4,:,:])  

#Se[0,:,:] = np.exp(-8/Se[0,:,:])
#Se[1,:,:] = np.exp(-/Se[1,:,:])  
#Se[3,:,:] = np.exp(-8/Se[3,:,:])  
#Se[4,:,:] = np.exp(-8/Se[4,:,:])  

R2 = np.load('Z_train_un.npy')
scaler = StandardScaler()
R2 = R2.reshape([R2.shape[0]*R2.shape[1],1])
scaler.fit(R2)
R2 = scaler.transform(R2)
R2 = R2.reshape([int(R2.shape[0]/NodeNo),NodeNo])

Corelation_mat = np.corrcoef(R2.T)
Corelation_mat[Corelation_mat<0] = 0
np.fill_diagonal(Corelation_mat,0)

Se[2,:,:] = Corelation_mat
np.save('Se',Se)
skijasi = pd.read_csv(str(staze[0]))
broj_label = 5
c = 1e-2
k = 0
testsize = 0.2 #0.25
testsize2 = 0.5 #0.2

skinovo = np.zeros([len(staze),skijasi.shape[0],broj_label])
for i in staze:
    skijasi = pd.read_csv(str(i))
    skijasi.set_index('date1',drop=False,inplace = True)
    for j1 in range(broj_label,skijasi.shape[0]):
        skinovo[k,j1,:] = skijasi.label.iloc[j1-broj_label:j1]
    k+=1
    

Noinst = skijasi.shape[0]
NodeNo = len(staze)
skinovo = np.zeros([Noinst,NodeNo*broj_label])
for i in staze:
    k=0
    skijasi = pd.read_csv(str(i))
    skijasi.set_index('date1',drop=False,inplace = True)
    for j1 in range(broj_label,skijasi.shape[0]):
        skinovo[j1,k*broj_label:(k+1)*broj_label] = skijasi.vreme_pros.iloc[j1-broj_label:j1]
    k+=1
#skinovo = pd.DataFrame(skinovo, index = skijasi.index.values)
skinovo_train_com, skinovo_test, y_train_com, y_test = train_test_split(skinovo, skijasi.label, test_size=testsize, random_state=31)
skinovo_train_un, skinovo_train_st, y_train_un, y_train_st = train_test_split(skinovo_train_com, y_train_com, test_size=testsize2, random_state=31)    


Se_test = np.zeros([skinovo_test.shape[0],1,len(staze),len(staze)])
i1=0
for i in range(NodeNo):
    for j1 in range(i1+1,NodeNo):
        print(i,j1)
        Se_test[:,0,i,j1] = np.exp(-c*np.linalg.norm(skinovo_test[:,i*broj_label:(i+1)*broj_label]- \
          skinovo_test[:,j1*broj_label:(j1+1)*broj_label], axis=1))
        Se_test[:,0,j1,i] = Se_test[:,0,i,j1]
    i1+=1
Se_train_st = np.zeros([skinovo_train_st.shape[0],1,len(staze),len(staze)])
i1=0
for i in range(NodeNo):
    for j1 in range(i1+1,NodeNo):
        print(i,j1)
        Se_train_st[:,0,i,j1] = np.exp(-c*np.linalg.norm(skinovo_train_st[:,i*broj_label:(i+1)*broj_label]- \
          skinovo_train_st[:,j1*broj_label:(j1+1)*broj_label], axis=1))
        Se_train_st[:,0,j1,i] = Se_train_st[:,0,i,j1]
    i1+=1



np.save('Se_train',Se_train_st)
np.save('Se_test',Se_test)        