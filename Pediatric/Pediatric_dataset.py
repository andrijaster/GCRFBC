# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:42:49 2018

@author: Andrija Master
"""
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


output = pd.read_csv('pediatricSID_CA_multilabel.csv')
atribute = pd.read_csv('pediatricSID_CA_data.csv')
atribute.set_index('ID',inplace = True)
atribute.reset_index(inplace = True,drop=True)
output.set_index('ID',inplace = True)
output.reset_index(inplace = True,drop=True)
output = output.astype(int)


No_class = output.shape[1]
bb = np.all(output == 0 ,axis=1)
bb = bb==False
output = output[bb]
output = shuffle(output,random_state=31)
atribute = atribute[bb]
atribute = shuffle(atribute,random_state=31)
atribute.reset_index(inplace = True, drop=True)
output.reset_index(inplace = True, drop=True)
