# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:15:22 2019

@author: Andrija Master
"""

def Strukturni(x_train, y_train, x_test, y_test):
        
    import time

    import numpy as np
    
    from sklearn.metrics import hamming_loss
    from sklearn.metrics import accuracy_score
    

    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    
    
    x_train = x_train.values
    y_train = y_train.values
    y_test = y_test.values
    x_test = x_test.values
        
    
    """ Define models """
    
    MLP = MLPClassifier()
    DT = DecisionTreeClassifier()
    
    """ Fit models """
    
    time_ST = np.zeros(2)

    start_time = time.time()
    DT.fit(x_train, y_train)
    y_DT = DT.predict(x_test)
    time_ST[0] = time.time() - start_time
    
    start_time = time.time()
    MLP.fit(x_train, y_train)
    y_MLP = MLP.predict(x_test)
    time_ST[1] = time.time() - start_time
  
    """ EVALUATE models """
    HL = np.zeros(2)
    ACC = np.zeros(2)
    
    HL[0] = hamming_loss(y_test,y_MLP)
    HL[1] = hamming_loss(y_test,y_DT)
    
    y_MLP = y_MLP.reshape([y_MLP.shape[0]*y_MLP.shape[1]])
    y_DT = y_DT.reshape([y_DT.shape[0]*y_DT.shape[1]])
    y_test = y_test.reshape([y_test.shape[0]*y_test.shape[1]])
    
    ACC[0] = accuracy_score(y_test,y_MLP)
    ACC[1] = accuracy_score(y_test,y_DT)
    
    return ACC, HL, time_ST

    
    






