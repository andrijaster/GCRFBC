# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:15:22 2019

@author: Andrija Master
"""


def Strukturni(x_train, y_train, x_test, y_test, No_class):
        
    import tensorflow as tf
    import numpy as np
    import gnn.gnn_utils as gnn_utils
    import gnn.GNN as GNN
    import Net_Strukturni as n
    
    import networkx as nx
    import scipy as sp
    
    """ GNN """
    g = nx.complete_graph(No_class)
    E_start = np.array(g.edges)
    E_start = np.hstack((E_start, np.ones([E_start.shape[0],1])*0))
    E_start_2 = np.asarray([[i, j, num] for j, i, num in E_start])
    E_start = np.vstack((E_start, E_start_2))
    N_tot = 3
    





