# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:15:22 2019

@author: Andrija Master
"""
#import numpy as np
#from collections import namedtuple
#SparseMatrix = namedtuple("SparseMatrix", "indices values dense_shape")
#
#def from_EN_to_GNN(E, N):
#    """
#    :param E: # E matrix - matrix of edges : [[id_p, id_c, graph_id],...]
#    :param N: # N matrix - [node_features, graph_id (to which the node belongs)]
#    :return: # L matrix - list of graph targets [tar_g_1, tar_g_2, ...]
#    """
#    N_full = N
#    N = N[:, :-1]  # avoid graph_id
#    e = E[:, :2]  # take only first tow columns => id_p, id_c
#    feat_temp = np.take(N, e, axis=0)  # take id_p and id_c  => (n_archs, 2, label_dim)
#    feat = np.reshape(feat_temp, [len(E), -1])  # (n_archs, 2*label_dim) => [[label_p, label_c], ...]
#    # creating input for gnn => [id_p, id_c, label_p, label_c]
#    inp = np.concatenate((E[:, 0:2], feat), axis=1)
#    # creating arcnode matrix, but transposed
#    """
#    1 1 0 0 0 0 0 
#    0 0 1 1 0 0 0
#    0 0 0 0 1 1 1    
#
#    """  # for the indices where to insert the ones, stack the id_p and the column id (single 1 for column)
#    arcnode = SparseMatrix(indices=np.stack((E[:, 0], np.arange(len(E))), axis=1),
#                           values=np.ones([len(E)]).astype(np.float32),
#                           dense_shape=[len(N), len(E)])
#
#    # get the number of graphs => from the graph_id
#    num_graphs = int(max(N_full[:, -1]) + 1)
#    # get all graph_ids
#    g_ids = N_full[:, -1]
#    g_ids = g_ids.astype(np.int32)
#
#    # creating graphnode matrix => create identity matrix get row corresponding to id of the graph
#    # graphnode = np.take(np.eye(num_graphs), g_ids, axis=0).T
#    # substitued with same code as before
#    graphnode = SparseMatrix(indices=np.stack((g_ids, np.arange(len(g_ids))), axis=1),
#                             values=np.ones([len(g_ids)]).astype(np.float32),
#                             dense_shape=[num_graphs, len(N)])
#
#    # print(graphnode.shape)
#
#    return inp, arcnode, graphnode



def Strukturni(x_train, y_train, x_test, y_test, Se_train, Se_test, No_class):
        
    import tensorflow as tf
    import numpy as np
    import gnn.gnn_utils as gnn_utils
    import gnn.GNN as GNN
    import Net_Strukturni as n
    
    import networkx as nx
    import scipy as sp
    
    """ GNN """
    x_train = x_train.values
    x_test = x_test.values
    g = nx.complete_graph(No_class)
    E_start = np.asarray(g.edges())
    E_start = np.hstack((E_start, np.ones([E_start.shape[0],1])*0))
    E_start_2 = np.asarray([[i, j, num] for j, i, num in E_start])
    E_start = np.vstack((E_start, E_start_2))
    
    E = E_start.copy()
    
    E_test = E_start.copy()
    
    N = np.zeros([No_class*x_train.shape[0],x_train.shape[1]+1])
    N[:No_class,:x_train.shape[1]] = np.tile(x_train[0,:],(2,1)) 
    
    N_test = np.zeros([No_class*x_train.shape[0],x_train.shape[1]+1])
    N_test[:No_class,:x_train.shape[1]] = np.tile(x_train[0,:],(2,1)) 
    for i in range(1,x_train.shape[0]):
        N[No_class*i:No_class*(i+1),:x_train.shape[1]] = np.tile(x_train[0,:],(2,1)) 
        N[No_class*i:No_class*(i+1),x_train.shape[1]] = i
        E_new = E_start.copy()
        E_new[:,:2] = E_new[:,:2] + No_class*i
        E_new[:,2] = E_new[:,2] + i
        E = np.vstack((E, E_new))
   
    for i in range(1,x_test.shape[0]):
        N_test[No_class*i:No_class*(i+1),:x_test.shape[1]] = np.tile(x_test[0,:],(2,1)) 
        N_test[No_class*i:No_class*(i+1),x_test.shape[1]] = i
        E_new_test = E_start.copy()
        E_new_test[:,:2] = E_new_test[:,:2] + No_class*i
        E_new_test[:,2] = E_new_test[:,2] + i
        E_test = np.vstack((E_test, E_new_test))
    
    
    E = E.astype('int32')
    E_test = E_test.astype('int32')
    inp, arcnode, graphnode = gnn_utils.from_EN_to_GNN(E, N)
    inp_test, arcnode_test, graphnode_test = gnn_utils.from_EN_to_GNN(E_test, N_test)
    input_train = np.zeros([Se_train.shape[0],Se_train.shape[1]])
    input_test = np.zeros([Se_test.shape[0],Se_test.shape[1]])
    m=0
    for i in range(Se_train.shape[0]):
        for k in range(Se_train.shape[2]):
                for j in range(k+1,Se_train.shape[3]):
                    input_train[m,:] = Se_train[i,:,k,j]
                    m+=1
    m=0                
    for i in range(Se_test.shape[0]):
        for k in range(Se_test.shape[2]):
                for j in range(k+1,Se_test.shape[3]):
                    input_test[m,:] = Se_test[i,:,k,j]    
                    m+=1
    aa = 22

    





