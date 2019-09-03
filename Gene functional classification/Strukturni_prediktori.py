# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:15:22 2019

@author: Andrija Master
"""

def Strukturni(x_train, y_train, x_test, y_test, Se_train, Se_test, No_class):
        
    import tensorflow as tf
    import numpy as np
    import gnn.gnn_utils as gnn_utils
    import gnn.GNN as GNN
    import Net_Strukturni as n
    
    import networkx as nx
    import scipy as sp
    
    """ GNN """
    """CREATE DATASET FOR GNN """
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
    inp = np.hstack((inp, input_train))
    inp_test = np.hstack((inp_test, input_test))
    aa = 22

    """ Calculate GNN """
    threshold = 0.01
    learning_rate = 0.01
    state_dim = 5
    tf.reset_default_graph()
    input_dim = inp.shape[1]
    output_dim = labels.shape[1]
    max_it = 50
    num_epoch = 1000
    optimizer = tf.train.AdamOptimizer
    
    # initialize state and output network
    net = n.Net(input_dim, state_dim, output_dim)
    
    # initialize GNN
    param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
    print(param)
    
    tensorboard = False
    
    g = GNN.GNN(net, input_dim, output_dim, state_dim,  max_it, optimizer, learning_rate, threshold, graph_based=False, param=param, config=config,
                tensorboard=tensorboard)
    
    # train the model
    count = 0
    
    ######
    
    for j in range(0, num_epoch):
        _, it = g.Train(inputs=inp, ArcNode=arcnode, target=labels, step=count)
    
        if count % 30 == 0:
            print("Epoch ", count)
            print("Training: ", g.Validate(inp, arcnode, labels, count))
    
            # end = time.time()
            # print("Epoch {} at time {}".format(j, end-start))
            # start = time.time()
    
        count = count + 1
    
    # evaluate on the test set
    # print("\nEvaluate: \n")
    # print(g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0])[0])
        





