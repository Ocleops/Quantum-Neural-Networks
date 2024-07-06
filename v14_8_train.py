#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:34:59 2021

@author: 
"""

# import os
# os.chdir(r'C:\Users\wu000\Downloads\QNN\new_code')
#os.chdir(r'C:\Users\alkis\Downloads\12.8.22\old')
import numpy as np
import matplotlib.pyplot as plt
import sys
from qiskit import *
# sys.path.append('scratch/ph5117/myrps')


from v14_8_QNN import QNN
from v14_8_State_preparation import state_preparation, get_train_states, get_test_states, get_circuits



if __name__ == "__main__":
    # Create QNN
    M = [4,1,4]
    D = ['trainlayer','trainlayer','conjugatelayer1']
    
    #gates = get_circuits(M[1], "bit_flip_channel")
    
    my_QNN = QNN(M, D)
    #my_QNN.draw()
    
    # Set parameters
    gamma  = 0.1
    cor_prob=0.2
    
    nb_train_states =50
    
    nb_val_states  = 20*20
    nb_test_states = 20*20
    
    batch_size = None
    shuffle_batch = True
    
    #K=np.load('k model = 414_new_state_in, p=0.8, optimazer = default, batch size = None, gm = 0.1, nb = 70, e = 0.1, lr = 0.2.npy')
    #my_QNN.set_K(K)
    
    #noisy channels and goals 
    input_state = 'Noise_Amplitude_Dumping'
    goal =  'Amplitude_Dumping' #put None if there is only one channel
          
    # Train the QNN
    nb_epochs = 100
    epsilon = 0.1
    lr = 0.2
    
    optimizer_name = 'default'  # options: default (SGD) /momentum /RMSprop /Adam /Nadam /Adamax
    
    optimizer = my_QNN.optimizers( optimizer_name , lr )#, beta1=0.999,beta2 = 0.95, it=100) # options: default (SGD) /momentum /RMSprop /Adam /Nadam /Adamax
    nb_shot = 2000
    
    early_stopping = True
    
    name =  'model = 414pres_conjlayer, p= '+str(cor_prob)+', optimazer = '+optimizer_name+', batch size = '+str(batch_size)+', gammma = '+str(gamma)+', nb = '+ str(nb_train_states)+', e = '+str(epsilon)+', lr = '+str(lr)+', shots='+str(nb_shot)+''
    

    # Get training and validation states
        # .M[l] returns the number of nodes of layer l
        
    #training / validation/ goal states
    training_states, goal_state = get_train_states(my_QNN.M[0],input_state ,goal , nb_train_states, gamma,  cor_prob )
   
    val_states = get_test_states(my_QNN.M[0], input_state, goal , nb_val_states, gamma, cor_prob  ) 
    

    # Test the fidelity before training
    test_states = get_test_states(my_QNN.M[0], input_state, goal , nb_test_states, gamma, cor_prob) 
 
    init_fid = my_QNN.test(test_states)
    print('Fidelity before training : {:.2f}'.format(np.mean(init_fid)))
    
    train_fid, val_fid = my_QNN.fit(training_states, nb_epochs, epsilon, optimizer= optimizer, p=gamma ,batch_size_train=batch_size,   
                           goal_state = goal_state , validation_states = val_states , save_best_val_fid = name,
                          run_in_batch = True, shuffle_batch = shuffle_batch ,name=name, early_stopping =  early_stopping) #, ran_phi_in_epoch = True , p = proba_error) # when ran_phi_in_epoch = True proba_error needs to be specified 
    

    
    # Test the fidelity after training
    final_fid = my_QNN.test(test_states)
    print('Fidelity after training : {:.2f}'.format(np.mean(final_fid)))