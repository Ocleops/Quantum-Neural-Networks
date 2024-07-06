#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:40:07 2021

@author: tomachache
"""

import numpy as np
from scipy import linalg as LA
from qiskit import *
from qiskit.quantum_info import random_statevector, Statevector, Operator
import random
# Various state preparation


def get_train_states(m, name , goal , nb_states , gamma=None , p=None, shuffle=False ):
    states = []
    goal_state=[]
    
    
    theta = [0, np.pi/2 ,np.pi/2] #, np.pi/2]   #[np.pi/2  ,    np.pi/2, np.pi/2]    #, np.pi/2]#,   np.pi/2,    np.pi/2  ]
    phi   = [0, np.pi/2  ,  0   ] #, np.pi ]   #[3*np.pi/2,  4*np.pi/3, np.pi/2]   #,   np.pi]#,   np.pi/2,    3*np.pi/2]
    
    #theta = [0, np.pi, np.pi/2, np.pi/2]
    #phi   = [0, 0    , 0      , np.pi  ]
    
    #nb_states = int(np.sqrt(nb_states))
    
    #phi = np.linspace(0, 2*np.pi,nb_states)
    #theta = np.linspace(0, np.pi,nb_states)
    
    
    #i=0
    #j=0
    for count, k in enumerate(range(nb_states)):
        states.append(( state_preparation(m, name, gamma, p, phi[count % len(phi) ], theta[count % len(theta)]),
                        state_preparation(m, goal, gamma, p, phi[count % len(phi) ], theta[count % len(theta)])) )
        
        goal_state.append( state_preparation(m, goal, p, phi[count % len(phi) ],theta[count % len(theta)]))
        
        #i +=1
        #j +=1                     
        #i =int(i%len(phi))
        #j =int(j%len(theta))
        
    if shuffle:
        random.shuffle(states)
    
    
    return states, goal_state
                

def get_test_states(m, name, goal, nb_states , gamma=None, p=None , shuffle=False):
    states =[]
    
    nb_states = int(np.sqrt(nb_states))
    

    PHI = np.linspace(0, 2*np.pi,nb_states)
    THETA = np.linspace(0, np.pi,nb_states)
    
    PHI, THETA = np.meshgrid(PHI,THETA )
    
    Phi=[]
    Theta=[]
    
    for i in range(len(PHI)):
        for j in range(len(THETA)): 
            phi = PHI[i,j]
            theta = THETA[i,j]
            
            states.append(( state_preparation(m, name, gamma, p, phi,theta),
                            state_preparation(m, goal, gamma, p,phi,theta)))
            
            Phi.append(phi)
            Theta.append(theta)
            
            
    if shuffle:
        random.shuffle(states)
    return states, Phi, Theta

    
def get_circuits(m,name):
    if name=="bit_flip_channel":
        circ1=QuantumCircuit(m, name="Bit Flip Channel")
        circ1=circ1.to_instruction()
        
        circ2=QuantumCircuit(m, name="Bit Flip to First")
        circ2.x(0)
        circ2=circ2.to_instruction()
        
        circ3=QuantumCircuit(m, name="Bit Flip to Second")
        circ3.x(1)
        circ3=circ3.to_instruction()
        
        circ4=QuantumCircuit(m, name="Bit Flip to Third")
        circ4.x(2)
        circ4=circ4.to_instruction()
        
        gates=[circ1,circ2,circ3,circ4]
    return gates

#------------------------------------------------------------------------------------------------------------------------

def state_preparation(m, name, gamma, p, phi = 0, theta=0): 
    # m : numberof qubits 
    # name : name of the state we want 
    # p : proba associated with noise
    # phi : angle of rotation on phi-GHZ
        
    circ = QuantumCircuit(m, name = 'State prep')
    
    if name == 'GHZ':
        circ.h(0)
        for k in range(1,m):
            circ.cx(0,k)
            
            
    elif name == 'Noise_Amplitude_Dumping':
        a = np.cos(theta/2)
        b = np.sin(theta/2)*np.exp((1.0j)*phi)
        
        state0000 = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        state1111 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
        state1100 = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        state0011 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
        state0111 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        state0100 = np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        state1011 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
        state1000 = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        state1101 = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
        state0001 = np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        state1110 = np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        state0010 = np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        
        prob_1 = np.random.rand(1)
        prob_2 = np.random.randint(0,m)
        
        
        if(prob_1 >= p):
            state0 = a*(1/np.sqrt(2)*(state0000 + (1-gamma)**2* state1111)) + b *(1-gamma)*(1/np.sqrt(2))*(state0011 + state1100)
            state0 = state0/LA.norm(state0)
            circ.initialize(state0)
        
        else:
            if(prob_2==0):
                state1 = np.sqrt((gamma*(1-gamma))/2)*(a*(1-gamma)*state0111 + b*state0100)
                state1 = state1/LA.norm(state1)
                circ.initialize(state1)
            
            elif(prob_2==1):
                state2 = np.sqrt((gamma*(1-gamma))/2)*(a*(1-gamma)*state1011 + b*state1000)
                state2 = state2/LA.norm(state2)
                circ.initialize(state2)
                
            elif(prob_2==2):
                state3 = np.sqrt((gamma*(1-gamma))/2)*(a*(1-gamma)*state1101 + b*state0001)
                state3 = state3/LA.norm(state3)
                circ.initialize(state3)
                
            else:
                state4 = np.sqrt((gamma*(1-gamma))/2)*(a*(1-gamma)*state1110 + b*state0010)
                state4 = state4/LA.norm(state4)
                circ.initialize(state4) 
                
    elif name == 'Amplitude_Dumping':
        initial_state = [np.cos(theta/2), np.exp((1.0j)*phi)*np.sin(theta/2)]
        circ.initialize(initial_state, 2)
        circ.ry(np.pi/2,0)
        circ.cx(2,3)
        circ.barrier()
        for i in range(1,4,1):
            circ.cx(0,i)  
            
    elif name == '1q_state':
        if m==1:
            circ.initialize( [ np.cos(theta/2), np.sin(theta/2)*np.exp(1.0j * phi) ] )
    
    elif name == 'enc':
        
        matrix = np.zeros((2,2), dtype=complex) 
        matrix[0][0]=np.cos(theta/2)
        matrix[1][0]=np.sin(theta/2)
        matrix[1][1]=np.cos(theta/2)
        matrix[0][1]=-np.sin(theta/2)
        circ.unitary(Operator(matrix), range(1))
        
        
        matrix1 = np.zeros((2,2),dtype=complex) 
        matrix1[0][0]=1
        matrix1[1][0]=0
        matrix1[1][1]=np.exp((1.0j)*phi)
        matrix1[0][1]=0
        
        circ.unitary(Operator(matrix1), range(1) )

        for i in range(1,m,1):
            circ.cx(0,i)

    elif name == 'enc_bit_flip':
        
        matrix = np.zeros((2,2), dtype=complex) 
        matrix[0][0]=np.cos(theta/2)
        matrix[1][0]=np.sin(theta/2)
        matrix[1][1]=np.cos(theta/2)
        matrix[0][1]=-np.sin(theta/2)
        circ.unitary(Operator(matrix), range(1))
        
        
        matrix1 = np.zeros((2,2),dtype=complex) 
        matrix1[0][0]=1
        matrix1[1][0]=0
        matrix1[1][1]=np.exp((1.0j)*phi)
        matrix1[0][1]=0
        
        circ.unitary(Operator(matrix1), range(1) )

        for i in range(1,m,1):
            circ.cx(0,i)
            
        circ.barrier()
        
        prob_1 = np.random.rand(1)
        prob_2 = np.random.randint(0,m)
        
        if prob_1 <= p: # flips each bit with proba p
            circ.x(prob_2)

    elif name == 'enc_bit_flip_all':
        
        matrix = np.zeros((2,2), dtype=complex) 
        matrix[0][0]=np.cos(theta/2)
        matrix[1][0]=np.sin(theta/2)
        matrix[1][1]=np.cos(theta/2)
        matrix[0][1]=-np.sin(theta/2)
        circ.unitary(Operator(matrix), range(1))
        
        
        matrix1 = np.zeros((2,2),dtype=complex) 
        matrix1[0][0]=1
        matrix1[1][0]=0
        matrix1[1][1]=np.exp((1.0j)*phi)
        matrix1[0][1]=0
        
        circ.unitary(Operator(matrix1), range(1) )

        for i in range(1,m,1):
            circ.cx(0,i)
            
        circ.barrier()
        
        prob_1 = np.random.rand(m)
        for k in range(0,m):            
            if prob_1[k] <= p: # flips each bit with proba p
                circ.x(k)
            
    elif name == 'random_state':
        circ = QuantumCircuit(m)   
        circ.initialize(random_statevector(2**m).data)
        
        
    elif name == 'state_0':
        pass
    
    elif name == 'state_1':
        circ.x(0)
        for k in range(1,m):
            circ.cx(0,k)    
                  
    elif name == 'noisy_GHZ_bitflip':
        prob = np.random.rand(m)
        circ.h(0)
        for k in range(1,m):
            circ.cx(0,k)
            if prob[k] <= p: # flips each bit with proba p
                circ.x(k)
        if prob[0] <= p:
            circ.x(0)
            
    
    elif name == 'noisy_GHZ_QDC':
        probas = [1 - 3*p/4, p/4, p/4, p/4]
        gate_inds = np.random.choice(np.arange(4), size = m, p = probas)
        circ.h(0)
        for k in range(1,m):
            circ.cx(0,k)
            if gate_inds[k] == 1:
                circ.x(k)
            elif gate_inds[k] == 2:
                circ.y(k)
            elif gate_inds[k] == 3:
                circ.z(k)
        if gate_inds[0] == 1:
            circ.x(0)
        elif gate_inds[0] == 2:
            circ.y(0)
        elif gate_inds[0] == 3:
            circ.z(0)
            
    elif name == 'noisy_-GHZ_QDC':
        probas = [1 - 3*p/4, p/4, p/4, p/4]
        gate_inds = np.random.choice(np.arange(4), size = m, p = probas)
        circ.x(0)
        circ.h(0)
        for k in range(1,m):
            circ.cx(0,k)
            if gate_inds[k] == 1:
                circ.x(k)
            elif gate_inds[k] == 2:
                circ.y(k)
            elif gate_inds[k] == 3:
                circ.z(k)
        if gate_inds[0] == 1:
            circ.x(0)
        elif gate_inds[0] == 2:
            circ.y(0)
        elif gate_inds[0] == 3:
            circ.z(0)
            
    elif name == 'noisy_phi-GHZ_QDC':
        probas = [1 - 3*p/4, p/4, p/4, p/4]
        gate_inds = np.random.choice(np.arange(4), size = m, p = probas)
        circ.h(0)
        circ.p(phi,0)
        
        for k in range(1,m):
            circ.cx(0,k)
            if gate_inds[k] == 1:
                circ.x(k)
            elif gate_inds[k] == 2:
                circ.y(k)
            elif gate_inds[k] == 3:
                circ.z(k)
        if gate_inds[0] == 1:
            circ.x(0)
        elif gate_inds[0] == 2:
            circ.y(0)
        elif gate_inds[0] == 3:
            circ.z(0)
    
    elif name =='noisy_phi-GHZ_bitflip':
        prob = np.random.rand(m)
        circ.h(0)
        circ.p(phi,0)
        for k in range(1,m):
            circ.cx(0,k)
            if prob[k] <= p: # flips each bit with proba p
                circ.x(k)
        if prob[0] <= p:
            circ.x(0)
        
    
    elif name == 'rigged_QDC': # QDC where 1st and 2nd qubits have different probas
        probas_rigged = [1-p, p/2, p/2, 0]
        probas_rigged2 = [1 - 29*p/30, 2*p/5, 2*p/5, p/6]
        probas = [1 - 3*p/4, p/4, p/4, p/4]
        gate_inds = np.random.choice(np.arange(4), size = m - 1, p = probas)
        gate_inds_r = np.random.choice(np.arange(4), p = probas_rigged)
        gate_inds_r2 = np.random.choice(np.arange(4), p = probas_rigged2)
        circ.h(0)
        circ.cx(0,1)
        if gate_inds_r2 == 1:
            circ.x(1)
        elif gate_inds_r2 == 2:
            circ.y(1)
        elif gate_inds_r2 == 3:
            circ.z(1)
        for k in range(2,m):
            circ.cx(0,k)
            if gate_inds[k-1] == 1:
                circ.x(k)
            elif gate_inds[k-1] == 2:
                circ.y(k)
            elif gate_inds[k-1] == 3:
                circ.z(k)
        if gate_inds_r == 1:
            circ.x(0)
        elif gate_inds_r == 2:
            circ.y(0)
        elif gate_inds_r == 3:
            circ.z(0)
    
    elif name == 'rigged_-QDC': # QDC where 1st and 2nd qubits have different probas
        probas_rigged = [1-p, p/2, p/2, 0]
        probas_rigged2 = [1 - 29*p/30, 2*p/5, 2*p/5, p/6]
        probas = [1 - 3*p/4, p/4, p/4, p/4]
        gate_inds = np.random.choice(np.arange(4), size = m - 1, p = probas)
        gate_inds_r = np.random.choice(np.arange(4), p = probas_rigged)
        gate_inds_r2 = np.random.choice(np.arange(4), p = probas_rigged2)
        circ.x(0)
        circ.h(0)
        circ.cx(0,1)
        if gate_inds_r2 == 1:
            circ.x(1)
        elif gate_inds_r2 == 2:
            circ.y(1)
        elif gate_inds_r2 == 3:
            circ.z(1)
        for k in range(2,m):
            circ.cx(0,k)
            if gate_inds[k-1] == 1:
                circ.x(k)
            elif gate_inds[k-1] == 2:
                circ.y(k)
            elif gate_inds[k-1] == 3:
                circ.z(k)
        if gate_inds_r == 1:
            circ.x(0)
        elif gate_inds_r == 2:
            circ.y(0)
        elif gate_inds_r == 3:
            circ.z(0)
    else:
        raise ValueError('Unrecognized name.')
            
    return circ
