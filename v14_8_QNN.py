#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:21:20 2021

@author: tomachache
"""

import numpy as np
from scipy import linalg
import time
import random
from qiskit import *
from qiskit.extensions import UnitaryGate

# Create Pauli matrices (base for unitary matrices used in QNNs)
def create_Pauli():
    I = np.eye(2)
    X = np.array([[0,1], [1,0]])
    Y = 1j * np.array([[0,-1], [1,0]])
    Z = np.array([[1,0], [0,-1]])

    return [I, X, Y, Z]
    

# Create all possible tensor products of m Pauli matrices
def tensor_Paulis(m):
    Pauli = create_Pauli()
    
    def get_tensors(m, U):
        if m == 0:
            return np.array(U)
        else:
            return get_tensors(m-1, [np.kron(u, P) for u in U for P in Pauli])
        
    return get_tensors(m, [np.eye(1)])

        

# Create a QNN
class QNN:
    def __init__(self, M, D, *args): # M : topology of the network ; e.g. M = [4,2,1,2,4]
        self.M = M
        self.D = D
    
        
        self.num_of_layers = len(M)
        

        self.gates = []
        assert D.count("gate")==len(args), "the amount of gates given is not correct"
        for i in args:     
            self.gates.append(i)
            
        self.linked_layers = []
        x = 0
        for i in D:
            if i !="trainlayer" and i!="gate" and i.rstrip(i[-1]) != "conjugatelayer":
                raise ValueError("D can contain only the strings: trainlayer, gate, conjugatelayer + a number")
            
            
            if (i.rstrip(i[-1]) == "conjugatelayer"):
                self.linked_layers.append( (int( i[-1] ) ,x))  
                
            if i!="gate":
                x+=1
                
        # Compute the width of the network (nb of qubits used in subroutine 2)
        w = 0
        for i in range(self.num_of_layers - 1):
            w = max(w, self.M[i] + self.M[i+1])
        self.W = w # width
        
        self.num_qubits = 1 + self.M[-1] + self.W # total nb of qubits required
        
        # Creating K (vector of all coefficients to learn)
        
        # First compute the nb of coeffs needed
        self.nb_coeffs = 0
        
        # Create and store the "basis" of gates' matrices (all the different tensor products of length M[i-1]+1 of Pauli's)
        # These basis are then multiplied with coeffs K, summed, and transform in a unitary U = e^{iS} then into a gate
        self.mat_basis = []
        
        
        self.gates_coord = []
        count_gate = 0
        pos_gate = 0
        for i in range(1, self.num_of_layers ):
            if D[i+count_gate]=='gate':
                self.gates_coord.append( pos_gate + 2 + i + 2 * count_gate )
                count_gate+=1
                
            if D[i+count_gate]=='trainlayer':
                self.nb_coeffs += self.M[i] * 4**(self.M[i-1]+1)
                self.mat_basis.append(tensor_Paulis(self.M[i-1]+1))
            
            if D[i+count_gate].rstrip(D[i+count_gate][-1])=='conjugatelayer':
                pos_gate += self.M[ int( self.D[i+count_gate][-1]) ] + self.M[i-1]
                self.mat_basis.append(tensor_Paulis(self.M[i-1]+1))
                
                continue
            
            pos_gate += self.M[i]+self.M[i-1]
            
          
            
        #print(self.nb_coeffs)
        # Choose required initialization for K
        #self.K = np.zeros(self.nb_coeffs)           # initialize at 0
        self.K = np.random.rand(self.nb_coeffs)     # initialize with random values in [0,1]
        
        # Choose backend and options
        self.backend = Aer.get_backend('qasm_simulator')
        self.backend_options = {'max_parallel_experiments': 0, 'max_parallel_threads': 0}
        
        # Choose nb of shots (S)
        self.shots = 1000 
        
        self.circ = None # record current circuit used
        self.show_circ = None # record compiled version of circuit (for plotting purpose)
        
        self.create_circuit() # create circuit


    def set_gates(self, *args):
        
        assert D.count("gate")==len(args), "the amount of gates given is not correct"
        
        for gate in args:
            self.gates.append(gate)
            
        return

    def set_K(self, K): # set pre-learned K
        
        assert len(K) == self.nb_coeffs
        self.K = K
        self.create_circuit()
        return
        
    def subroutine_2(self):  #sub_2 builds the network
        circ = QuantumCircuit(self.W, name = 'Quantum AE')
        
        free_qubits = self.M[0] # will allow to see which qubits are "free" for the next layer
        m_new = np.array(range(free_qubits)) # m_new is the list of "free" qubits
    
        cpt = 0
        count_gate = 0
        for layer in range(1, self.num_of_layers): 
            
        
            if  self.D[layer+count_gate]=="gate":
                circ.append( self.gates[count_gate][0], list(m_new)  )
                circ.barrier()
                count_gate+=1
           
            ########### conjugatelayer ############ 
            cj_free_qubits = self.M[0]
            cj_m_new = np.array(range(cj_free_qubits))
            cj_cpt = 0
            if self.D[ layer+count_gate ].rstrip( self.D[layer+count_gate][-1] ) =="conjugatelayer":                
                assert self.M[ int(self.D[ layer+count_gate ][-1]) ] + self.M[ int(self.D[ layer+count_gate ][-1]) - 1 ]  == self.M[ layer ] + self.M[ layer - 1 ]
                cj_layer = int(self.D[layer+count_gate][-1])
                
                cj_count_gate = 0
                for i in range(1, cj_layer ):
                    if  self.D[i+cj_count_gate]=="gate":
                        cj_count_gate+=1
                    
                    if self.D[ i+cj_count_gate ].rstrip( self.D[i+cj_count_gate][-1] ) =="conjugatelayer":
                        continue
                    
                    cj_cpt += self.M[i] * 4**(self.M[i-1]+1)
                
                for i in range(1, cj_layer + 1 ):
                    cj_m_old = list(cj_m_new)
                    cj_m_new = np.array(range(cj_free_qubits, cj_free_qubits + self.M[i])) % self.W
                    cj_free_qubits += self.M[i]
                    
                for gate in range(self.M[ cj_layer ]-1, -1, -1):
                    C_U = UnitaryGate(linalg.expm(-1j * np.sum(self.mat_basis[cj_layer-1] * self.K[gate*(4**(self.M[cj_layer-1]+1)) + cj_cpt : (gate+1)*(4**(self.M[cj_layer-1]+1)) + cj_cpt, None, None], axis = 0)),label="$(U_"+str(gate+1)+"^"+str(cj_layer)+")^\dagger$")
                    
                    m_tot = cj_m_old + [cj_m_new[gate]]
                    for i in range( len(m_tot)-1, 0, -1 ):
                        j = m_tot[i]
                        m_tot[i] = m_tot[i-1]
                        m_tot[i-1] = j
  
                    
                    circ.append(C_U, cj_m_old + [cj_m_new[gate]])
                
                m_new = cj_m_old
                free_qubits = ( cj_m_old[-1] + 1 )% self.W
                
                circ.reset( list(cj_m_new) )
                
                if layer < self.num_of_layers - 1: # don't put barrier at the end since there will be one already
                    circ.barrier() # if we want to set a barrier between each layer (not really useful) 
                continue
           
            ########### trainlayer #################
            
            m_old = list(m_new) # m_old is the list of qubits "taken" (i.e. used by the current layer)
            m_new = np.array(range(free_qubits, free_qubits + self.M[layer])) % self.W           
            free_qubits += self.M[layer]
            
            if self.D[layer+count_gate]=="trainlayer":
                for gate in range(self.M[layer]):
                    # Create unitary gates by multipliying K with the basis of gates' matrices
                    # self.mat_basis[i-1] : all the basis matrices for layer i
                    # self.K[j*(4**(self.M[i-1]+1)) + cpt : (j+1)*(4**(self.M[i-1]+1)) + cpt] : relevant coeffs
                    # multiply these 2, sum to get a matrix S, perform e^{iS}, then create the gate
                    C_U = UnitaryGate(linalg.expm(1j * np.sum(self.mat_basis[layer-1] * self.K[gate*(4**(self.M[layer-1]+1)) + cpt : (gate+1)*(4**(self.M[layer-1]+1)) + cpt, None, None], axis = 0)),label="$U_"+str(gate+1)+"^"+str(layer)+"$")
                    circ.append(C_U, m_old + [m_new[gate]])
                
                cpt += self.M[layer] * 4**(self.M[layer-1]+1)
                
                circ.reset(m_old)
                if layer < self.num_of_layers - 1: # don't put barrier at the end since there will be one already
                    circ.barrier() # if we want to set a barrier between each layer (not really useful) 
                # if we put barriers be careful as the positions of gates in the fit function will change
                
            # if we put barriers be careful as the positions of gates in the fit function will change
                
        circ.draw()
        return circ, m_new # that way we have the final relevant qubits
        
    def create_circuit(self): # create the circuit corresponding to the network
        circ = QuantumCircuit(self.num_qubits, 1)
        
        # Initialize empty states preparation
        input_state = QuantumCircuit(self.M[0], name = 'Input State')
        target_state = QuantumCircuit(self.M[-1], name = 'Target State')
        circ.append(input_state.to_instruction(), range(1 + self.M[-1], 1 + self.M[-1] + self.M[0]))
        circ.append(target_state.to_instruction(), range(1, 1 + self.M[-1]))
        circ.barrier()
        
        # Subroutine 2
        sub, out_qubits = self.subroutine_2() 
        circ.append(sub.to_instruction(), range(1 + self.M[-1], self.num_qubits))
        circ.barrier()
        
        # Subroutine 1
        circ.h(0)
        for k in range(self.M[-1]):
            circ.cswap(0, k + 1, 1 + self.M[-1] + out_qubits[k])
        circ.h(0)

        circ.barrier()
        circ.measure(0,0)
        self.show_circ = circ
        self.circ = circ.decompose()
        
    def return_circ(self):
        sub, out_qubits = self.subroutine_2() 
        #sub = sub.decompose()
        return sub
        
        
    def return_circ(self):
        sub, out_qubits = self.subroutine_2() 
        sub = sub.decompose()
        return sub
    
    def draw(self, *args):
        for i in args:
            self.circ.draw(i)
            return 
        self.circ.draw("mpl")
        
    def run(self): # run the circuit and output the fidelity
        result = execute(self.circ, self.backend, shots = self.shots).result()
        #result = execute(self.circ, self.backend, shots = self.shots, backend_options = self.backend_options).result()
        return (2 * result.get_counts(0)['0']/self.shots - 1)
    
    def run_multiple_circs(self, circs, batch_size, train = True): # run multiple circuits in batches
        n = len(circs)
        #result = execute(circs, self.backend, shots = self.shots).result()
        result = execute(circs, self.backend, shots = self.shots, backend_options = self.backend_options).result() 

        if train:
            assert n == batch_size * (self.nb_coeffs + 1)
            return np.mean([[2 * result.get_counts(i * batch_size + j)['0']/self.shots - 1 for j in range(batch_size)] for i in range(self.nb_coeffs + 1)], axis = 1) # nb_coeff + 1 to account for the original cost
        else: # we're calling this function in the test phase
            assert n == batch_size
            return np.array([2 * result.get_counts(i)['0']/self.shots - 1 for i in range(n)])
        
    def return_K(self):
        return self.K
    
    def set_nb_shots(self,s):
        self.shots = s

    def optimizers(self, name , lr ,  beta1 =0.9 , beta2 = 0.999, eps = 10**(-8), it=0): 
        global t 
        
        t = it
        
        def default(delta , eta = lr ):
            self.K += eta * delta
            
        def RMSprop(delta , beta = 0.9, eta = lr):
            global m
            m = beta * m + (1-beta)*delta**2
            
            self.K += eta/(np.sqrt(m)) * delta
            
        def momentum(delta , mu=0.9, eta = lr ):
            # add momentum
            global velocity 
            velocity = mu * velocity + eta * delta
            
            self.K += velocity
            
        def Adamax(delta ,  eta  = lr , beta_1 = beta1 , beta_2 = beta2  ):
            global m,v,t
            
            m = beta_1 * m + (1 - beta_1) * delta
            v = np.maximum( beta_2 * v , np.absolute(delta) + 10**(-8))
            t+=1
            
            self.K += eta/v * m/(1-beta_1**t)
            
        def Adam(delta ,  eta  = lr , beta_1 = beta1 , beta_2 = beta2 , eps = eps):  
            global m,v,t
            
            m = beta_1 * m + (1 - beta_1) * delta
            v = beta_2 * v + (1 - beta_2) * (delta**2)
            t += 1
            
            self.K += eta * (m/(1 - beta_1**t)) / (np.sqrt(v/(1 - beta_2**t)) + eps)
        
        def Nadam(delta , eta = lr , beta_1 = beta1 , beta_2 = beta2 , eps = eps ):
            global m,v,t
                 
            m = beta_1 * m + (1 - beta_1) * delta
            v = beta_2 * v + (1 - beta_2) * (delta**2)
            t+=1
            
            m_hat = m/(1-beta_1**t)
            v_hat = v/(1-beta_2**t)
            
            self.K += eta/(np.sqrt(v_hat)+eps) * beta_1 * m_hat + ( (1 - beta_1 ) * delta )/( 1 - beta_1**t ) 
            

        global m,v,velocity
        
        if (name == 'default'):
            return default
        
        elif (name == 'Adam'):                 
            m = np.zeros(self.nb_coeffs)
            v = np.zeros(self.nb_coeffs)
            return Adam
      
        elif (name == 'momentum'):
            velocity = np.zeros(self.nb_coeffs)
            
            return momentum 
        
        elif (name == 'Nadam'):
            m = np.zeros(self.nb_coeffs)
            v = np.zeros(self.nb_coeffs) 

            return Nadam
        
        elif (name == 'Adamax'):                 
            m = np.zeros(self.nb_coeffs)
            v = np.zeros(self.nb_coeffs)
        
            return Adam
        
        elif (name == 'RMSprop'):
             m = np.zeros(self.nb_coeffs)
             
             return RMSprop
         
        else:
            raise ValueError('Unrecognized optimizer name')


    def fit(self, training_states, nb_epochs, epsilon, optimizer,  p, batch_size_train = None, batch_size_test = None,
            goal_state = None, validation_states = None  , 
            run_in_batch = False, save_best_val_fid = None, nb_shots=1000 , early_stopping=True , shuffle_batch = False , name=None): 
        
        # train the network
        # training_states : list of pairs of circuit preparing input/target states
        # nb_epochs : nb of rounds 
        # epsilon : step used for the derivative approximation
        # eta : learning rate
        # batch_size_train/test : nb of states considered at each epoch for the training/testing at the end
        # goal_state : the 'goal' state we want, when we use the network as an AE (just to see if the network is improving at every step)
        # For instance, for the AE considered, the training_states will be pairs of noisy GHZ while the goal state is the true GHZ 
        # validation_states : states similar to input states (come from same noisy distribution), used for validation
        # use_Adam/momentum : whether to use Adam gradient ascent or SGD with momentum
        # run_in_batch : whether to run all the circuits in batches (may speed up computation, but not realistic physically)
        # save_best_val_fid : whether to save weights corresponding to maximum fidelity on validation states
        
        
        self.set_nb_shots(nb_shots)
        
        if validation_states is not None:
            assert goal_state is not None # validation states are used with the goal state
            
        
        N = len(training_states) # nb of training pairs
        
        if batch_size_train is None:
            batch_size_train = N
        
        if batch_size_train > N: 
            raise ValueError("batchsize is larger than the number of training states")

            
        C = [] # will save the cost (i.e. fidelity) after every epoch
        C_val = [] # same for validation states
        for epoch in range(nb_epochs):
            start = time.time()
            
            #if ran_states_in_epoch == True:
            #    training_states = get_train_states(self.M[0],'noisy_phi-GHZ_QDC',None , p , N , random_phi=True)
                
            batch_list = np.random.choice( int(N/batch_size_train), int(N/batch_size_train) , replace = False)
            
            if shuffle_batch:
                states = list(zip(training_states, goal_state))
                random.shuffle(states)
                training_states, goal_state =  zip(*states)
                training_states, goal_state = list(training_states), list(goal_state)
            
            
            for batch_num in batch_list:
                batch = np.arange(batch_size_train * batch_num, batch_size_train * batch_num + batch_size_train) # pick the batch
                
                circs = [] # will be used in case run_in_batch = True
                
                cost = 0
                
                if len(self.gates_coord)>0:
                    #m = np.random.rand( len(batch) )
                    
                    
                    n = []
                    n = [0 for j in range( int(len(batch)*(1-p)) )  ]
                    
                    for i in range(1, len(self.gates[0])):
                        n = n + [i for j in range( int(len(batch) * p /(len(self.gates[0])-1) ))]
                        
                    if len(n)!=len(batch):
                        while( len(n)< len(batch)):
                            n.append(0)
                    
                    random.shuffle(n)
                    
                    
                    #n = np.random.randint(1, len(self.gates[0]) , size = len(batch)) 
                    
                for l, k in enumerate( batch ): # replace the states preparations by the current one
                    # target state is the first gate and input is the second one
                    self.circ.data[1] = (training_states[k][0], self.circ.data[1][1], self.circ.data[1][2])
                    self.circ.data[0] = (training_states[k][1], self.circ.data[0][1], self.circ.data[0][2])

                    for count, coord in enumerate(self.gates_coord):   
                        #if m[l] > p:
                        self.circ.data[ coord ] = (self.gates[count][n[l]] , self.circ.data[coord][1], self.circ.data[coord][2])
                        #else:
                        #    self.circ.data[ coord ] = (self.gates[count][n[l]] , self.circ.data[coord][1], self.circ.data[coord][2])
                
                        
                    if run_in_batch:
                        circs.append(self.circ.copy())
                    else:
                        cost += self.run()
            
                
                cost /= batch_size_train
                
                
                delta = np.zeros(self.nb_coeffs)
                
                
                cp_cjl = [x[0] for x in self.linked_layers]
                cjl = [x[1] for x in self.linked_layers]
                
                cpt = 0
                pos_gate = 0 # record position of the gate we are currently modifying
                count_gate = 0
                for i in range(1, self.num_of_layers):  
                
                    if self.D[i+count_gate]=='gate':
                        count_gate+=1
                        
                    if self.D[ i+count_gate ].rstrip( self.D[i+count_gate][-1] ) =="conjugatelayer": 
                        pos_gate += self.M[int(self.D[i+count_gate][-1])]+self.M[i-1]
                        continue
                        
                    for j in range(self.M[i]): # nb of unitary gate at that layer
                        for v in range(4**(self.M[i-1]+1)): # nb of coeff of each gate of the current layer
                            # K is the vector of all coefficients (i.e. coeffs of all unitary matrices)
                            # K[v + j*(4**(self.M[i-1]+1)) + sum_{s = 1}^{i-1} (self.M[i] * 4**(self.M[i-1]+1))] is the v-th coeff of the j-th Unitary of the i-th layer
                            # Total nb of operations for backprop : nb_coeffs = sum_{i=1}^L M[i] * 4**(M[i-1]+1)
                            
                            self.K[v + j*(4**(self.M[i-1]+1)) + cpt] += epsilon
                            
                            # Instead of updating whole circuit : pop 1 gate then recreate it
                            # j-th gate (from 0 to M[i]-1) of the i-th (from 1 to L-1) layer is at position : j + sum_{s = 1}^{i-1} M[s]
                            # (without resets nor barriers)
                            # We have M[i-1] resets at the end of layer i (from 1 to L-1)
                            # So (without barriers) j-th gate of the i-th layer is at position : j + sum_{s = 1}^{i-1} (M[s]+M[s-1])
                            # Create a counter to record the gate's position : pos_gate = sum_{s = 1}^{i-1} (M[s]+M[s-1])
                            # Add + 2 to the position to account for state preparations
                            # + i for barriers
                            
                        
                            self.circ.data[j + pos_gate + 2 + i + 2*count_gate][0].params[0] = linalg.expm(1j * np.sum(self.mat_basis[i-1] * self.K[j*(4**(self.M[i-1]+1)) + cpt : (j+1)*(4**(self.M[i-1]+1)) + cpt, None, None], axis = 0))
                            

                            
                            if cp_cjl.count(i) == 1 :
                                layer = cjl[ cp_cjl.index(i) ]
                                count_gate_ = 0
                                pos_gate_ = 0
                                
                                
                                for i_ in range(1, layer):
                                    if self.D[i_+count_gate_]=='gate':
                                        count_gate_+=1
                                        
                                    if self.D[ i_+count_gate_ ].rstrip( self.D[i_+count_gate_][-1] ) =="conjugatelayer" : 
                                        pos_gate_ += self.M[int(self.D[i_+count_gate_][-1])]+self.M[i_-1]
                                        continue
                                    
                                    pos_gate_ += self.M[i_]+self.M[i_-1]
                                    
                                    
                                if self.D[layer+count_gate_]=='gate':
                                    count_gate_+=1
                                
                                
                                try:
                                    self.circ.data[j + pos_gate_ + 2 + layer + 2*count_gate_][0].params[0] = linalg.expm(-1j * np.sum(self.mat_basis[i-1] * self.K[j*(4**(self.M[i-1]+1)) + cpt : (j+1)*(4**(self.M[i-1]+1)) + cpt, None, None], axis = 0))
                                except:
                                    self.circ.data[j + pos_gate_ + 2 + layer + 2*count_gate_+1][0].params[0] = linalg.expm(-1j * np.sum(self.mat_basis[i-1] * self.K[j*(4**(self.M[i-1]+1)) + cpt : (j+1)*(4**(self.M[i-1]+1)) + cpt, None, None], axis = 0))
                                
                                
                                
                            
                            self.K[v + j*(4**(self.M[i-1]+1)) + cpt] -= epsilon
                            
                            
                            # Compute the new avg cost
                            new_cost = 0
                            for l, k in enumerate( batch ): # insert the state preparations, run the network, and delete them
                                self.circ.data[1] = (training_states[k][0], self.circ.data[1][1], self.circ.data[1][2])
                                self.circ.data[0] = (training_states[k][1], self.circ.data[0][1], self.circ.data[0][2])
                                
                                
                                for count, coord in enumerate(self.gates_coord):   
                                    #if m[l] > p:
                                    self.circ.data[ coord ] = (self.gates[count][n[l]] , self.circ.data[coord][1], self.circ.data[coord][2])
                                    #else:
                                    #    self.circ.data[ coord ] = (self.gates[count][n[l]] , self.circ.data[coord][1], self.circ.data[coord][2])
                                
                                if run_in_batch:
                                    circs.append(self.circ.copy())
                                else:
                                    new_cost +=  self.run()
                                
                            
                                
                            new_cost /= batch_size_train
                            
                            delta[v + j*(4**(self.M[i-1]+1)) + cpt] = (new_cost - cost)/epsilon
                            
                                
                    cpt += self.M[i] * 4**(self.M[i-1]+1)
                    pos_gate += self.M[i] + self.M[i-1]
                
                if run_in_batch:

                    delta_0 = self.run_multiple_circs(circs, batch_size_train)
                    delta = (delta_0[1:] - delta_0[0])/epsilon
                
                
                optimizer( delta = delta )
                
                # Now we re-create the circuit --> maybe faster to just re-run subroutine 2 and only replace this part
                self.create_circuit()
            
            if batch_size_test is None:
                batch_size_test = N
            
            # Do a final run to compute the new cost (for validation purpose)
            batch_test = np.random.choice(N, batch_size_test, replace = False)
            final_cost = 0
            test_circs = []
            
            if validation_states is not None:
                N_val = len(validation_states)
                
            if len(self.gates_coord)>0:
                m = np.random.rand( batch_size_test )
                n = np.random.randint(1, len(self.gates[0]) , size = batch_size_test)
            
            val_cost = 0
            val_circs = []
            for k in batch_test: # insert the state preparations, run the network, and delete them
                self.circ.data[1] = (training_states[k][0], self.circ.data[1][1], self.circ.data[1][2])  
                if goal_state is not None : 
                    self.circ.data[0] = (goal_state[k], self.circ.data[0][1], self.circ.data[0][2])
                else:
                    self.circ.data[0] = (training_states[k][1], self.circ.data[0][1], self.circ.data[0][2])    
                
                for count, coord in enumerate(self.gates_coord):   
                    if m[k] > p:
                        self.circ.data[ coord ] = (self.gates[count][0] , self.circ.data[coord][1], self.circ.data[coord][2])
                    else:
                        self.circ.data[ coord ] = (self.gates[count][n[k]] , self.circ.data[coord][1], self.circ.data[coord][2])
                
                if run_in_batch:
                    test_circs.append(self.circ.copy())
                else:
                    final_cost += self.run()
                    
            if len(self.gates_coord)>0:
                m = np.random.rand( N_val )
                n = np.random.randint(1, len(self.gates[0]) , size = N_val)
                    
            if validation_states is not None:
                for k in range(N_val):
                    self.circ.data[1] = (validation_states[k][0], self.circ.data[1][1], self.circ.data[1][2])  
                    self.circ.data[0] = (validation_states[k][1], self.circ.data[0][1], self.circ.data[0][2])   
                    
                    for count, coord in enumerate(self.gates_coord):   
                        if m[k] > p:
                            self.circ.data[ coord ] = (self.gates[count][0] , self.circ.data[coord][1], self.circ.data[coord][2])
                        else:
                            self.circ.data[ coord ] = (self.gates[count][n[k]] , self.circ.data[coord][1], self.circ.data[coord][2])
                    
                    if run_in_batch:
                        val_circs.append(self.circ.copy())
                    else:
                        val_cost += self.run()
                    
            if run_in_batch:
                final_cost = np.mean(self.run_multiple_circs(test_circs, batch_size_test, train = False))
            else:
                final_cost /= batch_size_test
            C.append(final_cost)
            
            if validation_states is not None:
                if run_in_batch:
                    val_cost = np.mean(self.run_multiple_circs(val_circs, N_val, train = False))
                else:
                    val_cost /= N_val  
                C_val.append(val_cost)
                
                if (save_best_val_fid is not None) and val_cost == np.max(C_val): # save weights corresponding to best epoch on val set
                    best_k =self.K
                    best_epoch = epoch+1
                    c=1
                    
            end = time.time()
            
            if(c==1):
                print('Epoch {} | Fidelity : {:.3f} | Validation Fidelity : {:.3f} | Time : {} s. Kerasetone !!!'.
                      format(epoch + 1, final_cost, val_cost, int(end-start)))
                c=0
                
            elif(c==0):
                print('Epoch {} | Fidelity : {:.3f} | Validation Fidelity : {:.3f} | Time : {} s.'.
                      format(epoch + 1, final_cost, val_cost, int(end-start)))
                
            if (epoch+5)%8==0 or epoch ==0:
                print("Estimated Running Time: {} days, {} hours, {} mins".format(int((end-start)*(nb_epochs-epoch)/(24*60*60)), int(((end-start)*(nb_epochs-epoch)/(60*60))%24), int(((end-start)*(nb_epochs-epoch)/60)%60)))

            if ((epoch+1)%8==0 or epoch ==0 ):
                # save models
                save_best_k ='k ' + save_best_val_fid +'.npy' # insert None if you don't want to save the best validation
                save_val      ='val '+ save_best_val_fid +'.npy'
                save_train    ='train ' + save_best_val_fid +'.npy'
                
                np.save(save_val, np.array(C))
                np.save(save_train, np.array(C_val))
                np.save(save_best_k, best_k)
                print('Model name:'+name)
                
 
            if (early_stopping and (epoch==0)):
                penalty = 0
                tolerence = 10
                
            if (early_stopping and (epoch!=0)):
                
                if (C_val[epoch] < C_val[epoch-1] ):
                    penalty+=1
                else:
                    penalty=0
                
                if (penalty > tolerence):
                    print("Terminated due to Early Stopping")
                    break
                    
            #if (epoch + 1) % 10 == 0: # allows to stop simulations without loosing everything if they are too long
             #   np.save('train_fid.npy', np.array(C))
              #  np.save('val_fid.npy', np.array(C_val))
        
        if save_best_val_fid is not None:
            self.K = best_k
            np.save(save_best_k, best_k)
            init_fid = self.test(validation_states)
            print('Best is saved. Best Epoch: '+str(best_epoch))
            print('Best Val Fidelity : {:.2f}'.format(np.mean(init_fid)))
        
            
        return C, C_val
    
    def test(self, test_states, run_in_batch = False, p = None): # test the network
        # test_states : list of pairs of circuit preparing input/target states
        N = len(test_states)
        
        if len(self.gates_coord)>0:
            m = np.random.rand( N )
            n = np.random.randint(1, len(self.gates[0]) , size = N)
        
        cost = []
        circs = []
        for k in range(N): # insert the state preparations, run the network, and delete them
            self.circ.data[1] = (test_states[k][0], self.circ.data[1][1], self.circ.data[1][2])
            self.circ.data[0] = (test_states[k][1], self.circ.data[0][1], self.circ.data[0][2]) 
            
            for count, coord in enumerate(self.gates_coord):   
                if m[k] > p:
                    self.circ.data[ coord ] = (self.gates[count][0] , self.circ.data[coord][1], self.circ.data[coord][2])
                else:
                    self.circ.data[ coord ] = (self.gates[count][n[k]] , self.circ.data[coord][1], self.circ.data[coord][2])
            
            if run_in_batch:
                circs.append(self.circ.copy())
            else:
                cost += [self.run()]
        if run_in_batch:
            cost = self.run_multiple_circs(circs, N, train = False)
        else:
            cost = np.array(cost)
        return cost
