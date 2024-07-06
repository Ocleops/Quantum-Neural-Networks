
# Quantum neural networks for the discovery and implementation of quantum error-correcting codes

## Description
This is the sumplementary code of the paper

doi: https://doi.org/10.1063/5.0157940.

For a detailed description of how Dissipative Neural Networks work, please read the following papers:

1. https://doi.org/10.1038/s41467-020-14454-2
2. https://doi.org/10.48550/arXiv.2012.14714

and also watch the youtube video:
https://www.youtube.com/watch?v=_M2GQAknykg&t=1505s


## Files
This repository is comprised by mainly 3 files

[v14_8_QNN](#v14_8_QNN):
- This is the class that containes all the necessary algorithms that are needed for training and testing our models. Detailed explanation of this class can be found in the appendices of the first paper.

[v14_8_State_preparation](#v14_8_State_preparation):
- This class creates the logical qubits to adress specific types of quantum noise such as Amplitude Damping. Also it is responsible for corrupting the logical qubits with noise.

[v14_8_train](#v14_8_train):
- This file is responsible for the training of the model. To do so, 2 lists have to be defined. The first list M, denotes the architecture of the model.
- M = [4,1,4] corresponds to 4 qubits in the input layer, 1 qubit in the latent space and 4 in the output layer. Note that this implementation allows for QNNs with arbitrary architecture, not just Autoencoders
- D = ['trainlayer', 'trainlyer', 'conjugatelayer1'] denotes the type of each layer in the QNN. Note that the first element of the list must always be 'trainlayer'. In the string 'conjugatelayer1', '1' denotes which unitary operator's hermitian conjugate will be used for that layer of the QNN. For example, here, the hermitian conjugate of the unitary operator that connects the input layer with the latent space, will be used to connect the latent space with the output layer.
- The parameter gamma is used in the case we are dealing with Amplitude Damping. It is the probability of a damping event to occur and it has to remain small.
- The parameter cor_prob is the probability for an error to occur in the codeword, depending on the type of noise we are working with. There are no restrictions on the values of this parameter, although higher values will affect the denoising abiliy of the networks.
-  To use the weights of our best performing model for denoising amplitude damping, just uncomment the lines:

```bash
1. #K=np.load('k model = 414_new_state_in, p=0.8, optimazer = default, batch size = None, gm = 0.1, nb = 70, e = 0.1, lr = 0.2.npy')

2. #my_QNN.set_K(K)
```

## Contact 
myronthk@gmail.com
