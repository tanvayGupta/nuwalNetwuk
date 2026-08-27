# Let's try the same XOR problem. But this time using Matrices
# I will still use sigmoid activation function

import numpy as np

END_LAYER = 2

inputOutputMap = {
[0,0] : 0, 
[0,1]: 1, 
[1,0]: 1, 
[1,1] : 0} #Input Dictionary

inputs = [[0,0],[0,1],[1,0],[1,1]]
outputs = [0,1,1,0]

randomizer = np.random.randint(0,4)

values = np.array(inputs[randomizer],
np.ones(1,3),
np.ones(1,1))

weights = np.array(np.random.randn(2,3),
np.random.randn(3,1)) 
# #So the first hidden layers has two inputs and three neurons. Then the output layer, got 3 inputs to this and one output. 
# because like the number of neurons in the first hidden layer is the number of neurons that act as input here.

bias = np.array(np.random.randn(1,3), 
 np.random.randn(1,1) ) #three biases, one for each neuron, one bias for last neuron

def sigmoidActivation(x):
    return (1/(1+np.exp(-x))) #Literally Sigmoid Function

def sigmoidDerivative(x): #Returns sigmoid derivative of x
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid * (1 - sigmoid)

def forwardProp(layer):
    z = np.matmul(values[layer],weights[layer]) + bias[layer]
    # activated = np.vectorize(sigmoidActivation)(z)  SO basically the exponential is already vecotrized, so I think the following line will work
    activated = sigmoidActivation(z)

    values[layer + 1] = activated

    return

def lossCalculation(output):
    diff = output - int(values[END_LAYER])
    #I will use MSE

    loss = (diff ** 2) 


def main():
    randomizer = np.random.randint(0,4)

    values[0] = inputs[randomizer]
    output = outputs[randomizer]

    forwardProp(0)
    forwardProp(1)







