# Let's try the same XOR problem. But this time using Matrices
# I will still use sigmoid activation function

import numpy as np

END_LAYER = 2
ALPHA = 1
ITERATIONS = 5000

inputs = np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
outputs = [0,1,1,0]
error = 0

randomizer = np.random.randint(0,4)


#thats how I can create ragged arrays...
values = np.empty(3,dtype=object)

values[0] = inputs[randomizer]
values[1] = np.ones((1, 3))
values[2] = np.ones((1, 1))

weights = np.empty(2,dtype=object)
weights[0] = np.random.randn(2,3)
weights[1] = np.random.randn(3,1)

wgradients = np.empty(2,dtype=object)
wgradients[0] = np.ones((2,3))
wgradients[1] = np.ones((3,1))
# #So the first hidden layers has two inputs and three neurons. Then the output layer, got 3 inputs to this and one output. 
# because like the number of neurons in the first hidden layer is the number of neurons that act as input here.

bgradients = np.empty(2,dtype=object)
bgradients[0] = np.ones((1,3))
bgradients[1] = np.ones((1,1))

bias = np.empty(2,dtype=object)
bias[0] = np.random.randn(1, 3)
bias[1] = np.random.randn(1, 1) #three biases, one for each neuron, one bias for last neuron

def sigmoidActivation(x):
    return (1/(1+np.exp(-x))) #Literally Sigmoid Function

def sigmoidDerivative(sigmoid): #Returns sigmoid derivative of x
    # sigmoid = 1/(1+np.exp(-x))
    return sigmoid * (1 - sigmoid)

def forwardProp(layer):
    z = np.matmul(values[layer],weights[layer]) + bias[layer]
    # activated = np.vectorize(sigmoidActivation)(z)  SO basically the exponential is already vecotrized, so I think the following line will work
    activated = sigmoidActivation(z)

    values[layer + 1] = activated
    return

def backProp0(layer):
    global error
    # delL_delyHat = 2 * error
    # delyHat_delSig = sigmoidDerivative(values[layer+1])

    bgradients[layer] = (-2 * error) * (sigmoidDerivative(values[layer+1]))
    wgradients[layer] = np.transpose(np.matmul(values[layer].T, bgradients[layer]))

def backProp(layer): #This also does gradient descent for now
    bgradients[layer] = np.transpose((bgradients[layer+1].T * weights[layer+1])) * sigmoidDerivative(values[layer+1])
    wgradients[layer] = np.transpose(np.matmul(values[layer].T, bgradients[layer]))

def gradientDescent(layer):
    # print(bgradients[layer])
    # print(wgradients[layer])
    bias[layer] = bias[layer] - ALPHA * bgradients[layer]
    weights[layer] = weights[layer] - ALPHA * wgradients[layer].T


def main(i):
    global error
    randomizer = np.random.randint(0,4)

    values[0] = inputs[randomizer]
    output = outputs[randomizer]

    forwardProp(0)
    forwardProp(1)

    error = output - values[END_LAYER]
    print(error[0][0]**2, f"error {i}")
    backProp0(1)
    backProp(0)
    gradientDescent(0)
    gradientDescent(1)
    return

for i in range(ITERATIONS):
    # print(i, "i")
    main(i)