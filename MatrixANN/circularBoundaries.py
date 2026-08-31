# For my next act, I'll try to let neural networks simulate non linear boundaries. Like circles
# So it will get two coordinates. And if it lies within the circle (x-1)^2 + (y-1)^2 = 0.5 ** 2
# So radius is 0.5 and the circle center is shifted. Yummy

#But, i'll also test something here
# Basically, if I have a boundary, then no matter what point i give, it will get it right. So I will train it on values from -2,2 for both x and y

from loggerCircle import save_results
import numpy as np
import time
start = time.time()

END_LAYER = 3
ALPHA = 0.1
ITERATIONS = 50000

INPUT_LAYER_NEURONS = 2
HIDDEN_LAYER_1_NEURONS = 4
HIDDEN_LAYER_2_NEURONS = 4
OUTPUT_LAYER_NEURONS = 1
TOTAL_HIDDEN_LAYERS = 3 # treat the output layer as a hidden layer
inputs = (np.random.rand(1,INPUT_LAYER_NEURONS) * 1.7)
error = 0

RADIUS = 0.6
OFF_X = 1
OFF_Y = 1

#thats how I can create ragged arrays...
values = np.empty(TOTAL_HIDDEN_LAYERS+1,dtype=object)

values[0] = inputs
values[1] = np.ones((1, HIDDEN_LAYER_1_NEURONS))
values[2] = np.ones((1, HIDDEN_LAYER_2_NEURONS))
values[3] = np.ones((1, OUTPUT_LAYER_NEURONS))

weights = np.empty(TOTAL_HIDDEN_LAYERS,dtype=object)
weights[0] = np.clip(np.random.randn(INPUT_LAYER_NEURONS,HIDDEN_LAYER_1_NEURONS), -1.5, 1.5)
weights[1] = np.clip(np.random.randn(HIDDEN_LAYER_1_NEURONS,HIDDEN_LAYER_2_NEURONS), -1.5, 1.5)
weights[2] = np.clip(np.random.randn(HIDDEN_LAYER_2_NEURONS,OUTPUT_LAYER_NEURONS), -1.5, 1.5)

wgradients = np.empty(TOTAL_HIDDEN_LAYERS,dtype=object)
wgradients[0] = np.ones((INPUT_LAYER_NEURONS,HIDDEN_LAYER_1_NEURONS))
wgradients[1] = np.ones((HIDDEN_LAYER_1_NEURONS,OUTPUT_LAYER_NEURONS))
# So the first hidden layers has two inputs and three neurons. Then the output layer, got 3 inputs to this and one output. 
# because like the number of neurons in the first hidden layer is the number of neurons that act as input here.

bgradients = np.empty(TOTAL_HIDDEN_LAYERS,dtype=object)
bgradients[0] = np.ones((1,HIDDEN_LAYER_1_NEURONS))
bgradients[1] = np.ones((1,OUTPUT_LAYER_NEURONS))

bias = np.empty(TOTAL_HIDDEN_LAYERS,dtype=object)
bias[0] = np.clip(np.random.randn(1, HIDDEN_LAYER_1_NEURONS), -1.5, 1.5)
bias[1] = np.clip(np.random.randn(1, OUTPUT_LAYER_NEURONS), -1.5, 1.5) #three biases, one for each neuron, one bias for last neuron

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
    bgradients[layer] = np.matmul(bgradients[layer+1], weights[layer+1].T) * sigmoidDerivative(values[layer+1])
    wgradients[layer] = np.matmul(values[layer].T, bgradients[layer])

def gradientDescent(layer):
    bias[layer] = bias[layer] - ALPHA * bgradients[layer]
    weights[layer] = weights[layer] - ALPHA * wgradients[layer].T


def main(i):
    global error
    inputs = (np.random.rand(1,INPUT_LAYER_NEURONS) * 1.7)

    values[0] = inputs
    if ((inputs[0][0] - OFF_X)**2 + (inputs[0][1]- OFF_Y)**2 < RADIUS**2):
        output = 0
    else:
        output = 1
    #0 inside the circle, 1 outside

    forwardProp(0)
    forwardProp(1)

    error = output - values[END_LAYER]
    # print(error[0][0]**2, f"error {i}")
    backProp0(1)
    backProp(0)
    gradientDescent(0)
    gradientDescent(1)
    return

for i in range(ITERATIONS):
    # print(i, "i")
    main(i)

end = time.time()
print("done", ALPHA)
print("Runtime:", end - start, "seconds")

#Now i just save these errors    


np.random.seed(42)
test_inputs = np.random.rand(100,1,2)
mae = 0
TOTAL_TEST_CASES=100
# test_inputs = np.array([[[0.5,1]],[[1,1]],[[1.2,1]],[[0,1]],[[1,1.6]],[[0.6,1]],[[1.59,1]],[[1,0.41]]])
for i in range(TOTAL_TEST_CASES):
    #The final four values are on the exact border. This is so fun. I had used < 0.25. SO basically the ones on the borders must come out as one
    #It will be fun to watch these last 4

    values[0] = test_inputs[i]
    forwardProp(0)
    forwardProp(1)

    if ((values[0][0][0] - OFF_X)**2 + (values[0][0][1]- OFF_Y)**2 < RADIUS**2):
        output = 0
    else:
        output = 1

    # print(output)
    error = output - values[END_LAYER]

    mae += abs(error[0][0])
    # print(round(error[0][0],2))

save_results(
    mae/TOTAL_TEST_CASES,
    ITERATIONS,
    ALPHA,
    HIDDEN_LAYER_1_NEURONS
)



