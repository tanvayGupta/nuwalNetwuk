import numpy as np
import time

start = time.time()
ALPHA = 0.08
MOMENTUM=1
EPOCHS=30
layers = [2,3,1]

#I would like to make a 2,3,1 neural network

class inputNode:

    def __init__(self, value, vertical,weights):
        self.alpha = ALPHA
        self.weights = np.array(weights)
        self.value = value
        self.vertical = vertical
        # self.outputs = self.value * self.weights
        # print(f"I am outputs {self.vertical} {self.outputs}")

    def get_weight(self,vertical):
        return self.weights[vertical]
    
    def set_value(self,value):
        self.value=value

    
class Node:

    def __init__(self, prev, vertical, weights):
        self.alpha = ALPHA
        self.weights= np.array(weights)
        self.prev = prev
        self.vertical = vertical
        self.delta=None
        # self.sigmoid_activation_function()

    def value_calc(self):  #Gets Z, the weighted sum
        values = np.array([node.value for node in self.prev])
        weights = np.array([node.weights[self.vertical] for node in self.prev])
        # print(values, "v")
        # print(weights, "w" )
        self.weighted_sum = np.dot(values,weights)
        # print(self.weighted_sum)

    def get_weight(self,vertical):
        return self.weights[vertical]
    
    def sigmoid_activation_function(self): #Gets the value of when we apply activation function on that z
        self.value_calc()
        self.value = 1/(1 + np.exp(-self.weighted_sum))
        # self.outputs = self.value * self.weights
        self.sigmoidDerivative = self.value * (1 - self.value)
    
    def deltaHidden(self, delta):
        self.delta = delta * self.weights[0] * self.sigmoidDerivative
        return self.delta
    
    def setDelta(self,delta):
        self.delta=delta

    def backpropagation(self):
        for node in self.prev:
            self.current_update = (self.value * self.delta * self.alpha)
            new_weight = ((node.weights[self.vertical]) * MOMENTUM) + self.current_update
            node.weights[self.vertical] = new_weight

node11 = inputNode(0, 0, [1.1, 1.2, 1.3]) #random values but easier for me to understand
node12 = inputNode(1, 1, [1.1, 1.2, 1.3])
node21 = Node([node11, node12], 0, [1])
node22 = Node([node11, node12], 1, [0.5])
node23 = Node([node11, node12], 2, [0.5])
node31 = Node([node21, node22, node23], 0, [1])

def main_loop(iterations,ideal):
    for _ in range(iterations):
        node21.sigmoid_activation_function()
        node22.sigmoid_activation_function()
        node23.sigmoid_activation_function()
        node31.sigmoid_activation_function()
        error = -ideal + node31.value
        # print(f"error {_}  {error}")
        delta = error * node31.sigmoidDerivative
        node31.setDelta(delta)
        node31.backpropagation()
        # print("node31 backprop succesful {_}")
        node21.deltaHidden(delta)
        node22.deltaHidden(delta)
        node23.deltaHidden(delta)
        node21.backpropagation()
        node22.backpropagation()
        node23.backpropagation()

for _ in range(EPOCHS):
    node11.set_value(0)
    node12.set_value(1)
    main_loop(20,1)
    node11.set_value(1)
    node12.set_value(0)
    main_loop(20,1)
    node11.set_value(0)
    node12.set_value(0)
    main_loop(20,0)
    node11.set_value(1)
    node12.set_value(1)
    main_loop(20,0)


# print(f"node 11 weights {node11.weights}")
# print(f"node 12 weights {node12.weights}")
# print(f"node 21 weights {node21.weights}")
# print(f"node 22 weights {node22.weights}")
# print(f"node 23 weights {node23.weights}")

end = time.time()

print("Runtime:", end - start, "seconds")

errorList=[]
def test(x,y,target):
    node11.set_value(x)
    node12.set_value(y)
    node21.sigmoid_activation_function()
    node22.sigmoid_activation_function()
    node23.sigmoid_activation_function()
    node31.sigmoid_activation_function()
    error = target - node31.value
    print(f"Just one simple test {abs(error)}")
    errorList.append(error)

test(0,0,0)
test(1,0,1)
test(0,1,1)
test(1,1,0)
errorList = np.array(errorList)



