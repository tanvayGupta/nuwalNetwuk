import numpy as np
ALPHA = 0.01
layers = [1,5,5,1]

#I would like to make a 1,5,5,1 neural network

class inputNode:

    def __init__(self, value, vertical):
        self.alpha = ALPHA
        self.weights = np.array([1,2,3,4,5])
        self.value = value
        self.vertical = vertical
        self.outputs = self.value * self.weights
        # print(f"I am outputs {self.vertical} {self.outputs}")

    def get_weight(self,vertical):
        return self.weights[vertical]
    
class Node:

    def __init__(self, prev, vertical):
        self.alpha = ALPHA
        self.weights = np.array([1,2,3])
        self.prev = prev
        self.vertical = vertical
        self.sigmoid_activation_function()

    def value_calc(self):
        values = np.array([node.outputs[self.vertical] for node in self.prev])
        print(f"I am values {values}")
        self.weighted_sum = np.sum(values)
        return self.weighted_sum

    def get_weight(self,vertical):
        return self.weights[vertical]
    
    def sigmoid_activation_function(self):
        self.value_calc()
        self.value = 1/(1 + np.exp(-self.weighted_sum))
        self.outputs = self.value * self.weights
        print(f"This is the output of this node {self.value}")
    
    
    

node10 = inputNode(7, 0)
node21 = Node([node10], 0)
node22 = Node([node10],1)
node23 = Node([node10],2)
node24 = Node([node10],3)
node25 = Node([node10],4)
node31 = Node([node21,node22,node23,node24,node25],0)

print(node31.value)
