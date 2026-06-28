import numpy as np
ALPHA = 0.01
layers = [1,5,5,1]

#I would like to make a 1,5,5,1 neural network

class inputNode:

    def __init__(self, value, vertical):
        self.alpha = ALPHA
        self.weights = np.array([0.5,0.5,0.5,0.5,0.5])
        self.value = value
        self.vertical = vertical
        self.outputs = self.value * self.weights
        # print(f"I am outputs {self.vertical} {self.outputs}")

    def get_weight(self,vertical):
        return self.weights[vertical]
    
class Node:

    def __init__(self, prev, vertical):
        self.alpha = ALPHA
        self.weights = np.array([0.5,0.5,0.5,0.5,0.5])
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
    
    def partial_derivative(self):
        return self.value * (1 - self.value)
    
    def delta(self, target):
        return (target - self.value) * self.partial_derivative()
    

    
    
    

node11 = inputNode(0.3, 0)
node21 = Node([node11], 0)
node22 = Node([node11],1)
node23 = Node([node11],2)
node24 = Node([node11],3)
node25 = Node([node11],4)
node2 = [node21,node22,node23,node24,node25]
node31 = Node(node2,0)
node32 = Node(node2,1)
node33 = Node(node2,2)
node34 = Node(node2,3)
node35 = Node(node2,4)
node3 = [node31,node32,node33,node34,node35]
node41 = Node(node3,0)

nodes = [node11] + node2 + node3 + [node41]


