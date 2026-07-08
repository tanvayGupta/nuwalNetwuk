#The first xor.py was absolute shit, I dont think I ever understood bakcpropagation properly, let's try again

import numpy as np
import time

start = time.time()

ALPHA = 0.08
EPOCHS = 1000
# BIAS_ALPHA = ALPHA/2
error = 0
layers = [2,3,1] #Nodes in layers in respective orders, iput, hidden and output

class inputNode:
    def __init__(self,value,vertical):
        self.activated_value = value
        self.vertical = vertical


    def set_value(self,value):
        self.activated_value = value

class Node:
    # layer is the hidden layer number, so the first hidden layer will have layer = 1, second hidden layer as layer = 2, so on and so forth...
    def __init__(self,prev,layer,vertical): 
        self.prevNodes = prev
        self.vertical = vertical
        self.weights = np.random.randn(layers[layer-1])
        self.activated_value = None
        self.bias = 0.5
        self.delta = None
        self.nextNodes = None
        self.delta = None
        
    def value_calc(self):
        self.prev_values = [node.activated_value for node in self.prevNodes]
        self.weighted_sum = np.dot(self.weights,self.prev_values) + self.bias
        self.activated_value = 1/(1 + np.exp(-self.weighted_sum)) #Sigmoid activation fxn

    def delta_calc(self):
        a = self.activated_value
        if self.nextNodes == None:
            self.delta = 2 * error * a * (1 - a)
        else:
            delta_next = [node.delta for node in self.nextNodes]
            correspondingNextNodeWeights = [node.weights[self.vertical] for node in self.nextNodes]
            sum_of_Pderivatives = np.dot(delta_next,correspondingNextNodeWeights)
            self.delta = sum_of_Pderivatives * a * (1-a)


    def grad_desc(self):
        update = [(ALPHA * self.delta * node.activated_value) for node in self.prevNodes]
        self.weights = self.weights - update

        self.bias = self.bias - ALPHA*self.delta
        
    
    
    def set_next(self,next):
        self.nextNodes = next


node11 = inputNode(0,0)
node12 = inputNode(0,1)
node21 = Node([node11,node12],1,0)
node22 = Node([node11,node12],1,1)
node23 = Node([node11,node12],1,2)
node31 = Node([node21,node22,node23],2,0)
node21.set_next([node31])
node22.set_next([node31])
node23.set_next([node31])


def main_loop(iterations,ideal):
    global error
    for _ in range(iterations):
        node21.value_calc()
        node22.value_calc()
        node23.value_calc()
        node31.value_calc()
        error = node31.activated_value - ideal
        node31.delta_calc()
        node21.delta_calc()
        node22.delta_calc()
        node23.delta_calc()
        node31.grad_desc()
        node21.grad_desc()
        node22.grad_desc()
        node23.grad_desc()

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

end = time.time()

print("Runtime:", end - start, "seconds")

errorList=[]
def test(x,y,target):
    node11.set_value(x)
    node12.set_value(y)
    node21.value_calc()
    node22.value_calc()
    node23.value_calc()
    node31.value_calc()
    error = target - node31.activated_value
    print(f"Just one simple test {error}")
    errorList.append(error)

test(0,0,0)
test(1,0,1)
test(0,1,1)
test(1,1,0)
errorList = np.array(errorList)


