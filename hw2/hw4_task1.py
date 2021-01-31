#hw4 task1
#2 hidden layers -- 4 neurons + 4 neurons
#1 output layers -- 3 neurons
import math
import random
import numpy as np
from collections import deque

# in this execution len(labels)=3, len(examples)= 150, len(examples[i])/len(example) = 4

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def MSE(y, index):
    mse = 0
    for i in range(len(labels)):
        mse += (t[index][i] - y[i]) * (t[index][i] - y[i]) 
    mse = mse/(len(labels))
    return mse
    
def Propagation(example,index,rate,mode,
                w1,w2,w3):
    
    z1 = [] #z1[1] z1[2] z1[3] inputs of output-layer neurons
    z2 = [] #z2[1] z2[2] z2[3] z2[4] inputs of 2nd hidden-layer neurons
    z3 = [] #z3[1] z3[2] z3[3] z3[4] inputs of 1st hidden-layer neurons
    h1 = [] #h1 outputs of 1st hidden-layer neurons
    h2 = [] #h2 outputs of 2nd hidden-layer neurons
    y = []  #output result of every example
    
    ### forward propagation, to generate output
    for i in range(len(example)):
        res = 0
        for j in range(len(example)):
            res += example[j] * w3[j][i]
        z3.append(res)
        h1.append(sigmoid(z3[i]))
    
    for i in range(len(example)):
        res = 0
        for j in range(len(example)):
            res += h1[j] * w2[j][i]
        z2.append(res)
        h2.append(sigmoid(z2[i]))
    
    for i in range(len(labels)):
        res = 0
        for j in range(len(example)):
            res += h2[j] * w1[j][i]
        z1.append(res)
        y.append(sigmoid(z1[i]))
    
    
    if(mode == 1):
        return MSE(y,index)
    elif(mode == 0):
        ### back propagation for updating weights
        delta1 = [0.0,0.0,0.0]
        delta2 = [0.0,0.0,0.0,0.0]
        delta3 = [0.0,0.0,0.0,0.0]
        
        #calculate delta for each neuron
        for i in range(len(labels)):
            delta1[i] = y[i]*(1-y[i])*(t[index][i]-y[i])
        for j in range(len(example)):
            res = 0
            for i in range(len(labels)):
                res += delta1[i] * w1[j][i]
            delta2[j] = h2[j] * (1-h2[j]) * res  
        for j in range(len(example)):
            res = 0
            for i in range(len(example)):
                res += delta2[i] * w2[j][i]
            delta3[j] = h1[j] * (1-h1[j]) * res
        
        
        #calculate weights for each neuron-neuron or neuron-attribute paths
        for i in range(len(example)):
            for j in range(len(example)):
                w3[j][i] = w3[j][i] + rate* delta3[i]* example[j]
        for i in range(len(example)):
            for j in range(len(example)):
                w2[j][i] = w2[j][i] + rate* delta2[i]* h1[j]
        for i in range(len(labels)):
            for j in range(len(example)):
                w1[j][i] = w1[j][i] + rate* delta1[i]* h2[j]
        
        #weight update finished
    return 0            

def Random_Weights(w1,w2,w3):
    #output weight
    for i in range(len(examples[0])):
        arr = []
        for j in range(len(labels)):
            arr.append(random.uniform(-0.1,0.1))
        w1.append(arr)
    
    #hidden weight 
    for i in range(len(examples[0])):
        arr = []
        for j in range(len(examples[0])):
            arr.append(random.uniform(-0.1,0.1))
        w2.append(arr)
        
    for i in range(len(examples[0])):
        arr = []
        for j in range(len(examples[0])):
            arr.append(random.uniform(-0.1,0.1))
        w3.append(arr)
    
    
    
def Neural_Network(rate):
    epoch = 0
    success = 0
    mse_avg = []
    mse = 0
    
    w1 = []# output weight
    w2 = []# hidden weight 2
    w3 = []# hidden weight 1

    Random_Weights(w1,w2,w3)
    
    while(success == 0):
        mode = 0 # training mode
        for i in range(len(examples)):
            Propagation(examples[i],i, rate,mode,
                        w1,w2,w3)
        mode = 1 # checking mode for mse
        mse = 0 
        for i in range(len(examples)):
            mse += Propagation(examples[i], i, rate, mode,
                               w1,w2,w3)
        mse_avg.append(mse/(len(examples)))
        print('rate:',rate,'epoch:',epoch+1,'  average mse:',mse_avg[epoch])
        if(epoch >= 1):
            abs_fc = abs((mse_avg[epoch] - mse_avg[epoch-1])/mse_avg[epoch-1])
            print('absolute_fraction_of_change:',abs_fc)
            if(abs_fc <= math.pow(10,-4)):
                success += 1
        epoch += 1
        
    return epoch

#read data from .txt file
name = 'iris.data.txt'
file = open(name, 'r')
examples = []
labels = []
y = []
t = [] #setosa, versicolor, virginica
for line in file.readlines():
    if(line[0]!='\n'):
        f_list = [float(i) for i in line[0:15].split(",") if i.strip()]
        examples.append(f_list[0:4])
        label = line[16:].strip()
        if(label == 'Iris-setosa'):
            t.append([1,0,0])
        elif(label == 'Iris-versicolor'):
            t.append([0,1,0])
        elif(label == 'Iris-virginica'):
            t.append([0,0,1])
        else:
            t = t 
        if(len(labels)==0 or (label != labels[len(labels)-1] and len(labels) != 0 )):
            labels.append(label)
        
print('\n The satisfied epoch is:',Neural_Network(0.1))
