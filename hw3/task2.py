import matplotlib.pyplot as plt
import random as rand

name = 'hw3_dataset.txt'
c = []
h = [0.0,0.0,0.0,0.0,0.0,
     0.0,0.0,0.0,0.0,0.0,
     0.0,0.0,0.0,0.0,0.0,
     0.0,0.0,0.0,0.0,0.0]

def check_classifier(w, i, attr_num):
    sigma = 0
    sigma += w[0] * x[0]
    for j in range(1, attr_num+1):
        sigma += w[j] * x[j][i]
    if(sigma> 0):
        h[i] = 1
    else:
        h[i] = 0
    if( h[i] == c[i]):
        return 1
    else:
        return 0

def perceptron(Eta,attr_num):
    epoch = 0 
    error = 1
    w = []
    for i in range(attr_num+1):
        w.append(0.2)
    while(error > 0):
        error = 0
        for i in range(20):
            if(check_classifier(w, i, attr_num) == 0):
                error += 1
            w[0]= w[0] + (c[i] - h[i])* Eta * x[0]
            for j in range(1,attr_num+1):
                w[j]= w[j] + (c[i] - h[i])* Eta * x[j][i]
        epoch += 1
    #below print weights, specified for task2
    print('N=',attr_num-5,'Eta=0.2')
    for i in range(6):
        print('w[',i,']=',w[i])
    #above print weights, specified for task2
    return epoch



N = [1,5,10,15,20]
y = [0,0,0,0,0]

for ii in range(5):
    #reload x from dataset every new N
    #label does not change
    x= [1.0,[],[],[],[],[]]

    file = open(name, 'r')
    for line in file.readlines():
        line.strip()
        for i in range(1,6):
            x[i].append(float(line[5+i-1]))
        sum = 0
        for i in range(len(line)):
            if(line[i]=='1' and i>=5):
                sum += 1
        if(sum>=3):
            c.append(1)
        else :
            c.append(0)
    file.close()

    
    for j in range(N[ii]):  #N[ii] refers to the current N in loop
        arr = []
        for k in range(20):
            arr.append(float(rand.randint(0,1)))
        x.append(arr)
    
    y[ii] = perceptron(0.2, 5+N[ii]) * 20
        
plt.plot(N,y,'ro')
plt.xlabel('N')
plt.ylabel('Number of example presentations')

