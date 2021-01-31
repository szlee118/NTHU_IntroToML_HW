import numpy as np

#firstly, read in files
name = ['training-data.txt','testing-data.txt']
Training_Set = []
Testing_Set = []
for i in range(len(name)):
    file = open(name[i], 'r')
    #get entire traning set
    examples = []
    labels = []
    for line in file.readlines():
        if(line[0]!='\n'):
            f_list = [float(i) for i in line[0:15].split(",") if i.strip()]
            examples.append(f_list[0:4])
        if(line[16:].strip() == "Iris-setosa"):
            labels.append(0)
        elif(line[16:].strip() == "Iris-versicolor"):
            labels.append(1)
        if(i == 0):
            Training_Set.append(examples)
            Training_Set.append(labels)
        else:
            Testing_Set.append(examples)
            Testing_Set.append(labels)


Training_Subset = [[],[],[],[],[],[],[],[],[],[]] #10 random subsets, is chosen later

#induce 3NN classifier
def KNN(data, label_no, k, x):
    
    distance = []                     #the list of distance between object x and each piece of data
    nearest = []                      #the list of labels that are closest to object x , in length of k
    
    #use the data structure of structured array in python, 
    #to relate distance and order of the point one-to-one,
    #easier for us to find the label of nearest point
    #thus, setting dtype is essential
    dtype = [('distance',float),('num', int)]
    
    #generate the list of data
    for i in range(len(data)):
        #formula of distance in python
        content = [(np.sqrt(np.sum(np.square(data[np.int(i)]-x))),np.int(i))]
        if i==0:
           distance = np.array(content, dtype=dtype)
        else:
           new_subarr = np.array(content, dtype=dtype)
           distance = np.concatenate((distance, new_subarr))
  
    #sort the array of distance to put the nearest points in front
    sorted_dis = np.sort(distance, order = 'distance')

    #pick up the k nearest points
    for i in range(k):
        ord_list = sorted_dis['num']
        index = ord_list[i]
        nearest.append(label_no[index])
        
    
    #choose the most frequent label in nearest[] list
    counts = np.bincount(nearest)
    res = np.argmax(counts)
    return label_no[res]

#normalize a list
def normalize_list(list_normal):
    total = 0
    for i in range(len(list_normal)):
        total += list_normal[i]
    for i in range(len(list_normal)):
        list_normal[i] = (list_normal[i]) / (total)
    return list_normal

#initialze variables in adaboost algorithm
e = []
prob= []
epsilon = 0
beta = 0
for i in range(9):
    e_assist = []
    for j in range(len(Training_Set[0])):
        e_assist.append(100)
    e.append(e_assist)
for i in range(len(Training_Set[0])):
    prob.append(1/len(Training_Set[0]))
    

weight = []

#Here is the main part
#I derive Training Subset iteratively for 9 nine times
#Use numpy.random.choice to slecet with probability list
#Update probability list(for 90 examples) & Append weight(for each Classifer) every iteration 
#Beta's formula is different from TextBook Version
for i in range(9):
    r= np.random.choice(
            range(90), 
            10,
            p=prob
        )
    examples_sub = []
    labels_sub = []
    for j in range(len(r)):
        examples_sub.append(Training_Set[0][r[j]])
        labels_sub.append(Training_Set[1][r[j]])
    Training_Subset[i].append(examples_sub)
    Training_Subset[i].append(labels_sub)
    
    epsilon = 0
    
    for j in range(len(Training_Set[0])):
        res = KNN(np.array(Training_Subset[i][0]),np.array(Training_Subset[i][1]),3,np.array(Training_Set[0][j]))
        if(res == Training_Set[1][j]):
            e[i][j] = 0
        else:
            e[i][j] = 1
        epsilon += prob[j] * e[i][j]
        
    beta = np.sqrt(epsilon/(1-epsilon))                  #Beta is calculated
    
    weight.append((1/2)*np.log((1-epsilon)/epsilon))     #weight is caculated
    
    for j in range(len(Training_Set[0])):
        if(beta == 0):
            prob[j] = 1/len(Training_Set[0])
        else:
            if(e[i][j] == 0):
                prob[j] = prob[j] * beta
            else:
                prob[j] = prob[j] / beta
    prob = normalize_list(prob)


error_master = 0
print('Final Weight:',weight)

#Here is the master classifier
#For original version, we need not do the perceptron
for j in range(len(Testing_Set[0])):
    setosa_vote = 0
    versicolor_vote = 0
    output_label = 100
    for i in range(9):
        res = KNN(np.array(Training_Subset[i][0]),np.array(Training_Subset[i][1]),3,np.array(Testing_Set[0][j]))
        if(res == 0):
            setosa_vote += weight[i]
        else:
            versicolor_vote += weight[i]
    if(setosa_vote > versicolor_vote):
        output_label = 0
    else:
        output_label = 1
    if(output_label != Testing_Set[1][j]):
        print('Incorrect: setosa vote is',setosa_vote,'versicolor vote is',versicolor_vote)
        error_master += 1
    else:
        print('Correct: setosa vote is',setosa_vote,'versicolor vote is',versicolor_vote)

print('The accuracy of Original Adaboost algorithm ', 1-error_master/len(Testing_Set[0]))        
        