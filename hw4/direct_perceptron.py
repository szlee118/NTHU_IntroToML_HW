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

# I assume that sigma(wi*xi) > 0 ---> setosa
#               sigma(wi*xi)<= 0 ---> versicolor
h = []
for i in range(90):
    h.append(100)

def check_classifier(w, i, attr_num, T):
    sigma = 0
    for j in range(attr_num):
        sigma += w[j] * T[0][i][j]
    if(sigma> 0):
        h[i] = 1
    else:
        h[i] = 0
    if( h[i] == T[1][i]):
        #print('correct')
        return 1
    else:
        #print('incorrect')
        return 0

w = []

def perceptron(Eta,attr_num):
    epoch = 0 
    error = 1
    for i in range(attr_num):
        w.append(0.2)
    while(error > 0):
        error = 0
        for i in range(len(Training_Set[0])):
            if(check_classifier(w, i, attr_num, Training_Set) == 0):
                error += 1
            for j in range(attr_num):
                w[j]= w[j] + (Training_Set[1][i] - h[i])* Eta * Training_Set[0][i][j]
            #print(w)
        epoch += 1
        if(epoch == 100):
            break
    #below print weights
    print('Eta=',Eta)
    for i in range(attr_num):
        print('w[',i,']=',w[i])
    #above print weights
    return error

perceptron(0.2,len(Training_Set[0][0]))
print('finish perceptron with 100 epochs')

checksum = 0
for i in range(len(Testing_Set[0])):
    checksum += check_classifier(w, i, len(Testing_Set[0][0]), Testing_Set)
checksum = checksum/len(Testing_Set[0])

print('The accuracy of Direct perceptron algorithm is', checksum)
