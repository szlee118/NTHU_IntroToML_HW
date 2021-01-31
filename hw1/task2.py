from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
d1=np.array(iris.data[0:30])
d2=np.array(iris.data[50:80])
d3=np.array(iris.data[100:130])
d4=np.array(iris.data[30:50])
d5=np.array(iris.data[80:100])
d6=np.array(iris.data[130:150])

l1=np.array(iris.target[0:30])
l2=np.array(iris.target[50:80])
l3=np.array(iris.target[100:130])
l4=np.array(iris.target[30:50])
l5=np.array(iris.target[80:100])
l6=np.array(iris.target[130:150])

train_data = np.concatenate((d1,d2,d3))
train_label = np.concatenate((l1,l2,l3))
test_data = np.concatenate((d4,d5,d6))
test_label = np.concatenate((l4,l5,l6))
label_name=iris.target_names

#same as task1
def Myclassifier(data, label_no, label_name, k, x):
    
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
    return label_name[res]



def Mypredictor(train_data, train_label, label_names, test_data, test_label):
    success = 0
    fail = 0
    for i in range(len(test_data)):
       res= Myclassifier(train_data,train_label, label_names, 1, test_data[i]) #1-NN implementation
       if res == label_names[test_label[i]]:
           success += 1
       else :
           fail += 1
    accuracy = success/(success+fail)
    print('Correct Cases: ',success)
    print('Incorrect Cases: ',fail)
    print('Accuracy: ',accuracy)       
    
Mypredictor(train_data, train_label, label_name, test_data, test_label)  
    

