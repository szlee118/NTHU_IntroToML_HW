from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

test_data1=[[7.0, 3.2, 4.7, 1.4]] # this is a piece of data used for training
test_data2=[[3.3, 3.3, 3.3, 3.3]] # this is a random piece of data
test_data3=[[2.2, 2.2, 4.4, 5.5]] # this is a random piece of data
test_data4=[[1.0, 2.0, 3.0, 4.0]] # this is a random piece of data
test_data5=[[1.5, 9.9, 0.1, 9.7]] # this is a random piece of data


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
    
print('1-NN')
print(test_data1,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,1, test_data1))
print(test_data2,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,1, test_data2))
print(test_data3,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,1, test_data3))
print(test_data4,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,1, test_data4))
print(test_data5,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,1, test_data5))
print('3-NN')
print(test_data1,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,3, test_data1))
print(test_data2,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,3, test_data2))
print(test_data3,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,3, test_data3))
print(test_data4,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,3, test_data4))
print(test_data5,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,3, test_data5))
print('10-NN')
print(test_data1,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,10, test_data1))
print(test_data2,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,10, test_data2))
print(test_data3,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,10, test_data3))
print(test_data4,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,10, test_data4))
print(test_data5,' is calssified as: ',Myclassifier(iris.data, iris.target, iris.target_names,10, test_data5))
