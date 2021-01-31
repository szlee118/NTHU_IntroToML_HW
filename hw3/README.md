# hw2 Computer Assignment Report
#### 106062328 李思佑

## Code design
### 1. Functions
- starting(frame) function -- Neural_Network()
- It runs an everlasting loop until the absolute fraction of change of mse reach certain level
- It calculates and returns # of epochs
- The only parameter is rate
- For each epoch, it calls Propagation() two times, in different mode.<br>1. training mode to update weights by back-propagation<br>2. checking mode only to execute forward-propagation and return mse to generate average mse for current epoch
<br><br>
![](./nn.png)
<br><br>

- main function -- Propagation()
- This function includes forward-propagation and back-propagation
- First part of the code generate output in each level of neural network, including using the sigmoid function
- Second part of the code is based on the mode passed in, the responsibility of neurons(delta[]) is calculated to be auxiliary of updating weights.
- Since there're only two hidden layers, each level of propagation is delievered seperately. If more hidden layers
<br><br>
![](./propagation.png)
<br><br>

- auxiliary functions -- sigmoid(), MSE()
<br><br>
![](./sm.png)
<br><br>

- initialize weights -- Random_Weights()
<br><br>
![](./rw.png)
<br><br>

- read files -- not general, specified for this homework
<br><br>
![](./rf.png)
<br><br>

2.Others
- Avoid using global variables and declare them in functions as much as possible.
- Choose to present len(example) in the general form instead of numerical 4, present len(label) in the general form instead of numerical 3, to make code more applicable to general usage.
- Some specific details are described in the comments of code.


## Output
### task1
- Although the absolute fraction of change is relatively low, the average mse is still high, which means the convergent result may still make wrong classifications. It might be a signal of local-minimum, since obtaining a classifier with both variables low might need larger # of epochs.
- Here I present two most frequent results of the execution.
- ![](./task1_0.png)
<br><br>

- ![](./task1_1.png)

<br>

### task2
- The execution time might be long(several minutes), the # of epochs range from single digit to thousands. Here I present three of the possible results(with random initial weights)
- From the result of the experiment, epoch is relatively low of learning rate == 0.1, other learning rates generate unstable results for different initial weights.

<br>

- ![](./task2.png)
<br><br>

- ![](./task2_0.png)
<br><br>

- ![](./task2_1.png)


 
