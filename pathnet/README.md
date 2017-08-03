pathnet
===========

Tensorflow Implementation of Pathnet from Google Deepmind.

Implementation is on Tensorflow r1.2

https://arxiv.org/pdf/1701.08734.pdf

"Agents are pathways (views) through the network which determine the subset of parameters that are used and updated by the forwards and backwards passes of the backpropogation algorithm. During learning, a tournament selection genetic algorithm is used to select pathways through the neural network for replication and mutation. Pathway fitness is the performance of that pathway measured according to a cost function. We demonstrate successful transfer learning; fixing the parameters along a path learned on task A and re-evolving a new population of paths for task B, allows task B to be learned faster than it could be learned from scratch or after fine-tuning."
Form Paper

![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/pathnet.PNG)

### Failure Story

Memory Leak Problem was happened without placeholder for geopath. Without placeholder, changing the value of tensor variable is to assign new memory, thus assigning new path for each generation caused memory leak and slow learning.

Binary MNIST classification tasks
-------------------

`
python binary_mnist_pathnet.py 
`

If you want to run that repeatly, then do as followed.

`
./auto_binary_mnist_pathnet.sh
`

### Settings
L, M, N, B and the number of populations are 3, 10, 3, 2 and 20, respectively (In paper, the number of populations is 64.). 
GradientDescent Method is used with learning rate=0.05 (In paper, learning rate=0.0001.).
Aggregation function between layers is average (In paper, that is summation.).
Skip connection, Resnet and linear modules are used for each layers except input layer.
Fixed path of first task is always activated when feed-forwarding the networks on second task (In paper, the path is not always activated.).
The learning is converaged, when training accuracy is over 99%.

Chrisantha Fernando (1st author of this paper)  and I checked the results of the paper was generated when the value is 20. Thus, I set that as 20.
I set bigger learning rate vaule than that of paper for getting results faster than before.
Higher learning rate can accelate network learning faster than positive transfer learning. For de-accelating converage, average function is used.
The author and I checked the paper results was generated when last aggregation function is average not summation (Except last one, others are summation.).
Fixed path activation is for generating more dramatic results than before.
For faster converage than before, lower converage accuracy then before(99.8%) is used.

B candidates use same data batchs.
geopath set and parameters except the ones on optimal path of first task are reset after finishing first task.


### Results
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/binary_mnist_1vs3_1vs2.PNG) 
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/binary_mnist_6vs7_4vs5.PNG) 
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/binary_mnist_4vs5_graph.PNG) 

The experiments are 1vs3 <-> 1vs2 and 4vs5 <-> 6vs7. 
The reason of selecting those classes is to check positive transfer learning whenever there are sharing class or not. 

1vs3 experiments showed first task and second task after 1vs2 converage generation means are about 168.25 and 82.64. 
Pathnet made about 2 times faster converage than that from the scratch.

1vs2 experiments showed first task and second task after 1vs3 converage generation means are about 196.60 and 118.32. 
Pathnet made about 1.7 times faster converage than that from the scratch.

4vs5 experiments showed first task and second task after 6vs7 converage generation means are about 270.68 and 149.31. 
Pathnet made about 1.8 times faster converage than that from the scratch.

6vs7 experiments showed first task and second task after 4vs5 converage generation means are about 93.69 and 55.91. 
Pathnet made about 1.7 times faster converage than that from the scratch.

Pathnet showed about 1.7~2 times better performance than that of "learning from scratch" on Binary MNIST Classification whenever there are sharing class or not.

CIFAR10 and SVHN classification tasks
-------------------

`
python cifar_svhn_pathnet.py 
`

If you want to run that repeatly, then do as followed.

`
./auto_cifar_svhn_pathnet.sh
`

### Settings
L, M, N, B and the number of populations are 3, 20, 5, 2 and 20, respectively. 
GradientDescent Method is used with learning rate=0.2 (With learning rate=0.05, this task can not be learned. Thus, higher learning rate than before is set).
The accuracy is checked after 500 epoches.

Except M, N and learning rate, other parameters are same to that of Binary MNIST classification task.


### Results
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/cifar_svhn.PNG) 

The experiments are CIFAR10 <-> SVHN.

CIFAR10 experiments showed first task and second task after SVHN accuracy means are about 38.56% and 41.75%. 
Pathnet made about 1.1 times higher accuracy than that from the scratch.

SVHN experiments showed first task and second task after CIFAR10 accuracy means are about 19.68% and 56.25%. 
Pathnet made about 2.86 times higher accuracy than that from the scratch.

Pathnet showed positive transfer learning performance for both of the datasets. For SVHN, quitely higher transfer learning performance than CIFAR10 is showed. Because, CIFAR10 dataset has more plenty of patterns than SVHN.

Atari Game (Pong)
-------------------

`
./auto_atari_pathnet.sh
`

This module is implemented by Distributed Tensorflow.
You can set the number of parameter server and worker in the shell script, and please before running that, check the port is idle (used port number is from 2222 to 2222+ps#+w#).

Basic code for A3C is based on https://github.com/miyosuda/async_deep_reinforce

### Settings
L, M, N are 4, 10, anf 4, respectively (same to the paper).
The feature for each conv layer is 8 (same to original ones from author, I did check that.)
B and the number ofpopulations are 3 and 10, respectively, which are different to the paper, because my server cannot run 64 worker parallelly, thus, I did decrease the number of populations and B.
Aggregation function between layers is summation for faster learning than average (In paper, that is summation.).

I implemented PathNet with Distributed Tensorflow by adding one worker for processing genetic algorithm.
The worker checks score set including each worker's one, and operates genetic algorithm (in here, that is tournament algorithm.). 
Those operation is processed per each 5 seconds.
As same to the paper, winner score is not initialized as -1000.

I apply LSTM layer after last layer of pathnet for learning the model more efficiently than original one. (LSTM layer is also initilized after the task.)
(LSTM layer makes really more efficient learning than before. I checked the model except LSTM, which are saturated at about 100M step, however the model having LSTM just needs about 20M step.)

I used just pong game for checking positive transfer learning (the parameters except fixed path are initialzed after first task.), assumed second pong game will be more quickly saturated than first one.
Each task learns pong game in 15M steps, and I checked score graph in tensorboard.

### Results
![alt tag](https://github.com/jaesik817/pathnet/blob/master/figures/pong.PNG) 

The experiments are just two times pong game for checking positive transfer learning (the parameters except fixed path are initialzed after first task.), and I assumed second pong game will be more quickly saturated than first one.
Each task learns pong game in 15M steps, and I checked score graph in tensorboard.

I can check second pong game was saturated i more quickly than first one.
