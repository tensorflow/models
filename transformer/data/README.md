### How to get the data

#### Cluttered MNIST

The cluttered MNIST dataset can be found here [1] or can be generated via [2].

Settings used for `cluttered_mnist.py` :

```python

ORG_SHP = [28, 28]
OUT_SHP = [40, 40]
NUM_DISTORTIONS = 8
dist_size = (5, 5) 

```

[1] https://github.com/daviddao/spatial-transformer-tensorflow

[2] https://github.com/skaae/recurrent-spatial-transformer-code/blob/master/MNIST_SEQUENCE/create_mnist_sequence.py