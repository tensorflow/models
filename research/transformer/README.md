# Spatial Transformer Network

The Spatial Transformer Network [1] allows the spatial manipulation of data within the network.

<div align="center">
  <img width="600px" src="http://i.imgur.com/ExGDVul.png"><br><br>
</div>

### API 

A Spatial Transformer Network implemented in Tensorflow 1.0 and based on [2].

#### How to use

<div align="center">
  <img src="http://i.imgur.com/gfqLV3f.png"><br><br>
</div>

```python
transformer(U, theta, out_size)
```
    
#### Parameters

    U : float 
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels]. 
    theta: float   
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network
        
    
#### Notes
To initialize the network to the identity transform init ``theta`` to :

```python
identity = np.array([[1., 0., 0.],
                    [0., 1., 0.]]) 
identity = identity.flatten()
theta = tf.Variable(initial_value=identity)
```        

#### Experiments

<div align="center">
  <img width="600px" src="http://i.imgur.com/HtCBYk2.png"><br><br>
</div>

We used cluttered MNIST. Left column are the input images, right are the attended parts of the image by an STN.

All experiments were run in Tensorflow 0.7.

### References

[1] Jaderberg, Max, et al. "Spatial Transformer Networks." arXiv preprint arXiv:1506.02025 (2015)

[2] https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
