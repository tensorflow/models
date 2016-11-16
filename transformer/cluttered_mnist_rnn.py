# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 20:19:21 2016

@author: serag
"""

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import tensorflow as tf
from spatial_transformer import transformer
from tensorflow.python.ops import rnn,rnn_cell
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot

# %% Load data


mnist_cluttered = np.load('./data/mnist_sequence3_sample_8distortions_9x9.npz')

X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

y_train = np.reshape(y_train,[y_train.size,1])
y_valid = np.reshape(y_valid,[y_valid.size,1])
y_test = np.reshape(y_test,[y_test.size,1])

# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=10)


# %% Placeholders for 100x100 resolution
x = tf.placeholder(tf.float32, [None, 10000])
y = tf.placeholder(tf.float32, [None, 10])

# %% Since x is currently [batch, height*width], we need to reshape to a
# 4-D tensor to use it in a convolutional graph.  If one component of
# `shape` is the special value -1, the size of that dimension is
# computed so that the total size remains constant.  Since we haven't
# defined the batch dimension's shape yet, we use -1 to denote this
# dimension should not change size.
x_tensor = tf.reshape(x, [-1, 100, 100, 1])

#%% localizaton network

keep_prob = tf.placeholder(tf.float32)


l_pool0_loc = tf.nn.max_pool(x_tensor,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

W_conv0_loc = weight_variable([3,3,1,20])

l_conv0_loc = tf.nn.conv2d(l_pool0_loc,W_conv0_loc,strides=[1,1,1,1],padding='VALID')

l_pool1_loc = tf.nn.max_pool(l_conv0_loc,ksize=[1,2,2,1],strides =[1,2,2,1],padding='VALID')

W_conv1_loc = weight_variable([3,3,20,20])                   

l_conv1_loc =  tf.nn.conv2d(l_pool1_loc,W_conv1_loc,strides=[1,1,1,1],padding='VALID')

l_conv1_loc = tf.nn.dropout(l_conv1_loc,keep_prob)

l_pool2_loc = tf.nn.max_pool(l_conv1_loc,ksize=[1,2,2,1],strides =[1,2,2,1],padding='VALID')

W_conv2_loc = weight_variable([3,3,20,20])

l_conv2_loc = tf.nn.conv2d(l_pool2_loc,W_conv2_loc,strides=[1,1,1,1],padding='VALID') 

l_conv2_loc = tf.reshape(l_conv2_loc,[-1 ,9*9*20 ])

# Replicate input for Gated Recurrent Unit
l_conv2_loc = tf.tile(l_conv2_loc,[1,3])

l_conv2_loc = tf.split(1,3,l_conv2_loc)

# Gated Recurrent Unit
gru_cell = rnn_cell.GRUCell(num_units=256)

output, state = rnn.rnn(gru_cell,inputs=l_conv2_loc,dtype=tf.float32)


output = tf.concat(0,output)


W_fc1_loc = weight_variable([256,6])


# Use identity transformation as starting point
initial = np.array([[1., 0, 0], [0, 1., 0]])
initial = initial.astype('float32')
initial = initial.flatten()
b_fc1_loc = tf.Variable(initial_value=initial)


l_fc1_loc = tf.add(tf.matmul(output,W_fc1_loc), b_fc1_loc)


# %% We'll create a spatial transformer module to identify discriminative
# patches

downsample = 3

out_size = (100/downsample, 100/downsample)


l_transform = transformer(tf.tile(x_tensor,[3,1,1,1]), l_fc1_loc, out_size)

# %% Classification Network


W_conv0_out = weight_variable([3,3,1,32])                   

l_conv0_out = tf.nn.conv2d(l_transform,W_conv0_out,strides=[1,1,1,1],padding='VALID')


l_pool1_out = tf.nn.max_pool(l_conv0_out,ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

                   
l_drp1_out = tf.nn.dropout(l_pool1_out,keep_prob)


W_conv1_out = weight_variable([3,3,32,32])                   

l_conv1_out = tf.nn.conv2d(l_drp1_out,W_conv1_out,strides=[1,1,1,1],padding='VALID')


l_pool2_out = tf.nn.max_pool(l_conv1_out,ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')


l_drp2_out = tf.nn.dropout(l_pool2_out,keep_prob)


W_conv2_out = weight_variable([3,3,32,32])                   

l_conv2_out = tf.nn.conv2d(l_drp2_out,W_conv2_out,strides=[1,1,1,1],padding='VALID')



# %% We'll now reshape so we can connect to a fully-connected layer:
l_conv2_out_flat = tf.reshape(l_conv2_out, [-1, 4*4*32])

# %% Create a fully-connected layer:
n_fc = 400
W_fc1 = weight_variable([4*4*32, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(l_conv2_out_flat, W_fc1) + b_fc1)

# %% And finally our softmax layer:
W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_logits = tf.matmul(h_fc1, W_fc2) + b_fc2

# %% Define loss/eval/training functions
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)
# %% Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())


# %% We'll now train in minibatches and report accuracy, loss:
iter_per_epoch = 100
n_epochs = 500

indices = np.linspace(0, 10000 - 1, iter_per_epoch)
indices_y = np.linspace(0, 10000 - 1, iter_per_epoch)*3
indices = indices.astype('int')
indices_y = indices_y.astype('int')


for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices_y[iter_i]:indices_y[iter_i+1]]

        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})

    print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
                                                     feed_dict={
                                                         x: X_valid,
                                                         y: Y_valid,
                                                         keep_prob: 1.0
                                                     })))
  
