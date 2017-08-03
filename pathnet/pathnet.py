from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np

def parameters_backup(var_list_to_learn):
  var_list_backup=np.zeros(len(var_list_to_learn),dtype=object);
  for i in range(len(var_list_to_learn)):
    var_list_backup[i]=var_list_to_learn[i].eval();
  return var_list_backup;

def parameters_update(sess,var_update_placeholders,var_update_ops,var_list_backup):
  for i in range(len(var_update_placeholders)):
    sess.run(var_update_ops[i],{var_update_placeholders[i]:var_list_backup[i]});
    
def geopath_insert(sess,geopath_update_placeholders,geopath_update_ops,candi,L,M):
  for i in range(L):
    for j in range(M):
      sess.run(geopath_update_ops[i,j],{geopath_update_placeholders[i,j]:candi[i,j]});

def geopath_initializer(L,M):
  geopath=np.zeros((L,M),dtype=object);
  for i in range(L):
    for j in range(M):
      geopath[i,j]=tf.Variable(1.0);
  return geopath;

def mutation(geopath,L,M,N):
  for i in range(L):
    for j in range(M):
      if(geopath[i,j]==1):
        rand_value=int(np.random.rand()*L*N);
        if(rand_value<=1):
          geopath[i,j]=0;
          rand_value2=np.random.randint(-2,2);
          rand_value2=rand_value2-2;
          if(((j+rand_value2)>=0)&((j+rand_value2)<M)):
            geopath[i,j+rand_value2]=1;
          else:
            if((j+rand_value2)<0):
              geopath[i,0]=1;
            elif((j+rand_value2)>=M):
              geopath[i,M-1]=1;
  return geopath;

def select_two_candi(M):
  selected=np.zeros(2,dtype=int);
  j=0;
  while j<=2:
    rand_value=int(np.random.rand()*M);
    if(j==0):
      selected[j]=rand_value;j+=1;
    else:
      if(selected[0]!=rand_value):
        selected[j]=rand_value;j+=1;
        break;
  return selected[0],selected[1];
  
def get_geopath(L,M,N):
  geopath=np.zeros((L,M),dtype=float);
  for i in range(L):
    j=0;
    #Active module # can be smaller than N
    while j<N:
      rand_value=int(np.random.rand()*M);
      if(geopath[i,rand_value]==0.0):
        geopath[i,rand_value]=1.0;j+=1;
  return geopath;
      

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def module_weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return [tf.Variable(initial)];

def module_bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return [tf.Variable(initial)];

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def module(input_tensor, weights, biases, layer_name, act=tf.nn.relu):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights[0]) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def module2(i,input_tensor, weights, biases, layer_name, act=tf.nn.relu):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # Skip Layer
    if(i%3==0):
      return input_tensor;
    # Linear Layer with relu
    elif(i%3==1):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        variable_summaries(weights[0])
      with tf.name_scope('biases'):
        variable_summaries(biases[0])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights[0]) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations
    # Residual Layer with relu
    elif(i%3==2):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        variable_summaries(weights[0])
      with tf.name_scope('biases'):
        variable_summaries(biases[0])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights[0]) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')+input_tensor
      tf.summary.histogram('activations', activations)
      return activations

def conv_module(input_tensor, weights, biases, stride, layer_name, act=tf.nn.relu):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.nn.conv2d(input_tensor,weights[0],strides=[1,stride,stride,1],padding="VALID") + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations
 
def nn_layer(input_tensor, weights, biases, layer_name, act=tf.nn.relu):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights[0])
    with tf.name_scope('biases'):
      variable_summaries(biases[0])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights[0]) + biases
      tf.summary.histogram('pre_activations', preactivate)
    return preactivate;
