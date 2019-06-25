#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp


# In[ ]:

# add location or path where model.ckpt file is located
for i in range(0,51): 
    variable_name = 'd' + str(i)
    variable_name = tf.get_variable(cp.list_variables(r'C:\tensorflow1\models\research\object_detection\training\fast-style-model.ckpt-done')[i][0],initializer = tf.constant(cp.list_variables(r'C:\tensorflow1\models\research\object_detection\training\fast-style-model.ckpt-done')[i][1]))
    
print(variable_name)


# In[ ]:


# initialize all of the variables
init_op = tf.global_variables_initializer()


# In[ ]:


with tf.Session() as sess:
    # initialize all of the variables in the session
    sess.run(init_op)
   


# In[ ]:


# create saver object
saver = tf.train.Saver()
for i, var in enumerate(saver._var_list):
    print('Var {}: {}'.format(i, var))


# In[ ]:


with tf.Session() as sess:
    # initialize all of the variables in the session
    sess.run(init_op)

    # save the variable in the disk
    saved_path = saver.save(sess, r'saved_variable')
    print('model saved in {}'.format(saved_path))


# In[ ]:


import os
os.getcwd()


# In[ ]:

# in listdir you can enter path where you want to save these file in your computer
import os 
for file in os.listdir('.'):
    if 'saved_variable' in file:
        print(file)

