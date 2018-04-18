# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import cv2
import tensorflow as tf

from trainers.AbstractNetworkTrainer import AbstractNetworkTrainer

class SimpleNetworkTrainer(AbstractNetworkTrainer):

	def __init__(self, network_name='PNet'):	
		AbstractNetworkTrainer.__init__(self, network_name)	

	def _train_model(self, base_lr, loss, data_num):

    		lr_factor = 0.1
    		global_step = tf.Variable(0, trainable=False)
    		#LR_EPOCH [8,14]
    		#boundaried [num_batch,num_batch]
    		boundaries = [int(epoch * data_num / self._config.BATCH_SIZE) for epoch in self._config.LR_EPOCH]
    		#lr_values[0.01,0.001,0.0001,0.00001]
    		lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(self._config.LR_EPOCH) + 1)]
    		#control learning rate
    		lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    		optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    		train_op = optimizer.minimize(loss, global_step)

    		return( train_op, lr_op )

	def _random_flip_images(self, image_batch, label_batch, landmark_batch):
    		if random.choice([0,1]) > 0:
        		num_images = image_batch.shape[0]
        		fliplandmarkindexes = np.where(label_batch==-2)[0]
        		flipposindexes = np.where(label_batch==1)[0]
        		#only flip
        		flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        		#random flip    
        		for i in flipindexes:
            			cv2.flip(image_batch[i],1,image_batch[i])        
        
        		#pay attention: flip landmark    
        		for i in fliplandmarkindexes:
            			landmark_ = landmark_batch[i].reshape((-1,2))
            			landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            			landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            			landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
            			landmark_batch[i] = landmark_.ravel()
        
    		return( image_batch, landmark_batch )

	def train(self, network_name, dataset_dir, model_train_dir):
		
		return(True)

