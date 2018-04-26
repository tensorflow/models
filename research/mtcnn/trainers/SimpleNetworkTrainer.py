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

import os
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.contrib import slim
from datetime import datetime

from trainers.AbstractNetworkTrainer import AbstractNetworkTrainer
from datasets.TensorFlowDataset import TensorFlowDataset

from nets.NetworkFactory import NetworkFactory

from losses.class_loss_ohem import class_loss_ohem
from losses.bounding_box_loss_ohem import bounding_box_loss_ohem
from losses.landmark_loss_ohem import landmark_loss_ohem

class SimpleNetworkTrainer(AbstractNetworkTrainer):

	def __init__(self, network_name='PNet'):	
		AbstractNetworkTrainer.__init__(self, network_name)	

	def _train_model(self, base_learning_rate, loss, data_num):

    		learning_rate_factor = 0.1
    		global_step = tf.Variable(0, name='global_step', trainable=False)
    		boundaries = [int(epoch * data_num / self._batch_size) for epoch in self._config.LR_EPOCH]
    		learning_rate_values = [base_learning_rate * (learning_rate_factor ** x) for x in range(0, len(self._config.LR_EPOCH) + 1)]
    		learning_rate_op = tf.train.piecewise_constant(global_step, boundaries, learning_rate_values)
    		optimizer = tf.train.MomentumOptimizer(learning_rate_op, 0.9)
    		train_op = optimizer.minimize(loss, global_step=global_step)

    		return( train_op, learning_rate_op )

	def _random_flip_images(self, image_batch, label_batch, landmark_batch):
    		if random.choice([0,1]) > 0:
        		num_images = image_batch.shape[0]
        		fliplandmarkindexes = np.where(label_batch==-2)[0]
        		flipposindexes = np.where(label_batch==1)[0]
        		flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))

        		for i in flipindexes:
            			cv2.flip(image_batch[i],1,image_batch[i])        
        
        		for i in fliplandmarkindexes:
            			landmark_ = landmark_batch[i].reshape((-1,2))
            			landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            			landmark_[[0, 1]] = landmark_[[1, 0]]
            			landmark_[[3, 4]] = landmark_[[4, 3]]
            			landmark_batch[i] = landmark_.ravel()
        
    		return( image_batch, landmark_batch )

	def _calculate_accuracy(self, cls_prob,label):
    		pred = tf.argmax(cls_prob,axis=1)
	    	label_int = tf.cast(label,tf.int64)
    		cond = tf.where(tf.greater_equal(label_int,0))
    		picked = tf.squeeze(cond)
    		label_picked = tf.gather(label_int,picked)
    		pred_picked = tf.gather(pred,picked)
    		accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    		return accuracy_op

	def _read_data(self, dataset_root_dir):
		dataset_dir = self.dataset_dir(dataset_root_dir)		
		tensorflow_file_name = self._image_list_file_name(dataset_dir)
		
		self._number_of_samples = 0
		self._number_of_samples = sum(1 for _ in tf.python_io.tf_record_iterator(tensorflow_file_name))
		
		image_size = self.network_size()
		tensorflow_dataset = TensorFlowDataset()
		return(tensorflow_dataset.read_tensorflow_file(tensorflow_file_name, self._batch_size, image_size))

	def train(self, network_name, dataset_root_dir, train_root_dir, base_learning_rate, max_number_of_epoch, log_every_n_steps):
		network_train_dir = self.network_train_dir(train_root_dir)
		if(not os.path.exists(network_train_dir)):
			os.makedirs(network_train_dir)
		
		image_size = self.network_size()	
	
		image_batch, label_batch, bbox_batch, landmark_batch = self._read_data(dataset_root_dir)		

		class_loss_ratio, bbox_loss_ratio, landmark_loss_ratio = NetworkFactory.loss_ratio(network_name)

    		input_image = tf.placeholder(tf.float32, shape=[self._batch_size, image_size, image_size, 3], name='input_image')
    		target_label = tf.placeholder(tf.float32, shape=[self._batch_size], name='target_label')
    		target_bounding_box = tf.placeholder(tf.float32, shape=[self._batch_size, 4], name='target_bounding_box')
    		target_landmarks = tf.placeholder(tf.float32,shape=[self._batch_size,10],name='target_landmarks')		

		output_class_probability, output_bounding_box, output_landmarks = self._network.setup_training_network(input_image)

		class_loss_op = class_loss_ohem(output_class_probability, target_label)
           	bounding_box_loss_op = bounding_box_loss_ohem(output_bounding_box, target_bounding_box, target_label)
            	landmark_loss_op = landmark_loss_ohem(output_landmarks, target_landmarks, target_label)

		class_accuracy_op = self._calculate_accuracy(output_class_probability, target_label)
		L2_loss_op = tf.add_n(tf.losses.get_regularization_losses())

		train_op, learning_rate_op = self._train_model(base_learning_rate, class_loss_ratio*class_loss_op + bbox_loss_ratio*bounding_box_loss_op + landmark_loss_ratio*landmark_loss_op + L2_loss_op, self._number_of_samples)

    		init = tf.global_variables_initializer()
    		self._session = tf.Session()

    		saver = tf.train.Saver(save_relative_paths=True)
    		self._session.run(init)

    		tf.summary.scalar("class_loss", class_loss_op)
    		tf.summary.scalar("bounding_box_loss",bounding_box_loss_op)
    		tf.summary.scalar("landmark_loss",landmark_loss_op)
    		tf.summary.scalar("class_accuracy",class_accuracy_op)
    		summary_op = tf.summary.merge_all()

    		logs_dir = os.path.join(network_train_dir, "logs")
		if(not os.path.exists(logs_dir)):
			os.makedirs(logs_dir)

    		summary_writer = tf.summary.FileWriter(logs_dir, self._session.graph)
    		coordinator = tf.train.Coordinator()

    		threads = tf.train.start_queue_runners(sess=self._session, coord=coordinator)
    		current_step = 0

    		max_number_of_steps = int(self._number_of_samples / self._batch_size + 1) * max_number_of_epoch
    		epoch = 0

		global_step = 0
		if( self._network.load_model(self._session, network_train_dir) ):
			model_path = self._network.model_path()		
			print( 'Model is restored from %s.' %( model_path ) )
			global_step = int(os.path.basename(model_path).split('-')[1])
		
		network_train_file_name = os.path.join(network_train_dir, self.network_name())	
    		self._session.graph.finalize()    

    		try:
        		for step in range(max_number_of_steps):
            			current_step = current_step + 1

            			if coordinator.should_stop():
                			break

            			image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array = self._session.run([image_batch, label_batch, bbox_batch, landmark_batch])
            			image_batch_array, landmark_batch_array = self._random_flip_images(image_batch_array, label_batch_array, landmark_batch_array)

             			_, _, summary = self._session.run(
							[train_op, learning_rate_op ,summary_op], 
							feed_dict={
								input_image:image_batch_array, 
								target_label:label_batch_array, 
								target_bounding_box:bbox_batch_array, 
								target_landmarks:landmark_batch_array
								})
            
            			if( (step+1) % log_every_n_steps == 0 ):
                			current_class_loss, current_bbox_loss, current_landmark_loss, current_L2_loss, current_lr, current_accuracy = self._session.run(
							[class_loss_op, bounding_box_loss_op, landmark_loss_op, L2_loss_op, learning_rate_op, class_accuracy_op],
							feed_dict={
								input_image:image_batch_array, 
								target_label:label_batch_array, 
								target_bounding_box:bbox_batch_array, 
								target_landmarks: landmark_batch_array
								})                
                			print("%s - step - %d accuracy - %3f, class loss - %4f, bbox loss - %4f, landmark loss - %4f, L2 loss - %4f, lr - %f " 
						% (datetime.now(), step+1, current_accuracy, current_class_loss, current_bbox_loss, current_landmark_loss, current_L2_loss, current_lr))

					summary_writer.add_summary(summary, global_step=(global_step + step) )

            			if( current_step * self._batch_size > self._number_of_samples*2 ):
                			epoch = epoch + 1
                			current_step = 0
                			saver.save(self._session, network_train_file_name, global_step=(global_step + epoch))            			
		except tf.errors.OutOfRangeError:
       			print("Error")
		finally:
       			coordinator.request_stop()
       			summary_writer.close()
		coordinator.join(threads)
		self._session.close()

		return(True)

