from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class AbstractFaceDetector(object):

	def __init__(self):
		self.is_training = False

	def setup_network(self, inputs):
		raise NotImplementedError('Must be implemented by the subclass.')

	def load_model(self, checkpoint_path):
		raise NotImplementedError('Must be implemented by the subclass.')

	def load_model_from(self, checkpoint_path):
		self.model_path = checkpoint_path

		saver = tf.train.Saver()
       		model_dictionary = '/'.join(checkpoint_path.split('/')[:-1])
       		check_point = tf.train.get_checkpoint_state(model_dictionary)
       		read_state = ( check_point and check_point.model_checkpoint_path )
       		assert  read_state, "Invalid parameter dictionary."
      		saver.restore(self.session, checkpoint_path)

	def detect(self, data_batch):	
		raise NotImplementedError('Must be implemented by the subclass.')
