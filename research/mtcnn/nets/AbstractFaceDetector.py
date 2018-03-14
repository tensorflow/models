from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class AbstractFaceDetector(object):

	def __init__(self):
		self.is_training = False

	def setup_network(self, inputs):
		raise NotImplementedError('Must be implemented by the subclass.')

	def load_model(self, checkpoint_path):
		raise NotImplementedError('Must be implemented by the subclass.')

	def detect(self, data_batch):	
		raise NotImplementedError('Must be implemented by the subclass.')
