from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class AbstractFaceDetector(object):

	def __init__(self):
		print('AbstractFaceDetector')
		self.network_size = 12

	def load_model(self, checkpoint_path):
		raise NotImplementedError('Must be implemented by the subclass.')
	
