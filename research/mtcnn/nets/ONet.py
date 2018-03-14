from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nets.AbstractFaceDetector import AbstractFaceDetector

class ONet(AbstractFaceDetector):

	def __init__(self, batch_size = 1):
		print('ONet')
		self.network_size = 48
		self.batch_size = batch_size

	def detect(self, image, boxes):
		return( np.array([]), np.array([]), np.array([]) )
