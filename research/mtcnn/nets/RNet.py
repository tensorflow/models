from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nets.AbstractFaceDetector import AbstractFaceDetector

class RNet(AbstractFaceDetector):

	def __init__(self):
		print('RNet')
		self.network_size = 24

	def detect(self, image, boxes):
		return( np.array([]), np.array([]), np.array([]) )
