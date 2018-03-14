from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nets.AbstractFaceDetector import AbstractFaceDetector

class PNet(AbstractFaceDetector):

	def __init__(self):
		print('PNet')

	def detect(self, image):
		return( np.array([]), np.array([]), np.array([]) )
