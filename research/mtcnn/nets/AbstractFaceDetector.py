from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class AbstractFaceDetector(object):

	def __init__(self):
		print('AbstractFaceDetector')
		self.setup()

	def setup(self):
		print('setup')
