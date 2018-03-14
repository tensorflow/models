from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from nets.PNet import PNet
from nets.RNet import RNet
from nets.ONet import ONet

class FaceDetector(object):

	def __init__(self):
		print('FaceDetector')
		self.pnet = PNet()
		self.rnet = RNet()
		self.onet = ONet()
		print('FaceDetector-done')

	def propose_faces(self, image):
		return(self.pnet.detect(image))

	def refine_faces(self, image, boxes_c):
		return(self.rnet.detect(image, boxes_c))

	def outpute_faces(self, image, boxes_c):
		return(self.onet.detect(image, boxes_c))

	def detect(self, image):
		boxes = None

		start_time = time.time()
		pnet_time = 0
        	if self.pnet:
            		boxes, boxes_c, _ = self.propose_faces(image)
            		if boxes_c is None:
                		return( np.array([]), np.array([]) )    
            		pnet_time = time.time() - start_time
            	
		start_time = time.time()
		rnet_time = 0
        	if self.rnet:
            		boxes, boxes_c, _ = self.refine_faces(image, boxes_c)
            		if boxes_c is None:
                		return( np.array([]),np.array([]) )    
            		rnet_time = time.time() - start_time

            	start_time = time.time()
		onet_time = 0
        	if self.onet:
            		boxes, boxes_c, landmark = self.outpute_faces(image, boxes_c)
            		if boxes_c is None:
                		return( np.array([]),np.array([]) )    
            		onet_time = time.time() - start_time

		print('detect')
		return(boxes_c, landmark)


