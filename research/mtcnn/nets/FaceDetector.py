from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import cv2
import numpy as np

from nets.nms import py_nms

from nets.PNet import PNet
from nets.RNet import RNet
from nets.ONet import ONet

class FaceDetector(object):

	def __init__(self):
		print('FaceDetector')
		self.min_face_size = 24
		self.threshold = [0.9, 0.6, 0.7]
		self.scale_factor = 0.79

		self.pnet = PNet()
		self.pnet.load_model('/git-space/mtcnn/research/mtcnn/data/mtcnn/PNet/PNet')

		self.rnet = RNet()
		self.onet = ONet()
		print('FaceDetector-done')

    	def generate_bbox(self, cls_map, reg, scale, threshold):
 
        	stride = 2
        	#stride = 4
        	cellsize = 12
        	#cellsize = 25

        	t_index = np.where(cls_map > threshold)

        	# find nothing
        	if t_index[0].size == 0:
            		return( np.array([]) )
        	#offset
        	dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        	reg = np.array([dx1, dy1, dx2, dy2])
        	score = cls_map[t_index[0], t_index[1]]
        	boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])

        	return( boundingbox.T )

	def processed_image(self, image, scale):
	        height, width, channels = image.shape
	        new_height = int(height * scale)
	        new_width = int(width * scale)
	        new_shape = (new_width, new_height)
	        resized_image = cv2.resize(image, new_shape, interpolation = cv2.INTER_LINEAR)
	        resized_image = (resized_image - 127.5) / 128
	        return( resized_image )

	def propose_faces(self, im):
        	h, w, c = im.shape
        	net_size = self.pnet.network_size
        
        	current_scale = float(net_size) / self.min_face_size  # find initial scale
        	# print("current_scale", net_size, self.min_face_size, current_scale)
        	im_resized = self.processed_image(im, current_scale)
        	current_height, current_width, _ = im_resized.shape
        	# fcn
        	all_boxes = list()
        	while min(current_height, current_width) > net_size:
            		#return the result predicted by pnet
            		#cls_cls_map : H*w*2
            		#reg: H*w*4
            		cls_cls_map, reg = self.pnet.detect(im_resized)
            		#boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            		boxes = self.generate_bbox(cls_cls_map[:, :,1], reg, current_scale, self.threshold[0])

            		current_scale *= self.scale_factor
            		im_resized = self.processed_image(im, current_scale)
            		current_height, current_width, _ = im_resized.shape

            		if boxes.size == 0:
                		continue
            		keep = py_nms(boxes[:, :5], 0.5, 'Union')
            		boxes = boxes[keep]
            		all_boxes.append(boxes)

        	if len(all_boxes) == 0:
            		return None, None, None

        	all_boxes = np.vstack(all_boxes)

        	# merge the detection from first stage
        	keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        	all_boxes = all_boxes[keep]
        	boxes = all_boxes[:, :5]

        	bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        	bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        	# refine the boxes
        	boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        	boxes_c = boxes_c.T

        	return( boxes, boxes_c, None )

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
		return(boxes_c, landmark)


