from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import cv2
import numpy as np

from nets.FaceDetector import FaceDetector

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--webcamera_id', type=int, help='Webcamera ID.', default=0)
	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.2)
	return(parser.parse_args(argv))

def main(args):
	print('main')
	face_detector = FaceDetector()
	webcamera = cv2.VideoCapture(args.webcamera_id)
	webcamera.set(3, 600)
	webcamera.set(4, 800)
	
	while True:
    		start_time = cv2.getTickCount()
    		status, current_frame = webcamera.read()
    		if status:
        		image = np.array(current_frame)
        		boxes_c, landmarks = face_detector.detect(image)

			end_time = cv2.getTickCount()
        		time_duration = (end_time - start_time) / cv2.getTickFrequency()
        		frames_per_sec = 1.0 / time_duration

        		cv2.putText(current_frame, '{:.3f}'.format(frames_per_sec), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
     
		        cv2.imshow("", current_frame)
        		if cv2.waitKey(1) & 0xFF == ord('q'):
            			break
    		else:
        		print('Error detecting the webcamera')
        		break

	webcamera.release()
	cv2.destroyAllWindows()
	print('main-done')

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))

