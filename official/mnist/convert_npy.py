#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from array import *

def main(unused_argv):

	output = []
	images = []

	filename_generate = True
	index = 0

	if FLAGS.images is not None:
		images = str.split(FLAGS.images)
	if FLAGS.output is not "":                                                 # check for output names and make sure outputs map to images
		output = str.split(FLAGS.output)
		filename_generate = False
		if len(output) != len(images):
			raise ValueError('The number of image files and output files must be the same.')

	if FLAGS.batch == "True":
		combined_arr = np.array([])                                            # we'll be adding up arrays

	for image_name in images:
		input_image = Image.open(image_name).convert('L')                      # convert to grayscale
		input_image = input_image.resize((28, 28))                             # resize the image, if needed
		width, height = input_image.size
		data_image = array('B')
		pixel = input_image.load()
		for x in range(0,width):
			for y in range(0,height):
				data_image.append(pixel[y,x])                                  # use the MNIST format
		np_image = np.array(data_image)
		img_arr = np.reshape(np_image, (1, 28, 28))
		img_arr = img_arr/float(255)                                           # use scale of [0, 1]
		if FLAGS.batch != "True":
			if filename_generate:
				np.save("image"+str(index), img_arr)                           # save each image with random filenames
			else:
				np.save(output[index], img_arr)                                # save each image with chosen filenames
			index = index+1
		else:
			if combined_arr.size == 0:
				combined_arr = img_arr
			else:
				combined_arr = np.concatenate((combined_arr, img_arr), axis=0) # add all image arrays to one array
	if FLAGS.batch == "True":
		if filename_generate:
			np.save("images"+str(index), combined_arr)                         # save batched images with random filename
		else:
			np.save(output[0], combined_arr)                                   # save batched images with chosen filename

class ImageArgParser(argparse.ArgumentParser):

  def __init__(self):
    super(ImageArgParser, self).__init__()

    self.add_argument(
        '--images',
        type=str,
        default="example3.png example5.png",
        help='Images in this directory to be converted.'
        )
    self.add_argument(
        '--output',
        type=str,
        default="",
        help='Name of the converted image.'
        )
    self.add_argument(
        '--batch',
        type=str,
        default="True",
        help='Combine images into one .npy file.'
        )

if __name__ == '__main__':
	parser = ImageArgParser()
	tf.logging.set_verbosity(tf.logging.INFO)
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)