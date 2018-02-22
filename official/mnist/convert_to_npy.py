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
import numpy as np
from PIL import Image
from array import *

def main(args):
	combined_arr = np.array([])
	images = str.split(args.images)
	for image_name in images:
		input_image = Image.open(image_name).convert('L').resize((28, 28)) # convert to grayscale, resize
		data_image = array('f')
		pixel = input_image.load()
		for x in range(input_image.size[0]):
			for y in range(input_image.size[1]):
				data_image.append(((pixel[y,x] - 255)*-1)/255.0) # use the MNIST format

		img_arr = np.reshape(np.array(data_image), (1, 28, 28)) # convert to reshaped numpy array
		if combined_arr.size == 0:
			combined_arr = img_arr
		else:
			combined_arr = np.concatenate((combined_arr, img_arr), axis=0) # add all image arrays to one array

	np.save("converted_images", combined_arr)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert an image to an MNIST-format numpy array.')
	parser.add_argument('images', action="store", type=str)
	args = parser.parse_args()
	main(args)