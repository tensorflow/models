# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Generates a basic dataset i.e. PNet dataset.

Usage:
```shell

$ python generate_basic_dataset.py \
	--annotation_file=/workspace/datasets/WIDER_train/wider_face_train.txt \
	--input_image_dir=/workspace/datasets/WIDER_train/images \ 
	--target_root_dir=/workspace/datasets/mtcnn 
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

from datasets.BasicDataset import BasicDataset

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--annotation_file', type=str, help='Input WIDER face dataset annotations file.', default=None)
	parser.add_argument('--input_image_dir', type=str, help='Input WIDER face dataset training image directory.', default=None)
	parser.add_argument('--target_root_dir', type=str, help='Output directory where output images and TensorFlow data files are saved.', default=None)
	parser.add_argument('--minimum_face', type=int, help='Minimum face size used for face detection.', default=12)
	return(parser.parse_args(argv))

def main(args):

	if(not args.annotation_file):
		raise ValueError('You must supply input WIDER face dataset annotations file with --annotation_file.')		

	if(not args.input_image_dir):
		raise ValueError('You must supply input WIDER face dataset training image directory with --input_image_dir.')		

	if(not args.target_root_dir):
		raise ValueError('You must supply output directory for storing output images and TensorFlow data files with --target_root_dir.')

	basic_dataset = BasicDataset()
	status = basic_dataset.generate(args.annotation_file, args.input_image_dir, args.minimum_face, args.target_root_dir)
	if(status):
		print('Basic dataset is generated at ' + args.target_root_dir)
	else:
		print('Error generating basic dataset.')

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))


