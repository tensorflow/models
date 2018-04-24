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

r"""Generates a hard dataset i.e. RNet or ONet dataset.

Usage:
```shell

$ python generate_hard_dataset.py \
	--network_name=RNet \ 
	--train_root_dir=./data/models/mtcnn/train \
	--annotation_image_dir=./data/WIDER_Face/WIDER_train/images \ 
	--annotation_file_name=./data/WIDER_Face/WIDER_train/wider_face_train.txt \
	--landmark_image_dir=./data/LFW_Landmark \
	--landmark_file_name=./data/LFW_Landmark/trainImageList.txt \
	--target_root_dir=./data/datasets/mtcnn \
	--minimum_face=24

$ python generate_hard_dataset.py \
	--network_name=ONet \ 
	--train_root_dir=./data/models/mtcnn/train \
	--annotation_image_dir=./data/WIDER_Face/WIDER_train/images \ 
	--annotation_file_name=./data/WIDER_Face/WIDER_train/wider_face_train.txt \
	--landmark_image_dir=./data/LFW_Landmark \
	--landmark_file_name=./data/LFW_Landmark/trainImageList.txt \
	--target_root_dir=./data/datasets/mtcnn \
	--minimum_face=48
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

from datasets.HardDataset import HardDataset

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--network_name', type=str, help='The name of the network.', default='ONet')    
	parser.add_argument('--train_root_dir', type=str, help='Input train root directory where model weights are saved.', default=None)

	parser.add_argument('--annotation_file_name', type=str, help='Input WIDER face dataset annotations file.', default=None)
	parser.add_argument('--annotation_image_dir', type=str, help='Input WIDER face dataset training image directory.', default=None)

	parser.add_argument('--landmark_image_dir', type=str, help='Input landmark dataset training image directory.', default=None)
	parser.add_argument('--landmark_file_name', type=str, help='Input landmark dataset annotation file.', default=None)

	parser.add_argument('--target_root_dir', type=str, help='Output directory where output images and TensorFlow data files are saved.', default=None)
	parser.add_argument('--minimum_face', type=int, help='Minimum face size used for face detection.', default=24)
	return(parser.parse_args(argv))

def main(args):

	if(not args.annotation_file_name):
		raise ValueError('You must supply input WIDER face dataset annotations file with --annotation_file_name.')
	if(not args.annotation_image_dir):
		raise ValueError('You must supply input WIDER face dataset training image directory with --annotation_image_dir.')		

	if(not args.landmark_image_dir):
		raise ValueError('You must supply input landmark dataset training image directory with --landmark_image_dir.')		
	if(not args.landmark_file_name):
		raise ValueError('You must supply input landmark dataset annotation file with --landmark_file_name.')	

	if(not args.target_root_dir):
		raise ValueError('You must supply output directory for storing output images and TensorFlow data files with --target_root_dir.')
	
	if( not (args.network_name in ['RNet', 'ONet']) ):
		raise ValueError('The network name should be either RNet or ONet.')

	hard_dataset = HardDataset(args.network_name)
	status = hard_dataset.generate(args.annotation_image_dir, args.annotation_file_name, args.landmark_image_dir, args.landmark_file_name, args.train_root_dir, args.minimum_face, args.target_root_dir)
	if(status):
		print(args.network_name + ' network dataset is generated at ' + args.target_root_dir)
	else:
		print('Error generating hard dataset.')

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))


