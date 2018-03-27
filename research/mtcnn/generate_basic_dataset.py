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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

from datasets.BasicDataset import BasicDataset

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--annotation_file', type=str, help='Input WIDER face dataset training annotations file.', default=None)
	parser.add_argument('--target_root_dir', type=str, help='Output directory where output images are saved.', default=None)
	return(parser.parse_args(argv))

def main(args):
	basic_dataset = BasicDataset()
	basic_dataset.generate()
	print('Main')

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))


