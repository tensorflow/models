# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Download CUB data for FineGAN.
"""

import os
import argparse
from absl import app
import tensorflow as tf
assert tf.version.VERSION.startswith('2.')

ap = argparse.ArgumentParser()
ap.add_argument('-dp', '--download_path', required=False, help='Download Path')
args = vars(ap.parse_args())

data_url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"

def download_data(download_path):
	path_to_zip = tf.keras.utils.get_file(
		'CUB_200_2011.tgz', cache_subdir=download_path,
		origin = data_url, extract=True)

	path_to_folder = os.path.join(os.path.dirname(path_to_zip), 'data/')

	return path_to_folder

if __name__ == '__main__':
	if args['download_path'] is not None:
		path = download_data(args["download_path"])
	else:
		cur = os.getcwd()
		path = download_data(cur)

