#!/usr/bin/python
#
# Derived from download_and_preprocess_flowers.sh
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
#
# ==============================================================================
"""
 Script to download and preprocess the flowers data set. This data set
 provides a demonstration for how to perform fine-tuning (i.e. tranfer
 learning) from one model to a new data set.

 This script provides a demonstration for how to prepare an arbitrary
 data set for training an Inception v3 model.

 We demonstrate this with the flowers data set which consists of images
 of labeled flower images from 5 classes:

 daisy, dandelion, roses, sunflowers, tulips

 The final output of this script are sharded TFRecord files containing
 serialized Example protocol buffers. See build_image_data.py for
 details of how the Example protocol buffer contains image data.

 usage:
  ./download_and_preprocess_flowers.py [data-dir]

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from subprocess import call
import os
import os.path
import shutil
import sys
import random


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: download_and_preprocess_flowers.sh [data dir]\n')
    sys.exit(-1)
data_dir = os.path.realpath(sys.argv[1])

# Create the output and temporary directories.
work_dir = __file__ + ".runfiles/inception/inception"
scratch_dir = data_dir + '/raw-data/'
try:
  os.makedirs(scratch_dir);
except OSError as e:
    print('Mkdir error: "%s" %s' % (scratch_dir,os.strerror(e.errno)))

# Download the flowers data.
data_url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
base_dir = os.path.dirname(os.path.realpath(__file__))
current_dir = os.getcwd();
try:
    os.chdir(data_dir)
except OSError as e:
    print(os.strerror(e.errno))
    sys.exit(-1)

tarball = "flower_photos.tgz"
if not os.path.isfile(tarball):
  print('Downloading flower data set.\n')
  call(['curl', '-o', tarball, data_url])
else:
  print('Skipping download of flower data.')

# Note the locations of the train and validation data.
train_directory = scratch_dir + 'train/'
validation_directory = scratch_dir + 'validation/'

# process
print("Organizing the validation data into sub-directories.")
labels_file = scratch_dir + 'labels.txt'

# Expands the data into the flower_photos/ directory and rename it as the
# train directory.
call(['tar', 'xf', 'flower_photos.tgz'])

shutil.rmtree(train_directory, True)
shutil.rmtree(validation_directory, True)

os.rename('flower_photos', train_directory)

# Generate a list of 5 labels: daisy, dandelion, roses, sunflowers, tulips
labels_file = scratch_dir + 'labels.txt'

labels = [d for d in os.listdir(train_directory) if os.path.isdir(os.path.join(train_directory, d))]
labels.sort()
with open(labels_file, 'w') as f:
  [f.write(label+'\n') for label in labels]


# Generate the validation data set.
os.makedirs(validation_directory)
for label in labels:
  validation_dir_for_label = validation_directory + label
  train_dir_for_label = train_directory + label
  os.makedirs(validation_dir_for_label)

  # Move the first randomly selected 100 images to the validation set.
  images = [f for f in os.listdir(train_dir_for_label)]
  random.shuffle(images)
  validation_images = images[0:99]

  for image in validation_images:
    os.rename(train_dir_for_label + '/' + image, validation_dir_for_label + '/' + image)


# Build the TFRecords version of the image data.
os.chdir(current_dir)
build_script = work_dir + "/build_image_data"
output_directory = data_dir
os.system(build_script +
  ' --train_directory="%s"'
  ' --validation_directory="%s"'
  ' --output_directory="%s"'
  ' --labels_file="%s"' %
  (train_directory, validation_directory, output_directory, labels_file));

