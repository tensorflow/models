# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
r"""Converts .nii files in LiTS dataset to .npy files.

This script should be run just once before running convert_lits.py.

The file is forked from:
https://github.com/tensorflow/tpu/blob/master/models/official/unet3d/data_preprocess/convert_lits_nii_to_npy.py
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import glob
import multiprocessing
import os

import nibabel as nib
import numpy as np


num_processes = 2
input_path = "Downloads/LiTS/Train/"  # where the .nii files are.
output_path = "Downloads/LiTS/Train_np/"  # where you want to put the npy files.


def process_one_file(image_path):
  """Convert one nii file to npy."""
  im_id = os.path.basename(image_path).split("volume-")[1].split(".nii")[0]
  label_path = image_path.replace("volume-", "segmentation-")

  image = nib.load(image_path).get_data().astype(np.float32)
  label = nib.load(label_path).get_data().astype(np.float32)

  print("image shape: {}, dtype: {}".format(image.shape, image.dtype))
  print("label shape: {}, dtype: {}".format(label.shape, label.dtype))

  np.save(os.path.join(output_path, "volume-{}.npy".format(im_id)), image)
  np.save(os.path.join(output_path, "segmentation-{}.npy".format(im_id)), label)


nii_dir = os.path.join(input_path, "volume-*")
p = multiprocessing.Pool(num_processes)
p.map(process_one_file, glob.glob(nii_dir))
