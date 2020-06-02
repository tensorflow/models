# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
"""Helper functions for DELF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from PIL import ImageFile
import tensorflow as tf

# To avoid PIL crashing for truncated (corrupted) images.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def RgbLoader(path):
  """Helper function to read image with PIL.

  Args:
    path: Path to image to be loaded.

  Returns:
    PIL image in RGB format.
  """
  with tf.io.gfile.GFile(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')

