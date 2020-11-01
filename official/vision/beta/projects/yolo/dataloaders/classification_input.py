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
"""TFDS Classification decoder and parser."""
# Import libraries
import tensorflow as tf

from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""
  def __init__(self):
      return 

  def decode(self, serialized_example):
    sample_dict = {
                'image/encoded': tf.io.encode_jpeg(serialized_example['image'], quality=100), 
                'image/class/label': serialized_example['label'], 
                }
    return sample_dict


