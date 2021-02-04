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
"""TFDS Classification decoders."""

import tensorflow as tf
from official.vision.beta.dataloaders import decoder


class ClassificationDecorder(decoder.Decoder):
  """A tf.Example decoder for tfds classification datasets."""

  def decode(self, serialized_example):
    sample_dict = {
        'image/encoded':
            tf.io.encode_jpeg(serialized_example['image'], quality=100),
        'image/class/label':
            serialized_example['label'],
    }
    return sample_dict


TFDS_ID_TO_DECODER_MAP = {
    'cifar10': ClassificationDecorder,
    'cifar100': ClassificationDecorder,
    'imagenet2012': ClassificationDecorder,
}
