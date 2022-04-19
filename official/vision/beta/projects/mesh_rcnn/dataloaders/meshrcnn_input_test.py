# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Test case for Mesh R-CNN dataloader configuration definition."""
import dataclasses
from typing import List, Optional, Union

import tensorflow as tf
from official.core import config_definitions as cfg
from official.core import input_reader
from official.modeling import hyperparams
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.dataloaders.pix3d_decoder import Pix3dDecoder
import official.vision.beta.projects.mesh_rcnn.dataloaders.meshrcnn_input as meshrcnn_input

@dataclasses.dataclass
class Parser(hyperparams.Config):
  """Dummy configuration for parser"""
  num_classes: int = 9
  min_level: int = 2
  max_level: int = 6
  num_scales: int = 1
  anchor_size: float = 8.0
  rpn_match_threshold=0.7
  rpn_unmatched_threshold=0.3
  rpn_batch_size_per_im=256
  rpn_fg_fraction=0.5
  aug_rand_hflip=False
  aug_scale_min=1.0
  aug_scale_max=1.0
  skip_crowd_during_training=True
  max_num_instances=1
  include_mask=False
  mask_crop_size=112
  dtype='float32'

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = r'C:\ML\Datasets\tfrecords\train\*'
  global_batch_size: int = 10
  is_training: bool = True
  dtype: str = 'float16'
  decoder = None
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = False

class MeshRCNNInputTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('training', True), ('testing', False))
  def test_meshrcnn_input(self, is_training):
    with tf.device('/CPU:0'):
      params = DataConfig(is_training=is_training)

      decoder = Pix3dDecoder(include_mask=True)
      anchors = [[12.0, 19.0], [31.0, 46.0], [96.0, 54.0], [46.0, 114.0],
                 [133.0, 127.0], [79.0, 225.0], [301.0, 150.0], [172.0, 286.0],
                 [348.0, 340.0]]
      masks = {'3': [0, 1, 2], '4': [3, 4, 5], '5': [6, 7, 8]}
      parser = meshrcnn_input.Parser(
          output_size = [800, 800],
          min_level = params.parser.min_level,
          max_level = params.parser.max_level,
          num_scales = params.parser.num_scales,
          aspect_ratios = [0.5, 1.0, 2.0],
          anchor_size = params.parser.anchor_size,
          rpn_match_threshold = params.parser.rpn_match_threshold,
          rpn_unmatched_threshold = params.parser.rpn_unmatched_threshold,
          rpn_batch_size_per_im = params.parser.rpn_batch_size_per_im,
          rpn_fg_fraction = params.parser.rpn_fg_fraction,
          aug_rand_hflip = params.parser.aug_rand_hflip,
          aug_scale_min = params.parser.aug_scale_min,
          aug_scale_max = params.parser.aug_scale_max,
          skip_crowd_during_training = params.parser.skip_crowd_during_training,
          max_num_instances = params.parser.max_num_instances,
          include_mask = params.parser.include_mask,
          mask_crop_size = params.parser.mask_crop_size,
          dtype = params.parser.dtype)

      reader = input_reader.InputReader(
          params,
          dataset_fn=tf.data.TFRecordDataset,
          decoder_fn=decoder.decode,
          parser_fn=parser.parse_fn(params.is_training))
      dataset = reader.read(input_context=None)

      # for i, element in enumerate(dataset):
      #   print(element[0])
      #   break
      for one_batch in dataset.batch(1):
        image, labels = one_batch
        break

      # for l, (i, j) in enumerate(dataset):
      #   if postprocess_fn:
      #     i, j = postprocess_fn(i, j)
      #   boxes = box_ops.xcycwh_to_yxyx(j['bbox'])
      #   self.assertTrue(tf.reduce_all(tf.math.logical_and(i >= 0, i <= 1)))
      #   if l > 10:
      #     break


if __name__ == '__main__':
  tf.test.main()