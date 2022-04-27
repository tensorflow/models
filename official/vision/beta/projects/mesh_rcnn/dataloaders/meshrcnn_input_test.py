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

import tensorflow as tf
from absl.testing import parameterized

import official.vision.beta.projects.mesh_rcnn.dataloaders.meshrcnn_input as meshrcnn_input
from official.core import config_definitions as cfg
from official.core import input_reader
from official.modeling import hyperparams
from official.vision.beta.projects.mesh_rcnn.dataloaders.pix3d_decoder import \
    Pix3dDecoder


@dataclasses.dataclass
class Parser(hyperparams.Config):
  """Dummy configuration for parser"""
  num_classes = 9
  min_level = 2
  max_level = 5
  num_scales = 1
  anchor_size = 8.0
  rpn_match_threshold = 0.7
  rpn_unmatched_threshold = 0.3
  rpn_batch_size_per_im = 256
  rpn_fg_fraction = 0.5
  aug_rand_hflip = False
  aug_scale_min = 1.0
  aug_scale_max = 1.0
  skip_crowd_during_training = True
  max_num_instances = 1
  max_num_verts = 108416
  max_num_faces = 126748
  max_num_voxels = 2097152
  include_mask = True
  mask_crop_size = 112
  dtype = 'float32'

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = r'C:\ML\Datasets\tfrecords\pix3d\test\*'
  global_batch_size: int = 10
  is_training: bool = True
  dtype: str = 'float16'
  decoder = None
  parser: Parser = Parser()
  shuffle_buffer_size: int = 100
  tfds_download: bool = False

class MeshRCNNInputTest(tf.test.TestCase, parameterized.TestCase):
  """Mesh R-CNN Data pipeline test"""

  @parameterized.named_parameters(('training', True), ('evaluation', False))
  def test_meshrcnn_input(self, is_training):
    with tf.device('/CPU:0'):
      params = DataConfig(is_training=is_training)
      decoder = Pix3dDecoder(include_mask=True)

      parser = meshrcnn_input.Parser(
          output_size=[800, 800],
          min_level=params.parser.min_level,
          max_level=params.parser.max_level,
          num_scales=params.parser.num_scales,
          aspect_ratios=[0.5, 1.0, 2.0],
          anchor_size=params.parser.anchor_size,
          rpn_match_threshold=params.parser.rpn_match_threshold,
          rpn_unmatched_threshold=params.parser.rpn_unmatched_threshold,
          rpn_batch_size_per_im=params.parser.rpn_batch_size_per_im,
          rpn_fg_fraction=params.parser.rpn_fg_fraction,
          aug_rand_hflip=params.parser.aug_rand_hflip,
          aug_scale_min=params.parser.aug_scale_min,
          aug_scale_max=params.parser.aug_scale_max,
          skip_crowd_during_training=params.parser.skip_crowd_during_training,
          max_num_instances=params.parser.max_num_instances,
          max_num_verts=params.parser.max_num_verts,
          max_num_faces=params.parser.max_num_faces,
          max_num_voxels=params.parser.max_num_voxels,
          include_mask=params.parser.include_mask,
          mask_crop_size=params.parser.mask_crop_size,
          dtype=params.parser.dtype)

      reader = input_reader.InputReader(
          params,
          dataset_fn=tf.data.TFRecordDataset,
          decoder_fn=decoder.decode,
          parser_fn=parser.parse_fn(params.is_training))
      dataset = reader.read(input_context=None)

      batch_size = params.global_batch_size
      for i, (image, labels) in enumerate(dataset):
        self.assertAllEqual(tf.shape(image), [10, 800, 800, 3])

        if is_training:
          anchor_boxes = labels['anchor_boxes']
          rpn_score_targets = labels['rpn_score_targets']
          rpn_box_targets = labels['rpn_box_targets']
          gt_boxes = labels['gt_boxes']
          gt_classes = labels['gt_classes']
          gt_voxel = labels['gt_voxel']
          gt_verts = labels['gt_verts']
          gt_faces = labels['gt_faces']
          gt_voxel_mask = labels['gt_voxel_mask']
          gt_verts_mask = labels['gt_verts_mask']
          gt_faces_mask = labels['gt_faces_mask']
          rot_mat = labels['rot_mat']
          trans_mat = labels['trans_mat']
          trans_mat = labels['trans_mat']
        else:
          anchor_boxes = labels['anchor_boxes']
          gt_boxes = labels['groundtruths']['boxes']
          gt_classes = labels['groundtruths']['classes']
          gt_voxel = labels['groundtruths']['voxel']
          gt_verts = labels['groundtruths']['verts']
          gt_faces = labels['groundtruths']['faces']
          gt_voxel_mask = labels['groundtruths']['voxel_mask']
          gt_verts_mask = labels['groundtruths']['verts_mask']
          gt_faces_mask = labels['groundtruths']['faces_mask']
          rot_mat = labels['rot_mat']
          trans_mat = labels['trans_mat']
          trans_mat = labels['trans_mat']

        for j in range(params.parser.min_level, params.parser.max_level + 1):
          self.assertAllEqual(
              tf.shape(anchor_boxes[str(j)]),
              [batch_size, 800 / (2 ** j), 800 / (2 ** j), 12])

          if is_training:
            self.assertAllEqual(
                tf.shape(rpn_score_targets[str(j)]),
                [batch_size, 800 / (2 ** j), 800 / (2 ** j), 3])

            self.assertAllEqual(
                tf.shape(rpn_box_targets[str(j)]),
                [batch_size, 800 / (2 ** j), 800 / (2 ** j), 12])

        self.assertNotEqual(
            tf.reduce_max(gt_boxes), -1)

        self.assertAllEqual(
            tf.shape(gt_boxes),
            [batch_size, params.parser.max_num_instances, 4])

        self.assertAllEqual(
            tf.shape(gt_classes),
            [batch_size, params.parser.max_num_instances])

        self.assertAllEqual(
            tf.shape(gt_voxel),
            [batch_size, params.parser.max_num_voxels, 3])

        self.assertAllEqual(
            tf.shape(gt_verts),
            [batch_size, params.parser.max_num_verts, 3])

        self.assertAllEqual(
            tf.shape(gt_faces),
            [batch_size, params.parser.max_num_faces, 3])

        self.assertAllEqual(
            tf.shape(gt_voxel_mask),
            [batch_size, params.parser.max_num_voxels])

        self.assertAllEqual(
            tf.shape(gt_verts_mask),
            [batch_size, params.parser.max_num_verts])

        self.assertAllEqual(
            tf.shape(gt_faces_mask),
            [batch_size, params.parser.max_num_faces])

        self.assertAllEqual(tf.shape(rot_mat), [batch_size, 3, 3])
        self.assertAllEqual(tf.shape(trans_mat), [batch_size, 3])
        self.assertAllEqual(tf.shape(trans_mat), [batch_size, 3])

        if i >= 5:
          break

if __name__ == '__main__':
  tf.test.main()
