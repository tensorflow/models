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

"""Tests for maskrcnn_input."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf
from official.core import input_reader
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.dataloaders import maskrcnn_input
from official.vision.beta.dataloaders import tf_example_decoder


class InputReaderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([1024, 1024], True, True, True),
      ([1024, 1024], True, False, True),
      ([1024, 1024], False, True, True),
      ([1024, 1024], False, False, True),
      ([1024, 1024], True, True, False),
      ([1024, 1024], True, False, False),
      ([1024, 1024], False, True, False),
      ([1024, 1024], False, False, False),
  )
  def testMaskRCNNInputReader(self,
                              output_size,
                              skip_crowd_during_training,
                              include_mask,
                              is_training):
    min_level = 3
    max_level = 7
    num_scales = 3
    aspect_ratios = [1.0, 2.0, 0.5]
    max_num_instances = 100
    batch_size = 2
    mask_crop_size = 112
    anchor_size = 4.0

    params = cfg.DataConfig(
        input_path='/placer/prod/home/snaggletooth/test/data/coco/val*',
        global_batch_size=batch_size,
        is_training=is_training)

    parser = maskrcnn_input.Parser(
        output_size=output_size,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=anchor_size,
        rpn_match_threshold=0.7,
        rpn_unmatched_threshold=0.3,
        rpn_batch_size_per_im=256,
        rpn_fg_fraction=0.5,
        aug_rand_hflip=True,
        aug_scale_min=0.8,
        aug_scale_max=1.2,
        skip_crowd_during_training=skip_crowd_during_training,
        max_num_instances=max_num_instances,
        include_mask=include_mask,
        mask_crop_size=mask_crop_size,
        dtype='bfloat16')

    decoder = tf_example_decoder.TfExampleDecoder(include_mask=include_mask)
    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read()
    iterator = iter(dataset)

    images, labels = next(iterator)

    np_images = images.numpy()
    np_labels = tf.nest.map_structure(lambda x: x.numpy(), labels)

    if is_training:
      self.assertAllEqual(np_images.shape,
                          [batch_size, output_size[0], output_size[1], 3])
      self.assertAllEqual(np_labels['image_info'].shape, [batch_size, 4, 2])
      self.assertAllEqual(np_labels['gt_boxes'].shape,
                          [batch_size, max_num_instances, 4])
      self.assertAllEqual(np_labels['gt_classes'].shape,
                          [batch_size, max_num_instances])
      if include_mask:
        self.assertAllEqual(np_labels['gt_masks'].shape,
                            [batch_size, max_num_instances,
                             mask_crop_size, mask_crop_size])
      for level in range(min_level, max_level + 1):
        stride = 2 ** level
        output_size_l = [output_size[0] / stride, output_size[1] / stride]
        anchors_per_location = num_scales * len(aspect_ratios)
        self.assertAllEqual(
            np_labels['rpn_score_targets'][level].shape,
            [batch_size, output_size_l[0], output_size_l[1],
             anchors_per_location])
        self.assertAllEqual(
            np_labels['rpn_box_targets'][level].shape,
            [batch_size, output_size_l[0], output_size_l[1],
             4 * anchors_per_location])
        self.assertAllEqual(
            np_labels['anchor_boxes'][level].shape,
            [batch_size, output_size_l[0], output_size_l[1],
             4 * anchors_per_location])
    else:
      self.assertAllEqual(np_images.shape,
                          [batch_size, output_size[0], output_size[1], 3])
      self.assertAllEqual(np_labels['image_info'].shape, [batch_size, 4, 2])


if __name__ == '__main__':
  tf.test.main()
