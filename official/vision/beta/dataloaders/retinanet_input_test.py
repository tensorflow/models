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
"""Tests for retinanet_parser.py."""

# Import libraries
from absl.testing import parameterized

import tensorflow as tf
from official.core import input_reader
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.dataloaders import retinanet_input
from official.vision.beta.dataloaders import tf_example_decoder


class RetinaNetInputTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([512, 640], True, True, True),
      ([640, 640], False, False, False),
  )
  def testRetinanetInputReader(self,
                               output_size,
                               skip_crowd_during_training,
                               use_autoaugment,
                               is_training):

    batch_size = 2
    min_level = 3
    max_level = 7
    num_scales = 3
    aspect_ratios = [0.5, 1.0, 2.0]
    anchor_size = 3
    max_num_instances = 100

    params = cfg.DataConfig(
        input_path='/placer/prod/home/snaggletooth/test/data/coco/val*',
        global_batch_size=batch_size,
        is_training=is_training)

    decoder = tf_example_decoder.TfExampleDecoder()
    parser = retinanet_input.Parser(
        output_size=output_size,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=anchor_size,
        skip_crowd_during_training=skip_crowd_during_training,
        use_autoaugment=use_autoaugment,
        max_num_instances=max_num_instances,
        dtype='bfloat16')

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read()

    iterator = iter(dataset)
    image, labels = next(iterator)
    np_image = image.numpy()
    np_labels = tf.nest.map_structure(lambda x: x.numpy(), labels)

    # Checks image shape.
    self.assertEqual(list(np_image.shape),
                     [batch_size, output_size[0], output_size[1], 3])
    # Checks keys in labels.
    if is_training:
      self.assertCountEqual(
          np_labels.keys(),
          ['cls_targets', 'box_targets', 'anchor_boxes', 'cls_weights',
           'box_weights', 'image_info'])
    else:
      self.assertCountEqual(
          np_labels.keys(),
          ['cls_targets', 'box_targets', 'anchor_boxes', 'cls_weights',
           'box_weights', 'groundtruths', 'image_info'])
    # Checks shapes of `image_info` and `anchor_boxes`.
    self.assertEqual(np_labels['image_info'].shape, (batch_size, 4, 2))
    n_anchors = 0
    for level in range(min_level, max_level + 1):
      stride = 2 ** level
      output_size_l = [output_size[0] / stride, output_size[1] / stride]
      anchors_per_location = num_scales * len(aspect_ratios)
      self.assertEqual(
          list(np_labels['anchor_boxes'][level].shape),
          [batch_size, output_size_l[0], output_size_l[1],
           4 * anchors_per_location])
      n_anchors += output_size_l[0] * output_size_l[1] * anchors_per_location
    # Checks shapes of training objectives.
    self.assertEqual(np_labels['cls_weights'].shape, (batch_size, n_anchors))
    for level in range(min_level, max_level + 1):
      stride = 2 ** level
      output_size_l = [output_size[0] / stride, output_size[1] / stride]
      anchors_per_location = num_scales * len(aspect_ratios)
      self.assertEqual(
          list(np_labels['cls_targets'][level].shape),
          [batch_size, output_size_l[0], output_size_l[1],
           anchors_per_location])
      self.assertEqual(
          list(np_labels['box_targets'][level].shape),
          [batch_size, output_size_l[0], output_size_l[1],
           4 * anchors_per_location])
    # Checks shape of groundtruths for eval.
    if not is_training:
      self.assertEqual(np_labels['groundtruths']['source_id'].shape,
                       (batch_size,))
      self.assertEqual(np_labels['groundtruths']['classes'].shape,
                       (batch_size, max_num_instances))
      self.assertEqual(np_labels['groundtruths']['boxes'].shape,
                       (batch_size, max_num_instances, 4))
      self.assertEqual(np_labels['groundtruths']['areas'].shape,
                       (batch_size, max_num_instances))
      self.assertEqual(np_labels['groundtruths']['is_crowds'].shape,
                       (batch_size, max_num_instances))


if __name__ == '__main__':
  tf.test.main()
