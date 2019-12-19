# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Tests for DeepLab model and some helper functions."""

import tensorflow as tf

from deeplab import common
from deeplab import model


class DeeplabModelTest(tf.test.TestCase):

  def testWrongDeepLabVariant(self):
    model_options = common.ModelOptions([])._replace(
        model_variant='no_such_variant')
    with self.assertRaises(ValueError):
      model._get_logits(images=[], model_options=model_options)

  def testBuildDeepLabv2(self):
    batch_size = 2
    crop_size = [41, 41]

    # Test with two image_pyramids.
    image_pyramids = [[1], [0.5, 1]]

    # Test two model variants.
    model_variants = ['xception_65', 'mobilenet_v2']

    # Test with two output_types.
    outputs_to_num_classes = {'semantic': 3,
                              'direction': 2}

    expected_endpoints = [['merged_logits'],
                          ['merged_logits',
                           'logits_0.50',
                           'logits_1.00']]
    expected_num_logits = [1, 3]

    for model_variant in model_variants:
      model_options = common.ModelOptions(outputs_to_num_classes)._replace(
          add_image_level_feature=False,
          aspp_with_batch_norm=False,
          aspp_with_separable_conv=False,
          model_variant=model_variant)

      for i, image_pyramid in enumerate(image_pyramids):
        g = tf.Graph()
        with g.as_default():
          with self.test_session(graph=g):
            inputs = tf.random_uniform(
                (batch_size, crop_size[0], crop_size[1], 3))
            outputs_to_scales_to_logits = model.multi_scale_logits(
                inputs, model_options, image_pyramid=image_pyramid)

            # Check computed results for each output type.
            for output in outputs_to_num_classes:
              scales_to_logits = outputs_to_scales_to_logits[output]
              self.assertListEqual(sorted(scales_to_logits.keys()),
                                   sorted(expected_endpoints[i]))

              # Expected number of logits = len(image_pyramid) + 1, since the
              # last logits is merged from all the scales.
              self.assertEqual(len(scales_to_logits), expected_num_logits[i])

  def testForwardpassDeepLabv3plus(self):
    crop_size = [33, 33]
    outputs_to_num_classes = {'semantic': 3}

    model_options = common.ModelOptions(
        outputs_to_num_classes,
        crop_size,
        output_stride=16
    )._replace(
        add_image_level_feature=True,
        aspp_with_batch_norm=True,
        logits_kernel_size=1,
        decoder_output_stride=[4],
        model_variant='mobilenet_v2')  # Employ MobileNetv2 for fast test.

    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        inputs = tf.random_uniform(
            (1, crop_size[0], crop_size[1], 3))
        outputs_to_scales_to_logits = model.multi_scale_logits(
            inputs,
            model_options,
            image_pyramid=[1.0])

        sess.run(tf.global_variables_initializer())
        outputs_to_scales_to_logits = sess.run(outputs_to_scales_to_logits)

        # Check computed results for each output type.
        for output in outputs_to_num_classes:
          scales_to_logits = outputs_to_scales_to_logits[output]
          # Expect only one output.
          self.assertEqual(len(scales_to_logits), 1)
          for logits in scales_to_logits.values():
            self.assertTrue(logits.any())

  def testBuildDeepLabWithDensePredictionCell(self):
    batch_size = 1
    crop_size = [33, 33]
    outputs_to_num_classes = {'semantic': 2}
    expected_endpoints = ['merged_logits']
    dense_prediction_cell_config = [
        {'kernel': 3, 'rate': [1, 6], 'op': 'conv', 'input': -1},
        {'kernel': 3, 'rate': [18, 15], 'op': 'conv', 'input': 0},
    ]
    model_options = common.ModelOptions(
        outputs_to_num_classes,
        crop_size,
        output_stride=16)._replace(
            aspp_with_batch_norm=True,
            model_variant='mobilenet_v2',
            dense_prediction_cell_config=dense_prediction_cell_config)
    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g):
        inputs = tf.random_uniform(
            (batch_size, crop_size[0], crop_size[1], 3))
        outputs_to_scales_to_model_results = model.multi_scale_logits(
            inputs,
            model_options,
            image_pyramid=[1.0])
        for output in outputs_to_num_classes:
          scales_to_model_results = outputs_to_scales_to_model_results[output]
          self.assertListEqual(
              list(scales_to_model_results), expected_endpoints)
          self.assertEqual(len(scales_to_model_results), 1)


if __name__ == '__main__':
  tf.test.main()
