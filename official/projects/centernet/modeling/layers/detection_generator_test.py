# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Centernet detection_generator."""

from collections.abc import Mapping, Sequence

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.centernet.modeling.layers import detection_generator


def _build_input_example(
    batch_size: int, height: int, width: int, num_classes: int, num_outputs: int
) -> Mapping[str, Sequence[tf.Tensor]]:
  """Builds a random input example for CenterNetDetectionGenerator.

  Args:
    batch_size: The batch size.
    height: The height of the feature_map.
    width: The width of the feature_map.
    num_classes: The number of classes to detect.
    num_outputs: The number of output heatmaps, which corresponds to the length
      of CenterNetHead's input_levels.

  Returns:
    A dictionary, mapping from feature names to sequences of tensors.
  """
  return {
      'ct_heatmaps': [
          tf.random.normal([batch_size, height, width, num_classes])
          for _ in range(num_outputs)
      ],
      'ct_size': [
          tf.random.normal([batch_size, height, width, 2])
          for _ in range(num_outputs)
      ],
      'ct_offset': [
          tf.random.normal([batch_size, height, width, 2])
          for _ in range(num_outputs)
      ],
  }


class CenterNetDetectionGeneratorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (1, 256),
      (1, 512),
      (2, 256),
      (2, 512),
  )
  def test_squered_image_forward(self, batch_size, input_image_dims):
    max_detections = 128
    num_classes = 80
    generator = detection_generator.CenterNetDetectionGenerator(
        input_image_dims=input_image_dims, max_detections=max_detections
    )
    test_input = _build_input_example(
        batch_size=batch_size,
        height=input_image_dims,
        width=input_image_dims,
        num_classes=num_classes,
        num_outputs=2,
    )

    output = generator(test_input)

    self.assert_detection_generator_output_shapes(
        output, batch_size, max_detections
    )

  @parameterized.parameters(
      (1, (256, 512)),
      (1, (512, 256)),
      (2, (256, 512)),
      (2, (512, 256)),
  )
  def test_rectangular_image_forward(self, batch_size, input_image_dims):
    max_detections = 128
    num_classes = 80
    generator = detection_generator.CenterNetDetectionGenerator(
        input_image_dims=input_image_dims, max_detections=max_detections
    )
    test_input = _build_input_example(
        batch_size=batch_size,
        height=input_image_dims[0],
        width=input_image_dims[1],
        num_classes=num_classes,
        num_outputs=2,
    )

    output = generator(test_input)

    self.assert_detection_generator_output_shapes(
        output, batch_size, max_detections
    )

  def assert_detection_generator_output_shapes(
      self,
      output: Mapping[str, tf.Tensor],
      batch_size: int,
      max_detections: int,
  ):
    self.assertAllEqual(output['boxes'].shape, (batch_size, max_detections, 4))
    self.assertAllEqual(output['classes'].shape, (batch_size, max_detections))
    self.assertAllEqual(
        output['confidence'].shape, (batch_size, max_detections)
    )
    self.assertAllEqual(output['num_detections'].shape, (batch_size,))

  @parameterized.parameters(
      (256,),
      (512,),
      ((256, 512),),
      ((512, 256),),
  )
  def test_serialize_deserialize(self, input_image_dims):
    kwargs = {
        'input_image_dims': input_image_dims,
        'net_down_scale': 4,
        'max_detections': 128,
        'peak_error': 1e-6,
        'peak_extract_kernel_size': 3,
        'class_offset': 1,
        'use_nms': False,
        'nms_pre_thresh': 0.1,
        'nms_thresh': 0.5,
    }

    generator = detection_generator.CenterNetDetectionGenerator(**kwargs)
    new_generator = detection_generator.CenterNetDetectionGenerator.from_config(
        generator.get_config()
    )

    self.assertAllEqual(generator.get_config(), new_generator.get_config())


if __name__ == '__main__':
  tf.test.main()
