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

"""Tests for tf_example_builder."""

from absl.testing import parameterized
import tensorflow as tf
from official.vision.data import fake_feature_generator
from official.vision.data import image_utils
from official.vision.data import tf_example_builder


class TfExampleBuilderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('RGB_PNG', 128, 64, 3, 'PNG', 3),
                                  ('RGB_RAW', 128, 128, 3, 'RAW', 0),
                                  ('RGB_JPEG', 64, 128, 3, 'JPEG', [2, 5]))
  def test_add_image_matrix_feature_success(self, height, width, num_channels,
                                            image_format, label):
    # Prepare test data.
    image_np = fake_feature_generator.generate_image_np(height, width,
                                                        num_channels)
    expected_image_bytes = image_utils.encode_image(image_np, image_format)
    hashed_image = bytes('10242048', 'ascii')

    # Run code logic.
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_image_matrix_feature(
        image_np, image_format, hashed_image, label=label)
    example = example_builder.example

    # Verify outputs.
    # Prefer to use string literal for feature keys to directly display the
    # structure of the expected tf.train.Example.
    if isinstance(label, int):
      expected_labels = [label]
    else:
      expected_labels = label
    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[expected_image_bytes])),
                    'image/format':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(image_format, 'ascii')])),
                    'image/height':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[height])),
                    'image/width':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[width])),
                    'image/channels':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[num_channels])),
                    'image/source_id':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[hashed_image])),
                    'image/class/label':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=expected_labels)),
                })), example)

  def test_add_image_matrix_feature_with_feature_prefix_success(self):
    height = 64
    width = 64
    num_channels = 1
    image_format = 'PNG'
    feature_prefix = 'depth'
    label = 8
    image_np = fake_feature_generator.generate_image_np(height, width,
                                                        num_channels)
    expected_image_bytes = image_utils.encode_image(image_np, image_format)
    hashed_image = bytes('10242048', 'ascii')

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_image_matrix_feature(
        image_np,
        image_format,
        hashed_image,
        feature_prefix=feature_prefix,
        label=label)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'depth/image/encoded':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[expected_image_bytes])),
                    'depth/image/format':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(image_format, 'ascii')])),
                    'depth/image/height':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[height])),
                    'depth/image/width':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[width])),
                    'depth/image/channels':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[num_channels])),
                    'depth/image/source_id':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[hashed_image])),
                    'depth/image/class/label':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label]))
                })), example)

  def test_add_encoded_raw_image_feature_success(self):
    height = 128
    width = 128
    num_channels = 3
    image_format = 'RAW'
    expected_image_bytes = bytes('image', 'ascii')
    hashed_image = bytes('16188651', 'ascii')

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_encoded_image_feature(expected_image_bytes, 'RAW',
                                              height, width, num_channels)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[expected_image_bytes])),
                    'image/format':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(image_format, 'ascii')])),
                    'image/height':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[height])),
                    'image/width':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[width])),
                    'image/channels':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[num_channels])),
                    'image/source_id':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[hashed_image]))
                })), example)

  def test_add_encoded_raw_image_feature_valueerror(self):
    image_format = 'RAW'
    image_bytes = tf.bfloat16.as_numpy_dtype
    image_np = fake_feature_generator.generate_image_np(1, 1, 1)
    image_np = image_np.astype(image_bytes)
    expected_image_bytes = image_utils.encode_image(image_np, image_format)

    example_builder = tf_example_builder.TfExampleBuilder()
    with self.assertRaises(ValueError):
      example_builder.add_encoded_image_feature(expected_image_bytes,
                                                image_format)

  @parameterized.product(
      miss_image_format=(True, False),
      miss_height=(True, False),
      miss_width=(True, False),
      miss_num_channels=(True, False),
      miss_label=(True, False))
  def test_add_encoded_image_feature_success(self, miss_image_format,
                                             miss_height, miss_width,
                                             miss_num_channels,
                                             miss_label):
    height = 64
    width = 64
    num_channels = 3
    image_format = 'PNG'
    image_np = fake_feature_generator.generate_image_np(height, width,
                                                        num_channels)
    image_bytes = image_utils.encode_image(image_np, image_format)
    # We don't test on image_source_id because encoding process becomes
    # non-deterministic.
    hashed_image = bytes('10242048', 'ascii')
    label = 5

    image_format = None if miss_image_format else image_format
    height = None if miss_height else height
    width = None if miss_width else width
    num_channels = None if miss_num_channels else num_channels
    label = None if miss_label else label

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_encoded_image_feature(
        image_bytes,
        image_format=image_format,
        height=height,
        width=width,
        num_channels=num_channels,
        image_source_id=hashed_image,
        label=label)
    example = example_builder.example

    expected_features = {
        'image/encoded':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_bytes])),
        'image/format':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[bytes('PNG', 'ascii')])),
        'image/height':
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[64])),
        'image/width':
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[64])),
        'image/channels':
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[3])),
        'image/source_id':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[hashed_image]))}
    if not miss_label:
      expected_features.update({
          'image/class/label':
              tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[label]))})
    self.assertProtoEquals(
        tf.train.Example(features=tf.train.Features(feature=expected_features)),
        example)

  @parameterized.named_parameters(('no_box', 0), ('10_boxes', 10))
  def test_add_normalized_boxes_feature(self, num_boxes):
    normalized_boxes_np = fake_feature_generator.generate_normalized_boxes_np(
        num_boxes)
    ymins, xmins, ymaxs, xmaxs = normalized_boxes_np.T.tolist()
    labels = fake_feature_generator.generate_classes_np(
        2, size=num_boxes).tolist()

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_boxes_feature(
        xmins, xmaxs, ymins, ymaxs, labels=labels, normalized=True)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/object/bbox/xmin':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=xmins)),
                    'image/object/bbox/ymin':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=ymins)),
                    'image/object/bbox/xmax':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=xmaxs)),
                    'image/object/bbox/ymax':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=ymaxs)),
                    'image/object/class/label':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=labels)),
                })), example)

  @parameterized.named_parameters(('no_box', 0), ('10_boxes', 10))
  def test_add_box_pixels_feature(self, num_boxes):
    height, width = 10, 10
    boxes_np = fake_feature_generator.generate_boxes_np(height, width,
                                                        num_boxes)
    ymins, xmins, ymaxs, xmaxs = boxes_np.T.tolist()
    labels = fake_feature_generator.generate_classes_np(
        2, size=num_boxes).tolist()

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_boxes_feature(
        xmins, xmaxs, ymins, ymaxs, labels=labels, normalized=False)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/object/bbox/xmin_pixels':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=xmins)),
                    'image/object/bbox/ymin_pixels':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=ymins)),
                    'image/object/bbox/xmax_pixels':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=xmaxs)),
                    'image/object/bbox/ymax_pixels':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=ymaxs)),
                    'image/object/class/label':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=labels)),
                })), example)

  @parameterized.named_parameters(('no_box', 0), ('10_boxes', 10))
  def test_add_normalized_boxes_feature_with_confidence_and_prefix(
      self, num_boxes):
    normalized_boxes_np = fake_feature_generator.generate_normalized_boxes_np(
        num_boxes)
    ymins, xmins, ymaxs, xmaxs = normalized_boxes_np.T.tolist()
    labels = fake_feature_generator.generate_classes_np(
        2, size=num_boxes).tolist()
    confidences = fake_feature_generator.generate_confidences_np(
        size=num_boxes).tolist()
    feature_prefix = 'predicted'

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_boxes_feature(
        xmins,
        xmaxs,
        ymins,
        ymaxs,
        labels=labels,
        confidences=confidences,
        normalized=True,
        feature_prefix=feature_prefix)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'predicted/image/object/bbox/xmin':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=xmins)),
                    'predicted/image/object/bbox/ymin':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=ymins)),
                    'predicted/image/object/bbox/xmax':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=xmaxs)),
                    'predicted/image/object/bbox/ymax':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=ymaxs)),
                    'predicted/image/object/class/label':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=labels)),
                    'predicted/image/object/bbox/confidence':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=confidences)),
                })), example)

  @parameterized.named_parameters(('no_mask', 128, 64, 0),
                                  ('10_masks', 64, 128, 10))
  def test_add_instance_mask_matrices_feature_success(self, height, width,
                                                      num_masks):
    # Prepare test data.
    instance_masks_np = fake_feature_generator.generate_instance_masks_np(
        height,
        width,
        boxes_np=fake_feature_generator.generate_boxes_np(
            height, width, num_masks),
        normalized=False)
    expected_instance_masks_bytes = list(
        map(lambda x: image_utils.encode_image(x, 'PNG'), instance_masks_np))

    # Run code logic.
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_instance_mask_matrices_feature(instance_masks_np)
    example = example_builder.example

    # Verify outputs.
    # Prefer to use string literal for feature keys to directly display the
    # structure of the expected tf.train.Example.
    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/object/mask':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=expected_instance_masks_bytes)),
                    'image/object/area':
                        # The box area is 4x smaller than the image, and the
                        # mask area is 2x smaller than the box.
                        tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=[height * width / 8] * num_masks))
                })), example)

  @parameterized.named_parameters(('with_mask_areas', True),
                                  ('without_mask_areas', False))
  def test_add_encoded_instance_masks_feature_success(self, has_mask_areas):
    height = 64
    width = 64
    image_format = 'PNG'
    mask_np = fake_feature_generator.generate_semantic_mask_np(height, width, 2)
    mask_bytes = image_utils.encode_image(mask_np, image_format)

    test_masks = [mask_bytes for _ in range(2)]
    mask_areas = [2040., 2040.] if has_mask_areas else None

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_encoded_instance_masks_feature(
        test_masks, mask_areas=mask_areas)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/object/mask':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=test_masks)),
                    'image/object/area':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=[2040., 2040.])),
                })), example)

  @parameterized.named_parameters(
      ('with_visualization_mask', 128, 64, True),
      ('without_visualization_mask', 64, 128, False))
  def test_add_semantic_mask_matrices_feature_success(self, height, width,
                                                      has_visualization_mask):
    # Prepare test data.
    semantic_mask_np = fake_feature_generator.generate_semantic_mask_np(
        height, width, 2)
    image_format = 'PNG'
    expected_feature_dict = {
        'image/segmentation/class/encoded':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[
                    image_utils.encode_image(semantic_mask_np, image_format)
                ])),
        'image/segmentation/class/format':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[bytes(image_format, 'ascii')])),
    }
    visualization_mask_np = None
    if has_visualization_mask:
      visualization_mask_np = fake_feature_generator.generate_image_np(
          height, width)
      expected_feature_dict.update({
          'image/segmentation/class/visualization/encoded':
              tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[
                      image_utils.encode_image(visualization_mask_np,
                                               image_format)
                  ])),
          'image/segmentation/class/visualization/format':
              tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[bytes(image_format, 'ascii')])),
      })

    # Run code logic.
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_semantic_mask_matrix_feature(semantic_mask_np,
                                                     image_format,
                                                     visualization_mask_np,
                                                     image_format)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(feature=expected_feature_dict)), example)

  @parameterized.named_parameters(('with_visualization_mask', True),
                                  ('without_visualization_mask', False))
  def test_add_encoded_semantic_mask_feature_success(self,
                                                     has_visualization_mask):
    height, width = 64, 64
    semantic_mask_np = fake_feature_generator.generate_semantic_mask_np(
        height, width, 2)
    image_format = 'PNG'
    encoded_semantic_mask = image_utils.encode_image(semantic_mask_np,
                                                     image_format)
    expected_feature_dict = {
        'image/segmentation/class/encoded':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[encoded_semantic_mask])),
        'image/segmentation/class/format':
            tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[bytes(image_format, 'ascii')])),
    }
    encoded_visualization_mask = None
    if has_visualization_mask:
      visualization_mask_np = fake_feature_generator.generate_image_np(
          height, width)
      encoded_visualization_mask = image_utils.encode_image(
          visualization_mask_np, image_format)
      expected_feature_dict.update({
          'image/segmentation/class/visualization/encoded':
              tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[encoded_visualization_mask])),
          'image/segmentation/class/visualization/format':
              tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[bytes(image_format, 'ascii')])),
      })

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_encoded_semantic_mask_feature(
        encoded_semantic_mask, image_format, encoded_visualization_mask,
        image_format)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(feature=expected_feature_dict)), example)

  def test_add_panoptic_mask_matrices_feature_success(self):
    # Prepare test data.
    height, width, num_instances = 64, 64, 10
    num_thing_classes, num_semantic_segmentation_classes = 3, 6
    image_format = 'PNG'

    normalized_boxes_np = fake_feature_generator.generate_normalized_boxes_np(
        num_instances)
    instance_masks_np = fake_feature_generator.generate_instance_masks_np(
        height, width, normalized_boxes_np)
    instance_classes_np = fake_feature_generator.generate_classes_np(
        num_thing_classes, num_instances)
    semantic_mask_np = fake_feature_generator.generate_semantic_mask_np(
        height, width, num_semantic_segmentation_classes)
    panoptic_category_mask_np, panoptic_instance_mask_np = (
        fake_feature_generator.generate_panoptic_masks_np(
            semantic_mask_np, instance_masks_np, instance_classes_np,
            num_thing_classes - 1))

    # Run code logic.
    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_panoptic_mask_matrix_feature(panoptic_category_mask_np,
                                                     panoptic_instance_mask_np,
                                                     image_format,
                                                     image_format)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/panoptic/category/encoded':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[
                                image_utils.encode_image(
                                    panoptic_category_mask_np, image_format)
                            ])),
                    'image/panoptic/category/format':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(image_format, 'ascii')])),
                    'image/panoptic/instance/encoded':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[
                                image_utils.encode_image(
                                    panoptic_instance_mask_np, image_format)
                            ])),
                    'image/panoptic/instance/format':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(image_format, 'ascii')])),
                })), example)

  def test_add_encoded_panoptic_mask_feature_success(self):
    # Prepare test data.
    height, width, num_instances = 64, 64, 10
    num_thing_classes, num_semantic_segmentation_classes = 3, 6
    image_format = 'PNG'

    normalized_boxes_np = fake_feature_generator.generate_normalized_boxes_np(
        num_instances)
    instance_masks_np = fake_feature_generator.generate_instance_masks_np(
        height, width, normalized_boxes_np)
    instance_classes_np = fake_feature_generator.generate_classes_np(
        num_thing_classes, num_instances)
    semantic_mask_np = fake_feature_generator.generate_semantic_mask_np(
        height, width, num_semantic_segmentation_classes)
    panoptic_category_mask_np, panoptic_instance_mask_np = (
        fake_feature_generator.generate_panoptic_masks_np(
            semantic_mask_np, instance_masks_np, instance_classes_np,
            num_thing_classes - 1))

    encoded_panoptic_category_mask = image_utils.encode_image(
        panoptic_category_mask_np, image_format)
    encoded_panoptic_instance_mask = image_utils.encode_image(
        panoptic_instance_mask_np, image_format)

    example_builder = tf_example_builder.TfExampleBuilder()
    example_builder.add_encoded_panoptic_mask_feature(
        encoded_panoptic_category_mask, encoded_panoptic_instance_mask,
        image_format, image_format)
    example = example_builder.example

    self.assertProtoEquals(
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/panoptic/category/encoded':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[encoded_panoptic_category_mask])),
                    'image/panoptic/category/format':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(image_format, 'ascii')])),
                    'image/panoptic/instance/encoded':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[encoded_panoptic_instance_mask])),
                    'image/panoptic/instance/format':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(image_format, 'ascii')])),
                })), example)


if __name__ == '__main__':
  tf.test.main()
