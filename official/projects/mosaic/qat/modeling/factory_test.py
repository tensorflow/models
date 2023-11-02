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

"""Tests for factory.py."""

# Import libraries

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.mosaic.modeling import mosaic_blocks
from official.projects.mosaic.modeling import mosaic_head
from official.projects.mosaic.modeling import mosaic_model
from official.projects.mosaic.qat.modeling import factory as qat_factory
from official.projects.qat.vision.configs import common
from official.vision.modeling import backbones
from official.vision.modeling.heads import segmentation_heads


class SegmentationModelBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, [4, 8], [3, 2], ['concat_merge', 'sum_merge']),
      (128, [1, 4, 8], [3, 2], ['concat_merge', 'sum_merge']),
      (128, [1, 4, 8], [3, 2], ['sum_merge', 'sum_merge']),
      (128, [1, 4, 8], [3, 2], ['concat_merge', 'concat_merge']),
      (512, [1, 4, 8, 16], [3, 2], ['concat_merge', 'sum_merge']),
      (256, [4, 8], [3, 2], ['concat_merge', 'sum_merge']),
      (256, [1, 4, 8], [3, 2], ['concat_merge', 'sum_merge']),
      (256, [1, 4, 8, 16], [3, 2], ['concat_merge', 'sum_merge']),
  )
  def test_mosaic_segmentation_model(self, input_size, pyramid_pool_bin_nums,
                                     decoder_input_levels,
                                     decoder_stage_merge_styles):
    """Test for building and calling of a MOSAIC segmentation network."""
    num_classes = 32
    tf_keras.backend.set_image_data_format('channels_last')
    backbone = backbones.MobileNet(model_id='MobileNetMultiAVGSeg')
    encoder_input_level = 4

    # Create a regular FP32 MOSAIC model.
    neck = mosaic_blocks.MosaicEncoderBlock(
        encoder_input_level=encoder_input_level,
        branch_filter_depths=[64, 64],
        conv_kernel_sizes=[3, 5],
        pyramid_pool_bin_nums=pyramid_pool_bin_nums)
    head = mosaic_head.MosaicDecoderHead(
        num_classes=num_classes,
        decoder_input_levels=decoder_input_levels,
        decoder_stage_merge_styles=decoder_stage_merge_styles,
        decoder_filters=[64, 64],
        decoder_projected_filters=[32, 32])
    mask_scoring_head = segmentation_heads.MaskScoring(
        num_classes=num_classes,
        num_convs=1,
        num_filters=32,
        fc_dims=128,
        num_fcs=2,
        fc_input_size=[8, 8],
    )

    model = mosaic_model.MosaicSegmentationModel(
        backbone=backbone,
        head=head,
        neck=neck,
        mask_scoring_head=mask_scoring_head,
    )

    inputs = np.random.rand(2, input_size, input_size, 3)
    input_specs = tf_keras.layers.InputSpec(shape=inputs.shape)
    expected_outputs = model(inputs)

    # Create a quantized MOSAIC model from the regular FP32 model instance.
    quantization_config = common.Quantization()
    quantized_model = qat_factory.build_qat_mosaic_model(
        model=model,
        quantization=quantization_config,
        input_specs=input_specs)

    actual_output = quantized_model(inputs)
    self.assertAllEqual(actual_output['logits'].numpy().shape,
                        expected_outputs['logits'].numpy().shape)

if __name__ == '__main__':
  tf.test.main()

