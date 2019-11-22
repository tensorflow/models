# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.meta_architectures.rfcn_meta_arch."""

import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch_test_lib
from object_detection.meta_architectures import rfcn_meta_arch


class RFCNMetaArchTest(
    faster_rcnn_meta_arch_test_lib.FasterRCNNMetaArchTestBase):

  def _get_second_stage_box_predictor_text_proto(
      self, share_box_across_classes=False):
    del share_box_across_classes
    box_predictor_text_proto = """
      rfcn_box_predictor {
        conv_hyperparams {
          op: CONV
          activation: NONE
          regularizer {
            l2_regularizer {
              weight: 0.0005
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    """
    return box_predictor_text_proto

  def _get_model(self, box_predictor, **common_kwargs):
    return rfcn_meta_arch.RFCNMetaArch(
        second_stage_rfcn_box_predictor=box_predictor, **common_kwargs)

  def _get_box_classifier_features_shape(self,
                                         image_size,
                                         batch_size,
                                         max_num_proposals,
                                         initial_crop_size,
                                         maxpool_stride,
                                         num_features):
    return (batch_size, image_size, image_size, num_features)


if __name__ == '__main__':
  tf.test.main()
