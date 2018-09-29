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

"""Tests for object_detection.predictors.heads.box_head."""
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.predictors.heads import keras_box_head
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case


class ConvolutionalKerasBoxHeadTest(test_case.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
    activation: NONE
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def test_prediction_size_depthwise_false(self):
    conv_hyperparams = self._build_conv_hyperparams()
    box_prediction_head = keras_box_head.ConvolutionalBoxHead(
        is_training=True,
        box_code_size=4,
        kernel_size=3,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=False,
        num_predictions_per_location=1,
        use_depthwise=False)
    image_feature = tf.random_uniform(
        [64, 17, 19, 1024], minval=-10.0, maxval=10.0, dtype=tf.float32)
    box_encodings = box_prediction_head(image_feature)
    self.assertAllEqual([64, 323, 1, 4], box_encodings.get_shape().as_list())

  # TODO(kaftan): Remove conditional after CMLE moves to TF 1.10

if __name__ == '__main__':
  tf.test.main()
