"""Tests for google3.third_party.tensorflow_models.object_detection.builders.target_assigner_builder."""
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

import tensorflow as tf

from google.protobuf import text_format


from object_detection.builders import target_assigner_builder
from object_detection.core import target_assigner
from object_detection.protos import target_assigner_pb2


class TargetAssignerBuilderTest(tf.test.TestCase):

  def test_build_a_target_assigner(self):
    target_assigner_text_proto = """
      matcher {
        argmax_matcher {matched_threshold: 0.5}
      }
      similarity_calculator {
        iou_similarity {}
      }
      box_coder {
        faster_rcnn_box_coder {}
      }
    """
    target_assigner_proto = target_assigner_pb2.TargetAssigner()
    text_format.Merge(target_assigner_text_proto, target_assigner_proto)
    target_assigner_instance = target_assigner_builder.build(
        target_assigner_proto)
    self.assertIsInstance(target_assigner_instance,
                          target_assigner.TargetAssigner)


if __name__ == '__main__':
  tf.test.main()
