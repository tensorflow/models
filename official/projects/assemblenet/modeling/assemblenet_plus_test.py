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

"""Tests for assemblenet++ network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.assemblenet.configs import assemblenet as asn_config
from official.projects.assemblenet.modeling import assemblenet_plus as asnp


class AssembleNetPlusTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((50, True, ''), (50, False, ''),
                            (50, False, 'peer'), (50, True, 'peer'),
                            (50, True, 'self'), (50, False, 'self'))
  def test_network_creation(self, depth, use_object_input, attention_mode):

    batch_size = 2
    num_frames = 32
    img_size = 64
    num_classes = 101  # ufc-101
    num_object_classes = 151  # 151 is for ADE-20k

    if use_object_input:
      vid_input = (batch_size * num_frames, img_size, img_size, 3)
      obj_input = (batch_size * num_frames, img_size, img_size,
                   num_object_classes)
      input_specs = (tf.keras.layers.InputSpec(shape=(vid_input)),
                     tf.keras.layers.InputSpec(shape=(obj_input)))
      vid_inputs = np.random.rand(batch_size * num_frames, img_size, img_size,
                                  3)
      obj_inputs = np.random.rand(batch_size * num_frames, img_size, img_size,
                                  num_object_classes)
      inputs = [vid_inputs, obj_inputs]
      # We are using the full_asnp50_structure, since we feed both video and
      # object.
      model_structure = asn_config.full_asnp50_structure  # Uses object input.
      edge_weights = asn_config.full_asnp_structure_weights
    else:
      # video input: (batch_size, FLAGS.num_frames, image_size, image_size, 3)
      input_specs = tf.keras.layers.InputSpec(
          shape=(batch_size, num_frames, img_size, img_size, 3))
      inputs = np.random.rand(batch_size, num_frames, img_size, img_size, 3)

      # Here, we are using model_structures.asn50_structure for AssembleNet++
      # instead of full_asnp50_structure. By using asn50_structure, it
      # essentially becomes AssembleNet++ without objects, only requiring RGB
      # inputs (and optical flow to be computed inside the model).
      model_structure = asn_config.asn50_structure
      edge_weights = asn_config.asn_structure_weights

    model = asnp.assemblenet_plus(
        assemblenet_depth=depth,
        num_classes=num_classes,
        num_frames=num_frames,
        model_structure=model_structure,
        model_edge_weights=edge_weights,
        input_specs=input_specs,
        use_object_input=use_object_input,
        attention_mode=attention_mode,
    )

    outputs = model(inputs)
    self.assertAllEqual(outputs.shape.as_list(), [batch_size, num_classes])


if __name__ == '__main__':
  tf.test.main()
