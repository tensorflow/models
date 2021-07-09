# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from official.vision.beta.projects.centernet.decoders import centernet_decoder


class CenterNetDecoderTest(tf.test.TestCase, parameterized.TestCase):
  
  def test_decoder_shape(self):
    task_config = {
        'ct_heatmaps': 90,
        'ct_offset': 2,
        'ct_size': 2,
    }
    input_specs = [
        tf.keras.layers.InputSpec(shape=(None, 128, 128, 256)).shape,
        tf.keras.layers.InputSpec(shape=(None, 128, 128, 256)).shape,
    ]
    
    decoder = centernet_decoder.CenterNetDecoder(
        task_outputs=task_config,
        input_specs=input_specs,
        num_inputs=2)
    
    config = decoder.get_config()
    self.assertEqual(config['heatmap_bias'], -2.19)
    
    # Output shape tests
    outputs = decoder([np.zeros((2, 128, 128, 256), dtype=np.float32),
                       np.zeros((2, 128, 128, 256), dtype=np.float32)])
    self.assertLen(outputs, 3)
    self.assertEqual(outputs['ct_heatmaps'][0].shape, (2, 128, 128, 90))
    self.assertEqual(outputs['ct_offset'][0].shape, (2, 128, 128, 2))
    self.assertEqual(outputs['ct_size'][0].shape, (2, 128, 128, 2))
    
    # Weight initialization tests
    tf.print("\n\n{}\n\n".format(decoder.layers))
    hm_bias_vector = np.asarray(decoder.layers[2].weights[-1])
    off_bias_vector = np.asarray(decoder.layers[4].weights[-1])
    size_bias_vector = np.asarray(decoder.layers[6].weights[-1])
    
    self.assertArrayNear(hm_bias_vector,
                         np.repeat(-2.19, repeats=90), err=1.00e-6)
    self.assertArrayNear(off_bias_vector,
                         np.repeat(0, repeats=2), err=1.00e-6)
    self.assertArrayNear(size_bias_vector,
                         np.repeat(0, repeats=2), err=1.00e-6)


if __name__ == '__main__':
  tf.test.main()
