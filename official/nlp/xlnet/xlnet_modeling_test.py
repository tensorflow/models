# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow as tf

from official.nlp.xlnet import xlnet_modeling


class PositionalEmbeddingLayerTest(tf.test.TestCase):

  def test_positional_embedding(self):
    """A low-dimensional example is tested.

     With len(pos_seq)=2 and d_model=4:

       pos_seq  = [[1.], [0.]]
       inv_freq = [1., 0.01]
       pos_seq x inv_freq = [[1, 0.01], [0., 0.]]
       pos_emb = [[sin(1.), sin(0.01), cos(1.), cos(0.01)],
                  [sin(0.), sin(0.), cos(0.), cos(0.)]]
               = [[0.84147096, 0.00999983, 0.54030228, 0.99994999],
                 [0., 0., 1., 1.]]
    """
    target = np.array([[[0.84147096, 0.00999983, 0.54030228, 0.99994999]],
                       [[0., 0., 1., 1.]]])
    d_model = 4
    pos_seq = tf.range(1, -1, -1.0)  # [1., 0.]
    pos_emb_layer = xlnet_modeling.PositionalEmbedding(d_model)
    pos_emb = pos_emb_layer(pos_seq, batch_size=None).numpy().astype(float)

    logging.info(pos_emb)
    self.assertAllClose(pos_emb, target)

if __name__ == "__main__":
  tf.test.main()
