# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

from official.nlp.xlnet import xlnet_config
from official.nlp.xlnet import xlnet_models


class XLNetModelsTest(tf.test.TestCase):

  def setUp(self):
    super(XLNetModelsTest, self).setUp()
    self._xlnet_test_config = xlnet_config.XLNetConfig(
        args_dict=dict(
            n_layer=2,
            d_model=4,
            n_head=1,
            d_head=2,
            d_inner=4,
            ff_activation='gelu',
            untie_r=True,
            n_token=32000))
    self._run_config = xlnet_config.RunConfig(
        is_training=True,
        use_tpu=False,
        dropout=0.0,
        dropout_att=0.0,
        init_method='normal',
        init_range=0.1,
        init_std=0.02,
        mem_len=0,
        reuse_len=4,
        bi_data=False,
        clamp_len=-1,
        same_length=False)

  def test_xlnet_base(self):
    xlnet_base = xlnet_models.get_xlnet_base(
        model_config=self._xlnet_test_config,
        run_config=self._run_config,
        attention_type='bi',
        two_stream=False,
        use_cls_mask=False)
    self.assertIsInstance(xlnet_base, tf.keras.layers.Layer)

  def test_xlnet_classifier(self):
    xlnet_classifier, xlnet_base = xlnet_models.classifier_model(
        model_config=self._xlnet_test_config,
        run_config=self._run_config,
        num_labels=2)
    self.assertIsInstance(xlnet_classifier, tf.keras.Model)
    self.assertIsInstance(xlnet_base, tf.keras.layers.Layer)


if __name__ == '__main__':
  tf.test.main()
