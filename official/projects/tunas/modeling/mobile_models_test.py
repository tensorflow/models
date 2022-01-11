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

"""Tests for mobile_models."""

from absl.testing import parameterized
import numpy as np
import pyglove as pg
from pyglove.tensorflow.keras import layers
import tensorflow as tf

from official.projects.tunas.modeling import mobile_models


class StaticModelTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `mobile_models.static_model`."""

  def testStaticModel(self):
    """Test static model creation."""
    with tf.compat.v1.Graph().as_default():
      tf.compat.v1.set_random_seed(0)
      model = mobile_models.mobilenet_v2()
      inputs1 = tf.ones([1, 224, 224, 3])
      inputs2 = tf.zeros([1, 224, 224, 3])
      outputs1 = model(inputs1)
      tf.print(model.summary())
      print(model.summary())
      print(model.layers)
      print(isinstance(model.layers[0],
                       pg.tensorflow.keras.layers.CompoundLayer))
      self.assertLen(model.trainable_variables, 158)
      num_trainable_params = np.sum([
          np.prod(var.get_shape().as_list())
          for var in model.trainable_variables
      ])
      self.assertEqual(num_trainable_params, 3506153)
      self.assertLen(model.get_updates_for(inputs1), 104)

      outputs2 = model(inputs2)
      self.assertLen(model.trainable_variables, 158)
      self.assertLen(model.get_updates_for(inputs2), 104)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.assertAllClose(
          self.evaluate(tf.reduce_sum(model.losses)), 0.68539262)
      self.evaluate(outputs1)
      self.evaluate(outputs2)

  def testMobileDetEdgeTPU(self):
    """Test MobileDet edge TPU static model."""
    with tf.compat.v1.Graph().as_default():
      tf.compat.v1.set_random_seed(0)
      model = mobile_models.mobiledet_edge_tpu()
      inputs = tf.ones([1, 224, 224, 3])
      outputs = model(inputs)
      self.assertLen(model.trainable_variables, 176)
      num_trainable_params = np.sum([
          np.prod(var.get_shape().as_list())
          for var in model.trainable_variables
      ])
      self.assertEqual(num_trainable_params, 3177497)
      self.assertLen(model.get_updates_for(inputs), 116)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.assertAllClose(self.evaluate(tf.reduce_sum(model.losses)), 0.78207)
      self.evaluate(outputs)

  @parameterized.parameters([
      (mobile_models.mnasnet, 158, 4384593, 104),
      (mobile_models.proxyless_nas_mobile, 185, 4081793, 122),
  ])
  def testTunasStaticModel(self,
                           model_builder,
                           num_trainable_variables,
                           num_params,
                           num_updates):
    """Test MNASNet static model."""
    with tf.compat.v1.Graph().as_default():
      tf.compat.v1.set_random_seed(0)
      model = model_builder()
      inputs = tf.ones([1, 224, 224, 3])
      outputs = model(inputs)
      self.assertLen(model.trainable_variables, num_trainable_variables)
      num_trainable_params = np.sum([
          np.prod(var.get_shape().as_list())
          for var in model.trainable_variables
      ])
      self.assertEqual(num_trainable_params, num_params)
      self.assertLen(model.get_updates_for(inputs), num_updates)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(outputs)

  def testMobileDetEdgeTPUMultipliers(self):
    """Test MobileDet edge TPU static model with multiplier arguments."""
    with tf.compat.v1.Graph().as_default():
      tf.compat.v1.set_random_seed(0)
      model = mobile_models.mobiledet_edge_tpu(
          filters_multipliers=(0.5, 0.625, 0.75, 1.0, 2.0, 3.0, 4.0),
          expansion_multipliers=(6, 8, 10))
      inputs = tf.ones([1, 224, 224, 3])
      outputs = model(inputs)
      self.assertLen(model.trainable_variables, 197)
      num_trainable_params = np.sum([
          np.prod(var.get_shape().as_list())
          for var in model.trainable_variables
      ])
      self.assertEqual(num_trainable_params, 3930105)
      self.assertLen(model.get_updates_for(inputs), 130)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.assertAllClose(self.evaluate(tf.reduce_sum(model.losses)), 1.014057)
      self.evaluate(outputs)

  @parameterized.parameters([
      mobile_models.mobilenet_v2,
      mobile_models.mobiledet_edge_tpu,
      mobile_models.mnasnet,
      mobile_models.proxyless_nas_mobile,
      mobile_models.proxyless_nas_cpu,
      mobile_models.proxyless_nas_gpu
  ])
  def testLayerNamesAreTheSame(self, model_builder):
    """Test variable names are the same with multiple calls."""
    def get_layer_names(model):
      def _is_layer_name(k, v, p):
        del v
        return isinstance(p, tf.keras.layers.Layer) and k.key == 'name'
      return pg.query(model, custom_selector=_is_layer_name)

    self.assertEqual(
        get_layer_names(model_builder()),
        get_layer_names(model_builder()))


class SearchModelTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `mobile_models.search_model`."""

  def testSearchModel(self):
    """Test search model."""
    search_model = mobile_models.mobilenet_v2_filters_search()
    dna_spec = pg.dna_spec(search_model)

    # The search space only contains 9 filters (2 conv + 7 blocks)
    self.assertLen(dna_spec.elements, 9)

    # Make sure MobileNetV2 is one point in the search space.
    # To do so, we first modify the search space by using the same momentum
    # for BatchNormalization, and remove the name for MobileNetV2.
    pg.patch_on_member(
        search_model, layers.BatchNormalization, 'momentum', 0.99)
    mobilenetv2 = mobile_models.mobilenet_v2()
    dna = pg.template(search_model).encode(
        mobilenetv2.rebind(name='mobilenet_v2_filters_search'))
    self.assertEqual(dna, pg.DNA.parse([2, 1, 1, 2, 3, 3, 3, 3, 3]))

  def testProxylessSearchModel(self):
    """Test proxyless search model."""
    search_model = mobile_models.proxylessnas_search()
    dna_spec = pg.dna_spec(search_model)

    # The search space only contains 9 filters (2 conv + 7 blocks)
    self.assertLen(dna_spec.elements, 22)

    # Make sure ProxylessNASMobile is one point in the search space.
    # To do so, we first modify the search space by using the same momentum
    # for BatchNormalization, and remove the name for ProxylessNASMobile.
    pg.patch_on_member(
        search_model, layers.BatchNormalization, 'momentum', 0.99)
    proxyless_nas_mobile = mobile_models.proxyless_nas_mobile()
    dna = pg.template(search_model).encode(
        proxyless_nas_mobile.rebind(name='proxylessnas_search'))
    self.assertEqual(
        dna,
        pg.DNA.parse(list(mobile_models.PROXYLESSNAS_MOBILE_OPERATIONS)))

if __name__ == '__main__':
  tf.test.main()
