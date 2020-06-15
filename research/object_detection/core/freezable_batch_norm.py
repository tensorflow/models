# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""A freezable batch norm layer that uses Keras batch normalization."""
import tensorflow.compat.v1 as tf


class FreezableBatchNorm(tf.keras.layers.BatchNormalization):
  """Batch normalization layer (Ioffe and Szegedy, 2014).

  This is a `freezable` batch norm layer that supports setting the `training`
  parameter in the __init__ method rather than having to set it either via
  the Keras learning phase or via the `call` method parameter. This layer will
  forward all other parameters to the default Keras `BatchNormalization`
  layer

  This is class is necessary because Object Detection model training sometimes
  requires batch normalization layers to be `frozen` and used as if it was
  evaluation time, despite still training (and potentially using dropout layers)

  Like the default Keras BatchNormalization layer, this will normalize the
  activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  Arguments:
    training: If False, the layer will normalize using the moving average and
      std. dev, without updating the learned avg and std. dev.
      If None or True, the layer will follow the keras BatchNormalization layer
      strategy of checking the Keras learning phase at `call` time to decide
      what to do.
    **kwargs: The keyword arguments to forward to the keras BatchNormalization
        layer constructor.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as input.

  References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  def __init__(self, training=None, **kwargs):
    super(FreezableBatchNorm, self).__init__(**kwargs)
    self._training = training

  def call(self, inputs, training=None):
    # Override the call arg only if the batchnorm is frozen. (Ignore None)
    if self._training is False:  # pylint: disable=g-bool-id-comparison
      training = self._training
    return super(FreezableBatchNorm, self).call(inputs, training=training)
