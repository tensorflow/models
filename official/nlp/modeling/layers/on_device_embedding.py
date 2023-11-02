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

"""Keras-based one-hot embedding layer."""
# pylint: disable=g-classes-have-attributes

import tensorflow as tf, tf_keras


@tf_keras.utils.register_keras_serializable(package="Text")
class OnDeviceEmbedding(tf_keras.layers.Layer):
  """Performs an embedding lookup suitable for accelerator devices.

  This layer uses either tf.gather or tf.one_hot to translate integer indices to
  float embeddings.

  Args:
    vocab_size: Number of elements in the vocabulary.
    embedding_width: Output size of the embedding layer.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
      lookup. Defaults to False (that is, using tf.gather). Setting this option
      to True may improve performance, especially on small vocabulary sizes, but
      will generally require more memory.
    scale_factor: Whether to scale the output embeddings. Defaults to None (that
      is, not to scale). Setting this option to a float will let values in
      output embeddings multiplied by scale_factor.
    weight_fallback_dtype: When keras mix precision inferred wrong dtype for
      variables, `weight_fallback_dtype` will be used to define the dtype of
      weights.
  """

  def __init__(self,
               vocab_size,
               embedding_width,
               initializer="glorot_uniform",
               use_one_hot=False,
               scale_factor=None,
               weight_fallback_dtype=tf.float32,
               **kwargs):

    super().__init__(**kwargs)
    self._vocab_size = vocab_size
    self._embedding_width = embedding_width
    self._initializer = initializer
    self._use_one_hot = use_one_hot
    self._scale_factor = scale_factor
    # Backup control of the weight dtype because Keras mix precision sometimes
    # depends on the input to infer the compute dtype, but the inputs of
    # this layer are int type.
    self._weight_fallback_dtype = weight_fallback_dtype

  def get_config(self):
    config = {
        "vocab_size": self._vocab_size,
        "embedding_width": self._embedding_width,
        "initializer": self._initializer,
        "use_one_hot": self._use_one_hot,
        "scale_factor": self._scale_factor,
        "weight_fallback_dtype": self._weight_fallback_dtype,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    if (
        self.dtype is not None
        and not tf.dtypes.as_dtype(self.dtype).is_floating
    ):
      # Keras failed to infer the right dtype.
      dtype = self._weight_fallback_dtype
    else:
      dtype = self.dtype
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self._vocab_size, self._embedding_width],
        initializer=self._initializer,
        dtype=dtype)

    super().build(input_shape)

  def call(self, inputs):
    flat_inputs = tf.reshape(inputs, [-1])
    if self._use_one_hot:
      dtype = self.compute_dtype
      if not tf.dtypes.as_dtype(dtype).is_floating:
        # TensorFlow 1 compatibility. In TF1, self.compute_dtype is int32
        # instead of a floating-point dtype, as the dtype is inferred from the
        # dtype of the inputs
        dtype = self._weight_fallback_dtype
      one_hot_data = tf.one_hot(
          flat_inputs, depth=self._vocab_size, dtype=dtype)
      embeddings = tf.matmul(one_hot_data, self.embeddings)
    else:
      embeddings = tf.gather(self.embeddings, flat_inputs)
    embeddings = tf.reshape(
        embeddings,
        tf.concat([tf.shape(inputs), [self._embedding_width]], axis=0))
    embeddings.set_shape(inputs.shape.as_list() + [self._embedding_width])
    if self._scale_factor:
      embeddings *= self._scale_factor
    return embeddings

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def embedding_width(self):
    return self._embedding_width
