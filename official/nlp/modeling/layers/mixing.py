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

"""Keras-based mixing layers.

Based on the mixing layers use by FNet
(https://aclanthology.org/2022.naacl-main.319/) and Sparse Mixers
(https://arxiv.org/abs/2205.12399).

Mixing layers can be used as drop in replacements for self-attention layers. For
interoperability with attention layers, we use the same `query` and `value` call
signature.

Note: These mixing layers currently only support encoder stacks. Decoder stacks
can be supported in the future by utilizing the `value` inputs.
"""

import enum
import functools
from typing import Callable, Tuple, Union

import gin
import numpy as np
from scipy import linalg
import tensorflow as tf

from official.modeling import tf_utils

_Initializer = Union[str, tf.keras.initializers.Initializer]

default_kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=2e-2)


@gin.constants_from_enum
class MixingMechanism(enum.Enum):
  """Determines the type of mixing layer.

  Possible options:
    FOURIER: Fourier Transform mixing.
    LINEAR: Mixing using dense matrix multiplications with learnable weights.
    HARTLEY: Hartley Transform mixing.
  """
  FOURIER = "fourier"
  HARTLEY = "hartley"
  LINEAR = "linear"


class MixingLayer(tf.keras.layers.Layer):
  """Mixing layer base class.

  This class cannot be used directly. It just specifies the API for mixing
  layer subclasses. For interoperability with attention layers, we use the same
  `query` and `value` call signature.

  Based on the mixing layers use by FNet
  (https://aclanthology.org/2022.naacl-main.319/) and Sparse Mixers
  (https://arxiv.org/abs/2205.12399).
  """

  def __init__(self, name: str = "mixing", **kwargs):
    """Initializes layer.

    Args:
      name: Name for layer.
      **kwargs: Keyword arguments.
    """
    super().__init__(name=name, **kwargs)

  def call(self, query: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
    """Calls the layer.

    Subclasses should return tensors of shape
    <float>[batch_size, max_seq_length, hidden_dim].

    Args:
      query: Batch of input embeddings, typically of shape <float>[batch_size,
        max_seq_length, hidden_dim].
      value: Unused. Included to match attention layer API.
      **kwargs: Optional arguments to catch unused attention keyword arguments.

    Raises:
      NotImplementedError. This class should not be called directly.
    """
    raise NotImplementedError("Abstract method")


class FourierTransformLayer(MixingLayer):
  """Fourier Transform layer.

  Applies 2D Fourier Transform over final two dimensions of `query` inputs -
  typically the sequence and hidden dimensions.
  """

  def __init__(self,
               use_fft: bool = False,
               name: str = "fourier_transform",
               **kwargs):
    """Initializes layer.

    Args:
      use_fft: Whether to use Fast Fourier Transform (True) or the Discrete
        Fourier Transform (DFT) matrix (False) to compute the Fourier Transform.
        See _pick_fourier_transform() for recommendations on when to use FFT or
        DFT.
      name: Name for layer.
      **kwargs: Keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self.use_fft = use_fft

  def build(self, input_shape: Tuple[int, ...]):
    """Picks the Fourier Transform implementation."""
    self.fourier_transform = _pick_fourier_transform(
        self.use_fft,
        max_seq_length=input_shape[-2],
        hidden_dim=input_shape[-1])

  def call(self, query: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
    """Applies layer to `query`.

    Args:
      query: Batch of input embeddings, typically of shape <float>[batch_size,
        max_seq_length, hidden_dim].
      value: Unused. Included to match attention layer API.
      **kwargs: Optional arguments to catch unused attention keyword arguments.

    Returns:
      Real part of discrete Fourier Transform of `query` inputs with shape
        <float32>[batch_size, max_seq_length, hidden_dim].
    """
    del value  # Ignored by encoder-only mixing layers
    query = tf.cast(query, tf.complex64)
    return tf.math.real(self.fourier_transform(query))


class HartleyTransformLayer(MixingLayer):
  """Hartley Transform layer.

  Applies 2D Hartley Transform over final two dimensions of `query` inputs -
  typically the sequence and hidden dimensions.
  """

  def __init__(self,
               use_fft: bool = False,
               name: str = "hartley_transform",
               **kwargs):
    """Initializes layer.

    Args:
      use_fft: Whether to use Fast Fourier Transform (True) or the Discrete
        Fourier Transform (DFT) matrix (False) to compute the Hartley Transform.
        See _pick_fourier_transform() for recommendations on when to use FFT or
        DFT.
      name: Name for layer.
      **kwargs: Keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self.use_fft = use_fft

  def build(self, input_shape: Tuple[int, ...]):
    """Picks the Fourier Transform implementation."""
    self.fourier_transform = _pick_fourier_transform(
        self.use_fft,
        max_seq_length=input_shape[-2],
        hidden_dim=input_shape[-1])

  def call(self, query: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
    """Applies layer to `query`.

    Args:
      query: Batch of input embeddings, typically of shape <float>[batch_size,
        max_seq_length, hidden_dim].
      value: Unused. Included to match attention layer API.
      **kwargs: Optional arguments to catch unused attention keyword arguments.

    Returns:
      Real part of discrete Hartley Transform of `query` inputs with shape
        <float32>[batch_size, max_seq_length, hidden_dim].
    """
    del value  # Ignored by encoder-only mixing layers
    query = tf.cast(query, tf.complex64)
    frequencies = self.fourier_transform(query)
    return tf.math.real(frequencies) - tf.math.imag(frequencies)


class LinearTransformLayer(MixingLayer):
  """Dense, linear transformation layer.

  Applies matrix multiplications over sequence and hidden dimensions.
  """

  def __init__(self,
               kernel_initializer: _Initializer = default_kernel_initializer,
               name: str = "linear_transform",
               **kwargs):
    """Initializes layer.

    Args:
      kernel_initializer: Initialization scheme for kernel.
      name: Name for layer.
      **kwargs: Keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self.kernel_initializer = kernel_initializer

  def build(self, input_shape: Tuple[int, ...]):
    """Creates the hidden and sequence matrix variables of the layer."""
    self.mat_hidden = self.add_weight(
        shape=(input_shape[-1], input_shape[-1]),
        initializer=tf_utils.clone_initializer(self.kernel_initializer),
        trainable=True,
        name="hidden_kernel")
    self.mat_seq = self.add_weight(
        shape=(input_shape[-2], input_shape[-2]),
        initializer=tf_utils.clone_initializer(self.kernel_initializer),
        trainable=True,
        name="seq_kernel")

  def call(self, query: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
    """Applies layer to `query`.

    Args:
      query: Batch of input embeddings, typically of shape <float>[batch_size,
        max_seq_length, hidden_dim].
      value: Unused. Included to match attention layer API.
      **kwargs: Optional arguments to catch unused attention keyword arguments.

    Returns:
      Linearly transformed `query` inputs with shape
        <float>[batch_size, max_seq_length, hidden_dim].
    """
    del value  # Ignored by encoder-only mixing layers

    return tf.einsum("bij,jk,ni->bnk", query, self.mat_hidden, self.mat_seq)


def _pick_fourier_transform(
    use_fft: bool, max_seq_length: int,
    hidden_dim: int) -> Callable[[tf.Tensor], tf.Tensor]:
  """Returns FFT or DFT Fourier Transform implementation.

  On TPUs, we recommend using the Discrete Fourier Transform (DFT) matrix
  (use_fft=False), except for very long sequence lengths. On GPUs and CPUs, the
  Fast Fourier Transform (use_fft=True) is generally optimal for all sequence
  lengths.

  Note: When using the FFT it is recommended to use a sequence length that is a
  power of 2.

  Args:
    use_fft: If True, return FFT. Otherwise, return DFT matrix.
    max_seq_length: Maximum sequence length of inputs. Only used if
      use_fft=False.
    hidden_dim: Size of hidden dimension of inputs. Only used if use_fft=False.

  Returns:
    Fourier Transform.
  """
  if use_fft:
    return tf.signal.fft2d
  else:
    dft_mat_seq = linalg.dft(max_seq_length).astype(np.complex64)
    dft_mat_hidden = linalg.dft(hidden_dim).astype(np.complex64)

    def two_dim_matmul(x: tf.Tensor, matrix_dim_one: tf.Tensor,
                       matrix_dim_two: tf.Tensor) -> tf.Tensor:
      """Applies 2D matrix multiplication to input tensors of rank >= 2."""
      return tf.einsum("...ij,jk,ni->...nk", tf.cast(x, tf.complex64),
                       matrix_dim_two, matrix_dim_one)

    return functools.partial(
        two_dim_matmul,
        matrix_dim_one=dft_mat_seq,
        matrix_dim_two=dft_mat_hidden)
