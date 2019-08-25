# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Contains the TFExampleDecoder.

The TFExampleDecode is a DataDecoder used to decode TensorFlow Example protos.
In order to do so each requested item must be paired with one or more Example
features that are parsed to produce the Tensor-based manifestation of the item.
"""

import tensorflow as tf
slim = tf.contrib.slim
data_decoder = slim.data_decoder


class TFSequenceExampleDecoder(data_decoder.DataDecoder):
  """A decoder for TensorFlow SequenceExamples.

  Decoding SequenceExample proto buffers is comprised of two stages:
  (1) Example parsing and (2) tensor manipulation.

  In the first stage, the tf.parse_single_sequence_example function is called
  with a list of FixedLenFeatures and SparseLenFeatures. These instances tell TF
  how to parse the example. The output of this stage is a set of tensors.

  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.

  To perform this decoding operation, a SequenceExampleDecoder is given a list
  of ItemHandlers. Each ItemHandler indicates the set of features for stage 1
  and contains the instructions for post_processing its tensors for stage 2.
  """

  def __init__(self, keys_to_context_features, keys_to_sequence_features,
               items_to_handlers):
    """Constructs the decoder.

    Args:
      keys_to_context_features: a dictionary from TF-SequenceExample context
        keys to either tf.VarLenFeature or tf.FixedLenFeature instances.
        See tensorflow's parsing_ops.py.
      keys_to_sequence_features: a dictionary from TF-SequenceExample sequence
        keys to either tf.VarLenFeature or tf.FixedLenSequenceFeature instances.
        See tensorflow's parsing_ops.py.
      items_to_handlers: a dictionary from items (strings) to ItemHandler
        instances. Note that the ItemHandler's are provided the keys that they
        use to return the final item Tensors.

    Raises:
      ValueError: if the same key is present for context features and sequence
        features.
    """
    unique_keys = set()
    unique_keys.update(keys_to_context_features)
    unique_keys.update(keys_to_sequence_features)
    if len(unique_keys) != (
        len(keys_to_context_features) + len(keys_to_sequence_features)):
      # This situation is ambiguous in the decoder's keys_to_tensors variable.
      raise ValueError('Context and sequence keys are not unique. \n'
                       ' Context keys: %s \n Sequence keys: %s' %
                       (list(keys_to_context_features.keys()),
                        list(keys_to_sequence_features.keys())))

    self._keys_to_context_features = keys_to_context_features
    self._keys_to_sequence_features = keys_to_sequence_features
    self._items_to_handlers = items_to_handlers

  def list_items(self):
    """See base class."""
    return self._items_to_handlers.keys()

  def decode(self, serialized_example, items=None):
    """Decodes the given serialized TF-SequenceExample.

    Args:
      serialized_example: a serialized TF-SequenceExample tensor.
      items: the list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.

    Returns:
      the decoded items, a list of tensor.
    """

    context, feature_list = tf.parse_single_sequence_example(
        serialized_example, self._keys_to_context_features,
        self._keys_to_sequence_features)

    # Reshape non-sparse elements just once:
    for k in self._keys_to_context_features:
      v = self._keys_to_context_features[k]
      if isinstance(v, tf.FixedLenFeature):
        context[k] = tf.reshape(context[k], v.shape)

    if not items:
      items = self._items_to_handlers.keys()

    outputs = []
    for item in items:
      handler = self._items_to_handlers[item]
      keys_to_tensors = {
          key: context[key] if key in context else feature_list[key]
          for key in handler.keys
      }
      outputs.append(handler.tensors_to_item(keys_to_tensors))
    return outputs
