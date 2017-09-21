# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Base wrapper class for performing inference with an image-to-text model.

Subclasses must implement the following methods:

  build_model():
    Builds the model for inference and returns the model object.

  feed_image():
    Takes an encoded image and returns the initial model state, where "state"
    is a numpy array whose specifics are defined by the subclass, e.g.
    concatenated LSTM state. It's assumed that feed_image() will be called
    precisely once at the start of inference for each image. Subclasses may
    compute and/or save per-image internal context in this method.

  inference_step():
    Takes a batch of inputs and states at a single time-step. Returns the
    softmax output corresponding to the inputs, and the new states of the batch.
    Optionally also returns metadata about the current inference step, e.g. a
    serialized numpy array containing activations from a particular model layer.

Client usage:
  1. Build the model inference graph via build_graph_from_config() or
     build_graph_from_proto().
  2. Call the resulting restore_fn to load the model checkpoint.
  3. For each image in a batch of images:
     a) Call feed_image() once to get the initial state.
     b) For each step of caption generation, call inference_step().
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path


import tensorflow as tf

# pylint: disable=unused-argument


class InferenceWrapperBase(object):
  """Base wrapper class for performing inference with an image-to-text model."""

  def __init__(self):
    pass

  def build_model(self, model_config):
    """Builds the model for inference.

    Args:
      model_config: Object containing configuration for building the model.

    Returns:
      model: The model object.
    """
    tf.logging.fatal("Please implement build_model in subclass")

  def _create_restore_fn(self, checkpoint_path, saver):
    """Creates a function that restores a model from checkpoint.

    Args:
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.
      saver: Saver for restoring variables from the checkpoint file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.

    Raises:
      ValueError: If checkpoint_path does not refer to a checkpoint file or a
        directory containing a checkpoint file.
    """
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_path:
        raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

    def _restore_fn(sess):
      tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Successfully loaded checkpoint: %s",
                      os.path.basename(checkpoint_path))

    return _restore_fn

  def build_graph_from_config(self, model_config, checkpoint_path):
    """Builds the inference graph from a configuration object.

    Args:
      model_config: Object containing configuration for building the model.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    tf.logging.info("Building model.")
    self.build_model(model_config)
    saver = tf.train.Saver()

    return self._create_restore_fn(checkpoint_path, saver)

  def build_graph_from_proto(self, graph_def_file, saver_def_file,
                             checkpoint_path):
    """Builds the inference graph from serialized GraphDef and SaverDef protos.

    Args:
      graph_def_file: File containing a serialized GraphDef proto.
      saver_def_file: File containing a serialized SaverDef proto.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    # Load the Graph.
    tf.logging.info("Loading GraphDef from file: %s", graph_def_file)
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(graph_def_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    # Load the Saver.
    tf.logging.info("Loading SaverDef from file: %s", saver_def_file)
    saver_def = tf.train.SaverDef()
    with tf.gfile.FastGFile(saver_def_file, "rb") as f:
      saver_def.ParseFromString(f.read())
    saver = tf.train.Saver(saver_def=saver_def)

    return self._create_restore_fn(checkpoint_path, saver)

  def feed_image(self, sess, encoded_image):
    """Feeds an image and returns the initial model state.

    See comments at the top of file.

    Args:
      sess: TensorFlow Session object.
      encoded_image: An encoded image string.

    Returns:
      state: A numpy array of shape [1, state_size].
    """
    tf.logging.fatal("Please implement feed_image in subclass")

  def inference_step(self, sess, input_feed, state_feed):
    """Runs one step of inference.

    Args:
      sess: TensorFlow Session object.
      input_feed: A numpy array of shape [batch_size].
      state_feed: A numpy array of shape [batch_size, state_size].

    Returns:
      softmax_output: A numpy array of shape [batch_size, vocab_size].
      new_state: A numpy array of shape [batch_size, state_size].
      metadata: Optional. If not None, a string containing metadata about the
        current inference step (e.g. serialized numpy array containing
        activations from a particular model layer.).
    """
    tf.logging.fatal("Please implement inference_step in subclass")

# pylint: enable=unused-argument
