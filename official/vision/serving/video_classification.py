# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Video classification input and model functions for serving/inference."""
from typing import Mapping, Dict, Text

import tensorflow as tf, tf_keras

from official.vision.dataloaders import video_input
from official.vision.serving import export_base
from official.vision.tasks import video_classification


class VideoClassificationModule(export_base.ExportModule):
  """Video classification Module."""

  def _build_model(self):
    input_params = self.params.task.train_data
    self._num_frames = input_params.feature_shape[0]
    self._stride = input_params.temporal_stride
    self._min_resize = input_params.min_image_size
    self._crop_size = input_params.feature_shape[1]

    self._output_audio = input_params.output_audio
    task = video_classification.VideoClassificationTask(self.params.task)
    return task.build_model()

  def _decode_tf_example(self, encoded_inputs: tf.Tensor):
    sequence_description = {
        # Each image is a string encoding JPEG.
        video_input.IMAGE_KEY:
            tf.io.FixedLenSequenceFeature((), tf.string),
    }
    if self._output_audio:
      sequence_description[self._params.task.validation_data.audio_feature] = (
          tf.io.VarLenFeature(dtype=tf.float32))
    _, decoded_tensors = tf.io.parse_single_sequence_example(
        encoded_inputs, {}, sequence_description)
    for key, value in decoded_tensors.items():
      if isinstance(value, tf.SparseTensor):
        decoded_tensors[key] = tf.sparse.to_dense(value)
    return decoded_tensors

  def _preprocess_image(self, image):
    image = video_input.process_image(
        image=image,
        is_training=False,
        num_frames=self._num_frames,
        stride=self._stride,
        num_test_clips=1,
        min_resize=self._min_resize,
        crop_size=self._crop_size,
        num_crops=1)
    image = tf.cast(image, tf.float32)  # Use config.
    features = {'image': image}
    return features

  def _preprocess_audio(self, audio):
    features = {}
    audio = tf.cast(audio, dtype=tf.float32)  # Use config.
    audio = video_input.preprocess_ops_3d.sample_sequence(
        audio, 20, random=False, stride=1)
    audio = tf.ensure_shape(
        audio, self._params.task.validation_data.audio_feature_shape)
    features['audio'] = audio
    return features

  @tf.function
  def inference_from_tf_example(
      self, encoded_inputs: tf.Tensor) -> Mapping[str, tf.Tensor]:
    with tf.device('cpu:0'):
      if self._output_audio:
        inputs = tf.map_fn(
            self._decode_tf_example, (encoded_inputs),
            fn_output_signature={
                video_input.IMAGE_KEY: tf.string,
                self._params.task.validation_data.audio_feature: tf.float32
            })
        return self.serve(inputs['image'], inputs['audio'])
      else:
        inputs = tf.map_fn(
            self._decode_tf_example, (encoded_inputs),
            fn_output_signature={
                video_input.IMAGE_KEY: tf.string,
            })
        return self.serve(inputs[video_input.IMAGE_KEY], tf.zeros([1, 1]))

  @tf.function
  def inference_from_image_tensors(
      self, input_frames: tf.Tensor) -> Mapping[str, tf.Tensor]:
    return self.serve(input_frames, tf.zeros([1, 1]))

  @tf.function
  def inference_from_image_audio_tensors(
      self, input_frames: tf.Tensor,
      input_audio: tf.Tensor) -> Mapping[str, tf.Tensor]:
    return self.serve(input_frames, input_audio)

  @tf.function
  def inference_from_image_bytes(self, inputs: tf.Tensor):
    raise NotImplementedError(
        'Video classification do not support image bytes input.')

  def serve(self, input_frames: tf.Tensor, input_audio: tf.Tensor):
    """Cast image to float and run inference.

    Args:
      input_frames: uint8 Tensor of shape [batch_size, None, None, 3]
      input_audio: float32

    Returns:
      Tensor holding classification output logits.
    """
    with tf.device('cpu:0'):
      inputs = tf.map_fn(
          self._preprocess_image, (input_frames),
          fn_output_signature={
              'image': tf.float32,
          })
      if self._output_audio:
        inputs.update(
            tf.map_fn(
                self._preprocess_audio, (input_audio),
                fn_output_signature={'audio': tf.float32}))
    logits = self.inference_step(inputs)
    if self.params.task.train_data.is_multilabel:
      probs = tf.math.sigmoid(logits)
    else:
      probs = tf.nn.softmax(logits)
    return {'logits': logits, 'probs': probs}

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    """Gets defined function signatures.

    Args:
      function_keys: A dictionary with keys as the function to create signature
        for and values as the signature keys when returns.

    Returns:
      A dictionary with key as signature key and value as concrete functions
        that can be used for tf.saved_model.save.
    """
    signatures = {}
    for key, def_name in function_keys.items():
      if key == 'image_tensor':
        input_signature = tf.TensorSpec(
            shape=[self._batch_size] + self._input_image_size + [3],
            dtype=tf.uint8,
            name='INPUT_FRAMES')
        signatures[
            def_name] = self.inference_from_image_tensors.get_concrete_function(
                input_signature)
      elif key == 'frames_audio':
        input_signature = [
            tf.TensorSpec(
                shape=[self._batch_size] + self._input_image_size + [3],
                dtype=tf.uint8,
                name='INPUT_FRAMES'),
            tf.TensorSpec(
                shape=[self._batch_size] +
                self.params.task.train_data.audio_feature_shape,
                dtype=tf.float32,
                name='INPUT_AUDIO')
        ]
        signatures[
            def_name] = self.inference_from_image_audio_tensors.get_concrete_function(
                input_signature)
      elif key == 'serve_examples' or key == 'tf_example':
        input_signature = tf.TensorSpec(
            shape=[self._batch_size], dtype=tf.string)
        signatures[
            def_name] = self.inference_from_tf_example.get_concrete_function(
                input_signature)
      else:
        raise ValueError('Unrecognized `input_type`')
    return signatures
