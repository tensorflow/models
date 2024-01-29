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

r"""Generates example dataset for post-training quantization.

Example command line to run the script:

```shell
python3 quantize_movinet.py \
--saved_model_dir=${SAVED_MODEL_DIR} \
--saved_model_with_states_dir=${SAVED_MODEL_WITH_STATES_DIR} \
--output_dataset_dir=${OUTPUT_DATASET_DIR} \
--output_tflite=${OUTPUT_TFLITE} \
--quantization_mode='int_float_fallback' \
--save_dataset_to_tfrecords=True
```

"""

import functools
from typing import Any, Callable, Mapping, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

from official.vision.configs import video_classification as video_classification_configs
from official.vision.tasks import video_classification

tf.enable_v2_behavior()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'saved_model_dir', None, 'The saved_model directory.')
flags.DEFINE_string(
    'saved_model_with_states_dir', None,
    'The directory to the saved_model with state signature. '
    'The saved_model_with_states is needed in order to get the initial state '
    'shape and dtype while saved_model is used for the quantization.')
flags.DEFINE_string(
    'output_tflite', '/tmp/output.tflite',
    'The output tflite file path.')
flags.DEFINE_integer(
    'temporal_stride', 5,
    'Temporal stride used to generate input videos.')
flags.DEFINE_integer(
    'num_frames', 50, 'Input videos number of frames.')
flags.DEFINE_integer(
    'image_size', 172, 'Input videos frame size.')
flags.DEFINE_string(
    'quantization_mode', None,
    'The quantization mode. Can be one of "float16", "int8",'
    '"int_float_fallback" or None.')
flags.DEFINE_integer(
    'num_calibration_videos', 100,
    'Number of videos to run to generate example datasets.')
flags.DEFINE_integer(
    'num_samples_per_video', 3,
    'Number of sample draw from one single video.')
flags.DEFINE_boolean(
    'save_dataset_to_tfrecords', False,
    'Whether to save representative dataset to the disk.')
flags.DEFINE_string(
    'output_dataset_dir', '/tmp/representative_dataset/',
    'The directory to store exported tfrecords.')
flags.DEFINE_integer(
    'max_saved_files', 100,
    'The maximum number of tfrecord files to save.')


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()  # BytesList won't unpack string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _build_tf_example(feature):
  return tf.train.Example(
      features=tf.train.Features(feature=feature)).SerializeToString()


def save_to_tfrecord(input_frame: tf.Tensor,
                     input_states: Mapping[str, tf.Tensor],
                     frame_index: int,
                     predictions: tf.Tensor,
                     output_states: Mapping[str, tf.Tensor],
                     groundtruth_label_id: tf.Tensor,
                     output_dataset_dir: str,
                     file_index: int):
  """Save results to tfrecord."""
  features = {}
  features['frame_id'] = _int64_feature([frame_index])
  features['groundtruth_label'] = _int64_feature(
      groundtruth_label_id.numpy().flatten().tolist())
  features['predictions'] = _float_feature(
      predictions.numpy().flatten().tolist())
  image_string = tf.io.encode_png(
      tf.squeeze(tf.cast(input_frame * 255., tf.uint8), axis=[0, 1]))
  features['image'] = _bytes_feature(image_string.numpy())

  # Input/Output states at time T
  for k, v in output_states.items():
    dtype = v[0].dtype
    if dtype == tf.int32:
      features['input/' + k] = _int64_feature(
          input_states[k].numpy().flatten().tolist())
      features['output/' + k] = _int64_feature(
          output_states[k].numpy().flatten().tolist())
    elif dtype == tf.float32:
      features['input/' + k] = _float_feature(
          input_states[k].numpy().flatten().tolist())
      features['output/' + k] = _float_feature(
          output_states[k].numpy().flatten().tolist())
    else:
      raise ValueError(f'Unrecongized dtype: {dtype}')

  tfe = _build_tf_example(features)
  record_file = '{}/movinet_stream_{:06d}.tfrecords'.format(
      output_dataset_dir, file_index)
  logging.info('Saving to %s.', record_file)
  with tf.io.TFRecordWriter(record_file) as writer:
    writer.write(tfe)


def get_dataset() -> tf.data.Dataset:
  """Gets dataset source."""
  config = video_classification_configs.video_classification_kinetics600()

  temporal_stride = FLAGS.temporal_stride
  num_frames = FLAGS.num_frames
  image_size = FLAGS.image_size
  feature_shape = (num_frames, image_size, image_size, 3)

  config.task.validation_data.global_batch_size = 1
  config.task.validation_data.feature_shape = feature_shape
  config.task.validation_data.temporal_stride = temporal_stride
  config.task.train_data.min_image_size = int(1.125 * image_size)
  config.task.validation_data.dtype = 'float32'
  config.task.validation_data.drop_remainder = False

  task = video_classification.VideoClassificationTask(config.task)

  valid_dataset = task.build_inputs(config.task.validation_data)
  valid_dataset = valid_dataset.map(lambda x, y: (x['image'], y))
  valid_dataset = valid_dataset.prefetch(32)
  return valid_dataset


def stateful_representative_dataset_generator(
    model: tf.keras.Model,
    dataset_iter: Any,
    init_states: Mapping[str, tf.Tensor],
    save_dataset_to_tfrecords: bool = False,
    max_saved_files: int = 100,
    output_dataset_dir: Optional[str] = None,
    num_samples_per_video: int = 3,
    num_calibration_videos: int = 100):
  """Generates sample input data with states.

  Args:
    model: the inference keras model.
    dataset_iter: the dataset source.
    init_states: the initial states for the model.
    save_dataset_to_tfrecords: whether to save the representative dataset to
      tfrecords on disk.
    max_saved_files: the max number of saved tfrecords files.
    output_dataset_dir: the directory to store the saved tfrecords.
    num_samples_per_video: number of randomly sampled frames per video.
    num_calibration_videos: number of calibration videos to run.

  Yields:
    A dictionary of model inputs.
  """
  counter = 0
  for i in range(num_calibration_videos):
    if i % 100 == 0:
      logging.info('Reading representative dateset id %d.', i)

    example_input, example_label = next(dataset_iter)
    groundtruth_label_id = tf.argmax(example_label, axis=-1)
    input_states = init_states
    # split video into frames along the temporal dimension.
    frames = tf.split(example_input, example_input.shape[1], axis=1)

    random_indices = np.random.randint(
        low=1, high=len(frames), size=num_samples_per_video)
    # always include the first frame
    random_indices[0] = 0
    random_indices = set(random_indices)

    for frame_index, frame in enumerate(frames):
      predictions, output_states = model({'image': frame, **input_states})
      if frame_index in random_indices:
        if save_dataset_to_tfrecords and counter < max_saved_files:
          save_to_tfrecord(
              input_frame=frame,
              input_states=input_states,
              frame_index=frame_index,
              predictions=predictions,
              output_states=output_states,
              groundtruth_label_id=groundtruth_label_id,
              output_dataset_dir=output_dataset_dir,
              file_index=counter)
        yield {'image': frame, **input_states}
        counter += 1

      # update states for the next inference step
      input_states = output_states


def get_tflite_converter(
    saved_model_dir: str,
    quantization_mode: str,
    representative_dataset: Optional[Callable[..., Any]] = None
) -> tf.lite.TFLiteConverter:
  """Gets tflite converter."""
  converter = tf.lite.TFLiteConverter.from_saved_model(
      saved_model_dir=saved_model_dir)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  if quantization_mode == 'float16':
    logging.info('Using float16 quantization.')
    converter.target_spec.supported_types = [tf.float16]

  elif quantization_mode == 'int8':
    logging.info('Using full interger quantization.')
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

  elif quantization_mode == 'int_float_fallback':
    logging.info('Using interger quantization with float-point fallback.')
    converter.representative_dataset = representative_dataset

  else:
    logging.info('Using dynamic range quantization.')
  return converter


def quantize_movinet(dataset_fn):
  """Quantizes Movinet."""
  valid_dataset = dataset_fn()
  dataset_iter = iter(valid_dataset)

  # Load model
  encoder = hub.KerasLayer(FLAGS.saved_model_with_states_dir, trainable=False)
  inputs = tf.keras.layers.Input(
      shape=[1, FLAGS.image_size, FLAGS.image_size, 3],
      dtype=tf.float32,
      name='image')

  # Define the state inputs, which is a dict that maps state names to tensors.
  init_states_fn = encoder.resolved_object.signatures['init_states']
  state_shapes = {
      name: ([s if s > 0 else None for s in state.shape], state.dtype)
      for name, state in init_states_fn(
          tf.constant([1, 1, FLAGS.image_size, FLAGS.image_size, 3])).items()
  }
  states_input = {
      name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
      for name, (shape, dtype) in state_shapes.items()
  }

  # The inputs to the model are the states and the video
  inputs = {**states_input, 'image': inputs}
  outputs = encoder(inputs)
  model = tf.keras.Model(inputs, outputs, name='movinet_stream')
  input_shape = tf.constant(
      [1, FLAGS.num_frames, FLAGS.image_size, FLAGS.image_size, 3])
  init_states = init_states_fn(input_shape)

  # config representative_datset_fn
  representative_dataset = functools.partial(
      stateful_representative_dataset_generator,
      model=model,
      dataset_iter=dataset_iter,
      init_states=init_states,
      save_dataset_to_tfrecords=FLAGS.save_dataset_to_tfrecords,
      max_saved_files=FLAGS.max_saved_files,
      output_dataset_dir=FLAGS.output_dataset_dir,
      num_samples_per_video=FLAGS.num_samples_per_video,
      num_calibration_videos=FLAGS.num_calibration_videos)

  converter = get_tflite_converter(
      saved_model_dir=FLAGS.saved_model_dir,
      quantization_mode=FLAGS.quantization_mode,
      representative_dataset=representative_dataset)

  logging.info('Converting...')
  tflite_buffer = converter.convert()
  return tflite_buffer


def main(_):
  tflite_buffer = quantize_movinet(dataset_fn=get_dataset)

  with open(FLAGS.output_tflite, 'wb') as f:
    f.write(tflite_buffer)

  logging.info('tflite model written to %s', FLAGS.output_tflite)

if __name__ == '__main__':
  flags.mark_flag_as_required('saved_model_dir')
  flags.mark_flag_as_required('saved_model_with_states_dir')
  app.run(main)
