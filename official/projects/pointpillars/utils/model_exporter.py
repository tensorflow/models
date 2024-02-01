# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""PointPillars model export utility function for serving/inference."""

import os
from typing import Any, Dict, Mapping, Optional, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.core import export_base
from official.core import train_utils
from official.projects.pointpillars.modeling import factory
from official.projects.pointpillars.utils import utils


def export_inference_graph(
    batch_size: int,
    params: cfg.ExperimentConfig,
    checkpoint_path: str,
    export_dir: str,
    export_module: Optional[export_base.ExportModule] = None,
):
  """Exports inference graph for PointPillars model.

  Saved model is stored at export_dir/saved_model, checkpoint is saved
  at export_dir/checkpoint, and params is saved at export_dir/params.yaml.

  Args:
    batch_size: An int number specifying batch size for inference.
      Saved PointPillars model doesn't support dynamic batch size.
      Only three batch sizes are acceptable:
      train batch size per replica, evaluation batch size per replica, and 1.
    params: An instance of cfg.ExperimentConfig.
    checkpoint_path: Trained checkpoint path or directory.
    export_dir: Export directory path.
    export_module: Optional export module to be used instead of using params
      to create one.
  """
  logging.info('Exporting model.')
  if not export_module:
    export_module = PointPillarsModule(
        params=params,
        batch_size=batch_size)
  # Disable custom_gradients to make trt-converter be able to work.
  # Consider to use tf_keras.models.save_model/load_model APIs to fix
  # the custom gradients saving problem.
  # https://github.com/tensorflow/tensorflow/issues/40166
  save_options = tf.saved_model.SaveOptions(experimental_custom_gradients=False)
  export_base.export(
      export_module,
      function_keys=['tensors'],
      export_savedmodel_dir=export_dir,
      checkpoint_path=checkpoint_path,
      timestamped=False,
      save_options=save_options)

  logging.info('Saving checkpoint.')
  ckpt = tf.train.Checkpoint(model=export_module.model)
  ckpt.save(os.path.join(export_dir, 'checkpoint', 'ckpt'))

  logging.info('Saving experiment params.')
  train_utils.serialize_config(params, export_dir)


def load_model_predict_fn(export_dir: str) -> Any:
  """Load PointPillars model from saved directory.

  Args:
    export_dir: Export directory path.
  Returns:
    predict_fn: A function can be run for model inference.
  """
  logging.info('Loading model from %s.', export_dir)
  model = tf.saved_model.load(export_dir)
  predict_fn = model.signatures[
      tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  return predict_fn


def random_input_tensors(
    batch_size: int,
    params: cfg.ExperimentConfig) -> Tuple[tf.Tensor, tf.Tensor]:
  """Create random input tensors for PointPillars model.

  Args:
    batch_size: An int number specifying batch size to inference.
    params: An instance of cfg.ExperimentConfig.
  Returns:
    pillars: A tensor for input.
    indices: A tensor for input.
  """
  model_config = params.task.model
  pillars_config = model_config.pillars
  pillars = tf.random.uniform(
      shape=[batch_size,
             pillars_config.num_pillars,
             pillars_config.num_points_per_pillar,
             pillars_config.num_features_per_point],
      minval=0.0,
      maxval=1.0,
      dtype=tf.float32,
      name='pillars')
  indices = tf.random.uniform(
      shape=[batch_size, pillars_config.num_pillars, 2],
      minval=0,
      maxval=model_config.image.height,
      dtype=tf.int32,
      name='indices')
  return pillars, indices


class PointPillarsModule(export_base.ExportModule):
  """PointPillars model export module."""

  def __init__(self, params: cfg.ExperimentConfig, batch_size: int):
    """Initialize the module.

    Args:
      params: Experiment params.
      batch_size: The batch size of the model input.
    """
    self._params = params
    self._batch_size = batch_size
    self._pillars_spec, self._indices_spec = self._build_input_specs()
    model = self._build_model()
    super().__init__(params=params, model=model)

  def _build_input_specs(
      self) -> Tuple[tf_keras.layers.InputSpec, tf_keras.layers.InputSpec]:
    pillars_config = self._params.task.model.pillars
    pillars_spec = tf_keras.layers.InputSpec(
        shape=(self._batch_size,
               pillars_config.num_pillars,
               pillars_config.num_points_per_pillar,
               pillars_config.num_features_per_point),
        dtype='float32')
    indices_spec = tf_keras.layers.InputSpec(
        shape=(self._batch_size,
               pillars_config.num_pillars,
               2),
        dtype='int32')
    return pillars_spec, indices_spec

  def _build_model(self) -> tf_keras.Model:
    logging.info('Building PointPillars model.')
    input_specs = {
        'pillars': self._pillars_spec, 'indices': self._indices_spec
    }
    model = factory.build_pointpillars(
        input_specs=input_specs,
        model_config=self._params.task.model,
        # Train and eval batch size will be ignored for inference.
        train_batch_size=1,
        eval_batch_size=1)
    return model

  def serve(self, pillars: tf.Tensor, indices: tf.Tensor) -> Mapping[str, Any]:
    """Run model inference.

    Args:
      pillars: A float32 tensor.
      indices: An int32 tensor.
    Returns:
      outputs: A dict of detected results.
    """
    # Build image_shape and anchor_boxes on CPU.
    with tf.device('cpu'):
      model_config = self._params.task.model
      image_size = [model_config.image.height,
                    model_config.image.width]

      image_shape = tf.tile(tf.expand_dims(
          image_size, axis=0), [self._batch_size, 1])

      anchor_sizes = [(a.length, a.width) for a in model_config.anchors]
      anchor_boxes = utils.generate_anchors(
          min_level=model_config.min_level,
          max_level=model_config.max_level,
          image_size=image_size,
          anchor_sizes=anchor_sizes)
      for l in anchor_boxes:
        anchor_boxes[l] = tf.tile(
            tf.expand_dims(anchor_boxes[l], axis=0),
            [self._batch_size, 1, 1, 1])

    # Run model.
    detections = self.model.call(
        pillars=pillars,
        indices=indices,
        image_shape=image_shape,
        anchor_boxes=anchor_boxes,
        training=None
    )
    outputs = {
        'detection_boxes': detections['boxes'],
        'detection_scores': detections['scores'],
        'detection_classes': detections['classes'],
        'num_detections': detections['num_detections']
    }
    # NOTE: Need to flatten attributes, because outputs for functions used as
    # signatures must be a single Tensor, a sequence of Tensors, or a dictionary
    # from string to Tensor.
    outputs.update(detections['attributes'])
    return outputs

  @tf.function
  def inference_from_tensors(
      self, pillars: tf.Tensor, indices: tf.Tensor) -> Mapping[str, Any]:
    return self.serve(pillars, indices)

  def get_inference_signatures(
      self, function_keys: Dict[str, str]) -> Mapping[str, Any]:
    """Gets defined function signatures.

    Args:
      function_keys: A dictionary with keys as the function to create signature
        for and values as the signature keys when returns.

    Returns:
      A dictionary with key as signature key and value as concrete functions
        that can be used for tf.saved_model.save.
    """
    signatures = {}
    for input_type, name in function_keys.items():
      if input_type == 'tensors':
        pillars = tf.TensorSpec(
            shape=self._pillars_spec.shape,
            dtype=self._pillars_spec.dtype,
            name='pillars'
        )
        indices = tf.TensorSpec(
            shape=self._indices_spec.shape,
            dtype=self._indices_spec.dtype,
            name='indices'
        )
        signatures[
            name] = self.inference_from_tensors.get_concrete_function(
                pillars, indices)
      else:
        raise ValueError('Unrecognized input_type: {}'.format(input_type))
    return signatures
