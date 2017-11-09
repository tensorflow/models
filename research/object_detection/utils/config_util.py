# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Functions for reading and updating configuration files."""

import tensorflow as tf

from google.protobuf import text_format

from object_detection.protos import eval_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2


def get_configs_from_pipeline_file(pipeline_config_path):
  """Reads configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text
      proto.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Value are the
      corresponding config objects.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

  configs = {}
  configs["model"] = pipeline_config.model
  configs["train_config"] = pipeline_config.train_config
  configs["train_input_config"] = pipeline_config.train_input_reader
  configs["eval_config"] = pipeline_config.eval_config
  configs["eval_input_config"] = pipeline_config.eval_input_reader

  return configs


def create_pipeline_proto_from_configs(configs):
  """Creates a pipeline_pb2.TrainEvalPipelineConfig from configs dictionary.

  This function nearly performs the inverse operation of
  get_configs_from_pipeline_file(). Instead of returning a file path, it returns
  a `TrainEvalPipelineConfig` object.

  Args:
    configs: Dictionary of configs. See get_configs_from_pipeline_file().

  Returns:
    A fully populated pipeline_pb2.TrainEvalPipelineConfig.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  pipeline_config.model.CopyFrom(configs["model"])
  pipeline_config.train_config.CopyFrom(configs["train_config"])
  pipeline_config.train_input_reader.CopyFrom(configs["train_input_config"])
  pipeline_config.eval_config.CopyFrom(configs["eval_config"])
  pipeline_config.eval_input_reader.CopyFrom(configs["eval_input_config"])
  return pipeline_config


def get_configs_from_multiple_files(model_config_path="",
                                    train_config_path="",
                                    train_input_config_path="",
                                    eval_config_path="",
                                    eval_input_config_path=""):
  """Reads training configuration from multiple config files.

  Args:
    model_config_path: Path to model_pb2.DetectionModel.
    train_config_path: Path to train_pb2.TrainConfig.
    train_input_config_path: Path to input_reader_pb2.InputReader.
    eval_config_path: Path to eval_pb2.EvalConfig.
    eval_input_config_path: Path to input_reader_pb2.InputReader.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Key/Values are
        returned only for valid (non-empty) strings.
  """
  configs = {}
  if model_config_path:
    model_config = model_pb2.DetectionModel()
    with tf.gfile.GFile(model_config_path, "r") as f:
      text_format.Merge(f.read(), model_config)
      configs["model"] = model_config

  if train_config_path:
    train_config = train_pb2.TrainConfig()
    with tf.gfile.GFile(train_config_path, "r") as f:
      text_format.Merge(f.read(), train_config)
      configs["train_config"] = train_config

  if train_input_config_path:
    train_input_config = input_reader_pb2.InputReader()
    with tf.gfile.GFile(train_input_config_path, "r") as f:
      text_format.Merge(f.read(), train_input_config)
      configs["train_input_config"] = train_input_config

  if eval_config_path:
    eval_config = eval_pb2.EvalConfig()
    with tf.gfile.GFile(eval_config_path, "r") as f:
      text_format.Merge(f.read(), eval_config)
      configs["eval_config"] = eval_config

  if eval_input_config_path:
    eval_input_config = input_reader_pb2.InputReader()
    with tf.gfile.GFile(eval_input_config_path, "r") as f:
      text_format.Merge(f.read(), eval_input_config)
      configs["eval_input_config"] = eval_input_config

  return configs


def get_number_of_classes(model_config):
  """Returns the number of classes for a detection model.

  Args:
    model_config: A model_pb2.DetectionModel.

  Returns:
    Number of classes.

  Raises:
    ValueError: If the model type is not recognized.
  """
  meta_architecture = model_config.WhichOneof("model")
  if meta_architecture == "faster_rcnn":
    return model_config.faster_rcnn.num_classes
  if meta_architecture == "ssd":
    return model_config.ssd.num_classes

  raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")


def get_optimizer_type(train_config):
  """Returns the optimizer type for training.

  Args:
    train_config: A train_pb2.TrainConfig.

  Returns:
    The type of the optimizer
  """
  return train_config.optimizer.WhichOneof("optimizer")


def get_learning_rate_type(optimizer_config):
  """Returns the learning rate type for training.

  Args:
    optimizer_config: An optimizer_pb2.Optimizer.

  Returns:
    The type of the learning rate.
  """
  return optimizer_config.learning_rate.WhichOneof("learning_rate")


def merge_external_params_with_configs(configs, hparams=None, **kwargs):
  """Updates `configs` dictionary based on supplied parameters.

  This utility is for modifying specific fields in the object detection configs.
  Say that one would like to experiment with different learning rates, momentum
  values, or batch sizes. Rather than creating a new config text file for each
  experiment, one can use a single base config file, and update particular
  values.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    hparams: A `HParams`.
    **kwargs: Extra keyword arguments that are treated the same way as
      attribute/value pairs in `hparams`. Note that hyperparameters with the
      same names will override keyword arguments.

  Returns:
    `configs` dictionary.
  """

  if hparams:
    kwargs.update(hparams.values())
  for key, value in kwargs.items():
    if key == "learning_rate":
      _update_initial_learning_rate(configs, value)
      tf.logging.info("Overwriting learning rate: %f", value)
    if key == "batch_size":
      _update_batch_size(configs, value)
      tf.logging.info("Overwriting batch size: %d", value)
    if key == "momentum_optimizer_value":
      _update_momentum_optimizer_value(configs, value)
      tf.logging.info("Overwriting momentum optimizer value: %f", value)
    if key == "classification_localization_weight_ratio":
      # Localization weight is fixed to 1.0.
      _update_classification_localization_weight_ratio(configs, value)
    if key == "focal_loss_gamma":
      _update_focal_loss_gamma(configs, value)
    if key == "focal_loss_alpha":
      _update_focal_loss_alpha(configs, value)
    if key == "train_steps":
      _update_train_steps(configs, value)
      tf.logging.info("Overwriting train steps: %d", value)
    if key == "eval_steps":
      _update_eval_steps(configs, value)
      tf.logging.info("Overwriting eval steps: %d", value)
    if key == "train_input_path":
      _update_input_path(configs["train_input_config"], value)
      tf.logging.info("Overwriting train input path: %s", value)
    if key == "eval_input_path":
      _update_input_path(configs["eval_input_config"], value)
      tf.logging.info("Overwriting eval input path: %s", value)
    if key == "label_map_path":
      if value:
        _update_label_map_path(configs, value)
        tf.logging.info("Overwriting label map path: %s", value)
  return configs


def _update_initial_learning_rate(configs, learning_rate):
  """Updates `configs` to reflect the new initial learning rate.

  The configs dictionary is updated in place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    learning_rate: Initial learning rate for optimizer.

  Raises:
    TypeError: if optimizer type is not supported, or if learning rate type is
      not supported.
  """

  optimizer_type = get_optimizer_type(configs["train_config"])
  if optimizer_type == "rms_prop_optimizer":
    optimizer_config = configs["train_config"].optimizer.rms_prop_optimizer
  elif optimizer_type == "momentum_optimizer":
    optimizer_config = configs["train_config"].optimizer.momentum_optimizer
  elif optimizer_type == "adam_optimizer":
    optimizer_config = configs["train_config"].optimizer.adam_optimizer
  else:
    raise TypeError("Optimizer %s is not supported." % optimizer_type)

  learning_rate_type = get_learning_rate_type(optimizer_config)
  if learning_rate_type == "constant_learning_rate":
    constant_lr = optimizer_config.learning_rate.constant_learning_rate
    constant_lr.learning_rate = learning_rate
  elif learning_rate_type == "exponential_decay_learning_rate":
    exponential_lr = (
        optimizer_config.learning_rate.exponential_decay_learning_rate)
    exponential_lr.initial_learning_rate = learning_rate
  elif learning_rate_type == "manual_step_learning_rate":
    manual_lr = optimizer_config.learning_rate.manual_step_learning_rate
    original_learning_rate = manual_lr.initial_learning_rate
    learning_rate_scaling = float(learning_rate) / original_learning_rate
    manual_lr.initial_learning_rate = learning_rate
    for schedule in manual_lr.schedule:
      schedule.learning_rate *= learning_rate_scaling
  else:
    raise TypeError("Learning rate %s is not supported." % learning_rate_type)


def _update_batch_size(configs, batch_size):
  """Updates `configs` to reflect the new training batch size.

  The configs dictionary is updated in place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    batch_size: Batch size to use for training (Ideally a power of 2). Inputs
      are rounded, and capped to be 1 or greater.
  """
  configs["train_config"].batch_size = max(1, int(round(batch_size)))


def _update_momentum_optimizer_value(configs, momentum):
  """Updates `configs` to reflect the new momentum value.

  Momentum is only supported for RMSPropOptimizer and MomentumOptimizer. For any
  other optimizer, no changes take place. The configs dictionary is updated in
  place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    momentum: New momentum value. Values are clipped at 0.0 and 1.0.

  Raises:
    TypeError: If the optimizer type is not `rms_prop_optimizer` or
    `momentum_optimizer`.
  """
  optimizer_type = get_optimizer_type(configs["train_config"])
  if optimizer_type == "rms_prop_optimizer":
    optimizer_config = configs["train_config"].optimizer.rms_prop_optimizer
  elif optimizer_type == "momentum_optimizer":
    optimizer_config = configs["train_config"].optimizer.momentum_optimizer
  else:
    raise TypeError("Optimizer type must be one of `rms_prop_optimizer` or "
                    "`momentum_optimizer`.")

  optimizer_config.momentum_optimizer_value = min(max(0.0, momentum), 1.0)


def _update_classification_localization_weight_ratio(configs, ratio):
  """Updates the classification/localization weight loss ratio.

  Detection models usually define a loss weight for both classification and
  objectness. This function updates the weights such that the ratio between
  classification weight to localization weight is the ratio provided.
  Arbitrarily, localization weight is set to 1.0.

  Note that in the case of Faster R-CNN, this same ratio is applied to the first
  stage objectness loss weight relative to localization loss weight.

  The configs dictionary is updated in place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    ratio: Desired ratio of classification (and/or objectness) loss weight to
      localization loss weight.
  """
  meta_architecture = configs["model"].WhichOneof("model")
  if meta_architecture == "faster_rcnn":
    model = configs["model"].faster_rcnn
    model.first_stage_localization_loss_weight = 1.0
    model.first_stage_objectness_loss_weight = ratio
    model.second_stage_localization_loss_weight = 1.0
    model.second_stage_classification_loss_weight = ratio
  if meta_architecture == "ssd":
    model = configs["model"].ssd
    model.loss.localization_weight = 1.0
    model.loss.classification_weight = ratio


def _get_classification_loss(model_config):
  """Returns the classification loss for a model."""
  meta_architecture = model_config.WhichOneof("model")
  if meta_architecture == "faster_rcnn":
    model = model_config.faster_rcnn
    classification_loss = model.second_stage_classification_loss
  if meta_architecture == "ssd":
    model = model_config.ssd
    classification_loss = model.loss.classification_loss
  else:
    raise TypeError("Did not recognize the model architecture.")
  return classification_loss


def _update_focal_loss_gamma(configs, gamma):
  """Updates the gamma value for a sigmoid focal loss.

  The configs dictionary is updated in place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    gamma: Exponent term in focal loss.

  Raises:
    TypeError: If the classification loss is not `weighted_sigmoid_focal`.
  """
  classification_loss = _get_classification_loss(configs["model"])
  classification_loss_type = classification_loss.WhichOneof(
      "classification_loss")
  if classification_loss_type != "weighted_sigmoid_focal":
    raise TypeError("Classification loss must be `weighted_sigmoid_focal`.")
  classification_loss.weighted_sigmoid_focal.gamma = gamma


def _update_focal_loss_alpha(configs, alpha):
  """Updates the alpha value for a sigmoid focal loss.

  The configs dictionary is updated in place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    alpha: Class weight multiplier for sigmoid loss.

  Raises:
    TypeError: If the classification loss is not `weighted_sigmoid_focal`.
  """
  classification_loss = _get_classification_loss(configs["model"])
  classification_loss_type = classification_loss.WhichOneof(
      "classification_loss")
  if classification_loss_type != "weighted_sigmoid_focal":
    raise TypeError("Classification loss must be `weighted_sigmoid_focal`.")
  classification_loss.weighted_sigmoid_focal.alpha = alpha


def _update_train_steps(configs, train_steps):
  """Updates `configs` to reflect new number of training steps."""
  configs["train_config"].num_steps = int(train_steps)


def _update_eval_steps(configs, eval_steps):
  """Updates `configs` to reflect new number of eval steps per evaluation."""
  configs["eval_config"].num_examples = int(eval_steps)


def _update_input_path(input_config, input_path):
  """Updates input configuration to reflect a new input path.

  The input_config object is updated in place, and hence not returned.

  Args:
    input_config: A input_reader_pb2.InputReader.
    input_path: A path to data or list of paths.

  Raises:
    TypeError: if input reader type is not `tf_record_input_reader`.
  """
  input_reader_type = input_config.WhichOneof("input_reader")
  if input_reader_type == "tf_record_input_reader":
    input_config.tf_record_input_reader.ClearField("input_path")
    if isinstance(input_path, list):
      input_config.tf_record_input_reader.input_path.extend(input_path)
    else:
      input_config.tf_record_input_reader.input_path.append(input_path)
  else:
    raise TypeError("Input reader type must be `tf_record_input_reader`.")


def _update_label_map_path(configs, label_map_path):
  """Updates the label map path for both train and eval input readers.

  The configs dictionary is updated in place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    label_map_path: New path to `StringIntLabelMap` pbtxt file.
  """
  configs["train_input_config"].label_map_path = label_map_path
  configs["eval_input_config"].label_map_path = label_map_path
