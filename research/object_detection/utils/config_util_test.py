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
"""Tests for object_detection.utils.config_util."""

import os

import tensorflow as tf

from google.protobuf import text_format

from object_detection.protos import eval_pb2
from object_detection.protos import image_resizer_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2
from object_detection.utils import config_util


def _write_config(config, config_path):
  """Writes a config object to disk."""
  config_text = text_format.MessageToString(config)
  with tf.gfile.Open(config_path, "wb") as f:
    f.write(config_text)


def _update_optimizer_with_constant_learning_rate(optimizer, learning_rate):
  """Adds a new constant learning rate."""
  constant_lr = optimizer.learning_rate.constant_learning_rate
  constant_lr.learning_rate = learning_rate


def _update_optimizer_with_exponential_decay_learning_rate(
    optimizer, learning_rate):
  """Adds a new exponential decay learning rate."""
  exponential_lr = optimizer.learning_rate.exponential_decay_learning_rate
  exponential_lr.initial_learning_rate = learning_rate


def _update_optimizer_with_manual_step_learning_rate(
    optimizer, initial_learning_rate, learning_rate_scaling):
  """Adds a learning rate schedule."""
  manual_lr = optimizer.learning_rate.manual_step_learning_rate
  manual_lr.initial_learning_rate = initial_learning_rate
  for i in range(3):
    schedule = manual_lr.schedule.add()
    schedule.learning_rate = initial_learning_rate * learning_rate_scaling**i


def _update_optimizer_with_cosine_decay_learning_rate(
    optimizer, learning_rate, warmup_learning_rate):
  """Adds a new cosine decay learning rate."""
  cosine_lr = optimizer.learning_rate.cosine_decay_learning_rate
  cosine_lr.learning_rate_base = learning_rate
  cosine_lr.warmup_learning_rate = warmup_learning_rate


class ConfigUtilTest(tf.test.TestCase):

  def test_get_configs_from_pipeline_file(self):
    """Test that proto configs can be read from pipeline config file."""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 10
    pipeline_config.train_config.batch_size = 32
    pipeline_config.train_input_reader.label_map_path = "path/to/label_map"
    pipeline_config.eval_config.num_examples = 20
    pipeline_config.eval_input_reader.queue_capacity = 100

    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    self.assertProtoEquals(pipeline_config.model, configs["model"])
    self.assertProtoEquals(pipeline_config.train_config,
                           configs["train_config"])
    self.assertProtoEquals(pipeline_config.train_input_reader,
                           configs["train_input_config"])
    self.assertProtoEquals(pipeline_config.eval_config,
                           configs["eval_config"])
    self.assertProtoEquals(pipeline_config.eval_input_reader,
                           configs["eval_input_config"])

  def test_create_pipeline_proto_from_configs(self):
    """Tests that proto can be reconstructed from configs dictionary."""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 10
    pipeline_config.train_config.batch_size = 32
    pipeline_config.train_input_reader.label_map_path = "path/to/label_map"
    pipeline_config.eval_config.num_examples = 20
    pipeline_config.eval_input_reader.queue_capacity = 100
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    pipeline_config_reconstructed = (
        config_util.create_pipeline_proto_from_configs(configs))
    self.assertEqual(pipeline_config, pipeline_config_reconstructed)

  def test_get_configs_from_multiple_files(self):
    """Tests that proto configs can be read from multiple files."""
    temp_dir = self.get_temp_dir()

    # Write model config file.
    model_config_path = os.path.join(temp_dir, "model.config")
    model = model_pb2.DetectionModel()
    model.faster_rcnn.num_classes = 10
    _write_config(model, model_config_path)

    # Write train config file.
    train_config_path = os.path.join(temp_dir, "train.config")
    train_config = train_config = train_pb2.TrainConfig()
    train_config.batch_size = 32
    _write_config(train_config, train_config_path)

    # Write train input config file.
    train_input_config_path = os.path.join(temp_dir, "train_input.config")
    train_input_config = input_reader_pb2.InputReader()
    train_input_config.label_map_path = "path/to/label_map"
    _write_config(train_input_config, train_input_config_path)

    # Write eval config file.
    eval_config_path = os.path.join(temp_dir, "eval.config")
    eval_config = eval_pb2.EvalConfig()
    eval_config.num_examples = 20
    _write_config(eval_config, eval_config_path)

    # Write eval input config file.
    eval_input_config_path = os.path.join(temp_dir, "eval_input.config")
    eval_input_config = input_reader_pb2.InputReader()
    eval_input_config.label_map_path = "path/to/another/label_map"
    _write_config(eval_input_config, eval_input_config_path)

    configs = config_util.get_configs_from_multiple_files(
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        train_input_config_path=train_input_config_path,
        eval_config_path=eval_config_path,
        eval_input_config_path=eval_input_config_path)
    self.assertProtoEquals(model, configs["model"])
    self.assertProtoEquals(train_config, configs["train_config"])
    self.assertProtoEquals(train_input_config,
                           configs["train_input_config"])
    self.assertProtoEquals(eval_config, configs["eval_config"])
    self.assertProtoEquals(eval_input_config,
                           configs["eval_input_config"])

  def _assertOptimizerWithNewLearningRate(self, optimizer_name):
    """Asserts successful updating of all learning rate schemes."""
    original_learning_rate = 0.7
    learning_rate_scaling = 0.1
    warmup_learning_rate = 0.07
    hparams = tf.contrib.training.HParams(learning_rate=0.15)
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    # Constant learning rate.
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    optimizer = getattr(pipeline_config.train_config.optimizer, optimizer_name)
    _update_optimizer_with_constant_learning_rate(optimizer,
                                                  original_learning_rate)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    optimizer = getattr(configs["train_config"].optimizer, optimizer_name)
    constant_lr = optimizer.learning_rate.constant_learning_rate
    self.assertAlmostEqual(hparams.learning_rate, constant_lr.learning_rate)

    # Exponential decay learning rate.
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    optimizer = getattr(pipeline_config.train_config.optimizer, optimizer_name)
    _update_optimizer_with_exponential_decay_learning_rate(
        optimizer, original_learning_rate)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    optimizer = getattr(configs["train_config"].optimizer, optimizer_name)
    exponential_lr = optimizer.learning_rate.exponential_decay_learning_rate
    self.assertAlmostEqual(hparams.learning_rate,
                           exponential_lr.initial_learning_rate)

    # Manual step learning rate.
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    optimizer = getattr(pipeline_config.train_config.optimizer, optimizer_name)
    _update_optimizer_with_manual_step_learning_rate(
        optimizer, original_learning_rate, learning_rate_scaling)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    optimizer = getattr(configs["train_config"].optimizer, optimizer_name)
    manual_lr = optimizer.learning_rate.manual_step_learning_rate
    self.assertAlmostEqual(hparams.learning_rate,
                           manual_lr.initial_learning_rate)
    for i, schedule in enumerate(manual_lr.schedule):
      self.assertAlmostEqual(hparams.learning_rate * learning_rate_scaling**i,
                             schedule.learning_rate)

    # Cosine decay learning rate.
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    optimizer = getattr(pipeline_config.train_config.optimizer, optimizer_name)
    _update_optimizer_with_cosine_decay_learning_rate(optimizer,
                                                      original_learning_rate,
                                                      warmup_learning_rate)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    optimizer = getattr(configs["train_config"].optimizer, optimizer_name)
    cosine_lr = optimizer.learning_rate.cosine_decay_learning_rate

    self.assertAlmostEqual(hparams.learning_rate, cosine_lr.learning_rate_base)
    warmup_scale_factor = warmup_learning_rate / original_learning_rate
    self.assertAlmostEqual(hparams.learning_rate * warmup_scale_factor,
                           cosine_lr.warmup_learning_rate)

  def testRMSPropWithNewLearingRate(self):
    """Tests new learning rates for RMSProp Optimizer."""
    self._assertOptimizerWithNewLearningRate("rms_prop_optimizer")

  def testMomentumOptimizerWithNewLearningRate(self):
    """Tests new learning rates for Momentum Optimizer."""
    self._assertOptimizerWithNewLearningRate("momentum_optimizer")

  def testAdamOptimizerWithNewLearningRate(self):
    """Tests new learning rates for Adam Optimizer."""
    self._assertOptimizerWithNewLearningRate("adam_optimizer")

  def testNewBatchSize(self):
    """Tests that batch size is updated appropriately."""
    original_batch_size = 2
    hparams = tf.contrib.training.HParams(batch_size=16)
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_config.batch_size = original_batch_size
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    new_batch_size = configs["train_config"].batch_size
    self.assertEqual(16, new_batch_size)

  def testNewBatchSizeWithClipping(self):
    """Tests that batch size is clipped to 1 from below."""
    original_batch_size = 2
    hparams = tf.contrib.training.HParams(batch_size=0.5)
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_config.batch_size = original_batch_size
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    new_batch_size = configs["train_config"].batch_size
    self.assertEqual(1, new_batch_size)  # Clipped to 1.0.

  def testNewMomentumOptimizerValue(self):
    """Tests that new momentum value is updated appropriately."""
    original_momentum_value = 0.4
    hparams = tf.contrib.training.HParams(momentum_optimizer_value=1.1)
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    optimizer_config = pipeline_config.train_config.optimizer.rms_prop_optimizer
    optimizer_config.momentum_optimizer_value = original_momentum_value
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    optimizer_config = configs["train_config"].optimizer.rms_prop_optimizer
    new_momentum_value = optimizer_config.momentum_optimizer_value
    self.assertAlmostEqual(1.0, new_momentum_value)  # Clipped to 1.0.

  def testNewClassificationLocalizationWeightRatio(self):
    """Tests that the loss weight ratio is updated appropriately."""
    original_localization_weight = 0.1
    original_classification_weight = 0.2
    new_weight_ratio = 5.0
    hparams = tf.contrib.training.HParams(
        classification_localization_weight_ratio=new_weight_ratio)
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.ssd.loss.localization_weight = (
        original_localization_weight)
    pipeline_config.model.ssd.loss.classification_weight = (
        original_classification_weight)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    loss = configs["model"].ssd.loss
    self.assertAlmostEqual(1.0, loss.localization_weight)
    self.assertAlmostEqual(new_weight_ratio, loss.classification_weight)

  def testNewFocalLossParameters(self):
    """Tests that the loss weight ratio is updated appropriately."""
    original_alpha = 1.0
    original_gamma = 1.0
    new_alpha = 0.3
    new_gamma = 2.0
    hparams = tf.contrib.training.HParams(
        focal_loss_alpha=new_alpha, focal_loss_gamma=new_gamma)
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    classification_loss = pipeline_config.model.ssd.loss.classification_loss
    classification_loss.weighted_sigmoid_focal.alpha = original_alpha
    classification_loss.weighted_sigmoid_focal.gamma = original_gamma
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    classification_loss = configs["model"].ssd.loss.classification_loss
    self.assertAlmostEqual(new_alpha,
                           classification_loss.weighted_sigmoid_focal.alpha)
    self.assertAlmostEqual(new_gamma,
                           classification_loss.weighted_sigmoid_focal.gamma)

  def testMergingKeywordArguments(self):
    """Tests that keyword arguments get merged as do hyperparameters."""
    original_num_train_steps = 100
    original_num_eval_steps = 5
    desired_num_train_steps = 10
    desired_num_eval_steps = 1
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_config.num_steps = original_num_train_steps
    pipeline_config.eval_config.num_examples = original_num_eval_steps
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(
        configs,
        train_steps=desired_num_train_steps,
        eval_steps=desired_num_eval_steps)
    train_steps = configs["train_config"].num_steps
    eval_steps = configs["eval_config"].num_examples
    self.assertEqual(desired_num_train_steps, train_steps)
    self.assertEqual(desired_num_eval_steps, eval_steps)

  def testGetNumberOfClasses(self):
    """Tests that number of classes can be retrieved."""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 20
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    number_of_classes = config_util.get_number_of_classes(configs["model"])
    self.assertEqual(20, number_of_classes)

  def testNewTrainInputPath(self):
    """Tests that train input path can be overwritten with single file."""
    original_train_path = ["path/to/data"]
    new_train_path = "another/path/to/data"
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    reader_config = pipeline_config.train_input_reader.tf_record_input_reader
    reader_config.input_path.extend(original_train_path)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(
        configs, train_input_path=new_train_path)
    reader_config = configs["train_input_config"].tf_record_input_reader
    final_path = reader_config.input_path
    self.assertEqual([new_train_path], final_path)

  def testNewTrainInputPathList(self):
    """Tests that train input path can be overwritten with multiple files."""
    original_train_path = ["path/to/data"]
    new_train_path = ["another/path/to/data", "yet/another/path/to/data"]
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    reader_config = pipeline_config.train_input_reader.tf_record_input_reader
    reader_config.input_path.extend(original_train_path)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(
        configs, train_input_path=new_train_path)
    reader_config = configs["train_input_config"].tf_record_input_reader
    final_path = reader_config.input_path
    self.assertEqual(new_train_path, final_path)

  def testNewLabelMapPath(self):
    """Tests that label map path can be overwritten in input readers."""
    original_label_map_path = "path/to/original/label_map"
    new_label_map_path = "path//to/new/label_map"
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    train_input_reader = pipeline_config.train_input_reader
    train_input_reader.label_map_path = original_label_map_path
    eval_input_reader = pipeline_config.eval_input_reader
    eval_input_reader.label_map_path = original_label_map_path
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(
        configs, label_map_path=new_label_map_path)
    self.assertEqual(new_label_map_path,
                     configs["train_input_config"].label_map_path)
    self.assertEqual(new_label_map_path,
                     configs["eval_input_config"].label_map_path)

  def testDontOverwriteEmptyLabelMapPath(self):
    """Tests that label map path will not by overwritten with empty string."""
    original_label_map_path = "path/to/original/label_map"
    new_label_map_path = ""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    train_input_reader = pipeline_config.train_input_reader
    train_input_reader.label_map_path = original_label_map_path
    eval_input_reader = pipeline_config.eval_input_reader
    eval_input_reader.label_map_path = original_label_map_path
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(
        configs, label_map_path=new_label_map_path)
    self.assertEqual(original_label_map_path,
                     configs["train_input_config"].label_map_path)
    self.assertEqual(original_label_map_path,
                     configs["eval_input_config"].label_map_path)

  def testNewMaskType(self):
    """Tests that mask type can be overwritten in input readers."""
    original_mask_type = input_reader_pb2.NUMERICAL_MASKS
    new_mask_type = input_reader_pb2.PNG_MASKS
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    train_input_reader = pipeline_config.train_input_reader
    train_input_reader.mask_type = original_mask_type
    eval_input_reader = pipeline_config.eval_input_reader
    eval_input_reader.mask_type = original_mask_type
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(
        configs, mask_type=new_mask_type)
    self.assertEqual(new_mask_type, configs["train_input_config"].mask_type)
    self.assertEqual(new_mask_type, configs["eval_input_config"].mask_type)

  def  test_get_image_resizer_config(self):
    """Tests that number of classes can be retrieved."""
    model_config = model_pb2.DetectionModel()
    model_config.faster_rcnn.image_resizer.fixed_shape_resizer.height = 100
    model_config.faster_rcnn.image_resizer.fixed_shape_resizer.width = 300
    image_resizer_config = config_util.get_image_resizer_config(model_config)
    self.assertEqual(image_resizer_config.fixed_shape_resizer.height, 100)
    self.assertEqual(image_resizer_config.fixed_shape_resizer.width, 300)

  def test_get_spatial_image_size_from_fixed_shape_resizer_config(self):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    image_resizer_config.fixed_shape_resizer.height = 100
    image_resizer_config.fixed_shape_resizer.width = 200
    image_shape = config_util.get_spatial_image_size(image_resizer_config)
    self.assertAllEqual(image_shape, [100, 200])

  def test_get_spatial_image_size_from_aspect_preserving_resizer_config(self):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    image_resizer_config.keep_aspect_ratio_resizer.min_dimension = 100
    image_resizer_config.keep_aspect_ratio_resizer.max_dimension = 600
    image_resizer_config.keep_aspect_ratio_resizer.pad_to_max_dimension = True
    image_shape = config_util.get_spatial_image_size(image_resizer_config)
    self.assertAllEqual(image_shape, [600, 600])

  def test_get_spatial_image_size_from_aspect_preserving_resizer_dynamic(self):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    image_resizer_config.keep_aspect_ratio_resizer.min_dimension = 100
    image_resizer_config.keep_aspect_ratio_resizer.max_dimension = 600
    image_shape = config_util.get_spatial_image_size(image_resizer_config)
    self.assertAllEqual(image_shape, [-1, -1])


if __name__ == "__main__":
  tf.test.main()
