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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import range
import tensorflow as tf

from google.protobuf import text_format

from object_detection.protos import eval_pb2
from object_detection.protos import image_resizer_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2
from object_detection.utils import config_util

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import training as contrib_training
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top


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

  def _create_and_load_test_configs(self, pipeline_config):
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    _write_config(pipeline_config, pipeline_config_path)
    return config_util.get_configs_from_pipeline_file(pipeline_config_path)

  def test_get_configs_from_pipeline_file(self):
    """Test that proto configs can be read from pipeline config file."""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 10
    pipeline_config.train_config.batch_size = 32
    pipeline_config.train_input_reader.label_map_path = "path/to/label_map"
    pipeline_config.eval_config.num_examples = 20
    pipeline_config.eval_input_reader.add().queue_capacity = 100

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
                           configs["eval_input_configs"])

  def test_create_configs_from_pipeline_proto(self):
    """Tests creating configs dictionary from pipeline proto."""

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 10
    pipeline_config.train_config.batch_size = 32
    pipeline_config.train_input_reader.label_map_path = "path/to/label_map"
    pipeline_config.eval_config.num_examples = 20
    pipeline_config.eval_input_reader.add().queue_capacity = 100

    configs = config_util.create_configs_from_pipeline_proto(pipeline_config)
    self.assertProtoEquals(pipeline_config.model, configs["model"])
    self.assertProtoEquals(pipeline_config.train_config,
                           configs["train_config"])
    self.assertProtoEquals(pipeline_config.train_input_reader,
                           configs["train_input_config"])
    self.assertProtoEquals(pipeline_config.eval_config, configs["eval_config"])
    self.assertProtoEquals(pipeline_config.eval_input_reader,
                           configs["eval_input_configs"])

  def test_create_pipeline_proto_from_configs(self):
    """Tests that proto can be reconstructed from configs dictionary."""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 10
    pipeline_config.train_config.batch_size = 32
    pipeline_config.train_input_reader.label_map_path = "path/to/label_map"
    pipeline_config.eval_config.num_examples = 20
    pipeline_config.eval_input_reader.add().queue_capacity = 100
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    pipeline_config_reconstructed = (
        config_util.create_pipeline_proto_from_configs(configs))
    self.assertEqual(pipeline_config, pipeline_config_reconstructed)

  def test_save_pipeline_config(self):
    """Tests that the pipeline config is properly saved to disk."""
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 10
    pipeline_config.train_config.batch_size = 32
    pipeline_config.train_input_reader.label_map_path = "path/to/label_map"
    pipeline_config.eval_config.num_examples = 20
    pipeline_config.eval_input_reader.add().queue_capacity = 100

    config_util.save_pipeline_config(pipeline_config, self.get_temp_dir())
    configs = config_util.get_configs_from_pipeline_file(
        os.path.join(self.get_temp_dir(), "pipeline.config"))
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
    self.assertProtoEquals(eval_input_config, configs["eval_input_configs"][0])

  def _assertOptimizerWithNewLearningRate(self, optimizer_name):
    """Asserts successful updating of all learning rate schemes."""
    original_learning_rate = 0.7
    learning_rate_scaling = 0.1
    warmup_learning_rate = 0.07
    hparams = contrib_training.HParams(learning_rate=0.15)
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

  def testGenericConfigOverride(self):
    """Tests generic config overrides for all top-level configs."""
    # Set one parameter for each of the top-level pipeline configs:
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.ssd.num_classes = 1
    pipeline_config.train_config.batch_size = 1
    pipeline_config.eval_config.num_visualizations = 1
    pipeline_config.train_input_reader.label_map_path = "/some/path"
    pipeline_config.eval_input_reader.add().label_map_path = "/some/path"
    pipeline_config.graph_rewriter.quantization.weight_bits = 1

    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    _write_config(pipeline_config, pipeline_config_path)

    # Override each of the parameters:
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    hparams = contrib_training.HParams(
        **{
            "model.ssd.num_classes": 2,
            "train_config.batch_size": 2,
            "train_input_config.label_map_path": "/some/other/path",
            "eval_config.num_visualizations": 2,
            "graph_rewriter_config.quantization.weight_bits": 2
        })
    configs = config_util.merge_external_params_with_configs(configs, hparams)

    # Ensure that the parameters have the overridden values:
    self.assertEqual(2, configs["model"].ssd.num_classes)
    self.assertEqual(2, configs["train_config"].batch_size)
    self.assertEqual("/some/other/path",
                     configs["train_input_config"].label_map_path)
    self.assertEqual(2, configs["eval_config"].num_visualizations)
    self.assertEqual(2,
                     configs["graph_rewriter_config"].quantization.weight_bits)

  def testNewBatchSize(self):
    """Tests that batch size is updated appropriately."""
    original_batch_size = 2
    hparams = contrib_training.HParams(batch_size=16)
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
    hparams = contrib_training.HParams(batch_size=0.5)
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_config.batch_size = original_batch_size
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    new_batch_size = configs["train_config"].batch_size
    self.assertEqual(1, new_batch_size)  # Clipped to 1.0.

  def testOverwriteBatchSizeWithKeyValue(self):
    """Tests that batch size is overwritten based on key/value."""
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_config.batch_size = 2
    configs = self._create_and_load_test_configs(pipeline_config)
    hparams = contrib_training.HParams(**{"train_config.batch_size": 10})
    configs = config_util.merge_external_params_with_configs(configs, hparams)
    new_batch_size = configs["train_config"].batch_size
    self.assertEqual(10, new_batch_size)

  def testKeyValueOverrideBadKey(self):
    """Tests that overwriting with a bad key causes an exception."""
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    configs = self._create_and_load_test_configs(pipeline_config)
    hparams = contrib_training.HParams(**{"train_config.no_such_field": 10})
    with self.assertRaises(ValueError):
      config_util.merge_external_params_with_configs(configs, hparams)

  def testOverwriteBatchSizeWithBadValueType(self):
    """Tests that overwriting with a bad valuye type causes an exception."""
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_config.batch_size = 2
    configs = self._create_and_load_test_configs(pipeline_config)
    # Type should be an integer, but we're passing a string "10".
    hparams = contrib_training.HParams(**{"train_config.batch_size": "10"})
    with self.assertRaises(TypeError):
      config_util.merge_external_params_with_configs(configs, hparams)

  def testNewMomentumOptimizerValue(self):
    """Tests that new momentum value is updated appropriately."""
    original_momentum_value = 0.4
    hparams = contrib_training.HParams(momentum_optimizer_value=1.1)
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
    hparams = contrib_training.HParams(
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
    hparams = contrib_training.HParams(
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
    desired_num_train_steps = 10
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_config.num_steps = original_num_train_steps
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"train_steps": desired_num_train_steps}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    train_steps = configs["train_config"].num_steps
    self.assertEqual(desired_num_train_steps, train_steps)

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
    override_dict = {"train_input_path": new_train_path}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
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
    override_dict = {"train_input_path": new_train_path}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
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
    eval_input_reader = pipeline_config.eval_input_reader.add()
    eval_input_reader.label_map_path = original_label_map_path
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"label_map_path": new_label_map_path}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    self.assertEqual(new_label_map_path,
                     configs["train_input_config"].label_map_path)
    for eval_input_config in configs["eval_input_configs"]:
      self.assertEqual(new_label_map_path, eval_input_config.label_map_path)

  def testDontOverwriteEmptyLabelMapPath(self):
    """Tests that label map path will not by overwritten with empty string."""
    original_label_map_path = "path/to/original/label_map"
    new_label_map_path = ""
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    train_input_reader = pipeline_config.train_input_reader
    train_input_reader.label_map_path = original_label_map_path
    eval_input_reader = pipeline_config.eval_input_reader.add()
    eval_input_reader.label_map_path = original_label_map_path
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"label_map_path": new_label_map_path}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    self.assertEqual(original_label_map_path,
                     configs["train_input_config"].label_map_path)
    self.assertEqual(original_label_map_path,
                     configs["eval_input_configs"][0].label_map_path)

  def testNewMaskType(self):
    """Tests that mask type can be overwritten in input readers."""
    original_mask_type = input_reader_pb2.NUMERICAL_MASKS
    new_mask_type = input_reader_pb2.PNG_MASKS
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    train_input_reader = pipeline_config.train_input_reader
    train_input_reader.mask_type = original_mask_type
    eval_input_reader = pipeline_config.eval_input_reader.add()
    eval_input_reader.mask_type = original_mask_type
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"mask_type": new_mask_type}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    self.assertEqual(new_mask_type, configs["train_input_config"].mask_type)
    self.assertEqual(new_mask_type, configs["eval_input_configs"][0].mask_type)

  def testUseMovingAverageForEval(self):
    use_moving_averages_orig = False
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.use_moving_averages = use_moving_averages_orig
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"eval_with_moving_averages": True}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    self.assertEqual(True, configs["eval_config"].use_moving_averages)

  def  testGetImageResizerConfig(self):
    """Tests that number of classes can be retrieved."""
    model_config = model_pb2.DetectionModel()
    model_config.faster_rcnn.image_resizer.fixed_shape_resizer.height = 100
    model_config.faster_rcnn.image_resizer.fixed_shape_resizer.width = 300
    image_resizer_config = config_util.get_image_resizer_config(model_config)
    self.assertEqual(image_resizer_config.fixed_shape_resizer.height, 100)
    self.assertEqual(image_resizer_config.fixed_shape_resizer.width, 300)

  def testGetSpatialImageSizeFromFixedShapeResizerConfig(self):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    image_resizer_config.fixed_shape_resizer.height = 100
    image_resizer_config.fixed_shape_resizer.width = 200
    image_shape = config_util.get_spatial_image_size(image_resizer_config)
    self.assertAllEqual(image_shape, [100, 200])

  def testGetSpatialImageSizeFromAspectPreservingResizerConfig(self):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    image_resizer_config.keep_aspect_ratio_resizer.min_dimension = 100
    image_resizer_config.keep_aspect_ratio_resizer.max_dimension = 600
    image_resizer_config.keep_aspect_ratio_resizer.pad_to_max_dimension = True
    image_shape = config_util.get_spatial_image_size(image_resizer_config)
    self.assertAllEqual(image_shape, [600, 600])

  def testGetSpatialImageSizeFromAspectPreservingResizerDynamic(self):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    image_resizer_config.keep_aspect_ratio_resizer.min_dimension = 100
    image_resizer_config.keep_aspect_ratio_resizer.max_dimension = 600
    image_shape = config_util.get_spatial_image_size(image_resizer_config)
    self.assertAllEqual(image_shape, [-1, -1])

  def testGetSpatialImageSizeFromConditionalShapeResizer(self):
    image_resizer_config = image_resizer_pb2.ImageResizer()
    image_resizer_config.conditional_shape_resizer.size_threshold = 100
    image_shape = config_util.get_spatial_image_size(image_resizer_config)
    self.assertAllEqual(image_shape, [-1, -1])

  def testGetMaxNumContextFeaturesFromModelConfig(self):
    model_config = model_pb2.DetectionModel()
    model_config.faster_rcnn.context_config.max_num_context_features = 10
    max_num_context_features = config_util.get_max_num_context_features(
        model_config)
    self.assertAllEqual(max_num_context_features, 10)

  def testGetContextFeatureLengthFromModelConfig(self):
    model_config = model_pb2.DetectionModel()
    model_config.faster_rcnn.context_config.context_feature_length = 100
    context_feature_length = config_util.get_context_feature_length(
        model_config)
    self.assertAllEqual(context_feature_length, 100)

  def testEvalShuffle(self):
    """Tests that `eval_shuffle` keyword arguments are applied correctly."""
    original_shuffle = True
    desired_shuffle = False

    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_input_reader.add().shuffle = original_shuffle
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"eval_shuffle": desired_shuffle}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    self.assertEqual(desired_shuffle, configs["eval_input_configs"][0].shuffle)

  def testTrainShuffle(self):
    """Tests that `train_shuffle` keyword arguments are applied correctly."""
    original_shuffle = True
    desired_shuffle = False

    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_input_reader.shuffle = original_shuffle
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"train_shuffle": desired_shuffle}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    train_shuffle = configs["train_input_config"].shuffle
    self.assertEqual(desired_shuffle, train_shuffle)

  def testOverWriteRetainOriginalImages(self):
    """Tests that `train_shuffle` keyword arguments are applied correctly."""
    original_retain_original_images = True
    desired_retain_original_images = False

    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.retain_original_images = (
        original_retain_original_images)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {
        "retain_original_images_in_eval": desired_retain_original_images
    }
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    retain_original_images = configs["eval_config"].retain_original_images
    self.assertEqual(desired_retain_original_images, retain_original_images)

  def testOverwriteAllEvalSampling(self):
    original_num_eval_examples = 1
    new_num_eval_examples = 10

    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_input_reader.add().sample_1_of_n_examples = (
        original_num_eval_examples)
    pipeline_config.eval_input_reader.add().sample_1_of_n_examples = (
        original_num_eval_examples)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"sample_1_of_n_eval_examples": new_num_eval_examples}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    for eval_input_config in configs["eval_input_configs"]:
      self.assertEqual(new_num_eval_examples,
                       eval_input_config.sample_1_of_n_examples)

  def testOverwriteAllEvalNumEpochs(self):
    original_num_epochs = 10
    new_num_epochs = 1

    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_input_reader.add().num_epochs = original_num_epochs
    pipeline_config.eval_input_reader.add().num_epochs = original_num_epochs
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"eval_num_epochs": new_num_epochs}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    for eval_input_config in configs["eval_input_configs"]:
      self.assertEqual(new_num_epochs, eval_input_config.num_epochs)

  def testUpdateMaskTypeForAllInputConfigs(self):
    original_mask_type = input_reader_pb2.NUMERICAL_MASKS
    new_mask_type = input_reader_pb2.PNG_MASKS

    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    train_config = pipeline_config.train_input_reader
    train_config.mask_type = original_mask_type
    eval_1 = pipeline_config.eval_input_reader.add()
    eval_1.mask_type = original_mask_type
    eval_1.name = "eval_1"
    eval_2 = pipeline_config.eval_input_reader.add()
    eval_2.mask_type = original_mask_type
    eval_2.name = "eval_2"
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"mask_type": new_mask_type}
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)

    self.assertEqual(configs["train_input_config"].mask_type, new_mask_type)
    for eval_input_config in configs["eval_input_configs"]:
      self.assertEqual(eval_input_config.mask_type, new_mask_type)

  def testErrorOverwritingMultipleInputConfig(self):
    original_shuffle = False
    new_shuffle = True
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    eval_1 = pipeline_config.eval_input_reader.add()
    eval_1.shuffle = original_shuffle
    eval_1.name = "eval_1"
    eval_2 = pipeline_config.eval_input_reader.add()
    eval_2.shuffle = original_shuffle
    eval_2.name = "eval_2"
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {"eval_shuffle": new_shuffle}
    with self.assertRaises(ValueError):
      configs = config_util.merge_external_params_with_configs(
          configs, kwargs_dict=override_dict)

  def testCheckAndParseInputConfigKey(self):
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_input_reader.add().name = "eval_1"
    pipeline_config.eval_input_reader.add().name = "eval_2"
    _write_config(pipeline_config, pipeline_config_path)
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    specific_shuffle_update_key = "eval_input_configs:eval_2:shuffle"
    is_valid_input_config_key, key_name, input_name, field_name = (
        config_util.check_and_parse_input_config_key(
            configs, specific_shuffle_update_key))
    self.assertTrue(is_valid_input_config_key)
    self.assertEqual(key_name, "eval_input_configs")
    self.assertEqual(input_name, "eval_2")
    self.assertEqual(field_name, "shuffle")

    legacy_shuffle_update_key = "eval_shuffle"
    is_valid_input_config_key, key_name, input_name, field_name = (
        config_util.check_and_parse_input_config_key(configs,
                                                     legacy_shuffle_update_key))
    self.assertTrue(is_valid_input_config_key)
    self.assertEqual(key_name, "eval_input_configs")
    self.assertEqual(input_name, None)
    self.assertEqual(field_name, "shuffle")

    non_input_config_update_key = "label_map_path"
    is_valid_input_config_key, key_name, input_name, field_name = (
        config_util.check_and_parse_input_config_key(
            configs, non_input_config_update_key))
    self.assertFalse(is_valid_input_config_key)
    self.assertEqual(key_name, None)
    self.assertEqual(input_name, None)
    self.assertEqual(field_name, "label_map_path")

    with self.assertRaisesRegexp(ValueError,
                                 "Invalid key format when overriding configs."):
      config_util.check_and_parse_input_config_key(
          configs, "train_input_config:shuffle")

    with self.assertRaisesRegexp(
        ValueError, "Invalid key_name when overriding input config."):
      config_util.check_and_parse_input_config_key(
          configs, "invalid_key_name:train_name:shuffle")

    with self.assertRaisesRegexp(
        ValueError, "Invalid input_name when overriding input config."):
      config_util.check_and_parse_input_config_key(
          configs, "eval_input_configs:unknown_eval_name:shuffle")

    with self.assertRaisesRegexp(
        ValueError, "Invalid field_name when overriding input config."):
      config_util.check_and_parse_input_config_key(
          configs, "eval_input_configs:eval_2:unknown_field_name")

  def testUpdateInputReaderConfigSuccess(self):
    original_shuffle = False
    new_shuffle = True
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.train_input_reader.shuffle = original_shuffle
    _write_config(pipeline_config, pipeline_config_path)
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    config_util.update_input_reader_config(
        configs,
        key_name="train_input_config",
        input_name=None,
        field_name="shuffle",
        value=new_shuffle)
    self.assertEqual(configs["train_input_config"].shuffle, new_shuffle)

    config_util.update_input_reader_config(
        configs,
        key_name="train_input_config",
        input_name=None,
        field_name="shuffle",
        value=new_shuffle)
    self.assertEqual(configs["train_input_config"].shuffle, new_shuffle)

  def testUpdateInputReaderConfigErrors(self):
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_input_reader.add().name = "same_eval_name"
    pipeline_config.eval_input_reader.add().name = "same_eval_name"
    _write_config(pipeline_config, pipeline_config_path)
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    with self.assertRaisesRegexp(ValueError,
                                 "Duplicate input name found when overriding."):
      config_util.update_input_reader_config(
          configs,
          key_name="eval_input_configs",
          input_name="same_eval_name",
          field_name="shuffle",
          value=False)

    with self.assertRaisesRegexp(
        ValueError, "Input name name_not_exist not found when overriding."):
      config_util.update_input_reader_config(
          configs,
          key_name="eval_input_configs",
          input_name="name_not_exist",
          field_name="shuffle",
          value=False)

    with self.assertRaisesRegexp(ValueError,
                                 "Unknown input config overriding."):
      config_util.update_input_reader_config(
          configs,
          key_name="eval_input_configs",
          input_name=None,
          field_name="shuffle",
          value=False)

  def testOverWriteRetainOriginalImageAdditionalChannels(self):
    """Tests that keyword arguments are applied correctly."""
    original_retain_original_image_additional_channels = True
    desired_retain_original_image_additional_channels = False

    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.eval_config.retain_original_image_additional_channels = (
        original_retain_original_image_additional_channels)
    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    override_dict = {
        "retain_original_image_additional_channels_in_eval":
            desired_retain_original_image_additional_channels
    }
    configs = config_util.merge_external_params_with_configs(
        configs, kwargs_dict=override_dict)
    retain_original_image_additional_channels = configs[
        "eval_config"].retain_original_image_additional_channels
    self.assertEqual(desired_retain_original_image_additional_channels,
                     retain_original_image_additional_channels)

  def testUpdateNumClasses(self):
    pipeline_config_path = os.path.join(self.get_temp_dir(), "pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.faster_rcnn.num_classes = 10

    _write_config(pipeline_config, pipeline_config_path)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    self.assertEqual(config_util.get_number_of_classes(configs["model"]), 10)

    config_util.merge_external_params_with_configs(
        configs, kwargs_dict={"num_classes": 2})

    self.assertEqual(config_util.get_number_of_classes(configs["model"]), 2)

  def testRemoveUnecessaryEma(self):
    input_dict = {
        "expanded_conv_10/project/act_quant/min":
            1,
        "FeatureExtractor/MobilenetV2_2/expanded_conv_5/expand/act_quant/min":
            2,
        "expanded_conv_10/expand/BatchNorm/gamma/min/ExponentialMovingAverage":
            3,
        "expanded_conv_3/depthwise/BatchNorm/beta/max/ExponentialMovingAverage":
            4,
        "BoxPredictor_1/ClassPredictor_depthwise/act_quant":
            5
    }

    no_ema_collection = ["/min", "/max"]

    output_dict = {
        "expanded_conv_10/project/act_quant/min":
            1,
        "FeatureExtractor/MobilenetV2_2/expanded_conv_5/expand/act_quant/min":
            2,
        "expanded_conv_10/expand/BatchNorm/gamma/min":
            3,
        "expanded_conv_3/depthwise/BatchNorm/beta/max":
            4,
        "BoxPredictor_1/ClassPredictor_depthwise/act_quant":
            5
    }

    self.assertEqual(
        output_dict,
        config_util.remove_unecessary_ema(input_dict, no_ema_collection))


if __name__ == "__main__":
  tf.test.main()
