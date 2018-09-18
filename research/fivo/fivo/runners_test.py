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

"""Tests for fivo.runners"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from fivo import runners
from fivo.models import base
from fivo.models import vrnn

FLAGS = tf.app.flags.FLAGS


class RunnersTest(tf.test.TestCase):

  def default_config(self):
    class Config(object):
      pass
    config = Config()
    config.model = "vrnn"
    config.latent_size = 64
    config.batch_size = 4
    config.num_samples = 4
    config.resampling_type = "multinomial"
    config.normalize_by_seq_len = True
    config.learning_rate = 0.0001
    config.max_steps = int(1e6)
    config.summarize_every = 50
    # Master must be "" to prevent state from persisting between sessions.
    config.master = ""
    config.task = 0
    config.ps_tasks = 0
    config.stagger_workers = True
    config.random_seed = 1234
    config.parallel_iterations = 1
    config.dataset_type = "pianoroll"
    config.data_dimension = None
    config.dataset_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data", "tiny_pianoroll.pkl")
    config.proposal_type = "filtering"
    return config

  def run_training_one_step(self, bound, dataset_type, data_dimension,
                            dataset_filename, dir_prefix, resampling_type,
                            model, batch_size=2, num_samples=3,
                            create_dataset_and_model_fn=(runners.create_dataset_and_model)):
    config = self.default_config()
    config.model = model
    config.resampling_type = resampling_type
    config.relaxed_resampling_temperature = 0.5
    config.bound = bound
    config.split = "train"
    config.dataset_type = dataset_type
    config.dataset_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test_data",
        dataset_filename)
    config.max_steps = 1
    config.batch_size = batch_size
    config.num_samples = num_samples
    config.latent_size = 4
    config.data_dimension = data_dimension
    config.logdir = os.path.join(tf.test.get_temp_dir(), "%s-%s-%s-%s" %
                                 (dir_prefix, bound, dataset_type, model))
    runners.run_train(config,
                      create_dataset_and_model_fn=create_dataset_and_model_fn)
    return config

  def dummmy_dataset_and_model_fn(self, *unused_args, **unused_kwargs):
    # We ignore the arguments in the dummy but need to preserve prototype.
    batch_elements = 5
    sequence_length = 4
    data_dimensions = 3
    dataset = tf.data.Dataset.from_tensors(
        tf.zeros((sequence_length, batch_elements, data_dimensions),
                 dtype=tf.float32))
    inputs = dataset.make_one_shot_iterator().get_next()
    targets = tf.zeros_like(inputs)
    lengths = tf.constant([sequence_length] * batch_elements)
    mean = tf.constant((0.0, 0.0, 0.0))
    model = vrnn.create_vrnn(data_dimensions, 1,
                             base.ConditionalNormalDistribution)
    return inputs, targets, lengths, model, mean

  def test_training_one_step_fivo_pianoroll_vrnn(self):
    self.run_training_one_step("fivo", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "multinomial", "vrnn")

  def test_training_one_step_iwae_pianoroll_vrnn(self):
    self.run_training_one_step("iwae", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "multinomial", "vrnn")

  def test_training_one_step_elbo_pianoroll_vrnn(self):
    self.run_training_one_step("elbo", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "multinomial", "vrnn")

  def test_training_one_step_fivo_speech_vrnn(self):
    self.run_training_one_step("fivo", "speech", 2, "tiny_speech_dataset.tfrecord",
                               "test-training", "multinomial", "vrnn")

  def test_training_one_step_iwae_speech_vrnn(self):
    self.run_training_one_step("iwae", "speech", 2, "tiny_speech_dataset.tfrecord",
                               "test-training", "multinomial", "vrnn")

  def test_training_one_step_elbo_speech_vrnn(self):
    self.run_training_one_step("elbo", "speech", 2, "tiny_speech_dataset.tfrecord",
                               "test-training", "multinomial", "vrnn")

  def test_training_one_step_fivo_pianoroll_srnn(self):
    self.run_training_one_step("fivo", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "multinomial", "srnn")

  def test_training_one_step_iwae_pianoroll_srnn(self):
    self.run_training_one_step("iwae", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "multinomial", "srnn")

  def test_training_one_step_elbo_pianoroll_srnn(self):
    self.run_training_one_step("elbo", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "multinomial", "srnn")

  def test_training_one_step_fivo_speech_srnn(self):
    self.run_training_one_step("fivo", "speech", 2, "tiny_speech_dataset.tfrecord",
                               "test-training", "multinomial", "srnn")

  def test_training_one_step_iwae_speech_srnn(self):
    self.run_training_one_step("iwae", "speech", 2, "tiny_speech_dataset.tfrecord",
                               "test-training", "multinomial", "srnn")

  def test_training_one_step_elbo_speech_srnn(self):
    self.run_training_one_step("elbo", "speech", 2, "tiny_speech_dataset.tfrecord",
                               "test-training", "multinomial", "srnn")

  def test_training_one_step_fivo_pianoroll_vrnn_relaxed(self):
    self.run_training_one_step("fivo", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "relaxed", "vrnn")

  def test_training_one_step_iwae_pianoroll_vrnn_relaxed(self):
    self.run_training_one_step("iwae", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "relaxed", "vrnn")

  def test_training_one_step_elbo_pianoroll_vrnn_relaxed(self):
    self.run_training_one_step("elbo", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "relaxed", "vrnn")

  def test_training_one_step_fivo_pianoroll_srnn_relaxed(self):
    self.run_training_one_step("fivo", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "relaxed", "srnn")

  def test_training_one_step_iwae_pianoroll_srnn_relaxed(self):
    self.run_training_one_step("iwae", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "relaxed", "srnn")

  def test_training_one_step_elbo_pianoroll_srnn_relaxed(self):
    self.run_training_one_step("elbo", "pianoroll", 88, "tiny_pianoroll.pkl",
                               "test-training", "relaxed", "srnn")

  def test_eval_vrnn(self):
    self.run_eval("vrnn")

  def test_eval_srnn(self):
    self.run_eval("srnn")

  def run_eval(self, model):
    config = self.run_training_one_step(
        "fivo", "pianoroll", 88, "tiny_pianoroll.pkl", "test-eval-" + model,
        "multinomial", model)
    config.split = "train"
    runners.run_eval(config)

  def test_sampling_vrnn(self):
    self.run_sampling("vrnn")

  def test_sampling_srnn(self):
    self.run_sampling("srnn")

  def run_sampling(self, model):
    """Test sampling from the model."""
    config = self.run_training_one_step(
        "fivo", "pianoroll", 88, "tiny_pianoroll.pkl", "test-sampling", "multinomial",
        model)
    config.prefix_length = 3
    config.sample_length = 6
    config.split = "train"
    config.sample_out_dir = None

    runners.run_sample(config)
    unused_samples = np.load(os.path.join(config.logdir, "samples.npz"))

  def test_training_with_custom_fn(self):
    self.run_training_one_step(
        "fivo", "pianoroll", 3, "tiny_pianoroll.pkl",
        "test-training-custom-fn", "multinomial", "vrnn", batch_size=5,
        create_dataset_and_model_fn=self.dummmy_dataset_and_model_fn)

  def test_eval_with_custom_fn(self):
    config = self.run_training_one_step(
        "fivo", "pianoroll", 1, "tiny_pianoroll.pkl",
        "test-eval-custom-fn", "multinomial", "vrnn", batch_size=1,
        create_dataset_and_model_fn=self.dummmy_dataset_and_model_fn)
    config.split = "train"
    runners.run_eval(
        config,
        create_dataset_and_model_fn=self.dummmy_dataset_and_model_fn)

  def test_sampling_with_custom_fn(self):
    config = self.run_training_one_step(
        "fivo", "pianoroll", 3, "tiny_pianoroll.pkl",
        "test-sample-custom-fn", "multinomial", "vrnn", batch_size=5,
        create_dataset_and_model_fn=self.dummmy_dataset_and_model_fn)
    config.prefix_length = 2
    config.sample_length = 3
    config.split = "train"
    config.sample_out_dir = None

    runners.run_sample(
        config,
        create_dataset_and_model_fn=self.dummmy_dataset_and_model_fn)
    unused_samples = np.load(os.path.join(config.logdir, "samples.npz"))


if __name__ == "__main__":
  tf.test.main()
