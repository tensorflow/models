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

"""A script to run training for sequential latent variable models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from fivo import ghmm_runners
from fivo import runners

# Shared flags.
tf.app.flags.DEFINE_enum("mode", "train",
                         ["train", "eval", "sample"],
                         "The mode of the binary.")
tf.app.flags.DEFINE_enum("model", "vrnn",
                         ["vrnn", "ghmm", "srnn"],
                         "Model choice.")
tf.app.flags.DEFINE_integer("latent_size", 64,
                            "The size of the latent state of the model.")
tf.app.flags.DEFINE_enum("dataset_type", "pianoroll",
                         ["pianoroll", "speech", "pose"],
                         "The type of dataset.")
tf.app.flags.DEFINE_string("dataset_path", "",
                           "Path to load the dataset from.")
tf.app.flags.DEFINE_integer("data_dimension", None,
                            "The dimension of each vector in the data sequence. "
                            "Defaults to 88 for pianoroll datasets and 200 for speech "
                            "datasets. Should not need to be changed except for "
                            "testing.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size.")
tf.app.flags.DEFINE_integer("num_samples", 4,
                            "The number of samples (or particles) for multisample "
                            "algorithms.")
tf.app.flags.DEFINE_string("logdir", "/tmp/smc_vi",
                           "The directory to keep checkpoints and summaries in.")
tf.app.flags.DEFINE_integer("random_seed", None,
                            "A random seed for seeding the TensorFlow graph.")
tf.app.flags.DEFINE_integer("parallel_iterations", 30,
                            "The number of parallel iterations to use for the while "
                            "loop that computes the bounds.")

# Training flags.
tf.app.flags.DEFINE_enum("bound", "fivo",
                         ["elbo", "iwae", "fivo", "fivo-aux"],
                         "The bound to optimize.")
tf.app.flags.DEFINE_boolean("normalize_by_seq_len", True,
                            "If true, normalize the loss by the number of timesteps "
                            "per sequence.")
tf.app.flags.DEFINE_float("learning_rate", 0.0002,
                          "The learning rate for ADAM.")
tf.app.flags.DEFINE_integer("max_steps", int(1e9),
                            "The number of gradient update steps to train for.")
tf.app.flags.DEFINE_integer("summarize_every", 50,
                            "The number of steps between summaries.")
tf.app.flags.DEFINE_enum("resampling_type", "multinomial",
                         ["multinomial", "relaxed"],
                         "The resampling strategy to use for training.")
tf.app.flags.DEFINE_float("relaxed_resampling_temperature", 0.5,
                          "The relaxation temperature for relaxed resampling.")
tf.app.flags.DEFINE_enum("proposal_type", "filtering",
                         ["prior", "filtering", "smoothing",
                          "true-filtering", "true-smoothing"],
                         "The type of proposal to use. true-filtering and true-smoothing "
                         "are only available for the GHMM. The specific implementation "
                         "of each proposal type is left to model-writers.")

# Distributed training flags.
tf.app.flags.DEFINE_string("master", "",
                           "The BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is used.")
tf.app.flags.DEFINE_boolean("stagger_workers", True,
                            "If true, bring one worker online every 1000 steps.")

# Evaluation flags.
tf.app.flags.DEFINE_enum("split", "train",
                         ["train", "test", "valid"],
                         "Split to evaluate the model on.")

# Sampling flags.
tf.app.flags.DEFINE_integer("sample_length", 50,
                            "The number of timesteps to sample for.")
tf.app.flags.DEFINE_integer("prefix_length", 25,
                            "The number of timesteps to condition the model on "
                            "before sampling.")
tf.app.flags.DEFINE_string("sample_out_dir", None,
                           "The directory to write the samples to. "
                           "Defaults to logdir.")

# GHMM flags.
tf.app.flags.DEFINE_float("variance", 0.1,
                          "The variance of the ghmm.")
tf.app.flags.DEFINE_integer("num_timesteps", 5,
                            "The number of timesteps to run the gmp for.")
FLAGS = tf.app.flags.FLAGS

PIANOROLL_DEFAULT_DATA_DIMENSION = 88
SPEECH_DEFAULT_DATA_DIMENSION = 200


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.model in ["vrnn", "srnn"]:
    if FLAGS.data_dimension is None:
      if FLAGS.dataset_type == "pianoroll":
        FLAGS.data_dimension = PIANOROLL_DEFAULT_DATA_DIMENSION
      elif FLAGS.dataset_type == "speech":
        FLAGS.data_dimension = SPEECH_DEFAULT_DATA_DIMENSION
    if FLAGS.mode == "train":
      runners.run_train(FLAGS)
    elif FLAGS.mode == "eval":
      runners.run_eval(FLAGS)
    elif FLAGS.mode == "sample":
      runners.run_sample(FLAGS)
  elif FLAGS.model == "ghmm":
    if FLAGS.mode == "train":
      ghmm_runners.run_train(FLAGS)
    elif FLAGS.mode == "eval":
      ghmm_runners.run_eval(FLAGS)

if __name__ == "__main__":
  tf.app.run(main)
