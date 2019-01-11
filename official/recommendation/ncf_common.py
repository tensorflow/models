# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""NCF framework to train and evaluate the NeuMF model.

The NeuMF model assembles both MF and MLP models under the NCF framework. Check
`neumf_model.py` for more details about the models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import heapq
import json
import logging
import math
import multiprocessing
import os
import signal
import typing

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from tensorflow.contrib.compiler import xla
from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import data_pipeline
from official.recommendation import data_preprocessing
from official.recommendation import neumf_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.logs import mlperf_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


FLAGS = flags.FLAGS


def get_inputs()
  if FLAGS.download_if_missing and not FLAGS.use_synthetic_data:
    movielens.download(FLAGS.dataset, FLAGS.data_dir)

  if FLAGS.seed is not None:
    np.random.seed(FLAGS.seed)

  params = parse_flags(FLAGS)
  total_training_cycle = FLAGS.train_epochs // FLAGS.epochs_between_evals

  if FLAGS.use_synthetic_data:
    producer = data_pipeline.DummyConstructor()
    num_train_steps = rconst.SYNTHETIC_BATCHES_PER_EPOCH
    num_eval_steps = rconst.SYNTHETIC_BATCHES_PER_EPOCH
  else:
    _, _, producer = data_preprocessing.instantiate_pipeline(
        dataset=FLAGS.dataset, data_dir=FLAGS.data_dir, params=params,
        constructor_type=FLAGS.constructor_type,
        deterministic=FLAGS.seed is not None)

    num_train_steps = (producer.train_batches_per_epoch //
                       params["batches_per_step"])
    num_eval_steps = (producer.eval_batches_per_epoch //
                      params["batches_per_step"])
    assert not producer.train_batches_per_epoch % params["batches_per_step"]
    assert not producer.eval_batches_per_epoch % params["batches_per_step"]

  return num_train_steps, num_eval_steps, producer

