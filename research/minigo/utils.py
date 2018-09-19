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
"""Utilities for MiniGo and DualNet model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import functools
import itertools
import math
import operator
import os
import random
import re
import string
import time

import tensorflow as tf  # pylint: disable=g-bad-import-order

# Regular expression of model number and name.
MODEL_NUM_REGEX = r'^\d{6}'  # model_num consists of six digits
# model_name consists of six digits followed by a dash and the model name
MODEL_NAME_REGEX = r'^\d{6}(-\w+)+'


def random_generator(size=6, chars=string.ascii_letters + string.digits):
  return ''.join(random.choice(chars) for x in range(size))


def generate_model_name(model_num):
  """Generate a full model name for the given model number.

  Args:
    model_num: The number/generation of the model.

  Returns:
    The model's full name: model_num-model_name.
  """
  if model_num == 0:  # Model number for bootstrap model
    new_name = 'bootstrap'
  else:
    new_name = random_generator()
  full_name = '{:06d}-{}'.format(model_num, new_name)
  return full_name


def detect_model_num(full_name):
  """Take the full name of a model and extract its model number.

  Args:
    full_name: The full name of a model.

  Returns:
    The model number. For example: '000000-bootstrap.index' => 0.
  """
  match = re.match(MODEL_NUM_REGEX, full_name)
  if match:
    return int(match.group())
  else:
    return None


def detect_model_name(full_name):
  """Take the full name of a model and extract its model name.

  Args:
    full_name: The full name of a model.

  Returns:
    The model name. For example: '000000-bootstrap.index' => '000000-bootstrap'.
  """
  match = re.match(MODEL_NAME_REGEX, full_name)
  if match:
    return match.group()
  else:
    return None


def get_models(models_dir):
  """Get all models.

  Args:
    models_dir: The directory of all models.

  Returns:
    A list of model number and names sorted increasingly. For example:
    [(13, 000013-modelname), (17, 000017-modelname), ...etc]
  """
  all_models = tf.gfile.Glob(os.path.join(models_dir, '*.meta'))
  model_filenames = [os.path.basename(m) for m in all_models]
  model_numbers_names = sorted([
      (detect_model_num(m), detect_model_name(m))
      for m in model_filenames])
  return model_numbers_names


def get_latest_model(models_dir):
  """Find the latest model.

  Args:
    models_dir: The directory of all models.

  Returns:
    The model number and name of the latest model. For example:
    (17, 000017-modelname)
  """
  models = get_models(models_dir)
  if models is None:
    models = [(0, '000000-bootstrap')]
  return models[-1]


def round_power_of_two(n):
  """Finds the nearest power of 2 to a number.

  Thus 84 -> 64, 120 -> 128, etc.

  Args:
    n: The given number.

  Returns:
    The nearest 2-power number to n.
  """
  return 2 ** int(round(math.log(n, 2)))


def parse_game_result(result):
  if re.match(r'[bB]\+', result):
    return 1
  elif re.match(r'[wW]\+', result):
    return -1
  else:
    return 0


def product(numbers):
  return functools.reduce(operator.mul, numbers)


def take_n(n, iterable):
  return list(itertools.islice(iterable, n))


def iter_chunks(chunk_size, iterator):
  iterator = iter(iterator)
  while True:
    next_chunk = take_n(chunk_size, iterator)
    # If len(iterable) % chunk_size == 0, don't return an empty chunk.
    if next_chunk:
      yield next_chunk
    else:
      break


def shuffler(iterator, pool_size=10**5, refill_threshold=0.9):
  yields_between_refills = round(pool_size * (1 - refill_threshold))
  # initialize pool; this step may or may not exhaust the iterator.
  pool = take_n(pool_size, iterator)
  while True:
    random.shuffle(pool)
    for _ in range(yields_between_refills):
      yield pool.pop()
    next_batch = take_n(yields_between_refills, iterator)
    if not next_batch:
      break
    pool.extend(next_batch)
  # finish consuming whatever's left - no need for further randomization.
  # yield from pool
  print(type(pool))
  for p in pool:
    yield p


@contextmanager
def timer(message):
  tick = time.time()
  yield
  tock = time.time()
  print('{}: {:.3} seconds'.foramt(message, (tock - tick)))


@contextmanager
def logged_timer(message):
  tick = time.time()
  yield
  tock = time.time()
  print('{}: {:.3} seconds'.format(message, (tock - tick)))
  tf.logging.info('{}: {:.3} seconds'.format(message, (tock - tick)))


class MiniGoDirectory(object):
  """The class to set up directories of MiniGo."""

  def __init__(self, base_dir):
    self.trained_models_dir = os.path.join(base_dir, 'trained_models')
    self.estimator_model_dir = os.path.join(base_dir, 'estimator_model_dir/')
    self.selfplay_dir = os.path.join(base_dir, 'data/selfplay/')
    self.holdout_dir = os.path.join(base_dir, 'data/holdout/')
    self.training_chunk_dir = os.path.join(base_dir, 'data/training_chunks/')
    self.sgf_dir = os.path.join(base_dir, 'sgf/')
    self.evaluate_dir = os.path.join(base_dir, 'sgf/evaluate/')
