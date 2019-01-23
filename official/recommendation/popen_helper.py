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
"""Helper file for running the async data generation process in OSS."""

import contextlib
import multiprocessing
import multiprocessing.pool


def get_forkpool(num_workers, init_worker=None, closing=True):
  pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker)
  return contextlib.closing(pool) if closing else pool


def get_threadpool(num_workers, init_worker=None, closing=True):
  pool = multiprocessing.pool.ThreadPool(processes=num_workers,
                                         initializer=init_worker)
  return contextlib.closing(pool) if closing else pool


class FauxPool(object):
  """Mimic a pool using for loops.

  This class is used in place of proper pools when true determinism is desired
  for testing or debugging.
  """
  def __init__(self, *args, **kwargs):
    pass

  def map(self, func, iterable, chunksize=None):
    return [func(i) for i in iterable]

  def imap(self, func, iterable, chunksize=1):
    for i in iterable:
      yield func(i)

  def close(self):
    pass

  def terminate(self):
    pass

  def join(self):
    pass

def get_fauxpool(num_workers, init_worker=None, closing=True):
  pool = FauxPool(processes=num_workers, initializer=init_worker)
  return contextlib.closing(pool) if closing else pool
