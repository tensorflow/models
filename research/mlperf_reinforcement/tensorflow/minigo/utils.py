# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for working with go games and coordinates"""

from collections import defaultdict
from contextlib import contextmanager
import functools
import itertools
import operator
import logging
import random
import re
import time


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
        for i in range(yields_between_refills):
            yield pool.pop()
        next_batch = take_n(yields_between_refills, iterator)
        if not next_batch:
            break
        pool.extend(next_batch)
    # finish consuming whatever's left - no need for further randomization.
    yield from pool


@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f seconds" % (message, (tock - tick)))


@contextmanager
def logged_timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f seconds" % (message, (tock - tick)))
    logging.info("%s: %.3f seconds" % (message, (tock - tick)))
