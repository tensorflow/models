# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Convolutional Gated Recurrent Networks for Algorithm Learning."""

import math
import random
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS

bins = [8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64, 128]
all_tasks = ["sort", "kvsort", "id", "rev", "rev2", "incr", "add", "left",
             "right", "left-shift", "right-shift", "bmul", "mul", "dup",
             "badd", "qadd", "search"]
forward_max = 128
log_filename = ""


def pad(l):
  for b in bins:
    if b >= l: return b
  return forward_max


train_set = {}
test_set = {}
for some_task in all_tasks:
  train_set[some_task] = []
  test_set[some_task] = []
  for all_max_len in xrange(10000):
    train_set[some_task].append([])
    test_set[some_task].append([])


def add(n1, n2, base=10):
  """Add two numbers represented as lower-endian digit lists."""
  k = max(len(n1), len(n2)) + 1
  d1 = n1 + [0 for _ in xrange(k - len(n1))]
  d2 = n2 + [0 for _ in xrange(k - len(n2))]
  res = []
  carry = 0
  for i in xrange(k):
    if d1[i] + d2[i] + carry < base:
      res.append(d1[i] + d2[i] + carry)
      carry = 0
    else:
      res.append(d1[i] + d2[i] + carry - base)
      carry = 1
  while res and res[-1] == 0:
    res = res[:-1]
  if res: return res
  return [0]


def init_data(task, length, nbr_cases, nclass):
  """Data initialization."""
  def rand_pair(l, task):
    """Random data pair for a task. Total length should be <= l."""
    k = (l-1)/2
    base = 10
    if task[0] == "b": base = 2
    if task[0] == "q": base = 4
    d1 = [np.random.randint(base) for _ in xrange(k)]
    d2 = [np.random.randint(base) for _ in xrange(k)]
    if task in ["add", "badd", "qadd"]:
      res = add(d1, d2, base)
    elif task in ["mul", "bmul"]:
      d1n = sum([d * (base ** i) for i, d in enumerate(d1)])
      d2n = sum([d * (base ** i) for i, d in enumerate(d2)])
      if task == "bmul":
        res = [int(x) for x in list(reversed(str(bin(d1n * d2n))))[:-2]]
      else:
        res = [int(x) for x in list(reversed(str(d1n * d2n)))]
    else:
      sys.exit()
    sep = [12]
    if task in ["add", "badd", "qadd"]: sep = [11]
    inp = [d + 1 for d in d1] + sep + [d + 1 for d in d2]
    return inp, [r + 1 for r in res]

  def rand_dup_pair(l):
    """Random data pair for duplication task. Total length should be <= l."""
    k = l/2
    x = [np.random.randint(nclass - 1) + 1 for _ in xrange(k)]
    inp = x + [0 for _ in xrange(l - k)]
    res = x + x + [0 for _ in xrange(l - 2*k)]
    return inp, res

  def rand_rev2_pair(l):
    """Random data pair for reverse2 task. Total length should be <= l."""
    inp = [(np.random.randint(nclass - 1) + 1,
            np.random.randint(nclass - 1) + 1) for _ in xrange(l/2)]
    res = [i for i in reversed(inp)]
    return [x for p in inp for x in p], [x for p in res for x in p]

  def rand_search_pair(l):
    """Random data pair for search task. Total length should be <= l."""
    inp = [(np.random.randint(nclass - 1) + 1,
            np.random.randint(nclass - 1) + 1) for _ in xrange(l-1/2)]
    q = np.random.randint(nclass - 1) + 1
    res = 0
    for (k, v) in reversed(inp):
      if k == q:
        res = v
    return [x for p in inp for x in p] + [q], [res]

  def rand_kvsort_pair(l):
    """Random data pair for key-value sort. Total length should be <= l."""
    keys = [(np.random.randint(nclass - 1) + 1, i) for i in xrange(l/2)]
    vals = [np.random.randint(nclass - 1) + 1 for _ in xrange(l/2)]
    kv = [(k, vals[i]) for (k, i) in keys]
    sorted_kv = [(k, vals[i]) for (k, i) in sorted(keys)]
    return [x for p in kv for x in p], [x for p in sorted_kv for x in p]

  def spec(inp):
    """Return the target given the input for some tasks."""
    if task == "sort":
      return sorted(inp)
    elif task == "id":
      return inp
    elif task == "rev":
      return [i for i in reversed(inp)]
    elif task == "incr":
      carry = 1
      res = []
      for i in xrange(len(inp)):
        if inp[i] + carry < nclass:
          res.append(inp[i] + carry)
          carry = 0
        else:
          res.append(1)
          carry = 1
      return res
    elif task == "left":
      return [inp[0]]
    elif task == "right":
      return [inp[-1]]
    elif task == "left-shift":
      return [inp[l-1] for l in xrange(len(inp))]
    elif task == "right-shift":
      return [inp[l+1] for l in xrange(len(inp))]
    else:
      print_out("Unknown spec for task " + str(task))
      sys.exit()

  l = length
  cur_time = time.time()
  total_time = 0.0
  for case in xrange(nbr_cases):
    total_time += time.time() - cur_time
    cur_time = time.time()
    if l > 10000 and case % 100 == 1:
      print_out("  avg gen time %.4f s" % (total_time / float(case)))
    if task in ["add", "badd", "qadd", "bmul", "mul"]:
      i, t = rand_pair(l, task)
      train_set[task][len(i)].append([i, t])
      i, t = rand_pair(l, task)
      test_set[task][len(i)].append([i, t])
    elif task == "dup":
      i, t = rand_dup_pair(l)
      train_set[task][len(i)].append([i, t])
      i, t = rand_dup_pair(l)
      test_set[task][len(i)].append([i, t])
    elif task == "rev2":
      i, t = rand_rev2_pair(l)
      train_set[task][len(i)].append([i, t])
      i, t = rand_rev2_pair(l)
      test_set[task][len(i)].append([i, t])
    elif task == "search":
      i, t = rand_search_pair(l)
      train_set[task][len(i)].append([i, t])
      i, t = rand_search_pair(l)
      test_set[task][len(i)].append([i, t])
    elif task == "kvsort":
      i, t = rand_kvsort_pair(l)
      train_set[task][len(i)].append([i, t])
      i, t = rand_kvsort_pair(l)
      test_set[task][len(i)].append([i, t])
    else:
      inp = [np.random.randint(nclass - 1) + 1 for i in xrange(l)]
      target = spec(inp)
      train_set[task][l].append([inp, target])
      inp = [np.random.randint(nclass - 1) + 1 for i in xrange(l)]
      target = spec(inp)
      test_set[task][l].append([inp, target])


def to_symbol(i):
  """Covert ids to text."""
  if i == 0: return ""
  if i == 11: return "+"
  if i == 12: return "*"
  return str(i-1)


def to_id(s):
  """Covert text to ids."""
  if s == "+": return 11
  if s == "*": return 12
  return int(s) + 1


def get_batch(max_length, batch_size, do_train, task, offset=None, preset=None):
  """Get a batch of data, training or testing."""
  inputs = []
  targets = []
  length = max_length
  if preset is None:
    cur_set = test_set[task]
    if do_train: cur_set = train_set[task]
    while not cur_set[length]:
      length -= 1
  pad_length = pad(length)
  for b in xrange(batch_size):
    if preset is None:
      elem = random.choice(cur_set[length])
      if offset is not None and offset + b < len(cur_set[length]):
        elem = cur_set[length][offset + b]
    else:
      elem = preset
    inp, target = elem[0], elem[1]
    assert len(inp) == length
    inputs.append(inp + [0 for l in xrange(pad_length - len(inp))])
    targets.append(target + [0 for l in xrange(pad_length - len(target))])
  res_input = []
  res_target = []
  for l in xrange(pad_length):
    new_input = np.array([inputs[b][l] for b in xrange(batch_size)],
                         dtype=np.int32)
    new_target = np.array([targets[b][l] for b in xrange(batch_size)],
                          dtype=np.int32)
    res_input.append(new_input)
    res_target.append(new_target)
  return res_input, res_target


def print_out(s, newline=True):
  """Print a message out and log it to file."""
  if log_filename:
    try:
      with gfile.GFile(log_filename, mode="a") as f:
        f.write(s + ("\n" if newline else ""))
    # pylint: disable=bare-except
    except:
      sys.stdout.write("Error appending to %s\n" % log_filename)
  sys.stdout.write(s + ("\n" if newline else ""))
  sys.stdout.flush()


def decode(output):
  return [np.argmax(o, axis=1) for o in output]


def accuracy(inpt, output, target, batch_size, nprint):
  """Calculate output accuracy given target."""
  assert nprint < batch_size + 1
  def task_print(inp, output, target):
    stop_bound = 0
    print_len = 0
    while print_len < len(target) and target[print_len] > stop_bound:
      print_len += 1
    print_out("    i: " + " ".join([str(i - 1) for i in inp if i > 0]))
    print_out("    o: " +
              " ".join([str(output[l] - 1) for l in xrange(print_len)]))
    print_out("    t: " +
              " ".join([str(target[l] - 1) for l in xrange(print_len)]))
  decoded_target = target
  decoded_output = decode(output)
  total = 0
  errors = 0
  seq = [0 for b in xrange(batch_size)]
  for l in xrange(len(decoded_output)):
    for b in xrange(batch_size):
      if decoded_target[l][b] > 0:
        total += 1
        if decoded_output[l][b] != decoded_target[l][b]:
          seq[b] = 1
          errors += 1
  e = 0  # Previous error index
  for _ in xrange(min(nprint, sum(seq))):
    while seq[e] == 0:
      e += 1
    task_print([inpt[l][e] for l in xrange(len(inpt))],
               [decoded_output[l][e] for l in xrange(len(decoded_target))],
               [decoded_target[l][e] for l in xrange(len(decoded_target))])
    e += 1
  for b in xrange(nprint - errors):
    task_print([inpt[l][b] for l in xrange(len(inpt))],
               [decoded_output[l][b] for l in xrange(len(decoded_target))],
               [decoded_target[l][b] for l in xrange(len(decoded_target))])
  return errors, total, sum(seq)


def safe_exp(x):
  perp = 10000
  if x < 100: perp = math.exp(x)
  if perp > 10000: return 10000
  return perp
