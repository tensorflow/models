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
"""Neural GPU -- data generation and batching utilities."""

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import program_utils

FLAGS = tf.app.flags.FLAGS

bins = [2 + bin_idx_i for bin_idx_i in xrange(256)]
all_tasks = ["sort", "kvsort", "id", "rev", "rev2", "incr", "add", "left",
             "right", "left-shift", "right-shift", "bmul", "mul", "dup",
             "badd", "qadd", "search", "progeval", "progsynth"]
log_filename = ""
vocab, rev_vocab = None, None


def pad(l):
  for b in bins:
    if b >= l: return b
  return bins[-1]


def bin_for(l):
  for i, b in enumerate(bins):
    if b >= l: return i
  return len(bins) - 1


train_set = {}
test_set = {}
for some_task in all_tasks:
  train_set[some_task] = []
  test_set[some_task] = []
  for all_max_len in xrange(10000):
    train_set[some_task].append([])
    test_set[some_task].append([])


def read_tmp_file(name):
  """Read from a file with the given name in our log directory or above."""
  dirname = os.path.dirname(log_filename)
  fname = os.path.join(dirname, name + ".txt")
  if not tf.gfile.Exists(fname):
    print_out("== not found file: " + fname)
    fname = os.path.join(dirname, "../" + name + ".txt")
  if not tf.gfile.Exists(fname):
    print_out("== not found file: " + fname)
    fname = os.path.join(dirname, "../../" + name + ".txt")
  if not tf.gfile.Exists(fname):
    print_out("== not found file: " + fname)
    return None
  print_out("== found file: " + fname)
  res = []
  with tf.gfile.GFile(fname, mode="r") as f:
    for line in f:
      res.append(line.strip())
  return res


def write_tmp_file(name, lines):
  dirname = os.path.dirname(log_filename)
  fname = os.path.join(dirname, name + ".txt")
  with tf.gfile.GFile(fname, mode="w") as f:
    for line in lines:
      f.write(line + "\n")


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
    k = int((l-1)/2)
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
    k = int(l/2)
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

  def prog_io_pair(prog, max_len, counter=0):
    try:
      ilen = np.random.randint(max_len - 3) + 1
      bound = max(15 - (counter / 20), 1)
      inp = [random.choice(range(-bound, bound)) for _ in range(ilen)]
      inp_toks = [program_utils.prog_rev_vocab[t]
                  for t in program_utils.tokenize(str(inp)) if t != ","]
      out = program_utils.evaluate(prog, {"a": inp})
      out_toks = [program_utils.prog_rev_vocab[t]
                  for t in program_utils.tokenize(str(out)) if t != ","]
      if counter > 400:
        out_toks = []
      if (out_toks and out_toks[0] == program_utils.prog_rev_vocab["["] and
          len(out_toks) != len([o for o in out if o == ","]) + 3):
        raise ValueError("generated list with too long ints")
      if (out_toks and out_toks[0] != program_utils.prog_rev_vocab["["] and
          len(out_toks) > 1):
        raise ValueError("generated one int but tokenized it to many")
      if len(out_toks) > max_len:
        raise ValueError("output too long")
      return (inp_toks, out_toks)
    except ValueError:
      return prog_io_pair(prog, max_len, counter+1)

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

  is_prog = task in ["progeval", "progsynth"]
  if is_prog:
    inputs_per_prog = 5
    program_utils.make_vocab()
    progs = read_tmp_file("programs_len%d" % (l / 10))
    if not progs:
      progs = program_utils.gen(l / 10, 1.2 * nbr_cases / inputs_per_prog)
      write_tmp_file("programs_len%d" % (l / 10), progs)
    prog_ios = read_tmp_file("programs_len%d_io" % (l / 10))
    nbr_cases = min(nbr_cases, len(progs) * inputs_per_prog) / 1.2
    if not prog_ios:
      # Generate program io data.
      prog_ios = []
      for pidx, prog in enumerate(progs):
        if pidx % 500 == 0:
          print_out("== generating io pairs for program %d" % pidx)
        if pidx * inputs_per_prog > nbr_cases * 1.2:
          break
        ptoks = [program_utils.prog_rev_vocab[t]
                 for t in program_utils.tokenize(prog)]
        ptoks.append(program_utils.prog_rev_vocab["_EOS"])
        plen = len(ptoks)
        for _ in xrange(inputs_per_prog):
          if task == "progeval":
            inp, out = prog_io_pair(prog, plen)
            prog_ios.append(str(inp) + "\t" + str(out) + "\t" + prog)
          elif task == "progsynth":
            plen = max(len(ptoks), 8)
            for _ in xrange(3):
              inp, out = prog_io_pair(prog, plen / 2)
              prog_ios.append(str(inp) + "\t" + str(out) + "\t" + prog)
      write_tmp_file("programs_len%d_io" % (l / 10), prog_ios)
    prog_ios_dict = {}
    for s in prog_ios:
      i, o, p = s.split("\t")
      i_clean = "".join([c for c in i if c.isdigit() or c == " "])
      o_clean = "".join([c for c in o if c.isdigit() or c == " "])
      inp = [int(x) for x in i_clean.split()]
      out = [int(x) for x in o_clean.split()]
      if inp and out:
        if p in prog_ios_dict:
          prog_ios_dict[p].append([inp, out])
        else:
          prog_ios_dict[p] = [[inp, out]]
    # Use prog_ios_dict to create data.
    progs = []
    for prog in prog_ios_dict:
      if len([c for c in prog if c == ";"]) <= (l / 10):
        progs.append(prog)
    nbr_cases = min(nbr_cases, len(progs) * inputs_per_prog) / 1.2
    print_out("== %d training cases on %d progs" % (nbr_cases, len(progs)))
    for pidx, prog in enumerate(progs):
      if pidx * inputs_per_prog > nbr_cases * 1.2:
        break
      ptoks = [program_utils.prog_rev_vocab[t]
               for t in program_utils.tokenize(prog)]
      ptoks.append(program_utils.prog_rev_vocab["_EOS"])
      plen = len(ptoks)
      dset = train_set if pidx < nbr_cases / inputs_per_prog else test_set
      for _ in xrange(inputs_per_prog):
        if task == "progeval":
          inp, out = prog_ios_dict[prog].pop()
          dset[task][bin_for(plen)].append([[ptoks, inp, [], []], [out]])
        elif task == "progsynth":
          plen, ilist = max(len(ptoks), 8), [[]]
          for _ in xrange(3):
            inp, out = prog_ios_dict[prog].pop()
            ilist.append(inp + out)
          dset[task][bin_for(plen)].append([ilist, [ptoks]])

  for case in xrange(0 if is_prog else nbr_cases):
    total_time += time.time() - cur_time
    cur_time = time.time()
    if l > 10000 and case % 100 == 1:
      print_out("  avg gen time %.4f s" % (total_time / float(case)))
    if task in ["add", "badd", "qadd", "bmul", "mul"]:
      i, t = rand_pair(l, task)
      train_set[task][bin_for(len(i))].append([[[], i, [], []], [t]])
      i, t = rand_pair(l, task)
      test_set[task][bin_for(len(i))].append([[[], i, [], []], [t]])
    elif task == "dup":
      i, t = rand_dup_pair(l)
      train_set[task][bin_for(len(i))].append([[i], [t]])
      i, t = rand_dup_pair(l)
      test_set[task][bin_for(len(i))].append([[i], [t]])
    elif task == "rev2":
      i, t = rand_rev2_pair(l)
      train_set[task][bin_for(len(i))].append([[i], [t]])
      i, t = rand_rev2_pair(l)
      test_set[task][bin_for(len(i))].append([[i], [t]])
    elif task == "search":
      i, t = rand_search_pair(l)
      train_set[task][bin_for(len(i))].append([[i], [t]])
      i, t = rand_search_pair(l)
      test_set[task][bin_for(len(i))].append([[i], [t]])
    elif task == "kvsort":
      i, t = rand_kvsort_pair(l)
      train_set[task][bin_for(len(i))].append([[i], [t]])
      i, t = rand_kvsort_pair(l)
      test_set[task][bin_for(len(i))].append([[i], [t]])
    elif task not in ["progeval", "progsynth"]:
      inp = [np.random.randint(nclass - 1) + 1 for i in xrange(l)]
      target = spec(inp)
      train_set[task][bin_for(l)].append([[inp], [target]])
      inp = [np.random.randint(nclass - 1) + 1 for i in xrange(l)]
      target = spec(inp)
      test_set[task][bin_for(l)].append([[inp], [target]])


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


def get_batch(bin_id, batch_size, data_set, height, offset=None, preset=None):
  """Get a batch of data, training or testing."""
  inputs, targets = [], []
  pad_length = bins[bin_id]
  for b in xrange(batch_size):
    if preset is None:
      elem = random.choice(data_set[bin_id])
      if offset is not None and offset + b < len(data_set[bin_id]):
        elem = data_set[bin_id][offset + b]
    else:
      elem = preset
    inpt, targett, inpl, targetl = elem[0], elem[1], [], []
    for inp in inpt:
      inpl.append(inp + [0 for _ in xrange(pad_length - len(inp))])
    if len(inpl) == 1:
      for _ in xrange(height - 1):
        inpl.append([0 for _ in xrange(pad_length)])
    for target in targett:
      targetl.append(target + [0 for _ in xrange(pad_length - len(target))])
    inputs.append(inpl)
    targets.append(targetl)
  res_input = np.array(inputs, dtype=np.int32)
  res_target = np.array(targets, dtype=np.int32)
  assert list(res_input.shape) == [batch_size, height, pad_length]
  assert list(res_target.shape) == [batch_size, 1, pad_length]
  return res_input, res_target


def print_out(s, newline=True):
  """Print a message out and log it to file."""
  if log_filename:
    try:
      with tf.gfile.GFile(log_filename, mode="a") as f:
        f.write(s + ("\n" if newline else ""))
    # pylint: disable=bare-except
    except:
      sys.stderr.write("Error appending to %s\n" % log_filename)
  sys.stdout.write(s + ("\n" if newline else ""))
  sys.stdout.flush()


def decode(output):
  return [np.argmax(o, axis=1) for o in output]


def accuracy(inpt_t, output, target_t, batch_size, nprint,
             beam_out=None, beam_scores=None):
  """Calculate output accuracy given target."""
  assert nprint < batch_size + 1
  inpt = []
  for h in xrange(inpt_t.shape[1]):
    inpt.extend([inpt_t[:, h, l] for l in xrange(inpt_t.shape[2])])
  target = [target_t[:, 0, l] for l in xrange(target_t.shape[2])]
  def tok(i):
    if rev_vocab and i < len(rev_vocab):
      return rev_vocab[i]
    return str(i - 1)
  def task_print(inp, output, target):
    stop_bound = 0
    print_len = 0
    while print_len < len(target) and target[print_len] > stop_bound:
      print_len += 1
    print_out("    i: " + " ".join([tok(i) for i in inp if i > 0]))
    print_out("    o: " +
              " ".join([tok(output[l]) for l in xrange(print_len)]))
    print_out("    t: " +
              " ".join([tok(target[l]) for l in xrange(print_len)]))
  decoded_target = target
  decoded_output = decode(output)
  # Use beam output if given and score is high enough.
  if beam_out is not None:
    for b in xrange(batch_size):
      if beam_scores[b] >= 10.0:
        for l in xrange(min(len(decoded_output), beam_out.shape[2])):
          decoded_output[l][b] = int(beam_out[b, 0, l])
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
  x = float(x)
  if x < 100: perp = math.exp(x)
  if perp > 10000: return 10000
  return perp
