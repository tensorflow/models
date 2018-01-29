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
"""Neural GPU."""

from __future__ import print_function

import math
import os
import random
import sys
import threading
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import program_utils
import data_utils as data
import neural_gpu as ngpu
import wmt_utils as wmt

tf.app.flags.DEFINE_float("lr", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("init_weight", 0.8, "Initial weights deviation.")
tf.app.flags.DEFINE_float("max_grad_norm", 4.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("cutoff", 1.2, "Cutoff at the gates.")
tf.app.flags.DEFINE_float("curriculum_ppx", 9.9, "Move curriculum if ppl < X.")
tf.app.flags.DEFINE_float("curriculum_seq", 0.3, "Move curriculum if seq < X.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Dropout that much.")
tf.app.flags.DEFINE_float("grad_noise_scale", 0.0, "Gradient noise scale.")
tf.app.flags.DEFINE_float("max_sampling_rate", 0.1, "Maximal sampling rate.")
tf.app.flags.DEFINE_float("length_norm", 0.0, "Length normalization.")
tf.app.flags.DEFINE_float("train_beam_freq", 0.0, "Beam-based training.")
tf.app.flags.DEFINE_float("train_beam_anneal", 20000, "How many steps anneal.")
tf.app.flags.DEFINE_integer("eval_beam_steps", 4, "How many beam steps eval.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "Steps per epoch.")
tf.app.flags.DEFINE_integer("nmaps", 64, "Number of floats in each cell.")
tf.app.flags.DEFINE_integer("vec_size", 64, "Size of word vectors.")
tf.app.flags.DEFINE_integer("train_data_size", 1000, "Training examples/len.")
tf.app.flags.DEFINE_integer("max_length", 40, "Maximum length.")
tf.app.flags.DEFINE_integer("random_seed", 125459, "Random seed.")
tf.app.flags.DEFINE_integer("nconvs", 2, "How many convolutions / 1 step.")
tf.app.flags.DEFINE_integer("kw", 3, "Kernel width.")
tf.app.flags.DEFINE_integer("kh", 3, "Kernel height.")
tf.app.flags.DEFINE_integer("height", 4, "Height.")
tf.app.flags.DEFINE_integer("mem_size", -1, "Memory size (sqrt)")
tf.app.flags.DEFINE_integer("soft_mem_size", 1024, "Softmax memory this size.")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of GPUs to use.")
tf.app.flags.DEFINE_integer("num_replicas", 1, "Number of replicas in use.")
tf.app.flags.DEFINE_integer("beam_size", 1, "Beam size during decoding. "
                            "If 0, no decoder, the non-extended Neural GPU.")
tf.app.flags.DEFINE_integer("max_target_vocab", 0,
                            "Maximal size of target vocabulary.")
tf.app.flags.DEFINE_integer("decode_offset", 0, "Offset for decoding.")
tf.app.flags.DEFINE_integer("task", -1, "Task id when running on borg.")
tf.app.flags.DEFINE_integer("nprint", 0, "How many test examples to print out.")
tf.app.flags.DEFINE_integer("eval_bin_print", 3, "How many bins step in eval.")
tf.app.flags.DEFINE_integer("mode", 0, "Mode: 0-train other-decode.")
tf.app.flags.DEFINE_bool("atrous", False, "Whether to use atrous convs.")
tf.app.flags.DEFINE_bool("layer_norm", False, "Do layer normalization.")
tf.app.flags.DEFINE_bool("quantize", False, "Whether to quantize variables.")
tf.app.flags.DEFINE_bool("do_train", True, "If false, only update memory.")
tf.app.flags.DEFINE_bool("rnn_baseline", False, "If true build an RNN instead.")
tf.app.flags.DEFINE_bool("simple_tokenizer", False,
                         "If true, tokenize on spaces only, digits are 0.")
tf.app.flags.DEFINE_bool("normalize_digits", True,
                         "Whether to normalize digits with simple tokenizer.")
tf.app.flags.DEFINE_integer("vocab_size", 16, "Joint vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp/", "Directory to store models.")
tf.app.flags.DEFINE_string("test_file_prefix", "", "Files to test (.en,.fr).")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_string("word_vector_file_en", "",
                           "Optional file with word vectors to start training.")
tf.app.flags.DEFINE_string("word_vector_file_fr", "",
                           "Optional file with word vectors to start training.")
tf.app.flags.DEFINE_string("problem", "wmt", "What problem are we solving?.")

tf.app.flags.DEFINE_integer("ps_tasks", 0, "Number of ps tasks used.")
tf.app.flags.DEFINE_string("master", "", "Name of the TensorFlow master.")

FLAGS = tf.app.flags.FLAGS
EXTRA_EVAL = 10
EVAL_LEN_INCR = 8
MAXLEN_F = 2.0


def zero_split(tok_list, append=None):
  """Split tok_list (list of ints) on 0s, append int to all parts if given."""
  res, cur, l = [], [], 0
  for tok in tok_list:
    if tok == 0:
      if append is not None:
        cur.append(append)
      res.append(cur)
      l = max(l, len(cur))
      cur = []
    else:
      cur.append(tok)
  if append is not None:
    cur.append(append)
  res.append(cur)
  l = max(l, len(cur))
  return res, l


def read_data(source_path, target_path, buckets, max_size=None, print_out=True):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    buckets: the buckets to use.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
      If set to 1, no data will be returned (empty lists of the right form).
    print_out: whether to print out status or not.

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in buckets]
  counter = 0
  if max_size != 1:
    with tf.gfile.GFile(source_path, mode="r") as source_file:
      with tf.gfile.GFile(target_path, mode="r") as target_file:
        source, target = source_file.readline(), target_file.readline()
        while source and target and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0 and print_out:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          source_ids, source_len = zero_split(source_ids)
          target_ids, target_len = zero_split(target_ids, append=wmt.EOS_ID)
          for bucket_id, size in enumerate(buckets):
            if source_len <= size and target_len <= size:
              data_set[bucket_id].append([source_ids, target_ids])
              break
          source, target = source_file.readline(), target_file.readline()
  return data_set


global_train_set = {"wmt": []}
train_buckets_scale = {"wmt": []}


def calculate_buckets_scale(data_set, buckets, problem):
  """Calculate buckets scales for the given data set."""
  train_bucket_sizes = [len(data_set[b]) for b in xrange(len(buckets))]
  train_total_size = max(1, float(sum(train_bucket_sizes)))

  # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  if problem not in train_buckets_scale:
    train_buckets_scale[problem] = []
  train_buckets_scale[problem].append(
      [sum(train_bucket_sizes[:i + 1]) / train_total_size
       for i in xrange(len(train_bucket_sizes))])
  return train_total_size


def read_data_into_global(source_path, target_path, buckets,
                          max_size=None, print_out=True):
  """Read data into the global variables (can be in a separate thread)."""
  # pylint: disable=global-variable-not-assigned
  global global_train_set, train_buckets_scale
  # pylint: enable=global-variable-not-assigned
  data_set = read_data(source_path, target_path, buckets, max_size, print_out)
  global_train_set["wmt"].append(data_set)
  train_total_size = calculate_buckets_scale(data_set, buckets, "wmt")
  if print_out:
    print("  Finished global data reading (%d)." % train_total_size)


def initialize(sess=None):
  """Initialize data and model."""
  global MAXLEN_F
  # Create training directory if it does not exist.
  if not tf.gfile.IsDirectory(FLAGS.train_dir):
    data.print_out("Creating training directory %s." % FLAGS.train_dir)
    tf.gfile.MkDir(FLAGS.train_dir)
  decode_suffix = "beam%dln%d" % (FLAGS.beam_size,
                                  int(100 * FLAGS.length_norm))
  if FLAGS.mode == 0:
    decode_suffix = ""
  if FLAGS.task >= 0:
    data.log_filename = os.path.join(FLAGS.train_dir,
                                     "log%d%s" % (FLAGS.task, decode_suffix))
  else:
    data.log_filename = os.path.join(FLAGS.train_dir, "neural_gpu/log")

  # Set random seed.
  if FLAGS.random_seed > 0:
    seed = FLAGS.random_seed + max(0, FLAGS.task)
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

  # Check data sizes.
  assert data.bins
  max_length = min(FLAGS.max_length, data.bins[-1])
  while len(data.bins) > 1 and data.bins[-2] >= max_length + EXTRA_EVAL:
    data.bins = data.bins[:-1]
  if sess is None and FLAGS.task == 0 and FLAGS.num_replicas > 1:
    if max_length > 60:
      max_length = max_length * 1 / 2  # Save memory on chief.
  min_length = min(14, max_length - 3) if FLAGS.problem == "wmt" else 3
  for p in FLAGS.problem.split("-"):
    if p in ["progeval", "progsynth"]:
      min_length = max(26, min_length)
  assert max_length + 1 > min_length
  while len(data.bins) > 1 and data.bins[-2] >= max_length + EXTRA_EVAL:
    data.bins = data.bins[:-1]

  # Create checkpoint directory if it does not exist.
  if FLAGS.mode == 0 or FLAGS.task < 0:
    checkpoint_dir = os.path.join(FLAGS.train_dir, "neural_gpu%s"
                                  % ("" if FLAGS.task < 0 else str(FLAGS.task)))
  else:
    checkpoint_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(checkpoint_dir):
    data.print_out("Creating checkpoint directory %s." % checkpoint_dir)
    tf.gfile.MkDir(checkpoint_dir)

  # Prepare data.
  if FLAGS.problem == "wmt":
    # Prepare WMT data.
    data.print_out("Preparing WMT data in %s" % FLAGS.data_dir)
    if FLAGS.simple_tokenizer:
      MAXLEN_F = 3.5
      (en_train, fr_train, en_dev, fr_dev,
       en_path, fr_path) = wmt.prepare_wmt_data(
           FLAGS.data_dir, FLAGS.vocab_size,
           tokenizer=wmt.space_tokenizer,
           normalize_digits=FLAGS.normalize_digits)
    else:
      (en_train, fr_train, en_dev, fr_dev,
       en_path, fr_path) = wmt.prepare_wmt_data(
           FLAGS.data_dir, FLAGS.vocab_size)

    # Read data into buckets and compute their sizes.
    fr_vocab, rev_fr_vocab = wmt.initialize_vocabulary(fr_path)
    data.vocab = fr_vocab
    data.rev_vocab = rev_fr_vocab
    data.print_out("Reading development and training data (limit: %d)."
                   % FLAGS.max_train_data_size)
    dev_set = {}
    dev_set["wmt"] = read_data(en_dev, fr_dev, data.bins)
    def data_read(size, print_out):
      read_data_into_global(en_train, fr_train, data.bins, size, print_out)
    data_read(50000, False)
    read_thread_small = threading.Thread(
        name="reading-data-small", target=lambda: data_read(900000, False))
    read_thread_small.start()
    read_thread_full = threading.Thread(
        name="reading-data-full",
        target=lambda: data_read(FLAGS.max_train_data_size, True))
    read_thread_full.start()
    data.print_out("Data reading set up.")
  else:
    # Prepare algorithmic data.
    en_path, fr_path = None, None
    tasks = FLAGS.problem.split("-")
    data_size = FLAGS.train_data_size
    for t in tasks:
      data.print_out("Generating data for %s." % t)
      if t in ["progeval", "progsynth"]:
        data.init_data(t, data.bins[-1], 20 * data_size, FLAGS.vocab_size)
        if len(program_utils.prog_vocab) > FLAGS.vocab_size - 2:
          raise ValueError("Increase vocab_size to %d for prog-tasks."
                           % (len(program_utils.prog_vocab) + 2))
        data.rev_vocab = program_utils.prog_vocab
        data.vocab = program_utils.prog_rev_vocab
      else:
        for l in xrange(max_length + EXTRA_EVAL - 1):
          data.init_data(t, l, data_size, FLAGS.vocab_size)
        data.init_data(t, data.bins[-2], data_size, FLAGS.vocab_size)
        data.init_data(t, data.bins[-1], data_size, FLAGS.vocab_size)
      if t not in global_train_set:
        global_train_set[t] = []
      global_train_set[t].append(data.train_set[t])
      calculate_buckets_scale(data.train_set[t], data.bins, t)
    dev_set = data.test_set

  # Grid-search parameters.
  lr = FLAGS.lr
  init_weight = FLAGS.init_weight
  max_grad_norm = FLAGS.max_grad_norm
  if sess is not None and FLAGS.task > -1:
    def job_id_factor(step):
      """If jobid / step mod 3 is 0, 1, 2: say 0, 1, -1."""
      return ((((FLAGS.task / step) % 3) + 1) % 3) - 1
    lr *= math.pow(2, job_id_factor(1))
    init_weight *= math.pow(1.5, job_id_factor(3))
    max_grad_norm *= math.pow(2, job_id_factor(9))

  # Print out parameters.
  curriculum = FLAGS.curriculum_seq
  msg1 = ("layers %d kw %d h %d kh %d batch %d noise %.2f"
          % (FLAGS.nconvs, FLAGS.kw, FLAGS.height, FLAGS.kh,
             FLAGS.batch_size, FLAGS.grad_noise_scale))
  msg2 = ("cut %.2f lr %.3f iw %.2f cr %.2f nm %d d%.4f gn %.2f %s"
          % (FLAGS.cutoff, lr, init_weight, curriculum, FLAGS.nmaps,
             FLAGS.dropout, max_grad_norm, msg1))
  data.print_out(msg2)

  # Create model and initialize it.
  tf.get_variable_scope().set_initializer(
      tf.orthogonal_initializer(gain=1.8 * init_weight))
  max_sampling_rate = FLAGS.max_sampling_rate if FLAGS.mode == 0 else 0.0
  o = FLAGS.vocab_size if FLAGS.max_target_vocab < 1 else FLAGS.max_target_vocab
  ngpu.CHOOSE_K = FLAGS.soft_mem_size
  do_beam_model = FLAGS.train_beam_freq > 0.0001 and FLAGS.beam_size > 1
  beam_size = FLAGS.beam_size if FLAGS.mode > 0 and not do_beam_model else 1
  beam_size = min(beam_size, FLAGS.beam_size)
  beam_model = None
  def make_ngpu(cur_beam_size, back):
    return ngpu.NeuralGPU(
        FLAGS.nmaps, FLAGS.vec_size, FLAGS.vocab_size, o,
        FLAGS.dropout, max_grad_norm, FLAGS.cutoff, FLAGS.nconvs,
        FLAGS.kw, FLAGS.kh, FLAGS.height, FLAGS.mem_size,
        lr / math.sqrt(FLAGS.num_replicas), min_length + 3, FLAGS.num_gpus,
        FLAGS.num_replicas, FLAGS.grad_noise_scale, max_sampling_rate,
        atrous=FLAGS.atrous, do_rnn=FLAGS.rnn_baseline,
        do_layer_norm=FLAGS.layer_norm, beam_size=cur_beam_size, backward=back)
  if sess is None:
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      model = make_ngpu(beam_size, True)
      if do_beam_model:
        tf.get_variable_scope().reuse_variables()
        beam_model = make_ngpu(FLAGS.beam_size, False)
  else:
    model = make_ngpu(beam_size, True)
    if do_beam_model:
      tf.get_variable_scope().reuse_variables()
      beam_model = make_ngpu(FLAGS.beam_size, False)

  sv = None
  if sess is None:
    # The supervisor configuration has a few overriden options.
    sv = tf.train.Supervisor(logdir=checkpoint_dir,
                             is_chief=(FLAGS.task < 1),
                             saver=model.saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=15 * 60,
                             global_step=model.global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = sv.PrepareSession(FLAGS.master, config=config)

  data.print_out("Created model. Checkpoint dir %s" % checkpoint_dir)

  # Load model from parameters if a checkpoint exists.
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
    data.print_out("Reading model parameters from %s"
                   % ckpt.model_checkpoint_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
  elif sv is None:
    sess.run(tf.global_variables_initializer())
    data.print_out("Initialized variables (no supervisor mode).")
  elif FLAGS.task < 1 and FLAGS.mem_size > 0:
    # sess.run(model.mem_norm_op)
    data.print_out("Created new model and normalized mem (on chief).")

  # Return the model and needed variables.
  return (model, beam_model, min_length, max_length, checkpoint_dir,
          (global_train_set, dev_set, en_path, fr_path), sv, sess)


def m_step(model, beam_model, sess, batch_size, inp, target, bucket, nsteps, p):
  """Evaluation multi-step for program synthesis."""
  state, scores, hist = None, [[-11.0 for _ in xrange(batch_size)]], []
  for _ in xrange(nsteps):
    # Get the best beam (no training, just forward model).
    new_target, new_first, new_inp, new_scores = get_best_beam(
        beam_model, sess, inp, target,
        batch_size, FLAGS.beam_size, bucket, hist, p, test_mode=True)
    hist.append(new_first)
    _, _, _, state = model.step(sess, inp, new_target, False, state=state)
    inp = new_inp
    scores.append([max(scores[-1][i], new_scores[i])
                   for i in xrange(batch_size)])
  # The final step with the true target.
  loss, res, _, _ = model.step(sess, inp, target, False, state=state)
  return loss, res, new_target, scores[1:]


def single_test(bin_id, model, sess, nprint, batch_size, dev, p, print_out=True,
                offset=None, beam_model=None):
  """Test model on test data of length l using the given session."""
  if not dev[p][bin_id]:
    data.print_out("  bin %d (%d)\t%s\tppl NA errors NA seq-errors NA"
                   % (bin_id, data.bins[bin_id], p))
    return 1.0, 1.0, 0.0
  inpt, target = data.get_batch(
      bin_id, batch_size, dev[p], FLAGS.height, offset)
  if FLAGS.beam_size > 1 and beam_model:
    loss, res, new_tgt, scores = m_step(
        model, beam_model, sess, batch_size, inpt, target, bin_id,
        FLAGS.eval_beam_steps, p)
    score_avgs = [sum(s) / float(len(s)) for s in scores]
    score_maxs = [max(s) for s in scores]
    score_str = ["(%.2f, %.2f)" % (score_avgs[i], score_maxs[i])
                 for i in xrange(FLAGS.eval_beam_steps)]
    data.print_out("  == scores (avg, max): %s" % "; ".join(score_str))
    errors, total, seq_err = data.accuracy(inpt, res, target, batch_size,
                                           nprint, new_tgt, scores[-1])
  else:
    loss, res, _, _ = model.step(sess, inpt, target, False)
    errors, total, seq_err = data.accuracy(inpt, res, target, batch_size,
                                           nprint)
  seq_err = float(seq_err) / batch_size
  if total > 0:
    errors = float(errors) / total
  if print_out:
    data.print_out("  bin %d (%d)\t%s\tppl %.2f errors %.2f seq-errors %.2f"
                   % (bin_id, data.bins[bin_id], p, data.safe_exp(loss),
                      100 * errors, 100 * seq_err))
  return (errors, seq_err, loss)


def assign_vectors(word_vector_file, embedding_key, vocab_path, sess):
  """Assign the embedding_key variable from the given word vectors file."""
  # For words in the word vector file, set their embedding at start.
  if not tf.gfile.Exists(word_vector_file):
    data.print_out("Word vector file does not exist: %s" % word_vector_file)
    sys.exit(1)
  vocab, _ = wmt.initialize_vocabulary(vocab_path)
  vectors_variable = [v for v in tf.trainable_variables()
                      if embedding_key == v.name]
  if len(vectors_variable) != 1:
    data.print_out("Word vector variable not found or too many.")
    sys.exit(1)
  vectors_variable = vectors_variable[0]
  vectors = vectors_variable.eval()
  data.print_out("Pre-setting word vectors from %s" % word_vector_file)
  with tf.gfile.GFile(word_vector_file, mode="r") as f:
    # Lines have format: dog 0.045123 -0.61323 0.413667 ...
    for line in f:
      line_parts = line.split()
      # The first part is the word.
      word = line_parts[0]
      if word in vocab:
        # Remaining parts are components of the vector.
        word_vector = np.array(map(float, line_parts[1:]))
        if len(word_vector) != FLAGS.vec_size:
          data.print_out("Warn: Word '%s', Expecting vector size %d, "
                         "found %d" % (word, FLAGS.vec_size,
                                       len(word_vector)))
        else:
          vectors[vocab[word]] = word_vector
  # Assign the modified vectors to the vectors_variable in the graph.
  sess.run([vectors_variable.initializer],
           {vectors_variable.initializer.inputs[1]: vectors})


def print_vectors(embedding_key, vocab_path, word_vector_file):
  """Print vectors from the given variable."""
  _, rev_vocab = wmt.initialize_vocabulary(vocab_path)
  vectors_variable = [v for v in tf.trainable_variables()
                      if embedding_key == v.name]
  if len(vectors_variable) != 1:
    data.print_out("Word vector variable not found or too many.")
    sys.exit(1)
  vectors_variable = vectors_variable[0]
  vectors = vectors_variable.eval()
  l, s = vectors.shape[0], vectors.shape[1]
  data.print_out("Printing %d word vectors from %s to %s."
                 % (l, embedding_key, word_vector_file))
  with tf.gfile.GFile(word_vector_file, mode="w") as f:
    # Lines have format: dog 0.045123 -0.61323 0.413667 ...
    for i in xrange(l):
      f.write(rev_vocab[i])
      for j in xrange(s):
        f.write(" %.8f" % vectors[i][j])
      f.write("\n")


def get_bucket_id(train_buckets_scale_c, max_cur_length, data_set):
  """Get a random bucket id."""
  # Choose a bucket according to data distribution. Pick a random number
  # in [0, 1] and use the corresponding interval in train_buckets_scale.
  random_number_01 = np.random.random_sample()
  bucket_id = min([i for i in xrange(len(train_buckets_scale_c))
                   if train_buckets_scale_c[i] > random_number_01])
  while bucket_id > 0 and not data_set[bucket_id]:
    bucket_id -= 1
  for _ in xrange(10 if np.random.random_sample() < 0.9 else 1):
    if data.bins[bucket_id] > max_cur_length:
      random_number_01 = min(random_number_01, np.random.random_sample())
      bucket_id = min([i for i in xrange(len(train_buckets_scale_c))
                       if train_buckets_scale_c[i] > random_number_01])
      while bucket_id > 0 and not data_set[bucket_id]:
        bucket_id -= 1
  return bucket_id


def score_beams(beams, target, inp, history, p,
                print_out=False, test_mode=False):
  """Score beams."""
  if p == "progsynth":
    return score_beams_prog(beams, target, inp, history, print_out, test_mode)
  elif test_mode:
    return beams[0], 10.0 if str(beams[0][:len(target)]) == str(target) else 0.0
  else:
    history_s = [str(h) for h in history]
    best, best_score, tgt, eos_id = None, -1000.0, target, None
    if p == "wmt":
      eos_id = wmt.EOS_ID
    if eos_id and eos_id in target:
      tgt = target[:target.index(eos_id)]
    for beam in beams:
      if eos_id and eos_id in beam:
        beam = beam[:beam.index(eos_id)]
      l = min(len(tgt), len(beam))
      score = len([i for i in xrange(l) if tgt[i] == beam[i]]) / float(len(tgt))
      hist_score = 20.0 if str([b for b in beam if b > 0]) in history_s else 0.0
      if score < 1.0:
        score -= hist_score
      if score > best_score:
        best = beam
        best_score = score
    return best, best_score


def score_beams_prog(beams, target, inp, history, print_out=False,
                     test_mode=False):
  """Score beams for program synthesis."""
  tgt_prog = linearize(target, program_utils.prog_vocab, True, 1)
  hist_progs = [linearize(h, program_utils.prog_vocab, True, 1)
                for h in history]
  tgt_set = set(target)
  if print_out:
    print("target: ", tgt_prog)
  inps, tgt_outs = [], []
  for i in xrange(3):
    ilist = [inp[i + 1, l] for l in xrange(inp.shape[1])]
    clist = [program_utils.prog_vocab[x] for x in ilist if x > 0]
    olist = clist[clist.index("]") + 1:]  # outputs
    clist = clist[1:clist.index("]")]     # inputs
    inps.append([int(x) for x in clist])
    if olist[0] == "[":  # olist may be [int] or just int
      tgt_outs.append(str([int(x) for x in olist[1:-1]]))
    else:
      if len(olist) == 1:
        tgt_outs.append(olist[0])
      else:
        print([program_utils.prog_vocab[x] for x in ilist if x > 0])
        print(olist)
        print(tgt_prog)
        print(program_utils.evaluate(tgt_prog, {"a": inps[-1]}))
        print("AAAAA")
        tgt_outs.append(olist[0])
  if not test_mode:
    for _ in xrange(7):
      ilen = np.random.randint(len(target) - 3) + 1
      inps.append([random.choice(range(-15, 15)) for _ in range(ilen)])
    tgt_outs.extend([program_utils.evaluate(tgt_prog, {"a": inp})
                     for inp in inps[3:]])
  best, best_prog, best_score = None, "", -1000.0
  for beam in beams:
    b_prog = linearize(beam, program_utils.prog_vocab, True, 1)
    b_set = set(beam)
    jsim = len(tgt_set & b_set) / float(len(tgt_set | b_set))
    b_outs = [program_utils.evaluate(b_prog, {"a": inp}) for inp in inps]
    errs = len([x for x in b_outs if x == "ERROR"])
    imatches = len([i for i in xrange(3) if b_outs[i] == tgt_outs[i]])
    perfect = 10.0 if imatches == 3 else 0.0
    hist_score = 20.0 if b_prog in hist_progs else 0.0
    if test_mode:
      score = perfect - errs
    else:
      matches = len([i for i in xrange(10) if b_outs[i] == tgt_outs[i]])
      score = perfect + matches + jsim - errs
    if score < 10.0:
      score -= hist_score
    # print b_prog
    # print "jsim: ", jsim, " errs: ", errs, " mtchs: ", matches, " s: ", score
    if score > best_score:
      best = beam
      best_prog = b_prog
      best_score = score
  if print_out:
    print("best score: ", best_score, " best prog: ", best_prog)
  return best, best_score


def get_best_beam(beam_model, sess, inp, target, batch_size, beam_size,
                  bucket, history, p, test_mode=False):
  """Run beam_model, score beams, and return the best as target and in input."""
  _, output_logits, _, _ = beam_model.step(
      sess, inp, target, None, beam_size=FLAGS.beam_size)
  new_targets, new_firsts, scores, new_inp = [], [], [], np.copy(inp)
  for b in xrange(batch_size):
    outputs = []
    history_b = [[h[b, 0, l] for l in xrange(data.bins[bucket])]
                 for h in history]
    for beam_idx in xrange(beam_size):
      outputs.append([int(o[beam_idx * batch_size + b])
                      for o in output_logits])
    target_t = [target[b, 0, l] for l in xrange(data.bins[bucket])]
    best, best_score = score_beams(
        outputs, [t for t in target_t if t > 0], inp[b, :, :],
        [[t for t in h if t > 0] for h in history_b], p, test_mode=test_mode)
    scores.append(best_score)
    if 1 in best:  # Only until _EOS.
      best = best[:best.index(1) + 1]
    best += [0 for _ in xrange(len(target_t) - len(best))]
    new_targets.append([best])
    first, _ = score_beams(
        outputs, [t for t in target_t if t > 0], inp[b, :, :],
        [[t for t in h if t > 0] for h in history_b], p, test_mode=True)
    if 1 in first:  # Only until _EOS.
      first = first[:first.index(1) + 1]
    first += [0 for _ in xrange(len(target_t) - len(first))]
    new_inp[b, 0, :] = np.array(first, dtype=np.int32)
    new_firsts.append([first])
  # Change target if we found a great answer.
  new_target = np.array(new_targets, dtype=np.int32)
  for b in xrange(batch_size):
    if scores[b] >= 10.0:
      target[b, 0, :] = new_target[b, 0, :]
  new_first = np.array(new_firsts, dtype=np.int32)
  return new_target, new_first, new_inp, scores


def train():
  """Train the model."""
  batch_size = FLAGS.batch_size * FLAGS.num_gpus
  (model, beam_model, min_length, max_length, checkpoint_dir,
   (train_set, dev_set, en_vocab_path, fr_vocab_path), sv, sess) = initialize()
  with sess.as_default():
    quant_op = model.quantize_op
    max_cur_length = min(min_length + 3, max_length)
    prev_acc_perp = [1000000 for _ in xrange(5)]
    prev_seq_err = 1.0
    is_chief = FLAGS.task < 1
    do_report = False

    # Main traning loop.
    while not sv.ShouldStop():
      global_step, max_cur_length, learning_rate = sess.run(
          [model.global_step, model.cur_length, model.lr])
      acc_loss, acc_l1, acc_total, acc_errors, acc_seq_err = 0.0, 0.0, 0, 0, 0
      acc_grad_norm, step_count, step_c1, step_time = 0.0, 0, 0, 0.0

      # For words in the word vector file, set their embedding at start.
      bound1 = FLAGS.steps_per_checkpoint - 1
      if FLAGS.word_vector_file_en and global_step < bound1 and is_chief:
        assign_vectors(FLAGS.word_vector_file_en, "embedding:0",
                       en_vocab_path, sess)
        if FLAGS.max_target_vocab < 1:
          assign_vectors(FLAGS.word_vector_file_en, "target_embedding:0",
                         en_vocab_path, sess)

      if FLAGS.word_vector_file_fr and global_step < bound1 and is_chief:
        assign_vectors(FLAGS.word_vector_file_fr, "embedding:0",
                       fr_vocab_path, sess)
        if FLAGS.max_target_vocab < 1:
          assign_vectors(FLAGS.word_vector_file_fr, "target_embedding:0",
                         fr_vocab_path, sess)

      for _ in xrange(FLAGS.steps_per_checkpoint):
        step_count += 1
        step_c1 += 1
        global_step = int(model.global_step.eval())
        train_beam_anneal = global_step / float(FLAGS.train_beam_anneal)
        train_beam_freq = FLAGS.train_beam_freq * min(1.0, train_beam_anneal)
        p = random.choice(FLAGS.problem.split("-"))
        train_set = global_train_set[p][-1]
        bucket_id = get_bucket_id(train_buckets_scale[p][-1], max_cur_length,
                                  train_set)
        # Prefer longer stuff 60% of time if not wmt.
        if np.random.randint(100) < 60 and FLAGS.problem != "wmt":
          bucket1 = get_bucket_id(train_buckets_scale[p][-1], max_cur_length,
                                  train_set)
          bucket_id = max(bucket1, bucket_id)

        # Run a step and time it.
        start_time = time.time()
        inp, target = data.get_batch(bucket_id, batch_size, train_set,
                                     FLAGS.height)
        noise_param = math.sqrt(math.pow(global_step + 1, -0.55) *
                                prev_seq_err) * FLAGS.grad_noise_scale
        # In multi-step mode, we use best from beam for middle steps.
        state, new_target, scores, history = None, None, None, []
        while (FLAGS.beam_size > 1 and
               train_beam_freq > np.random.random_sample()):
          # Get the best beam (no training, just forward model).
          new_target, new_first, new_inp, scores = get_best_beam(
              beam_model, sess, inp, target,
              batch_size, FLAGS.beam_size, bucket_id, history, p)
          history.append(new_first)
          # Training step with the previous input and the best beam as target.
          _, _, _, state = model.step(sess, inp, new_target, FLAGS.do_train,
                                      noise_param, update_mem=True, state=state)
          # Change input to the new one for the next step.
          inp = new_inp
          # If all results are great, stop (todo: not to wait for all?).
          if FLAGS.nprint > 1:
            print(scores)
          if sum(scores) / float(len(scores)) >= 10.0:
            break
        # The final step with the true target.
        loss, res, gnorm, _ = model.step(
            sess, inp, target, FLAGS.do_train, noise_param,
            update_mem=True, state=state)
        step_time += time.time() - start_time
        acc_grad_norm += 0.0 if gnorm is None else float(gnorm)

        # Accumulate statistics.
        acc_loss += loss
        acc_l1 += loss
        errors, total, seq_err = data.accuracy(
            inp, res, target, batch_size, 0, new_target, scores)
        if FLAGS.nprint > 1:
          print("seq_err: ", seq_err)
        acc_total += total
        acc_errors += errors
        acc_seq_err += seq_err

        # Report summary every 10 steps.
        if step_count + 3 > FLAGS.steps_per_checkpoint:
          do_report = True  # Don't polute plot too early.
        if is_chief and step_count % 10 == 1 and do_report:
          cur_loss = acc_l1 / float(step_c1)
          acc_l1, step_c1 = 0.0, 0
          cur_perp = data.safe_exp(cur_loss)
          summary = tf.Summary()
          summary.value.extend(
              [tf.Summary.Value(tag="log_perplexity", simple_value=cur_loss),
               tf.Summary.Value(tag="perplexity", simple_value=cur_perp)])
          sv.SummaryComputed(sess, summary, global_step)

      # Normalize and print out accumulated statistics.
      acc_loss /= step_count
      step_time /= FLAGS.steps_per_checkpoint
      acc_seq_err = float(acc_seq_err) / (step_count * batch_size)
      prev_seq_err = max(0.0, acc_seq_err - 0.02)  # No noise at error < 2%.
      acc_errors = float(acc_errors) / acc_total if acc_total > 0 else 1.0
      t_size = float(sum([len(x) for x in train_set])) / float(1000000)
      msg = ("step %d step-time %.2f train-size %.3f lr %.6f grad-norm %.4f"
             % (global_step + 1, step_time, t_size, learning_rate,
                acc_grad_norm / FLAGS.steps_per_checkpoint))
      data.print_out("%s len %d ppl %.6f errors %.2f sequence-errors %.2f" %
                     (msg, max_cur_length, data.safe_exp(acc_loss),
                      100*acc_errors, 100*acc_seq_err))

      # If errors are below the curriculum threshold, move curriculum forward.
      is_good = FLAGS.curriculum_ppx > data.safe_exp(acc_loss)
      is_good = is_good and FLAGS.curriculum_seq > acc_seq_err
      if is_good and is_chief:
        if FLAGS.quantize:
          # Quantize weights.
          data.print_out("  Quantizing parameters.")
          sess.run([quant_op])
        # Increase current length (until the next with training data).
        sess.run(model.cur_length_incr_op)
        # Forget last perplexities if we're not yet at the end.
        if max_cur_length < max_length:
          prev_acc_perp.append(1000000)

      # Lower learning rate if we're worse than the last 5 checkpoints.
      acc_perp = data.safe_exp(acc_loss)
      if acc_perp > max(prev_acc_perp[-5:]) and is_chief:
        sess.run(model.lr_decay_op)
      prev_acc_perp.append(acc_perp)

      # Save checkpoint.
      if is_chief:
        checkpoint_path = os.path.join(checkpoint_dir, "neural_gpu.ckpt")
        model.saver.save(sess, checkpoint_path,
                         global_step=model.global_step)

        # Run evaluation.
        bin_bound = 4
        for p in FLAGS.problem.split("-"):
          total_loss, total_err, tl_counter = 0.0, 0.0, 0
          for bin_id in xrange(len(data.bins)):
            if bin_id < bin_bound or bin_id % FLAGS.eval_bin_print == 1:
              err, _, loss = single_test(bin_id, model, sess, FLAGS.nprint,
                                         batch_size * 4, dev_set, p,
                                         beam_model=beam_model)
              if loss > 0.0:
                total_loss += loss
                total_err += err
                tl_counter += 1
          test_loss = total_loss / max(1, tl_counter)
          test_err = total_err / max(1, tl_counter)
          test_perp = data.safe_exp(test_loss)
          summary = tf.Summary()
          summary.value.extend(
              [tf.Summary.Value(tag="test/%s/loss" % p, simple_value=test_loss),
               tf.Summary.Value(tag="test/%s/error" % p, simple_value=test_err),
               tf.Summary.Value(tag="test/%s/perplexity" % p,
                                simple_value=test_perp)])
          sv.SummaryComputed(sess, summary, global_step)


def linearize(output, rev_fr_vocab, simple_tokenizer=None, eos_id=wmt.EOS_ID):
  # If there is an EOS symbol in outputs, cut them at that point (WMT).
  if eos_id in output:
    output = output[:output.index(eos_id)]
  # Print out French sentence corresponding to outputs.
  if simple_tokenizer or FLAGS.simple_tokenizer:
    vlen = len(rev_fr_vocab)
    def vget(o):
      if o < vlen:
        return rev_fr_vocab[o]
      return "UNK"
    return " ".join([vget(o) for o in output])
  else:
    return wmt.basic_detokenizer([rev_fr_vocab[o] for o in output])


def evaluate():
  """Evaluate an existing model."""
  batch_size = FLAGS.batch_size * FLAGS.num_gpus
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    (model, beam_model, _, _, _,
     (_, dev_set, en_vocab_path, fr_vocab_path), _, sess) = initialize(sess)
    for p in FLAGS.problem.split("-"):
      for bin_id in xrange(len(data.bins)):
        if (FLAGS.task >= 0 and bin_id > 4) or (FLAGS.nprint == 0 and
                                                bin_id > 8 and p == "wmt"):
          break
        single_test(bin_id, model, sess, FLAGS.nprint, batch_size, dev_set, p,
                    beam_model=beam_model)
    path = FLAGS.test_file_prefix
    xid = "" if FLAGS.task < 0 else ("%.4d" % (FLAGS.task+FLAGS.decode_offset))
    en_path, fr_path = path + ".en" + xid, path + ".fr" + xid
    # Evaluate the test file if they exist.
    if path and tf.gfile.Exists(en_path) and tf.gfile.Exists(fr_path):
      data.print_out("Translating test set %s" % en_path)
      # Read lines.
      en_lines, fr_lines = [], []
      with tf.gfile.GFile(en_path, mode="r") as f:
        for line in f:
          en_lines.append(line.strip())
      with tf.gfile.GFile(fr_path, mode="r") as f:
        for line in f:
          fr_lines.append(line.strip())
      # Tokenize and convert to ids.
      en_vocab, _ = wmt.initialize_vocabulary(en_vocab_path)
      _, rev_fr_vocab = wmt.initialize_vocabulary(fr_vocab_path)
      if FLAGS.simple_tokenizer:
        en_ids = [wmt.sentence_to_token_ids(
            l, en_vocab, tokenizer=wmt.space_tokenizer,
            normalize_digits=FLAGS.normalize_digits)
                  for l in en_lines]
      else:
        en_ids = [wmt.sentence_to_token_ids(l, en_vocab) for l in en_lines]
      # Translate.
      results = []
      for idx, token_ids in enumerate(en_ids):
        if idx % 5 == 0:
          data.print_out("Translating example %d of %d." % (idx, len(en_ids)))
        # Which bucket does it belong to?
        buckets = [b for b in xrange(len(data.bins))
                   if data.bins[b] >= len(token_ids)]
        if buckets:
          result, result_cost = [], 100000000.0
          for bucket_id in buckets:
            if data.bins[bucket_id] > MAXLEN_F * len(token_ids) + EVAL_LEN_INCR:
              break
            # Get a 1-element batch to feed the sentence to the model.
            used_batch_size = 1  # batch_size
            inp, target = data.get_batch(
                bucket_id, used_batch_size, None, FLAGS.height,
                preset=([token_ids], [[]]))
            loss, output_logits, _, _ = model.step(
                sess, inp, target, None, beam_size=FLAGS.beam_size)
            outputs = [int(o[0]) for o in output_logits]
            loss = loss[0] - (data.bins[bucket_id] * FLAGS.length_norm)
            if FLAGS.simple_tokenizer:
              cur_out = outputs
              if wmt.EOS_ID in cur_out:
                cur_out = cur_out[:cur_out.index(wmt.EOS_ID)]
              res_tags = [rev_fr_vocab[o] for o in cur_out]
              bad_words, bad_brack = wmt.parse_constraints(token_ids, res_tags)
              loss += 1000.0 * bad_words + 100.0 * bad_brack
            # print (bucket_id, loss)
            if loss < result_cost:
              result = outputs
              result_cost = loss
          final = linearize(result, rev_fr_vocab)
          results.append("%s\t%s\n" % (final, fr_lines[idx]))
          # print result_cost
          sys.stderr.write(results[-1])
          sys.stderr.flush()
        else:
          sys.stderr.write("TOOO_LONG\t%s\n" % fr_lines[idx])
          sys.stderr.flush()
      if xid:
        decode_suffix = "beam%dln%dn" % (FLAGS.beam_size,
                                         int(100 * FLAGS.length_norm))
        with tf.gfile.GFile(path + ".res" + decode_suffix + xid, mode="w") as f:
          for line in results:
            f.write(line)


def mul(l):
  res = 1.0
  for s in l:
    res *= s
  return res


def interactive():
  """Interactively probe an existing model."""
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Initialize model.
    (model, _, _, _, _, (_, _, en_path, fr_path), _, _) = initialize(sess)
    # Load vocabularies.
    en_vocab, rev_en_vocab = wmt.initialize_vocabulary(en_path)
    _, rev_fr_vocab = wmt.initialize_vocabulary(fr_path)
    # Print out vectors and variables.
    if FLAGS.nprint > 0 and FLAGS.word_vector_file_en:
      print_vectors("embedding:0", en_path, FLAGS.word_vector_file_en)
    if FLAGS.nprint > 0 and FLAGS.word_vector_file_fr:
      print_vectors("target_embedding:0", fr_path, FLAGS.word_vector_file_fr)
    total = 0
    for v in tf.trainable_variables():
      shape = v.get_shape().as_list()
      total += mul(shape)
      print(v.name, shape, mul(shape))
    print(total)
    # Start interactive loop.
    sys.stdout.write("Input to Neural GPU Translation Model.\n")
    sys.stdout.write("> ")
    sys.stdout.flush()
    inpt = sys.stdin.readline(), ""
    while inpt:
      cures = []
      # Get token-ids for the input sentence.
      if FLAGS.simple_tokenizer:
        token_ids = wmt.sentence_to_token_ids(
            inpt, en_vocab, tokenizer=wmt.space_tokenizer,
            normalize_digits=FLAGS.normalize_digits)
      else:
        token_ids = wmt.sentence_to_token_ids(inpt, en_vocab)
      print([rev_en_vocab[t] for t in token_ids])
      # Which bucket does it belong to?
      buckets = [b for b in xrange(len(data.bins))
                 if data.bins[b] >= max(len(token_ids), len(cures))]
      if cures:
        buckets = [buckets[0]]
      if buckets:
        result, result_cost = [], 10000000.0
        for bucket_id in buckets:
          if data.bins[bucket_id] > MAXLEN_F * len(token_ids) + EVAL_LEN_INCR:
            break
          glen = 1
          for gen_idx in xrange(glen):
            # Get a 1-element batch to feed the sentence to the model.
            inp, target = data.get_batch(
                bucket_id, 1, None, FLAGS.height, preset=([token_ids], [cures]))
            loss, output_logits, _, _ = model.step(
                sess, inp, target, None, beam_size=FLAGS.beam_size,
                update_mem=False)
            # If it is a greedy decoder, outputs are argmaxes of output_logits.
            if FLAGS.beam_size > 1:
              outputs = [int(o) for o in output_logits]
            else:
              loss = loss[0] - (data.bins[bucket_id] * FLAGS.length_norm)
              outputs = [int(np.argmax(logit, axis=1))
                         for logit in output_logits]
            print([rev_fr_vocab[t] for t in outputs])
            print(loss, data.bins[bucket_id])
            print(linearize(outputs, rev_fr_vocab))
            cures.append(outputs[gen_idx])
            print(cures)
            print(linearize(cures, rev_fr_vocab))
          if FLAGS.simple_tokenizer:
            cur_out = outputs
            if wmt.EOS_ID in cur_out:
              cur_out = cur_out[:cur_out.index(wmt.EOS_ID)]
            res_tags = [rev_fr_vocab[o] for o in cur_out]
            bad_words, bad_brack = wmt.parse_constraints(token_ids, res_tags)
            loss += 1000.0 * bad_words + 100.0 * bad_brack
          if loss < result_cost:
            result = outputs
            result_cost = loss
        print("FINAL", result_cost)
        print([rev_fr_vocab[t] for t in result])
        print(linearize(result, rev_fr_vocab))
      else:
        print("TOOO_LONG")
      sys.stdout.write("> ")
      sys.stdout.flush()
      inpt = sys.stdin.readline(), ""


def main(_):
  if FLAGS.mode == 0:
    train()
  elif FLAGS.mode == 1:
    evaluate()
  else:
    interactive()

if __name__ == "__main__":
  tf.app.run()
