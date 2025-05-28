# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

# -*- coding: utf-8 -*-
"""Script to pre-process pre-training data into tfrecords."""

import json
import os
import random

from absl import app
from absl import flags
from absl import logging

import numpy as np

import tensorflow.compat.v1 as tf
import sentencepiece as spm
from official.legacy.xlnet import preprocess_utils

FLAGS = flags.FLAGS


special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

VOCAB_SIZE = 32000
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def format_filename(prefix, bsz_per_host, seq_len, bi_data, suffix,
                    mask_alpha=5, mask_beta=1, reuse_len=None, uncased=False,
                    fixed_num_predict=None):
  """docs."""
  if reuse_len is None:
    reuse_len_str = ""
  else:
    reuse_len_str = "reuse-{}.".format(reuse_len)
  if not uncased:
    uncased_str = ""
  else:
    uncased_str = "uncased."
  if bi_data:
    bi_data_str = "bi"
  else:
    bi_data_str = "uni"
  if fixed_num_predict is not None:
    fnp_str = "fnp-{}.".format(fixed_num_predict)
  else:
    fnp_str = ""

  file_name = "{}.bsz-{}.seqlen-{}.{}{}{}.alpha-{}.beta-{}.{}{}".format(
      prefix, bsz_per_host, seq_len, reuse_len_str, uncased_str, bi_data_str,
      mask_alpha, mask_beta, fnp_str, suffix)

  return file_name


def _create_data(idx, input_paths):
  """Creates data."""
  # Load sentence-piece model
  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.sp_path)

  input_shards = []
  total_line_cnt = 0
  for input_path in input_paths:
    input_data, sent_ids = [], []
    sent_id, line_cnt = True, 0
    logging.info("Processing %s", input_path)
    for line in tf.gfile.Open(input_path):
      if line_cnt % 100000 == 0:
        logging.info("Loading line %d", line_cnt)
      line_cnt += 1

      if not line.strip():
        if FLAGS.use_eod:
          sent_id = not sent_id
          cur_sent = [EOD_ID]
        else:
          continue
      else:
        if FLAGS.from_raw_text:
          cur_sent = preprocess_utils.preprocess_text(
              line.strip(), lower=FLAGS.uncased)
          cur_sent = preprocess_utils.encode_ids(sp, cur_sent)
        else:
          cur_sent = list(map(int, line.strip().split()))

      input_data.extend(cur_sent)
      sent_ids.extend([sent_id] * len(cur_sent))
      sent_id = not sent_id

    logging.info("Finish with line %d", line_cnt)
    if line_cnt == 0:
      continue

    input_data = np.array(input_data, dtype=np.int64)
    sent_ids = np.array(sent_ids, dtype=bool)

    total_line_cnt += line_cnt
    input_shards.append((input_data, sent_ids))

  logging.info("[Task %d] Total number line: %d", idx, total_line_cnt)

  tfrecord_dir = os.path.join(FLAGS.save_dir, "tfrecords")

  filenames, num_batch = [], 0

  # Randomly shuffle input shards (with a fixed but distinct random seed)
  np.random.seed(100 * FLAGS.task + FLAGS.pass_id)

  perm_indices = np.random.permutation(len(input_shards))
  logging.info("Using perm indices %s for pass %d",
               perm_indices.tolist(), FLAGS.pass_id)

  input_data_list, sent_ids_list = [], []
  prev_sent_id = None
  for perm_idx in perm_indices:
    input_data, sent_ids = input_shards[perm_idx]
    # make sure the `send_ids[0] == not prev_sent_id`
    if prev_sent_id is not None and sent_ids[0] == prev_sent_id:
      sent_ids = np.logical_not(sent_ids)

    # append to temporary list
    input_data_list.append(input_data)
    sent_ids_list.append(sent_ids)

    # update `prev_sent_id`
    prev_sent_id = sent_ids[-1]

  input_data = np.concatenate(input_data_list)
  sent_ids = np.concatenate(sent_ids_list)

  file_name, cur_num_batch = create_tfrecords(
      save_dir=tfrecord_dir,
      basename="{}-{}-{}".format(FLAGS.split, idx, FLAGS.pass_id),
      data=[input_data, sent_ids],
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      bi_data=FLAGS.bi_data,
      sp=sp,
  )

  filenames.append(file_name)
  num_batch += cur_num_batch

  record_info = {
      "filenames": filenames,
      "num_batch": num_batch
  }

  return record_info


def create_data(_):
  """Creates pretrain data."""
  # Validate FLAGS
  assert FLAGS.bsz_per_host % FLAGS.num_core_per_host == 0
  if not FLAGS.use_tpu:
    FLAGS.num_core_per_host = 1  # forced to be one

  # Make workdirs
  if not tf.gfile.Exists(FLAGS.save_dir):
    tf.gfile.MakeDirs(FLAGS.save_dir)

  tfrecord_dir = os.path.join(FLAGS.save_dir, "tfrecords")
  if not tf.gfile.Exists(tfrecord_dir):
    tf.gfile.MakeDirs(tfrecord_dir)

  # Create and dump corpus_info from task 0
  if FLAGS.task == 0 and FLAGS.pass_id == 0:
    corpus_info = {
        "vocab_size": VOCAB_SIZE,
        "bsz_per_host": FLAGS.bsz_per_host,
        "num_core_per_host": FLAGS.num_core_per_host,
        "seq_len": FLAGS.seq_len,
        "reuse_len": FLAGS.reuse_len,
        "uncased": FLAGS.uncased,
        "bi_data": FLAGS.bi_data,
        "mask_alpha": FLAGS.mask_alpha,
        "mask_beta": FLAGS.mask_beta,
        "num_predict": FLAGS.num_predict,
        "use_eod": FLAGS.use_eod,
        "sp_path": FLAGS.sp_path,
        "input_glob": FLAGS.input_glob,
    }
    corpus_info_path = os.path.join(FLAGS.save_dir, "corpus_info.json")
    with tf.gfile.Open(corpus_info_path, "w") as fp:
      json.dump(corpus_info, fp)

  # Interleavely split the work into FLAGS.num_task splits
  file_paths = sorted(tf.gfile.Glob(FLAGS.input_glob))
  logging.info("Use glob: %s", FLAGS.input_glob)
  logging.info("Find %d files: %s", len(file_paths), file_paths)

  task_file_paths = file_paths[FLAGS.task::FLAGS.num_task]
  if not task_file_paths:
    logging.info("Exit: task %d has no file to process.", FLAGS.task)
    return

  logging.info("Task %d process %d files: %s",
               FLAGS.task, len(task_file_paths), task_file_paths)
  record_info = _create_data(FLAGS.task, task_file_paths)

  record_prefix = "record_info-{}-{}-{}".format(
      FLAGS.split, FLAGS.task, FLAGS.pass_id)
  record_name = format_filename(
      prefix=record_prefix,
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      reuse_len=FLAGS.reuse_len,
      bi_data=FLAGS.bi_data,
      suffix="json",
      uncased=FLAGS.uncased,
      fixed_num_predict=FLAGS.num_predict)
  record_info_path = os.path.join(tfrecord_dir, record_name)

  with tf.gfile.Open(record_info_path, "w") as fp:
    json.dump(record_info, fp)


def batchify(data, bsz_per_host, sent_ids=None):
  """Creates batches."""
  num_step = len(data) // bsz_per_host
  data = data[:bsz_per_host * num_step]
  data = data.reshape(bsz_per_host, num_step)
  if sent_ids is not None:
    sent_ids = sent_ids[:bsz_per_host * num_step]
    sent_ids = sent_ids.reshape(bsz_per_host, num_step)

  if sent_ids is not None:
    return data, sent_ids
  return data


def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
  """Split two segments from `data` starting from the index `begin_idx`."""

  data_len = data.shape[0]
  if begin_idx + tot_len >= data_len:
    logging.info("[_split_a_and_b] returns None: "
                 "begin_idx %d + tot_len %d >= data_len %d",
                 begin_idx, tot_len, data_len)
    return None

  end_idx = begin_idx + 1
  cut_points = []
  while end_idx < data_len:
    if sent_ids[end_idx] != sent_ids[end_idx - 1]:
      if end_idx - begin_idx >= tot_len: break
      cut_points.append(end_idx)
    end_idx += 1

  a_begin = begin_idx
  if len(cut_points) == 0 or random.random() < 0.5:  # pylint:disable=g-explicit-length-test
    label = 0
    if len(cut_points) == 0:  # pylint:disable=g-explicit-length-test
      a_end = end_idx
    else:
      a_end = random.choice(cut_points)

    b_len = max(1, tot_len - (a_end - a_begin))
    # (zihangd): `data_len - 1` to account for extend_target
    b_begin = random.randint(0, data_len - 1 - b_len)
    b_end = b_begin + b_len
    while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
      b_begin -= 1
    # (zihangd): `data_len - 1` to account for extend_target
    while b_end < data_len - 1 and sent_ids[b_end - 1] == sent_ids[b_end]:
      b_end += 1

    new_begin = a_end
  else:
    label = 1
    a_end = random.choice(cut_points)
    b_begin = a_end
    b_end = end_idx

    new_begin = b_end

  while a_end - a_begin + b_end - b_begin > tot_len:
    if a_end - a_begin > b_end - b_begin:
      # delete the right side only for the LM objective
      a_end -= 1
    else:
      b_end -= 1

  ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

  if extend_target:
    if a_end >= data_len or b_end >= data_len:
      logging.info("[_split_a_and_b] returns None: "
                   "a_end %d or b_end %d >= data_len %d",
                   a_end, b_end, data_len)
      return None
    a_target = data[a_begin + 1: a_end + 1]
    b_target = data[b_begin: b_end + 1]
    ret.extend([a_target, b_target])

  return ret


def _is_start_piece(piece):
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  if (piece.startswith("▁") or piece.startswith("<")
      or piece in special_pieces):
    return True
  else:
    return False


def _sample_mask(sp, seg, reverse=False, max_gram=5, goal_num_predict=None):
  """Samples `goal_num_predict` tokens for partial prediction."""
  seg_len = len(seg)
  mask = np.array([False] * seg_len, dtype=bool)

  num_predict = 0

  ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
  pvals = 1. / np.arange(1, max_gram + 1)
  pvals /= pvals.sum(keepdims=True)

  if reverse:
    seg = np.flip(seg, 0)

  cur_len = 0
  while cur_len < seg_len:
    if goal_num_predict is not None and num_predict >= goal_num_predict: break

    n = np.random.choice(ngrams, p=pvals)
    if goal_num_predict is not None:
      n = min(n, goal_num_predict - num_predict)
    ctx_size = (n * FLAGS.mask_alpha) // FLAGS.mask_beta
    l_ctx = np.random.choice(ctx_size)
    r_ctx = ctx_size - l_ctx

    # Find the start position of a complete token
    beg = cur_len + l_ctx
    while beg < seg_len and not _is_start_piece(sp.IdToPiece(seg[beg].item())):
      beg += 1
    if beg >= seg_len:
      break

    # Find the end position of the n-gram (start pos of the n+1-th gram)
    end = beg + 1
    cnt_ngram = 1
    while end < seg_len:
      cnt_ngram += 1
      if cnt_ngram > n:
        break
      end += 1
    if end >= seg_len:
      break

    # Update
    mask[beg:end] = True
    num_predict += end - beg

    cur_len = end + r_ctx

  while goal_num_predict is not None and num_predict < goal_num_predict:
    i = np.random.randint(seg_len)
    if not mask[i]:
      mask[i] = True
      num_predict += 1

  if reverse:
    mask = np.flip(mask, 0)

  return mask


def _sample_mask_ngram(sp, seg, reverse=False, max_gram=5,
                       goal_num_predict=None):
  """Sample `goal_num_predict` tokens for partial prediction."""

  seg_len = len(seg)
  mask = np.array([False] * seg_len, dtype=bool)

  num_predict = 0

  ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
  pvals = 1. / np.arange(1, max_gram + 1)
  pvals /= pvals.sum(keepdims=True)

  if reverse:
    seg = np.flip(seg, 0)

  cur_len = 0
  while cur_len < seg_len:
    if goal_num_predict is not None and num_predict >= goal_num_predict: break

    n = np.random.choice(ngrams, p=pvals)
    if goal_num_predict is not None:
      n = min(n, goal_num_predict - num_predict)
    ctx_size = (n * FLAGS.mask_alpha) // FLAGS.mask_beta
    l_ctx = np.random.choice(ctx_size)
    r_ctx = ctx_size - l_ctx

    # Find the start position of a complete token
    beg = cur_len + l_ctx
    while beg < seg_len and not _is_start_piece(sp.IdToPiece(seg[beg].item())):
      beg += 1
    if beg >= seg_len:
      break

    # Find the end position of the n-gram (start pos of the n+1-th gram)
    end = beg
    cnt_ngram = 0
    while end < seg_len:
      if _is_start_piece(sp.IdToPiece(seg[end].item())):
        cnt_ngram += 1
        if cnt_ngram > n:
          break

      # select current piece
      mask[end] = True

      # update the end pointer and increment num_predict
      end += 1
      num_predict += 1

      if goal_num_predict is not None and num_predict >= goal_num_predict:
        break

    cur_len = end + r_ctx

  while goal_num_predict is not None and num_predict < goal_num_predict:
    i = np.random.randint(seg_len)
    if not mask[i]:
      mask[i] = True
      num_predict += 1

  if reverse:
    mask = np.flip(mask, 0)

  return mask


def create_tfrecords(save_dir, basename, data, bsz_per_host, seq_len,
                     bi_data, sp):
  """Creates TFRecords."""
  data, sent_ids = data[0], data[1]

  num_core = FLAGS.num_core_per_host
  bsz_per_core = bsz_per_host // num_core

  if bi_data:
    assert bsz_per_host % (2 * FLAGS.num_core_per_host) == 0
    fwd_data, fwd_sent_ids = batchify(data, bsz_per_host // 2, sent_ids)

    fwd_data = fwd_data.reshape(num_core, 1, bsz_per_core // 2, -1)
    fwd_sent_ids = fwd_sent_ids.reshape(num_core, 1, bsz_per_core // 2, -1)

    bwd_data = fwd_data[:, :, :, ::-1]
    bwd_sent_ids = fwd_sent_ids[:, :, :, ::-1]

    data = np.concatenate(
        [fwd_data, bwd_data], 1).reshape(bsz_per_host, -1)
    sent_ids = np.concatenate(
        [fwd_sent_ids, bwd_sent_ids], 1).reshape(bsz_per_host, -1)
  else:
    data, sent_ids = batchify(data, bsz_per_host, sent_ids)

  logging.info("Raw data shape %s.", data.shape)

  file_name = format_filename(
      prefix=basename,
      bsz_per_host=bsz_per_host,
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="tfrecords",
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      reuse_len=FLAGS.reuse_len,
      uncased=FLAGS.uncased,
      fixed_num_predict=FLAGS.num_predict
  )
  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.python_io.TFRecordWriter(save_path)
  logging.info("Start writing %s.", save_path)

  num_batch = 0
  reuse_len = FLAGS.reuse_len

  # [sep] x 2 + [cls]
  assert reuse_len < seq_len - 3

  data_len = data.shape[1]
  sep_array = np.array([SEP_ID], dtype=np.int64)
  cls_array = np.array([CLS_ID], dtype=np.int64)

  i = 0
  while i + seq_len <= data_len:
    if num_batch % 500 == 0:
      logging.info("Processing batch %d", num_batch)

    all_ok = True
    features = []
    for idx in range(bsz_per_host):
      inp = data[idx, i: i + reuse_len]
      tgt = data[idx, i + 1: i + reuse_len + 1]

      results = _split_a_and_b(
          data[idx],
          sent_ids[idx],
          begin_idx=i + reuse_len,
          tot_len=seq_len - reuse_len - 3,
          extend_target=True)
      if results is None:
        logging.info("Break out with seq idx %d", i)
        all_ok = False
        break

      # unpack the results
      (a_data, b_data, label, _, a_target, b_target) = tuple(results)

      # sample ngram spans to predict
      reverse = bi_data and (idx // (bsz_per_core // 2)) % 2 == 1
      if FLAGS.num_predict is None:
        num_predict_0 = num_predict_1 = None
      else:
        num_predict_1 = FLAGS.num_predict // 2
        num_predict_0 = FLAGS.num_predict - num_predict_1
      mask_0 = _sample_mask(sp, inp, reverse=reverse,
                            goal_num_predict=num_predict_0)
      mask_1 = _sample_mask(sp, np.concatenate([a_data, sep_array, b_data,
                                                sep_array, cls_array]),
                            reverse=reverse, goal_num_predict=num_predict_1)

      # concatenate data
      cat_data = np.concatenate([inp, a_data, sep_array, b_data,
                                 sep_array, cls_array])
      seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] +
                [1] * b_data.shape[0] + [1] + [2])
      assert cat_data.shape[0] == seq_len
      assert mask_0.shape[0] == seq_len // 2
      assert mask_1.shape[0] == seq_len // 2

      # the last two CLS's are not used, just for padding purposes
      tgt = np.concatenate([tgt, a_target, b_target, cls_array, cls_array])
      assert tgt.shape[0] == seq_len

      is_masked = np.concatenate([mask_0, mask_1], 0)
      if FLAGS.num_predict is not None:
        assert np.sum(is_masked) == FLAGS.num_predict

      feature = {
          "input": _int64_feature(cat_data),
          "is_masked": _int64_feature(is_masked),
          "target": _int64_feature(tgt),
          "seg_id": _int64_feature(seg_id),
          "label": _int64_feature([label]),
      }
      features.append(feature)

    if all_ok:
      assert len(features) == bsz_per_host
      for feature in features:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())
      num_batch += 1
    else:
      break

    i += reuse_len

  record_writer.close()
  logging.info("Done writing %s. Num of batches: %d", save_path, num_batch)

  return save_path, num_batch


################
# get_input_fn #
################
def _convert_example(example, use_bfloat16):
  """Cast int64 into int32 and float32 to bfloat16 if use_bfloat16."""
  for key in list(example.keys()):
    val = example[key]
    if tf_keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    if val.dtype == tf.int64:
      val = tf.cast(val, tf.int32)
    if use_bfloat16 and val.dtype == tf.float32:
      val = tf.cast(val, tf.bfloat16)

    example[key] = val


def parse_files_to_dataset(parser, file_names, split, num_batch, num_hosts,
                           host_id, num_core_per_host, bsz_per_core):
  """Parses files to a dataset."""
  del num_batch
  # list of file pathes
  num_files = len(file_names)
  num_files_per_host = num_files // num_hosts
  my_start_file_id = host_id * num_files_per_host
  my_end_file_id = (host_id + 1) * num_files_per_host
  if host_id == num_hosts - 1:
    my_end_file_id = num_files
  file_paths = file_names[my_start_file_id: my_end_file_id]
  logging.info("Host %d handles %d files", host_id, len(file_paths))

  assert split == "train"
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # file-level shuffle
  if len(file_paths) > 1:
    dataset = dataset.shuffle(len(file_paths))

  # Note: we cannot perform sample-level shuffle here because this will violate
  # the consecutive requirement of data stream.
  dataset = tf.data.TFRecordDataset(dataset)

  # Note: since we are doing online preprocessing, the parsed result of
  # the same input at each time will be different. Thus, cache processed data
  # is not helpful. It will use a lot of memory and lead to contrainer OOM.
  # So, change to cache non-parsed raw data instead.
  dataset = dataset.cache().map(parser).repeat()
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(num_core_per_host * bsz_per_core)

  return dataset


def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
  """Samples a permutation of the factorization order, and create a mask.

  Args:
    inputs: int64 Tensor in shape [seq_len], input ids.
    targets: int64 Tensor in shape [seq_len], target ids.
    is_masked: bool Tensor in shape [seq_len]. True means being selected
      for partial prediction.
    perm_size: the length of longest permutation. Could be set to be reuse_len.
      Should not be larger than reuse_len or there will be data leaks.
    seq_len: int, sequence length.

  Returns:
    The permutation mask, new targets, target mask, and new inputs.

  """

  # Generate permutation indices
  index = tf.range(seq_len, dtype=tf.int64)
  index = tf.transpose(tf.reshape(index, [-1, perm_size]))
  index = tf.random_shuffle(index)
  index = tf.reshape(tf.transpose(index), [-1])

  # `perm_mask` and `target_mask`
  # non-functional tokens
  non_func_tokens = tf.logical_not(tf.logical_or(
      tf.equal(inputs, SEP_ID),
      tf.equal(inputs, CLS_ID)))

  non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
  masked_or_func_tokens = tf.logical_not(non_mask_tokens)

  # Set the permutation indices of non-masked (& non-funcional) tokens to the
  # smallest index (-1):
  # (1) they can be seen by all other positions
  # (2) they cannot see masked positions, so there won"t be information leak
  smallest_index = -tf.ones([seq_len], dtype=tf.int64)
  rev_index = tf.where(non_mask_tokens, smallest_index, index)

  # Create `target_mask`: non-funcional and maksed tokens
  # 1: use mask as input and have loss
  # 0: use token (or [SEP], [CLS]) as input and do not have loss
  target_tokens = tf.logical_and(masked_or_func_tokens, non_func_tokens)
  target_mask = tf.cast(target_tokens, tf.float32)

  # Create `perm_mask`
  # `target_tokens` cannot see themselves
  self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)

  # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
  # 0: can attend if i > j or j is non-masked
  perm_mask = tf.logical_and(
      self_rev_index[:, None] <= rev_index[None, :],
      masked_or_func_tokens)
  perm_mask = tf.cast(perm_mask, tf.float32)

  # new target: [next token] for LM and [curr token] (self) for PLM
  new_targets = tf.concat([inputs[0: 1], targets[: -1]],
                          axis=0)

  # construct inputs_k
  inputs_k = inputs

  # construct inputs_q
  inputs_q = target_mask

  return perm_mask, new_targets, target_mask, inputs_k, inputs_q


def get_dataset(params, num_hosts, num_core_per_host, split, file_names,
                num_batch, seq_len, reuse_len, perm_size, mask_alpha,
                mask_beta, use_bfloat16=False, num_predict=None):
  """Gets the dataset."""

  del mask_alpha
  del mask_beta
  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

    #### Function used to parse tfrecord
  def parser(record):
    """function used to parse tfrecord."""

    record_spec = {
        "input": tf.FixedLenFeature([seq_len], tf.int64),
        "target": tf.FixedLenFeature([seq_len], tf.int64),
        "seg_id": tf.FixedLenFeature([seq_len], tf.int64),
        "label": tf.FixedLenFeature([1], tf.int64),
        "is_masked": tf.FixedLenFeature([seq_len], tf.int64),
    }

    # retrieve serialized example
    example = tf.parse_single_example(
        serialized=record,
        features=record_spec)

    inputs = example.pop("input")
    target = example.pop("target")
    is_masked = tf.cast(example.pop("is_masked"), tf.bool)

    non_reuse_len = seq_len - reuse_len
    assert perm_size <= reuse_len and perm_size <= non_reuse_len

    perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
        inputs[:reuse_len],
        target[:reuse_len],
        is_masked[:reuse_len],
        perm_size,
        reuse_len)

    perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
        inputs[reuse_len:],
        target[reuse_len:],
        is_masked[reuse_len:],
        perm_size,
        non_reuse_len)

    perm_mask_0 = tf.concat([perm_mask_0, tf.ones([reuse_len, non_reuse_len])],
                            axis=1)
    perm_mask_1 = tf.concat([tf.zeros([non_reuse_len, reuse_len]), perm_mask_1],
                            axis=1)
    perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=0)
    target = tf.concat([target_0, target_1], axis=0)
    target_mask = tf.concat([target_mask_0, target_mask_1], axis=0)
    input_k = tf.concat([input_k_0, input_k_1], axis=0)
    input_q = tf.concat([input_q_0, input_q_1], axis=0)

    if num_predict is not None:
      indices = tf.range(seq_len, dtype=tf.int64)
      bool_target_mask = tf.cast(target_mask, tf.bool)
      indices = tf.boolean_mask(indices, bool_target_mask)

      ##### extra padding due to CLS/SEP introduced after prepro
      actual_num_predict = tf.shape(indices)[0]
      pad_len = num_predict - actual_num_predict

      ##### target_mapping
      target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
      paddings = tf.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
      target_mapping = tf.concat([target_mapping, paddings], axis=0)
      example["target_mapping"] = tf.reshape(target_mapping,
                                             [num_predict, seq_len])

      ##### target
      target = tf.boolean_mask(target, bool_target_mask)
      paddings = tf.zeros([pad_len], dtype=target.dtype)
      target = tf.concat([target, paddings], axis=0)
      example["target"] = tf.reshape(target, [num_predict])

      ##### target mask
      target_mask = tf.concat(
          [tf.ones([actual_num_predict], dtype=tf.float32),
           tf.zeros([pad_len], dtype=tf.float32)],
          axis=0)
      example["target_mask"] = tf.reshape(target_mask, [num_predict])
    else:
      example["target"] = tf.reshape(target, [seq_len])
      example["target_mask"] = tf.reshape(target_mask, [seq_len])

    # reshape back to fixed shape
    example["perm_mask"] = tf.reshape(perm_mask, [seq_len, seq_len])
    example["input_k"] = tf.reshape(input_k, [seq_len])
    example["input_q"] = tf.reshape(input_q, [seq_len])

    _convert_example(example, use_bfloat16)

    for k, v in example.items():
      logging.info("%s: %s", k, v)

    return example

  # Get dataset
  dataset = parse_files_to_dataset(
      parser=parser,
      file_names=file_names,
      split=split,
      num_batch=num_batch,
      num_hosts=num_hosts,
      host_id=host_id,
      num_core_per_host=num_core_per_host,
      bsz_per_core=bsz_per_core)

  return dataset


def get_input_fn(
    tfrecord_dir,
    split,
    bsz_per_host,
    seq_len,
    reuse_len,
    bi_data,
    num_hosts=1,
    num_core_per_host=1,
    perm_size=None,
    mask_alpha=None,
    mask_beta=None,
    uncased=False,
    num_passes=None,
    use_bfloat16=False,
    num_predict=None):
  """Gets the input function."""

  # Merge all record infos into a single one
  record_glob_base = format_filename(
      prefix="record_info-{}-*".format(split),
      bsz_per_host=bsz_per_host,
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="json",
      mask_alpha=mask_alpha,
      mask_beta=mask_beta,
      reuse_len=reuse_len,
      uncased=uncased,
      fixed_num_predict=num_predict)

  record_info = {"num_batch": 0, "filenames": []}

  tfrecord_dirs = tfrecord_dir.split(",")
  logging.info("Use the following tfrecord dirs: %s", tfrecord_dirs)

  for idx, record_dir in enumerate(tfrecord_dirs):
    record_glob = os.path.join(record_dir, record_glob_base)
    logging.info("[%d] Record glob: %s", idx, record_glob)

    record_paths = sorted(tf.gfile.Glob(record_glob))
    logging.info("[%d] Num of record info path: %d", idx, len(record_paths))

    cur_record_info = {"num_batch": 0, "filenames": []}

    for record_info_path in record_paths:
      if num_passes is not None:
        record_info_name = os.path.basename(record_info_path)
        fields = record_info_name.split(".")[0].split("-")
        pass_id = int(fields[-1])
        if len(fields) == 5 and pass_id >= num_passes:
          logging.info("Skip pass %d: %s", pass_id, record_info_name)
          continue

      with tf.gfile.Open(record_info_path, "r") as fp:
        info = json.load(fp)
        if num_passes is not None:
          eff_num_passes = min(num_passes, len(info["filenames"]))
          ratio = eff_num_passes / len(info["filenames"])
          cur_record_info["num_batch"] += int(info["num_batch"] * ratio)
          cur_record_info["filenames"] += info["filenames"][:eff_num_passes]
        else:
          cur_record_info["num_batch"] += info["num_batch"]
          cur_record_info["filenames"] += info["filenames"]

    # overwrite directory for `cur_record_info`
    new_filenames = []
    for filename in cur_record_info["filenames"]:
      basename = os.path.basename(filename)
      new_filename = os.path.join(record_dir, basename)
      new_filenames.append(new_filename)
    cur_record_info["filenames"] = new_filenames

    logging.info("[Dir %d] Number of chosen batches: %s",
                 idx, cur_record_info["num_batch"])
    logging.info("[Dir %d] Number of chosen files: %s",
                 idx, len(cur_record_info["filenames"]))
    logging.info(cur_record_info["filenames"])

    # add `cur_record_info` to global `record_info`
    record_info["num_batch"] += cur_record_info["num_batch"]
    record_info["filenames"] += cur_record_info["filenames"]

  logging.info("Total number of batches: %d", record_info["num_batch"])
  logging.info("Total number of files: %d", len(record_info["filenames"]))
  logging.info(record_info["filenames"])

  def input_fn(params):
    """docs."""
    assert params["batch_size"] * num_core_per_host == bsz_per_host

    dataset = get_dataset(
        params=params,
        num_hosts=num_hosts,
        num_core_per_host=num_core_per_host,
        split=split,
        file_names=record_info["filenames"],
        num_batch=record_info["num_batch"],
        seq_len=seq_len,
        reuse_len=reuse_len,
        perm_size=perm_size,
        mask_alpha=mask_alpha,
        mask_beta=mask_beta,
        use_bfloat16=use_bfloat16,
        num_predict=num_predict)

    return dataset

  return input_fn, record_info


def define_flags():
  """Defines relevant flags."""
  flags.DEFINE_bool("use_tpu", True, help="whether to use TPUs")
  flags.DEFINE_integer("bsz_per_host", 32, help="batch size per host.")
  flags.DEFINE_integer("num_core_per_host", 8, help="num TPU cores per host.")

  flags.DEFINE_integer("seq_len", 512,
                       help="Sequence length.")
  flags.DEFINE_integer("reuse_len", 256,
                       help="Number of token that can be reused as memory. "
                       "Could be half of `seq_len`.")
  flags.DEFINE_bool("uncased", False, help="Use uncased inputs or not.")
  flags.DEFINE_bool("bi_data", True,
                    help="whether to create bidirectional data")
  flags.DEFINE_integer("mask_alpha", default=6,
                       help="How many tokens to form a group.")
  flags.DEFINE_integer("mask_beta", default=1,
                       help="How many tokens to mask within each group.")
  flags.DEFINE_bool("use_eod", True,
                    help="whether to append EOD at the end of a doc.")
  flags.DEFINE_bool("from_raw_text", True,
                    help="Whether the input is raw text or encoded ids.")
  flags.DEFINE_integer("num_predict", default=85,
                       help="Num of tokens to predict.")

  flags.DEFINE_string("input_glob", "data/example/*.txt",
                      help="Input file glob.")
  flags.DEFINE_string("sp_path", "", help="Path to the sentence piece model.")
  flags.DEFINE_string("save_dir", "proc_data/example",
                      help="Directory for saving the processed data.")
  flags.DEFINE_enum("split", "train", ["train", "dev", "test"],
                    help="Save the data as which split.")

  flags.DEFINE_integer("pass_id", 0, help="ID of the current pass."
                       "Different passes sample different negative segment.")
  flags.DEFINE_integer("num_task", 1, help="Number of total tasks.")
  flags.DEFINE_integer("task", 0, help="The Task ID. This value is used when "
                       "using multiple workers to identify each worker.")


if __name__ == "__main__":
  define_flags()
  logging.set_verbosity(logging.INFO)
  app.run(create_data)
