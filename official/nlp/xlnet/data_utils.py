# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities used for data preparation."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import collections
import json
import os
from absl import logging

import numpy as np
import tensorflow as tf


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
SEG_ID_P = 0
SEG_ID_Q = 1
SEG_ID_CLS = 2
SEG_ID_PAD = 3


OnlineMaskingConfig = collections.namedtuple("OnlineMaskingConfig", [
    "sample_strategy", "max_num_tokens", "min_num_tokens", "max_num_words",
    "min_num_words"])


def file_based_input_fn_builder(input_file, name_to_features, batch_size,
                                is_training):
  """Creates an `input_fn` closure."""

  logging.info("Input tfrecord file %s", input_file)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def input_fn():
    """Returns dataset for training/evaluation."""
    num_threads = 8
    if isinstance(input_file, str):
      d = tf.data.TFRecordDataset(input_file)
      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        d = d.shuffle(2048)
        d = d.repeat()
    else:
      cycle_length = min(num_threads, len(input_file))
      d = tf.data.Dataset.from_tensor_slices(input_file)
      # file level shuffle
      d = d.shuffle(len(input_file)).repeat()

      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))

      if is_training:
        # sample level shuffle
        d = d.shuffle(buffer_size=2048)

    # TODO(b/138223458): Hard-code drop_remainder=True to get around the bug
    # that under TPU strategy, setting drop_remainder=False in
    # tf.data.Dataset.batch() while data_size can be divided by global
    # batch_size will trigger dynamic_dimension related TPU compilation error.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_threads,
            drop_remainder=True))

    # When `input_file` is a path to a single file or a list
    # containing a single path, disable auto sharding so that
    # same input file is sent to all workers.
    if isinstance(input_file, str) or len(input_file) == 1:
      options = tf.data.Options()
      options.experimental_distribute.auto_shard_policy = (
          tf.data.experimental.AutoShardPolicy.OFF)
      d = d.with_options(options)

    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    return d

  return input_fn


def create_classification_dataset(file_path, seq_length, batch_size,
                                  is_training):
  """Creates input dataset from (tf)records files for pretraining."""
  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.float32),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }

  input_fn = file_based_input_fn_builder(file_path, name_to_features,
                                         batch_size, is_training)
  dataset = input_fn()
  return dataset


def create_squad_dataset(file_path, seq_length, batch_size, is_training):
  """Creates input dataset from (tf)records files for pretraining."""
  name_to_features = {
      "unique_ids": tf.io.FixedLenFeature([], tf.int64),
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.float32),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "cls_index": tf.io.FixedLenFeature([], tf.int64),
      "p_mask": tf.io.FixedLenFeature([seq_length], tf.float32)
  }

  if is_training:
    name_to_features["start_positions"] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features["is_impossible"] = tf.io.FixedLenFeature([], tf.float32)

  input_fn = file_based_input_fn_builder(file_path, name_to_features,
                                         batch_size, is_training)
  dataset = input_fn()
  return dataset


def get_input_iterator(input_fn, strategy):
  """Returns distributed dataset iterator."""

  # When training with TPU pods, datasets needs to be cloned across
  # workers. Since Dataset instance cannot be cloned in eager mode, we instead
  # pass callable that returns a dataset.
  input_data = input_fn()
  if callable(input_data):
    iterator = iter(
        strategy.experimental_distribute_datasets_from_function(input_data))
  else:
    iterator = iter(strategy.experimental_distribute_dataset(input_data))
  return iterator


def get_classification_input_data(batch_size, seq_len, strategy, is_training,
                                  file_path):
  """Returns input dataset from input file string."""

  # When using TPU pods, we need to clone dataset across
  # workers and need to pass in function that returns the dataset rather
  # than passing dataset instance itself.
  use_dataset_fn = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
  if use_dataset_fn:
    if batch_size % strategy.num_replicas_in_sync != 0:
      raise ValueError(
          "Batch size must be divisible by number of replicas : {}".format(
              strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    batch_size = int(batch_size / strategy.num_replicas_in_sync)

  def _dataset_fn(ctx=None):
    del ctx

    train_dataset = create_classification_dataset(
        file_path=file_path,
        seq_length=seq_len,
        batch_size=batch_size,
        is_training=is_training)
    return train_dataset

  return _dataset_fn if use_dataset_fn else _dataset_fn()


def get_squad_input_data(batch_size, seq_len, q_len, strategy, is_training,
                         file_path):
  """Returns input dataset from input file string."""

  # When using TPU pods, we need to clone dataset across
  # workers and need to pass in function that returns the dataset rather
  # than passing dataset instance itself.
  use_dataset_fn = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
  if use_dataset_fn:
    if batch_size % strategy.num_replicas_in_sync != 0:
      raise ValueError(
          "Batch size must be divisible by number of replicas : {}".format(
              strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    batch_size = int(batch_size / strategy.num_replicas_in_sync)

  if is_training:
    input_glob = os.path.join(
        file_path,
        "spiece.model.*.slen-{}.qlen-{}.train.tf_record".format(seq_len, q_len))

    global_input_paths = tf.io.gfile.glob(input_glob)
  else:
    global_input_paths = file_path

  def _dataset_fn(ctx=None):
    del ctx

    train_dataset = create_squad_dataset(
        file_path=global_input_paths,
        seq_length=seq_len,
        batch_size=batch_size,
        is_training=is_training)
    return train_dataset

  return _dataset_fn if use_dataset_fn else _dataset_fn()


def _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len, num_predict):
  """Turn beg and end indices into actual mask."""
  non_func_mask = tf.logical_and(
      tf.not_equal(inputs, SEP_ID),
      tf.not_equal(inputs, CLS_ID))
  all_indices = tf.where(
      non_func_mask,
      tf.range(tgt_len, dtype=tf.int64),
      tf.constant(-1, shape=[tgt_len], dtype=tf.int64))
  candidate_matrix = tf.cast(
      tf.logical_and(
          all_indices[None, :] >= beg_indices[:, None],
          all_indices[None, :] < end_indices[:, None]),
      tf.float32)
  cumsum_matrix = tf.reshape(
      tf.cumsum(tf.reshape(candidate_matrix, [-1])),
      [-1, tgt_len])
  masked_matrix = tf.cast(cumsum_matrix <= num_predict, tf.float32)
  target_mask = tf.reduce_sum(candidate_matrix * masked_matrix, axis=0)
  is_masked = tf.cast(target_mask, tf.bool)

  return is_masked, target_mask


def _word_span_mask(inputs, tgt_len, num_predict, min_num_words,
                    max_num_words, boundary):
  """Sample whole word spans as prediction targets."""
  # Note: 1.2 is the token-to-word ratio
  mask_alpha = tgt_len / num_predict / 1.2
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(min_num_words, max_num_words + 1)
  probs = np.array([1.0 /  (i + 1) for i in span_len_seq])
  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)

  # Sample `num_predict` words here: note that this is over sampling
  span_lens = tf.random.categorical(
      logits=logits[None],
      num_samples=num_predict,
      dtype=tf.int64,
  )[0] + min_num_words

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_float = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_float * (mask_alpha - 1)

  left_ctx_len = round_to_int(left_ctx_len)
  right_offset = round_to_int(span_lens_float * mask_alpha) - left_ctx_len

  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + span_lens

  # Remove out of range indices
  max_boundary_index = tf.cast(tf.shape(boundary)[0] - 1, tf.int64)
  valid_idx_mask = end_indices < max_boundary_index
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  beg_indices = tf.gather(boundary, beg_indices)
  end_indices = tf.gather(boundary, end_indices)

  # Shuffle valid indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _token_span_mask(inputs, tgt_len, num_predict, min_num_tokens,
                     max_num_tokens):
  """Sample token spans as prediction targets."""
  mask_alpha = tgt_len / num_predict
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(min_num_tokens, max_num_tokens + 1)
  probs = np.array([1.0 /  (i + 1) for i in span_len_seq])

  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)
  span_lens = tf.random.categorical(
      logits=logits[None],
      num_samples=num_predict,
      dtype=tf.int64,
  )[0] + min_num_tokens

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_float = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_float * (mask_alpha - 1)
  left_ctx_len = round_to_int(left_ctx_len)

  # Compute the offset from left start to the right end
  right_offset = round_to_int(span_lens_float * mask_alpha) - left_ctx_len

  # Get the actual begin and end indices
  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + span_lens

  # Remove out of range indices
  valid_idx_mask = end_indices < tgt_len
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  # Shuffle valid indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _whole_word_mask(inputs, tgt_len, num_predict, boundary):
  """Sample whole words as prediction targets."""
  pair_indices = tf.concat([boundary[:-1, None], boundary[1:, None]], axis=1)
  cand_pair_indices = tf.random.shuffle(pair_indices)[:num_predict]
  beg_indices = cand_pair_indices[:, 0]
  end_indices = cand_pair_indices[:, 1]

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _single_token_mask(inputs, tgt_len, num_predict):
  """Sample individual tokens as prediction targets."""
  all_indices = tf.range(tgt_len, dtype=tf.int64)
  non_func_mask = tf.logical_and(
      tf.not_equal(inputs, SEP_ID),
      tf.not_equal(inputs, CLS_ID))
  non_func_indices = tf.boolean_mask(all_indices, non_func_mask)

  masked_pos = tf.random.shuffle(non_func_indices)
  masked_pos = tf.contrib.framework.sort(masked_pos[:num_predict])
  target_mask = tf.sparse_to_dense(
      sparse_indices=masked_pos,
      output_shape=[tgt_len],
      sparse_values=1.0,
      default_value=0.0)

  is_masked = tf.cast(target_mask, tf.bool)

  return is_masked, target_mask


def _online_sample_masks(inputs, tgt_len, num_predict, online_masking_config,
                         boundary=None):
  """Sample target positions to predict."""
  logging.info("Online sample with strategy: `%s`.",
               online_masking_config.sample_strategy)
  if online_masking_config.sample_strategy == "single_token":
    return _single_token_mask(inputs, tgt_len, num_predict)
  elif online_masking_config.sample_strategy == "whole_word":
    assert boundary is not None, "whole word sampling requires `boundary`"
    return _whole_word_mask(inputs, tgt_len, num_predict, boundary)
  elif online_masking_config.sample_strategy == "token_span":
    return _token_span_mask(inputs, tgt_len, num_predict,
                            online_masking_config.min_num_tokens,
                            online_masking_config.max_num_tokens)
  elif online_masking_config.sample_strategy == "word_span":
    assert boundary is not None, "word span sampling requires `boundary`"
    return _word_span_mask(inputs, tgt_len, num_predict,
                           online_masking_config.min_num_words,
                           online_masking_config.max_num_words,
                           boundary)
  else:
    raise NotImplementedError


def create_pretrain_dataset(file_names,
                            bsz_per_core,
                            seq_len,
                            reuse_len,
                            perm_size,
                            leak_ratio,
                            online_masking_config,
                            num_predict=None,
                            input_pipeline_context=None):
  """Creates pretrain dataset."""

  def parser(record):
    """Function used to parse tfrecord."""

    record_spec = {
        "input": tf.io.FixedLenFeature([seq_len], tf.int64),
        "seg_id": tf.io.FixedLenFeature([seq_len], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }

    if online_masking_config.sample_strategy in ["whole_word", "word_span"]:
      logging.info("Add `boundary` spec for %s",
                   online_masking_config.sample_strategy)
      record_spec["boundary"] = tf.io.VarLenFeature(tf.int64)

    # retrieve serialized example
    example = tf.io.parse_single_example(
        serialized=record, features=record_spec)

    inputs = example.pop("input")
    if online_masking_config.sample_strategy in ["whole_word", "word_span"]:
      boundary = tf.sparse.to_dense(example.pop("boundary"))
    else:
      boundary = None
    is_masked, _ = _online_sample_masks(
        inputs, seq_len, num_predict, online_masking_config, boundary=boundary)

    if reuse_len > 0:
      ##### Use memory
      # permutate the reuse and non-reuse parts separately
      non_reuse_len = seq_len - reuse_len
      assert reuse_len % perm_size == 0 and non_reuse_len % perm_size == 0

      # Creates permutation mask and target mask for the first reuse_len tokens.
      # The tokens in this part are reused from the last sequence.
      perm_mask_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
          inputs[:reuse_len], is_masked[:reuse_len], perm_size, reuse_len,
          leak_ratio)

      # Creates permutation mask and target mask for the rest of tokens in
      # current example, which are concatentation of two new segments.
      perm_mask_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
          inputs[reuse_len:], is_masked[reuse_len:], perm_size, non_reuse_len,
          leak_ratio)

      perm_mask_0 = tf.concat(
          [perm_mask_0, tf.ones([reuse_len, non_reuse_len])], axis=1)
      perm_mask_1 = tf.concat(
          [tf.zeros([non_reuse_len, reuse_len]), perm_mask_1], axis=1)
      perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=0)
      target_mask = tf.concat([target_mask_0, target_mask_1], axis=0)
      input_k = tf.concat([input_k_0, input_k_1], axis=0)
      input_q = tf.concat([input_q_0, input_q_1], axis=0)
    else:
      ##### Do not use memory
      assert seq_len % perm_size == 0
      # permutate the entire sequence together
      perm_mask, target_mask, input_k, input_q = _local_perm(
          inputs, is_masked, perm_size, seq_len, leak_ratio)

    # reshape back to fixed shape
    example["perm_mask"] = tf.reshape(perm_mask, [seq_len, seq_len])
    example["input_k"] = tf.reshape(input_k, [seq_len])
    example["input_q"] = tf.reshape(input_q, [seq_len])

    # Directly use raw inputs as the target
    target = inputs

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

    for key in list(example.keys()):
      val = example[key]
      if tf.keras.backend.is_sparse(val):
        val = tf.sparse.to_dense(val)
      if val.dtype == tf.int64:
        val = tf.cast(val, tf.int32)

      example[key] = val

    for k, v in example.items():
      logging.info("%s: %s", k, v)

    return example

  dataset = parse_files_to_dataset(
      parser=parser,
      file_paths=file_names,
      bsz_per_core=bsz_per_core,
      sequential=reuse_len > 0,
      input_pipeline_context=input_pipeline_context)

  return dataset


def format_filename(prefix, suffix, bsz_per_host, seq_len, reuse_len=None,
                    uncased=False):
  """Generates input file name pattern."""
  if reuse_len is not None and reuse_len > 0:
    reuse_str = "reuse-{}.".format(reuse_len)
    bsz_str = "hostbsz-{}.".format(bsz_per_host)
  else:
    reuse_str = ""
    bsz_str = ""

  if not uncased:
    case_str = ""
  else:
    case_str = "uncased."

  file_name = "{}.seq-{}.{}{}{}{}".format(
      prefix, seq_len, reuse_str, bsz_str, case_str, suffix)

  return file_name


def get_pretrain_input_data(batch_size,
                            seq_len,
                            strategy,
                            file_path,
                            reuse_len,
                            perm_size,
                            leak_ratio,
                            num_predict,
                            uncased,
                            online_masking_config,
                            num_hosts=1):
  """Returns input dataset from input file string."""

  # When using TPU pods, we need to clone dataset across
  # workers and need to pass in function that returns the dataset rather
  # than passing dataset instance itself.
  use_dataset_fn = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
  split = "train"
  bsz_per_host = int(batch_size / num_hosts)
  record_glob_base = format_filename(
      prefix="meta.{}.pass-*".format(split),
      suffix="json*",
      bsz_per_host=bsz_per_host,
      seq_len=seq_len,
      reuse_len=reuse_len,
      uncased=uncased)

  def _get_num_batch(info):
    if "num_batch" in info:
      return info["num_batch"]
    elif "num_example" in info:
      return info["num_example"] / bsz_per_host
    else:
      raise ValueError("Do not have sample info.")

  if use_dataset_fn:
    if batch_size % strategy.num_replicas_in_sync != 0:
      raise ValueError(
          "Batch size must be divisible by number of replicas : {}".format(
              strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    batch_size = int(batch_size / strategy.num_replicas_in_sync)

  record_info = {"num_batch": 0, "filenames": []}

  tfrecord_dirs = file_path.split(",")
  logging.info("Use the following tfrecord dirs: %s", tfrecord_dirs)

  for idx, record_dir in enumerate(tfrecord_dirs):
    record_glob = os.path.join(record_dir, record_glob_base)
    logging.info("[%d] Record glob: %s", idx, record_glob)

    record_paths = sorted(tf.io.gfile.glob(record_glob))
    logging.info("[%d] Num of record info path: %d", idx, len(record_paths))

    cur_record_info = {"num_batch": 0, "filenames": []}

    for record_info_path in record_paths:
      with tf.io.gfile.GFile(record_info_path, "r") as fp:
        info = json.load(fp)
        cur_record_info["num_batch"] += int(_get_num_batch(info))
        cur_record_info["filenames"] += info["filenames"]

    # overwrite directory for `cur_record_info`
    new_filenames = []
    for filename in cur_record_info["filenames"]:
      basename = os.path.basename(filename)
      new_filename = os.path.join(record_dir, basename)
      new_filenames.append(new_filename)
    cur_record_info["filenames"] = new_filenames

    logging.info("[Dir %d] Number of chosen batches: %s", idx,
                 cur_record_info["num_batch"])
    logging.info("[Dir %d] Number of chosen files: %s", idx,
                 len(cur_record_info["filenames"]))
    logging.info(cur_record_info["filenames"])

    # add `cur_record_info` to global `record_info`
    record_info["num_batch"] += cur_record_info["num_batch"]
    record_info["filenames"] += cur_record_info["filenames"]

  logging.info("Total number of batches: %d", record_info["num_batch"])
  logging.info("Total number of files: %d", len(record_info["filenames"]))
  logging.info(record_info["filenames"])

  def _dataset_fn(ctx=None):
    """Function that can create a pretrain dataset."""

    train_dataset = create_pretrain_dataset(
        file_names=record_info["filenames"],
        bsz_per_core=batch_size,
        seq_len=seq_len,
        reuse_len=reuse_len,
        perm_size=perm_size,
        leak_ratio=leak_ratio,
        online_masking_config=online_masking_config,
        num_predict=num_predict,
        input_pipeline_context=ctx)
    return train_dataset

  return _dataset_fn if use_dataset_fn else _dataset_fn()


def parse_files_to_dataset(parser,
                           file_paths,
                           bsz_per_core,
                           sequential,
                           input_pipeline_context=None):
  """Creates the dataset given file paths."""

  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # Note: we cannot perform sample-level shuffle here because this will violate
  # the consecutive requirement of data stream.

  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)
  # file-level shuffle
  if len(file_paths) > 1:
    dataset = dataset.shuffle(len(file_paths))

  if sequential:
    # Note: cannot perform sample-level shuffle here because this will violate
    # the consecutive requirement of data stream.
    dataset = tf.data.TFRecordDataset(dataset)
  else:
    # `cycle_length` is the number of parallel files that get read.
    cycle_length = min(8, len(file_paths))
    logging.info("Interleave %d files", cycle_length)

    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=True,
            cycle_length=cycle_length))
    buffer_size = 2048
    logging.info("Perform sample-level shuffle with size %d", buffer_size)
    dataset = dataset.shuffle(buffer_size=buffer_size)

  # (zihang): since we are doing online preprocessing, the parsed result of
  # the same input at each time will be different. Thus, cache processed data
  # is not helpful. It will use a lot of memory and lead to contrainer OOM.
  # So, change to cache non-parsed raw data instead.
  dataset = dataset.cache().map(parser).repeat()
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


def _local_perm(inputs, is_masked, perm_size, seq_len, leak_ratio):
  """Samples a permutation of the factorization order.

     Creates perm_mask and target_mask accordingly.

  Args:
    inputs: int64 Tensor in shape [seq_len], input ids.
    is_masked: bool Tensor in shape [seq_len]. True means being selected for
      partial prediction.
    perm_size: the length of longest permutation. Could be set to be reuse_len.
      Should not be larger than reuse_len or there will be data leaks.
    seq_len: int, sequence length.
    leak_ratio: float, percent of masked tokens that are leaked.

  Returns:
    perm_mask: float32 Tensor in shape [seq_len, seq_len] consisted of 0 and 1.
    If perm_mask[i][j] == 1, it means the ith token (in original order) cannot
    attend to the jth token
    (in original order). This case will happen only when the ith token's
    permutated position <= the jth token's permutated position,
    and the jth token is masked or is func token. If perm_mask[i][j] == 0, it
    means the ith token (in original order) can attend to the jth token
    (in original order). Note that non-masked tokens can be attended by all
    other tokens, which is different from the description in original paper.
    target_mask: float32 Tensor in shape [seq_len] consisted of 0 and 1. If
    target_mask[i] == 1,
    the ith token needs to be predicted and mask will be used as input. This
    token will count for loss.
    If target_mask[i] == 0, token (or [SEP], [CLS]) will be used as input. This
    token will not count for loss.
    inputs_k: int64 Tensor in shape [seq_len], input ids.
    inputs_q: float32 Tensor in shape [seq_len], the same as target_mask.

  """

  # Generate permutation indices
  index = tf.range(seq_len, dtype=tf.int64)
  index = tf.transpose(tf.reshape(index, [-1, perm_size]))
  index = tf.random.shuffle(index)
  index = tf.reshape(tf.transpose(index), [-1])

  # non-functional tokens
  non_func_tokens = tf.logical_not(tf.logical_or(
      tf.equal(inputs, SEP_ID),
      tf.equal(inputs, CLS_ID)))
  masked_tokens = tf.logical_and(is_masked, non_func_tokens)
  non_masked_or_func_tokens = tf.logical_not(masked_tokens)

  smallest_index = -2 * tf.ones([seq_len], dtype=tf.int64)

  # Similar to BERT, randomly leak some masked tokens
  if leak_ratio > 0:
    leak_tokens = tf.logical_and(
        masked_tokens,
        tf.random.uniform([seq_len], maxval=1.0) < leak_ratio)
    can_attend_self = tf.logical_or(non_masked_or_func_tokens, leak_tokens)
  else:
    can_attend_self = non_masked_or_func_tokens
  to_index = tf.where(can_attend_self, smallest_index, index)
  from_index = tf.where(can_attend_self, to_index + 1, to_index)

  # For masked tokens, can attend if i > j
  # For context tokens, always can attend each other
  can_attend = from_index[:, None] > to_index[None, :]

  # In modeling, 1 indicates cannot attend. Hence, reverse the value here.
  perm_mask = 1.0 - tf.cast(can_attend, tf.float32)

  # Only masked tokens are included in the loss
  target_mask = tf.cast(masked_tokens, tf.float32)

  # construct inputs_k
  inputs_k = inputs

  # construct inputs_q
  inputs_q = masked_tokens

  return perm_mask, target_mask, inputs_k, inputs_q
