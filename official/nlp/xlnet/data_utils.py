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

import json
import os
from absl import logging

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
      options.experimental_distribute.auto_shard = False
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


def _get_input_iterator(input_fn, strategy):
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


def create_pretrain_dataset(file_names,
                            bsz_per_core,
                            seq_len,
                            reuse_len,
                            perm_size,
                            num_predict=None,
                            input_pipeline_context=None):
  """Creates pretrain dataset."""

  def parser(record):
    """Function used to parse tfrecord."""

    record_spec = {
        "input": tf.io.FixedLenFeature([seq_len], tf.int64),
        "target": tf.io.FixedLenFeature([seq_len], tf.int64),
        "seg_id": tf.io.FixedLenFeature([seq_len], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
        "is_masked": tf.io.FixedLenFeature([seq_len], tf.int64),
    }

    # retrieve serialized example
    example = tf.io.parse_single_example(
        serialized=record, features=record_spec)

    inputs = example.pop("input")
    target = example.pop("target")
    is_masked = tf.cast(example.pop("is_masked"), tf.bool)

    non_reuse_len = seq_len - reuse_len
    # perm_size should not be larger than reuse_len or non_reuse_len otherwise
    # there will be data leaks.
    assert perm_size <= reuse_len and perm_size <= non_reuse_len

    # Creates permutation mask and target mask for the first reuse_len tokens.
    # The tokens in this part are reused from the last sequence.
    perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
        inputs[:reuse_len], target[:reuse_len], is_masked[:reuse_len],
        perm_size, reuse_len)

    # Creates permutation mask and target mask for the rest of tokens in
    # current example, which are concatentation of two new segments.
    perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
        inputs[reuse_len:], target[reuse_len:], is_masked[reuse_len:],
        perm_size, non_reuse_len)

    perm_mask_0 = tf.concat(
        [perm_mask_0, tf.ones([reuse_len, non_reuse_len])], axis=1)
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
      target_mask = tf.concat([
          tf.ones([actual_num_predict], dtype=tf.float32),
          tf.zeros([pad_len], dtype=tf.float32)
      ],
                              axis=0)
      example["target_mask"] = tf.reshape(target_mask, [num_predict])
    else:
      example["target"] = tf.reshape(target, [seq_len])
      example["target_mask"] = tf.reshape(target_mask, [seq_len])

    # reshape back to fixed shape
    example["perm_mask"] = tf.reshape(perm_mask, [seq_len, seq_len])
    example["input_k"] = tf.reshape(input_k, [seq_len])
    example["input_q"] = tf.reshape(input_q, [seq_len])

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
      input_pipeline_context=input_pipeline_context)

  return dataset


def format_filename(prefix,
                    bsz_per_host,
                    seq_len,
                    bi_data,
                    suffix,
                    mask_alpha=5,
                    mask_beta=1,
                    reuse_len=None,
                    uncased=False,
                    fixed_num_predict=None):
  """Generates input file name pattern."""
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


def get_pretrain_input_data(batch_size,
                            seq_len,
                            strategy,
                            file_path,
                            reuse_len,
                            perm_size,
                            mask_alpha,
                            mask_beta,
                            num_predict,
                            bi_data,
                            uncased,
                            num_hosts=1):
  """Returns input dataset from input file string."""

  # When using TPU pods, we need to clone dataset across
  # workers and need to pass in function that returns the dataset rather
  # than passing dataset instance itself.
  use_dataset_fn = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
  split = "train"
  record_glob_base = format_filename(
      prefix="record_info-{}-*".format(split),
      bsz_per_host=int(batch_size / num_hosts),
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="json",
      mask_alpha=mask_alpha,
      mask_beta=mask_beta,
      reuse_len=reuse_len,
      uncased=uncased,
      fixed_num_predict=num_predict)

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
        cur_record_info["num_batch"] += info["num_batch"]
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
        num_predict=num_predict,
        input_pipeline_context=ctx)
    return train_dataset

  return _dataset_fn if use_dataset_fn else _dataset_fn()


def parse_files_to_dataset(parser,
                           file_paths,
                           bsz_per_core,
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

  dataset = tf.data.TFRecordDataset(dataset)
  # (zihang): since we are doing online preprocessing, the parsed result of
  # the same input at each time will be different. Thus, cache processed data
  # is not helpful. It will use a lot of memory and lead to contrainer OOM.
  # So, change to cache non-parsed raw data instead.
  dataset = dataset.cache().map(parser).repeat()
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
  """Samples a permutation of the factorization order.

     Creates perm_mask and target_mask accordingly.

  Args:
    inputs: int64 Tensor in shape [seq_len], input ids.
    targets: int64 Tensor in shape [seq_len], target ids.
    is_masked: bool Tensor in shape [seq_len]. True means being selected for
      partial prediction.
    perm_size: the length of longest permutation. Could be set to be reuse_len.
      Should not be larger than reuse_len or there will be data leaks.
    seq_len: int, sequence length.

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
    new_targets: int64 Tensor in shape [seq_len], target token ids to be
    predicted in XLNet.
    In XLNet, target doesn't need to be shifted one position.
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

  # `perm_mask` and `target_mask`
  # non-functional tokens
  non_func_tokens = tf.logical_not(
      tf.logical_or(tf.equal(inputs, SEP_ID), tf.equal(inputs, CLS_ID)))

  non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
  masked_or_func_tokens = tf.logical_not(non_mask_tokens)

  # Set the permutation indices of non-masked (& non-funcional) tokens to the
  # smallest index (-1):
  # (1) they can be seen by all other positions
  # (2) they cannot see masked positions, so there won"t be information leak
  smallest_index = -tf.ones([seq_len], dtype=tf.int64)
  rev_index = tf.where(non_mask_tokens, smallest_index, index)

  # Create `target_mask`: non-funcional and masked tokens
  # 1: use mask as input and have loss
  # 0: use token (or [SEP], [CLS]) as input and do not have loss
  target_tokens = tf.logical_and(masked_or_func_tokens, non_func_tokens)
  target_mask = tf.cast(target_tokens, tf.float32)

  # Create `perm_mask`
  # `target_tokens` cannot see themselves
  self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)

  # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
  # 0: can attend if i > j or j is non-masked
  perm_mask = tf.logical_and(self_rev_index[:, None] <= rev_index[None, :],
                             masked_or_func_tokens)
  perm_mask = tf.cast(perm_mask, tf.float32)

  # new target: [next token] for LM and [curr token] (self) for PLM
  new_targets = tf.concat([inputs[0:1], targets[:-1]], axis=0)

  # construct inputs_k
  inputs_k = inputs

  # construct inputs_q
  inputs_q = target_mask

  return perm_mask, new_targets, target_mask, inputs_k, inputs_q
