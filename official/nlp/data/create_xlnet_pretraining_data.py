# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Create LM TF examples for XLNet."""

import dataclasses
import json
import math
import os

import random
from typing import Iterable, Mapping, List, Optional, Tuple
import unicodedata

# Import libraries

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.tools import tokenization

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

FLAGS = flags.FLAGS

flags.DEFINE_integer("seq_length", 512,
                     help="Sequence length.")
flags.DEFINE_integer("reuse_length", 256,
                     help="Number of token that can be reused as memory. "
                     "Could be half of `seq_len`.")
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")
flags.DEFINE_string(
    "save_dir", None,
    "Directory for saving processed data.")
flags.DEFINE_string("sp_model_file", "",
                    "The path to the model used by sentence piece tokenizer.")
flags.DEFINE_bool("use_eod_token", True,
                  "Whether or not to include EOD tokens.")
flags.DEFINE_bool("bi_data", True, "Whether or not to use bi-directional data.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer("per_host_batch_size", 32, "Batch size per host.")
flags.DEFINE_integer("num_cores_per_host", 16,
                     "The number of (TPU) cores per host.")
flags.DEFINE_string("prefix", "", "Filename prefix.")
flags.DEFINE_string("suffix", "", "Filename suffix.")

flags.DEFINE_integer("task_id", None,
                     "The id of the current task.")
flags.DEFINE_integer("num_tasks", None,
                     "The total number of tasks.")
flags.DEFINE_integer("num_passes", 1, "The number of times to run the script.")


@dataclasses.dataclass
class TrainingInstance:
  """Representation of a single XLNet Pretraining instance."""
  data: Iterable[int]
  segment_ids: Iterable[int]
  boundary_indices: Iterable[int]
  label: int

  def to_feature(self) -> Mapping[str, tf.train.Feature]:
    feat = lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x))
    return dict(
        input_word_ids=feat(self.data),
        input_type_ids=feat(self.segment_ids),
        boundary_indices=feat(self.boundary_indices),
        label=feat([self.label]))

  def to_example(self) -> tf.train.Example:
    return tf.train.Example(
        features=tf.train.Features(feature=self.to_feature()))

  def __str__(self):
    def seq_to_str(seq):
      return " ".join([str(x) for x in seq])

    s = ""
    s += "tokens: %s\n" % seq_to_str(self.data)
    s += "segment_ids: %s\n" % seq_to_str(self.segment_ids)
    s += "boundary_indices: %s\n" % seq_to_str(self.boundary_indices)
    s += "label: %s\n" % self.label
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def _preprocess_line(line: str, do_lower_case: bool = False) -> str:
  """Preprocesses an individual raw text line.

  This function will:
    - Remove extraneous spaces.
    - Replace `` with ", and '' with ".
    - Replaces accents.
    - Applies lower casing.

  Args:
    line: The input line to preprocess.
    do_lower_case: Whether or not to lower case the text.

  Returns:
    The preprocessed line.

  """
  line = " ".join(line.split())
  line = line.replace("``", "\"").replace("''", "\"")

  # Replace accents.
  line = unicodedata.normalize("NFKD", line)
  line = "".join([c for c in line if not unicodedata.combining(c)])

  if do_lower_case:
    line = line.lower()
  return line


def preprocess_and_tokenize_input_files(
    input_files: Iterable[str],
    tokenizer: tokenization.FullSentencePieceTokenizer,
    use_eod: bool = True,
    do_lower_case: bool = False,
    log_example_freq: int = 100000) -> List[Tuple[np.array, np.array]]:
  """Preprocesses and encodes raw text from input files.

  This function preprocesses raw text and encodes them into tokens using a
  `SentencePieceModel` tokenization method. This also provides the sentence
  indicator for each token.

  Args:
    input_files: The list of input file names.
    tokenizer: The SentencePiece tokenizer that has the attribute `sp_model`.
    use_eod: Whether or not to use an EOD indicator. If `False`, then EOD is
      not included.
    do_lower_case: Whether or not to apply lower casing during raw text
      preprocessing.
    log_example_freq: The optional field for how many lines to process before
      emitting an info log.

  Returns:
    The preprocessed list. Each entry in the list is a tuple consisting of
    the token IDs and the sentence IDs.

  """
  all_data = []
  eod_symbol = special_symbols["<eod>"]

  total_number_of_lines = 0

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    line_count = 0
    logging.info("Preprocessing %s", input_file)

    all_tokens = []
    all_sentence_ids = []

    sentence_id = True

    with tf.io.gfile.GFile(input_file, "rb") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break

        line_count += 1
        if line_count % log_example_freq == 0:
          logging.info("Loading line %d", line_count)

        line = line.strip()

        if not line:
          if use_eod:
            token_ids = [eod_symbol]
            sentence_id = not sentence_id
          else:
            continue
        else:
          preprocessed_line = _preprocess_line(
              line=line, do_lower_case=do_lower_case)
          token_ids = tokenization.encode_ids(
              sp_model=tokenizer.sp_model, text=preprocessed_line)

        all_tokens.extend(token_ids)
        all_sentence_ids.extend([sentence_id] * len(token_ids))
        sentence_id = not sentence_id
      logging.info("Finished processing %s. Number of lines: %d",
                   input_file, line_count)
      if line_count == 0:
        continue
      total_number_of_lines += line_count
      all_tokens = np.array(all_tokens, dtype=np.int64)
      all_sentence_ids = np.array(all_sentence_ids, dtype=bool)
      all_data.append((all_tokens, all_sentence_ids))

  logging.info("Completed text preprocessing. Total number of lines: %d",
               total_number_of_lines)
  return all_data


def _reshape_to_batch_dimensions(
    tokens: np.array,
    sentence_ids: np.array,
    per_host_batch_size: int) -> Tuple[np.array, np.array]:
  """Truncates and reshapes input data with a batch major dimension.

  Args:
    tokens: The input token ids. This should have the same shape as
      `sentence_ids`.
    sentence_ids: The input sentence ids. This should have the same shape as
      `token_ids`.
    per_host_batch_size: The target per-host batch size.

  Returns:
    The tuple of reshaped tokens and sentence_ids.
  """
  num_steps = len(tokens) // per_host_batch_size
  truncated_data_length = num_steps * per_host_batch_size

  logging.info("per_host_batch_size: %d", per_host_batch_size)
  logging.info("num_steps: %d", num_steps)
  def truncate_and_reshape(a):
    return a[:truncated_data_length].reshape((per_host_batch_size, num_steps))

  return (truncate_and_reshape(tokens), truncate_and_reshape(sentence_ids))


def _create_a_and_b_segments(
    tokens: np.array,
    sentence_ids: np.array,
    begin_index: int,
    total_length: int,
    no_cut_probability: float = 0.5):
  """Splits segments A and B from a single instance of tokens and sentence ids.

  Args:
    tokens: The 1D input token ids. This represents an individual entry within a
      batch.
    sentence_ids: The 1D input sentence ids. This represents an individual entry
      within a batch. This should be the same length as `tokens`.
    begin_index: The reference beginning index to split data.
    total_length: The target combined length of segments A and B.
    no_cut_probability: The probability of not cutting a segment despite
      a cut possibly existing.

  Returns:
    A tuple consisting of A data, B data, and label.

  """
  data_length = tokens.shape[0]
  if begin_index + total_length >= data_length:
    logging.info("[_create_segments]: begin_index %d + total_length %d >= "
                 "data_length %d", begin_index, total_length, data_length)
    return None

  end_index = begin_index + 1
  cut_indices = []

  # Identify all indices where sentence IDs change from one to the next.
  while end_index < data_length:
    if sentence_ids[end_index] != sentence_ids[end_index - 1]:
      if end_index - begin_index >= total_length:
        break
      cut_indices.append(end_index)
    end_index += 1

  a_begin = begin_index

  if not cut_indices or random.random() < no_cut_probability:
    # Segments A and B are contained within the same sentence.
    label = 0
    if not cut_indices:
      a_end = end_index
    else:
      a_end = random.choice(cut_indices)
    b_length = max(1, total_length - (a_end - a_begin))
    b_begin = random.randint(0, data_length - 1 - b_length)
    b_end = b_begin + b_length

    while b_begin > 0 and sentence_ids[b_begin - 1] == sentence_ids[b_begin]:
      b_begin -= 1
    while (b_end < data_length - 1 and
           sentence_ids[b_end - 1] == sentence_ids[b_end]):
      b_end += 1
  else:
    # Segments A and B are different sentences.
    label = 1
    a_end = random.choice(cut_indices)
    b_begin = a_end
    b_end = end_index

  while a_end - a_begin + b_end - b_begin > total_length:
    if a_end - a_begin > b_end - b_begin:
      # Delete only the right side for the LM objective.
      a_end -= 1
    else:
      b_end -= 1
  if a_end >= data_length or b_end >= data_length:
    logging.info("[_create_segments]: a_end %d or b_end %d >= data_length %d",
                 a_end, b_end, data_length)
    return None

  a_data = tokens[a_begin: a_end]
  b_data = tokens[b_begin: b_end]
  return a_data, b_data, label


def _is_functional_piece(piece: str) -> bool:
  return piece != "<unk>" and piece.startswith("<") and piece.endswith(">")


def _is_start_piece(piece: str) -> bool:
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  if (piece.startswith("▁") or piece in special_pieces):
    return True
  else:
    return False


def _get_boundary_indices(
    data: np.array,
    tokenizer: tokenization.FullSentencePieceTokenizer) -> np.array:
  """Gets the boundary indices of whole words."""
  seq_length = len(data)
  boundary_indices = []
  for index, piece in enumerate(tokenizer.convert_ids_to_tokens(data.tolist())):
    if _is_start_piece(piece) and not _is_functional_piece(piece):
      boundary_indices.append(index)
  boundary_indices.append(seq_length)
  return boundary_indices


def _convert_tokens_to_instances(
    tokens: np.array,
    sentence_ids: np.array,
    per_host_batch_size: int,
    seq_length: int,
    reuse_length: int,
    bi_data: bool,
    tokenizer: tokenization.FullSentencePieceTokenizer,
    num_cores_per_host: int = 0,
    logging_frequency: int = 500) -> List[TrainingInstance]:
  """Converts tokens and sentence IDs into individual training instances.

  The format of data in the XLNet pretraining task is very similar to the
  BERT pretraining task. Two segments A and B are randomly sampled, and the
  contatenation of A and B into a single sequence is used to perform
  language modeling.

  To create an XLNet Pretraining instance from a single long sequence, S:
  - Create a segment of length `reuse_length`. This first segment represents
    past tokens. During modeling, this segment is used to cache obtained
    content representations for the segment recurrence mechanism.
  - Similar to BERT, create a segment of length `seq_length` - `reuse_length`
    composed of A and B segments.
    For XLNet, the order is "A", "SEP", "B", "SEP", "CLS".

  Args:
    tokens: All tokens concatenated into a single list.
    sentence_ids: All sentence IDs concatenated into a single list.
    per_host_batch_size: The target batch size per host.
    seq_length: The max sequence length.
    reuse_length: The number of tokens to use from the previous segment.
    bi_data: Whether or not to use bidirectional data.
    tokenizer: The SentencePiece tokenizer that has the attribute `sp_model`.
    num_cores_per_host: The number of cores per host. This is required if
      `bi_data` = `True`.
    logging_frequency: The frequency at which to log status updates.

  Returns:
    A list of `TrainingInstance` objects.
  """
  instances = []

  per_core_batch_size = (per_host_batch_size // num_cores_per_host
                         if bi_data else None)

  if bi_data:
    logging.info("Bi-directional data enabled.")
    assert per_host_batch_size % (2 * num_cores_per_host) == 0
    forward_tokens, forward_sentence_ids = _reshape_to_batch_dimensions(
        tokens=tokens,
        sentence_ids=sentence_ids,
        per_host_batch_size=per_host_batch_size // 2)
    forward_data_shape = (num_cores_per_host, 1, per_core_batch_size // 2, -1)

    forward_tokens = forward_tokens.reshape(forward_data_shape)
    forward_sentence_ids = forward_sentence_ids.reshape(forward_data_shape)

    backwards_tokens = forward_tokens[:, :, :, ::-1]
    backwards_sentence_ids = forward_sentence_ids[:, :, :, ::-1]

    tokens = np.concatenate([forward_tokens, backwards_tokens], 1).reshape(
        per_host_batch_size, -1)
    sentence_ids = np.concatenate(
        [forward_sentence_ids, backwards_sentence_ids]).reshape(
            per_host_batch_size, -1)
  else:
    logging.info("Bi-directional data disabled.")
    tokens, sentence_ids = _reshape_to_batch_dimensions(
        tokens=tokens,
        sentence_ids=sentence_ids,
        per_host_batch_size=per_host_batch_size)

  logging.info("Tokens shape: %s", tokens.shape)

  data_length = tokens.shape[1]
  sep = np.array([special_symbols["<sep>"]], dtype=np.int64)
  cls = np.array([special_symbols["<cls>"]], dtype=np.int64)
  # 2 sep, 1 cls
  num_special_tokens = 3

  data_index = 0
  batch_number = 0
  step_size = reuse_length if reuse_length else seq_length
  num_batches = math.ceil(data_length / step_size)

  while data_index + seq_length <= data_length:
    if batch_number % logging_frequency == 0:
      logging.info("Processing batch %d of %d", batch_number, num_batches)

    for batch_index in range(per_host_batch_size):
      previous_segment_tokens = tokens[
          batch_index, data_index: data_index + reuse_length]

      results = _create_a_and_b_segments(
          tokens=tokens[batch_index],
          sentence_ids=sentence_ids[batch_index],
          begin_index=data_index + reuse_length,
          total_length=seq_length - reuse_length - num_special_tokens)

      if results is None:
        logging.info("Stopping at data index: %d", data_index)
        break
      a_data, b_data, label = results

      data = np.concatenate(
          [previous_segment_tokens, a_data, sep, b_data, sep, cls])
      a_length = a_data.shape[0]
      b_length = b_data.shape[0]
      segment_ids = ([0] * (reuse_length + a_length) + [0]
                     + [1] * b_length + [1] + [2])
      boundary_indices = _get_boundary_indices(tokenizer=tokenizer,
                                               data=data)
      assert len(data) == seq_length
      assert len(segment_ids) == seq_length
      assert len(boundary_indices) > 0  # pylint: disable=g-explicit-length-test

      instances.append(TrainingInstance(
          data=data,
          segment_ids=segment_ids,
          boundary_indices=boundary_indices,
          label=label))
    batch_number += 1
    data_index += step_size
  return instances


def write_instances_to_tfrecord(
    instances: Iterable[TrainingInstance],
    save_path: str):
  """Writes instances to TFRecord."""
  record_writer = tf.io.TFRecordWriter(save_path)
  logging.info("Start writing to %s.", save_path)

  for i, instance in enumerate(instances):
    if i < 5:
      logging.info("Instance %d: %s", i, str(instance))
    record_writer.write(instance.to_example().SerializeToString())

  record_writer.close()
  logging.info("Done writing %s.", save_path)


def shuffle_and_combine_preprocessed_data(
    all_data: List[Tuple[np.array, np.array]]) -> Tuple[np.array, np.array]:
  """Shuffles and combines preprocessed token/sentence IDs from documents."""
  document_permutation = np.random.permutation(len(all_data))

  previous_sentence_id = None

  all_tokens, all_sentence_ids = [], []
  for document_index in document_permutation:
    tokens, sentence_ids = all_data[document_index]
    # pylint: disable=g-explicit-length-test
    if len(tokens) == 0:
      continue
    if (previous_sentence_id is not None and
        sentence_ids[0] == previous_sentence_id):
      sentence_ids = np.logical_not(sentence_ids)

    all_tokens.append(tokens)
    all_sentence_ids.append(sentence_ids)

    previous_sentence_id = sentence_ids[-1]

  return np.concatenate(all_tokens), np.concatenate(all_sentence_ids)


def get_tfrecord_name(
    per_host_batch_size: int,
    num_cores_per_host: int,
    seq_length: int,
    bi_data: bool,
    reuse_length: int,
    do_lower_case: bool,
    use_eod_token: bool,
    prefix: str = "",
    suffix: str = "",
    pass_id: int = 0,
    num_passes: int = 1,
    task_id: int = None,
    num_tasks: int = None) -> str:
  """Formats the resulting TFRecord name based on provided inputs."""
  components = []
  if prefix:
    components.append(prefix)
  components.append("seqlen-{}".format(seq_length))
  if reuse_length == 0:
    components.append("memless")
  else:
    components.append("reuse-{}".format(reuse_length))
  components.append("bs-{}".format(per_host_batch_size))
  components.append("cores-{}".format(num_cores_per_host))

  if do_lower_case:
    components.append("uncased")
  else:
    components.append("cased")
  if use_eod_token:
    components.append("eod")
  if bi_data:
    components.append("bi")
  else:
    components.append("uni")

  if suffix:
    components.append(suffix)

  s = "_".join(components) + ".tfrecord"
  if num_passes == 1 and task_id is None:
    return s

  if task_id is None:
    num_tasks = 1
    task_id = 0

  current_shard = task_id * num_passes + pass_id
  total_shards = num_tasks * num_passes
  return s + "-{}-of-{}".format(current_shard, total_shards)


def create_tfrecords(
    tokenizer: tokenization.FullSentencePieceTokenizer,
    input_file_or_files: str,
    use_eod_token: bool,
    do_lower_case: bool,
    per_host_batch_size: int,
    seq_length: int,
    reuse_length: int,
    bi_data: bool,
    num_cores_per_host: int,
    save_dir: str,
    prefix: str = "",
    suffix: str = "",
    num_tasks: Optional[int] = None,
    task_id: Optional[int] = None,
    num_passes: int = 1):
  """Runs the end-to-end preprocessing pipeline."""

  logging.info("Input configuration:")
  logging.info("input file(s): %s", input_file_or_files)
  logging.info("use_eod_token: %s", use_eod_token)
  logging.info("do_lower_case: %s", do_lower_case)
  logging.info("per_host_batch_size: %d", per_host_batch_size)
  logging.info("seq_length: %d", seq_length)
  logging.info("reuse_length: %d", reuse_length)
  logging.info("bi_data: %s", bi_data)
  logging.info("num_cores_per_host: %d", num_cores_per_host)
  logging.info("save_dir: %s", save_dir)
  if task_id is not None and num_tasks is not None:
    logging.info("task_id: %d", task_id)
    logging.info("num_tasks: %d", num_tasks)

  input_files = []
  for input_pattern in input_file_or_files.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  logging.info("*** Reading from input files ***")
  for input_file in input_files:
    logging.info("  %s", input_file)

  logging.info("Shuffling the files with a fixed random seed.")
  np.random.shuffle(input_files)
  if num_tasks is not None:
    assert task_id is not None
    logging.info("Total number of input files: %d", len(input_files))
    logging.info("Splitting into %d shards of %d files each.",
                 num_tasks, len(input_files) // num_tasks)
    input_files = input_files[task_id::num_tasks]

  all_data = preprocess_and_tokenize_input_files(
      input_files=input_files,
      tokenizer=tokenizer,
      use_eod=use_eod_token,
      do_lower_case=do_lower_case)
  for pass_id in range(num_passes):
    logging.info("Beginning pass %d of %d", pass_id, num_passes)
    tokens, sentence_ids = shuffle_and_combine_preprocessed_data(all_data)

    assert len(tokens) == len(sentence_ids)

    filename = get_tfrecord_name(
        per_host_batch_size=per_host_batch_size,
        num_cores_per_host=num_cores_per_host,
        seq_length=seq_length,
        bi_data=bi_data,
        use_eod_token=use_eod_token,
        reuse_length=reuse_length,
        do_lower_case=do_lower_case,
        prefix=prefix,
        suffix=suffix,
        pass_id=pass_id,
        num_passes=num_passes,
        num_tasks=num_tasks,
        task_id=task_id)
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
      # If the path already exists, then we were probably preempted but
      # previously wrote this file.
      logging.info("%s already exists, skipping this batch.", save_path)
    else:
      instances = _convert_tokens_to_instances(
          tokenizer=tokenizer,
          tokens=tokens,
          sentence_ids=sentence_ids,
          per_host_batch_size=per_host_batch_size,
          seq_length=seq_length,
          reuse_length=reuse_length,
          bi_data=bi_data,
          num_cores_per_host=num_cores_per_host)
      write_instances_to_tfrecord(instances=instances, save_path=save_path)

  if task_id is None or task_id == 0:
    corpus_info = {
        "vocab_size": 32000,
        "per_host_batch_size": per_host_batch_size,
        "num_cores_per_host": num_cores_per_host,
        "seq_length": seq_length,
        "reuse_length": reuse_length,
        "do_lower_case": do_lower_case,
        "bi_data": bi_data,
        "use_eod_token": use_eod_token,
    }
    corpus_fname = os.path.basename(filename) + ".json"
    corpus_destination = os.path.join(save_dir, corpus_fname)
    logging.info("Saving corpus info to %s", corpus_destination)

    with tf.io.gfile.GFile(corpus_destination, "w") as fp:
      json.dump(corpus_info, fp)


def main(_):
  tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
  create_tfrecords(
      tokenizer=tokenizer,
      input_file_or_files=FLAGS.input_file,
      use_eod_token=FLAGS.use_eod_token,
      do_lower_case=FLAGS.do_lower_case,
      per_host_batch_size=FLAGS.per_host_batch_size,
      seq_length=FLAGS.seq_length,
      reuse_length=FLAGS.reuse_length,
      bi_data=FLAGS.bi_data,
      num_cores_per_host=FLAGS.num_cores_per_host,
      save_dir=FLAGS.save_dir,
      prefix=FLAGS.prefix,
      suffix=FLAGS.suffix,
      num_tasks=FLAGS.num_tasks,
      task_id=FLAGS.task_id,
      num_passes=FLAGS.num_passes)


if __name__ == "__main__":
  np.random.seed(0)
  logging.set_verbosity(logging.INFO)
  app.run(main)
