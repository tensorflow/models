# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Input pipe for feeding examples to a Seq2Label model graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from protos import seq2label_pb2
import seq2label_utils

DNA_BASES = tuple('ACGT')
NUM_DNA_BASES = len(DNA_BASES)
# Possible FASTA characters/IUPAC ambiguity codes.
# See https://en.wikipedia.org/wiki/Nucleic_acid_notation.
AMBIGUITY_CODES = {
    'K': 'GT',
    'M': 'AC',
    'R': 'AG',
    'Y': 'CT',
    'S': 'CG',
    'W': 'AT',
    'B': 'CGT',
    'V': 'ACG',
    'H': 'ACT',
    'D': 'AGT',
    'X': 'ACGT',
    'N': 'ACGT'
}


def load_dataset_info(dataset_info_path):
  """Load a `Seq2LabelDatasetInfo` from a serialized text proto file."""
  dataset_info = seq2label_pb2.Seq2LabelDatasetInfo()
  with tf.gfile.Open(dataset_info_path, 'r') as f:
    text_format.Parse(f.read(), dataset_info)
  return dataset_info


class _InputEncoding(object):
  """A helper class providing the graph operations needed to encode input.

  Instantiation of an _InputEncoding will write on the default TF graph, so it
  should only be instantiated inside the `input_fn`.

  Attributes:
    mode: `tf.estimator.ModeKeys`; the execution mode {TRAIN, EVAL, INFER}.
    targets: list of strings; the names of the labels of interest (e.g.
      "species").
    dna_bases: a tuple of the recognized DNA alphabet.
    n_bases: the size of the DNA alphabet.
    all_characters: list of recognized alphabet, including ambiguity codes.
    label_values: a tuple of strings, the possible label values of the
      prediction target.
    n_labels: the size of label_values
    fixed_read_length: an integer value of the statically-known read length, or
     None if the read length is to be determined dynamically.
  """

  def __init__(self,
               dataset_info,
               mode,
               targets,
               noise_rate=0.0,
               fixed_read_length=None):
    self.mode = mode
    self.targets = targets
    self.dna_bases = DNA_BASES
    self.n_bases = NUM_DNA_BASES
    self.all_characters = list(DNA_BASES) + sorted(AMBIGUITY_CODES.keys())
    self.character_encodings = np.concatenate(
        [[self._character_to_base_distribution(char)]
         for char in self.all_characters],
        axis=0)
    all_legal_label_values = seq2label_utils.get_all_label_values(dataset_info)
    # TF lookup tables.
    self.characters_table = tf.contrib.lookup.index_table_from_tensor(
        mapping=self.all_characters)
    self.label_tables = {
        target: tf.contrib.lookup.index_table_from_tensor(
            all_legal_label_values[target])
        for target in targets
    }
    self.fixed_read_length = fixed_read_length
    self.noise_rate = noise_rate

  def _character_to_base_distribution(self, char):
    """Maps the given character to a probability distribution over DNA bases.

    Args:
      char: character to be encoded as a probability distribution over bases.

    Returns:
      Array of size (self.n_bases,) representing the identity of the given
      character as a distribution over the possible DNA bases, self.dna_bases.

    Raises:
      ValueError: if the given character is not contained in the recognized
        alphabet, self.all_characters.
    """
    if char not in self.all_characters:
      raise ValueError(
          'Base distribution requested for unrecognized character %s.' % char)
    possible_bases = AMBIGUITY_CODES[char] if char in AMBIGUITY_CODES else char
    base_indices = [self.dna_bases.index(base) for base in possible_bases]
    probability_weight = 1.0 / len(possible_bases)
    distribution = np.zeros((self.n_bases))
    distribution[base_indices] = probability_weight
    return distribution

  def encode_read(self, string_seq):
    """Converts the input read sequence to one-hot encoding.

    Args:
      string_seq: tf.String; input read sequence.

    Returns:
      Input read sequence as a one-hot encoded Tensor, with depth and ordering
      of one-hot encoding determined by the given bases. Ambiguous characters
      such as "N" and "S" are encoded as a probability distribution over the
      possible bases they represent.
    """
    with tf.variable_scope('encode_read'):
      read = tf.string_split([string_seq], delimiter='').values
      read = self.characters_table.lookup(read)
      read = tf.cast(tf.gather(self.character_encodings, read), tf.float32)
      if self.fixed_read_length:
        read = tf.reshape(read, (self.fixed_read_length, self.n_bases))
      return read

  def encode_label(self, target, string_label):
    """Converts the label value to an integer encoding.

    Args:
      target: str; the target name.
      string_label: tf.String; value of the label for the current input read.

    Returns:
      Given label value as an index into the possible_target_values.
    """
    with tf.variable_scope('encode_label/{}'.format(target)):
      return tf.cast(self.label_tables[target].lookup(string_label), tf.int32)

  def _empty_label(self):
    return tf.constant((), dtype=tf.int32, shape=())

  def parse_single_tfexample(self, serialized_example):
    """Parses a tf.train.Example proto to a one-hot encoded read, label pair.

    Injects noise into the incoming tf.train.Example's read sequence
    when noise_rate is non-zero.

    Args:
      serialized_example: string; the serialized tf.train.Example proto
        containing the read sequence and label value of interest as
        tf.FixedLenFeatures.

    Returns:
      Tuple (features, labels) of dicts for the input features and prediction
      targets.
    """
    with tf.variable_scope('parse_single_tfexample'):
      features_spec = {'sequence': tf.FixedLenFeature([], tf.string)}
      for target in self.targets:
        features_spec[target] = tf.FixedLenFeature([], tf.string)
      features = tf.parse_single_example(
          serialized_example, features=features_spec)
      if self.noise_rate > 0.0:
        read_sequence = tf.py_func(seq2label_utils.add_read_noise,
                                   [features['sequence'], self.noise_rate],
                                   (tf.string))
      else:
        read_sequence = features['sequence']
      read_sequence = self.encode_read(read_sequence)
      read_features = {'sequence': read_sequence}
      if self.mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        label = {
            target: self.encode_label(target, features[target])
            for target in self.targets
        }
      else:
        label = {target: self._empty_label() for target in self.targets}
      return read_features, label


class InputDataset(object):
  """A class providing access to input data for the Seq2Label model.

  Attributes:
    mode: `tf.estimator.ModeKeys`; the execution mode {TRAIN, EVAL, INFER}.
    targets: list of strings; the names of the labels of interest (e.g.
      "species").
    dataset_info: a `Seq2LabelDatasetInfo` message reflecting the dataset
      metadata.
    initializer: the TF initializer op for the underlying iterator, which
      will rewind the iterator.
    is_train: Boolean indicating whether or not the execution mode is TRAIN.
  """

  def __init__(self,
               mode,
               targets,
               dataset_info,
               train_epochs=None,
               noise_rate=0.0,
               random_seed=None,
               input_tfrecord_files=None,
               fixed_read_length=None,
               ensure_constant_batch_size=False,
               num_parallel_calls=32):
    """Constructor for InputDataset.

    Args:
      mode: `tf.estimator.ModeKeys`; the execution mode {TRAIN, EVAL, INFER}.
      targets: list of strings; the names of the labels of interest (e.g.
        "species").
      dataset_info: a `Seq2LabelDatasetInfo` message reflecting the dataset
        metadata.
      train_epochs: the number of training epochs to perform, if mode==TRAIN.
      noise_rate: float [0.0, 1.0] specifying rate at which to inject
        base-flipping noise into the read sequences.
      random_seed: seed to be used for shuffling, if mode==TRAIN.
      input_tfrecord_files: a list of filenames for TFRecords of TF examples.
      fixed_read_length: an integer value of the statically-known read length,
        or None if the read length is to be determined dynamically.  The read
        length must be known statically for TPU execution.
      ensure_constant_batch_size: ensure a constant batch size at the expense of
        discarding the last "short" batch.  This also gives us a statically
        constant batch size, which is essential for e.g. the TPU platform.
      num_parallel_calls: the number of dataset elements to process in parallel.
        If None, elements will be processed sequentially.
    """
    self.input_tfrecord_files = input_tfrecord_files
    self.mode = mode
    self.targets = targets
    self.dataset_info = dataset_info
    self._train_epochs = train_epochs
    self._noise_rate = noise_rate
    self._random_seed = random_seed
    if random_seed is not None:
      np.random.seed(random_seed)
    self._fixed_read_length = fixed_read_length
    self._ensure_constant_batch_size = ensure_constant_batch_size
    self._num_parallel_calls = num_parallel_calls

  @staticmethod
  def from_tfrecord_files(input_tfrecord_files, *args, **kwargs):
    return InputDataset(
        *args, input_tfrecord_files=input_tfrecord_files, **kwargs)

  @property
  def is_train(self):
    return self.mode == tf.estimator.ModeKeys.TRAIN

  def input_fn(self, params):
    """Supplies input for the model.

    This function supplies input to our model as a function of the mode.

    Args:
      params: a dictionary, containing:
        - params['batch_size']: the integer batch size.

    Returns:
      A tuple of two values as follows:
       1) the *features* dict, containing a tensor value for keys as follows:
            - "sequence" - the encoded read input sequence.
       2) the *labels* dict. containing a key for `target`, whose value is:
           - a string Tensor value (in TRAIN/EVAL mode), or
           - a blank Tensor (PREDICT mode).
    """
    randomize_input = self.is_train
    batch_size = params['batch_size']

    encoding = _InputEncoding(
        self.dataset_info,
        self.mode,
        self.targets,
        noise_rate=self._noise_rate,
        fixed_read_length=self._fixed_read_length)

    dataset = tf.data.TFRecordDataset(self.input_tfrecord_files)
    dataset = dataset.map(
        encoding.parse_single_tfexample,
        num_parallel_calls=self._num_parallel_calls)

    dataset = dataset.repeat(self._train_epochs if self.is_train else 1)
    if randomize_input:
      dataset = dataset.shuffle(
          buffer_size=max(1000, batch_size), seed=self._random_seed)

    if self._ensure_constant_batch_size:
      # Only take batches of *exactly* size batch_size; then we get a
      # statically knowable batch shape.
      dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
      dataset = dataset.batch(batch_size)

    # Prefetch to allow infeed to be in parallel with model computations.
    dataset = dataset.prefetch(2)

    # Use initializable iterator to support table lookups.
    iterator = dataset.make_initializable_iterator()
    self.initializer = iterator.initializer
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    features, labels = iterator.get_next()
    return (features, labels)
