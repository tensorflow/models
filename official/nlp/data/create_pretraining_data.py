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

"""Create masked LM/next sentence masked_lm TF examples for BERT."""

import collections
import itertools
import random

# Import libraries

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf, tf_keras

from official.nlp.tools import tokenization

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_enum(
    "tokenization",
    "WordPiece",
    ["WordPiece", "SentencePiece"],
    "Specifies the tokenizer implementation, i.e., whether to use WordPiece "
    "or SentencePiece tokenizer. Canonical BERT uses WordPiece tokenizer, "
    "while ALBERT uses SentencePiece tokenizer.",
)

flags.DEFINE_string(
    "vocab_file",
    None,
    "For WordPiece tokenization, the vocabulary file of the tokenizer.",
)

flags.DEFINE_string(
    "sp_model_file",
    "",
    "For SentencePiece tokenization, the path to the model of the tokenizer.",
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask",
    False,
    "Whether to use whole word masking rather than per-token masking.",
)

flags.DEFINE_integer(
    "max_ngram_size", None,
    "Mask contiguous whole words (n-grams) of up to `max_ngram_size` using a "
    "weighting scheme to favor shorter n-grams. "
    "Note: `--do_whole_word_mask=True` must also be set when n-gram masking.")

flags.DEFINE_bool(
    "gzip_compress", False,
    "Whether to use `GZIP` compress option to get compressed TFRecord files.")

flags.DEFINE_bool(
    "use_v2_feature_names", False,
    "Whether to use the feature names consistent with the models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files,
                                    gzip_compress, use_v2_feature_names):
  """Creates TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(
        tf.io.TFRecordWriter(
            output_file, options="GZIP" if gzip_compress else ""))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    if use_v2_feature_names:
      features["input_word_ids"] = create_int_feature(input_ids)
      features["input_type_ids"] = create_int_feature(segment_ids)
    else:
      features["input_ids"] = create_int_feature(input_ids)
      features["segment_ids"] = create_int_feature(segment_ids)

    features["input_mask"] = create_int_feature(input_mask)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      logging.info("*** Example ***")
      logging.info("tokens: %s", " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        logging.info("%s: %s", feature_name, " ".join([str(x) for x in values]))

  for writer in writers:
    writer.close()

  logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(
    input_files,
    tokenizer,
    processor_text_fn,
    max_seq_length,
    dupe_factor,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    rng,
    do_whole_word_mask=False,
    max_ngram_size=None,
):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.io.gfile.GFile(input_file, "rb") as reader:
      for line in reader:
        line = processor_text_fn(line)

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng,
              do_whole_word_mask, max_ngram_size))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng,
    do_whole_word_mask=False,
    max_ngram_size=None):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng,
             do_whole_word_mask, max_ngram_size)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

# A _Gram is a [half-open) interval of token indices which form a word.
# E.g.,
#   words:  ["The", "doghouse"]
#   tokens: ["The", "dog", "##house"]
#   grams:  [(0,1), (1,3)]
_Gram = collections.namedtuple("_Gram", ["begin", "end"])


def _window(iterable, size):
  """Helper to create a sliding window iterator with a given size.

  E.g.,
    input = [1, 2, 3, 4]
    _window(input, 1) => [1], [2], [3], [4]
    _window(input, 2) => [1, 2], [2, 3], [3, 4]
    _window(input, 3) => [1, 2, 3], [2, 3, 4]
    _window(input, 4) => [1, 2, 3, 4]
    _window(input, 5) => None

  Args:
    iterable: elements to iterate over.
    size: size of the window.

  Yields:
    Elements of `iterable` batched into a sliding window of length `size`.
  """
  i = iter(iterable)
  window = []
  try:
    for e in range(0, size):
      window.append(next(i))
    yield window
  except StopIteration:
    # handle the case where iterable's length is less than the window size.
    return
  for e in i:
    window = window[1:] + [e]
    yield window


def _contiguous(sorted_grams):
  """Test whether a sequence of grams is contiguous.

  Args:
    sorted_grams: _Grams which are sorted in increasing order.
  Returns:
    True if `sorted_grams` are touching each other.

  E.g.,
    _contiguous([(1, 4), (4, 5), (5, 10)]) == True
    _contiguous([(1, 2), (4, 5)]) == False
  """
  for a, b in _window(sorted_grams, 2):
    if a.end != b.begin:
      return False
  return True


def _masking_ngrams(grams, max_ngram_size, max_masked_tokens, rng):
  """Create a list of masking {1, ..., n}-grams from a list of one-grams.

  This is an extension of 'whole word masking' to mask multiple, contiguous
  words such as (e.g., "the red boat").

  Each input gram represents the token indices of a single word,
     words:  ["the", "red", "boat"]
     tokens: ["the", "red", "boa", "##t"]
     grams:  [(0,1), (1,2), (2,4)]

  For a `max_ngram_size` of three, possible outputs masks include:
    1-grams: (0,1), (1,2), (2,4)
    2-grams: (0,2), (1,4)
    3-grams; (0,4)

  Output masks will not overlap and contain less than `max_masked_tokens` total
  tokens.  E.g., for the example above with `max_masked_tokens` as three,
  valid outputs are,
       [(0,1), (1,2)]  # "the", "red" covering two tokens
       [(1,2), (2,4)]  # "red", "boa", "##t" covering three tokens

  The length of the selected n-gram follows a zipf weighting to
  favor shorter n-gram sizes (weight(1)=1, weight(2)=1/2, weight(3)=1/3, ...).

  Args:
    grams: List of one-grams.
    max_ngram_size: Maximum number of contiguous one-grams combined to create
      an n-gram.
    max_masked_tokens: Maximum total number of tokens to be masked.
    rng: `random.Random` generator.

  Returns:
    A list of n-grams to be used as masks.
  """
  if not grams:
    return None

  grams = sorted(grams)
  num_tokens = grams[-1].end

  # Ensure our grams are valid (i.e., they don't overlap).
  for a, b in _window(grams, 2):
    if a.end > b.begin:
      raise ValueError("overlapping grams: {}".format(grams))

  # Build map from n-gram length to list of n-grams.
  ngrams = {i: [] for i in range(1, max_ngram_size+1)}
  for gram_size in range(1, max_ngram_size+1):
    for g in _window(grams, gram_size):
      if _contiguous(g):
        # Add an n-gram which spans these one-grams.
        ngrams[gram_size].append(_Gram(g[0].begin, g[-1].end))

  # Shuffle each list of n-grams.
  for v in ngrams.values():
    rng.shuffle(v)

  # Create the weighting for n-gram length selection.
  # Stored cumulatively for `random.choices` below.
  cummulative_weights = list(
      itertools.accumulate([1./n for n in range(1, max_ngram_size+1)]))

  output_ngrams = []
  # Keep a bitmask of which tokens have been masked.
  masked_tokens = [False] * num_tokens
  # Loop until we have enough masked tokens or there are no more candidate
  # n-grams of any length.
  # Each code path should ensure one or more elements from `ngrams` are removed
  # to guarantee this loop terminates.
  while (sum(masked_tokens) < max_masked_tokens and
         sum(len(s) for s in ngrams.values())):
    # Pick an n-gram size based on our weights.
    sz = random.choices(range(1, max_ngram_size+1),
                        cum_weights=cummulative_weights)[0]

    # Ensure this size doesn't result in too many masked tokens.
    # E.g., a two-gram contains _at least_ two tokens.
    if sum(masked_tokens) + sz > max_masked_tokens:
      # All n-grams of this length are too long and can be removed from
      # consideration.
      ngrams[sz].clear()
      continue

    # All of the n-grams of this size have been used.
    if not ngrams[sz]:
      continue

    # Choose a random n-gram of the given size.
    gram = ngrams[sz].pop()
    num_gram_tokens = gram.end-gram.begin

    # Check if this would add too many tokens.
    if num_gram_tokens + sum(masked_tokens) > max_masked_tokens:
      continue

    # Check if any of the tokens in this gram have already been masked.
    if sum(masked_tokens[gram.begin:gram.end]):
      continue

    # Found a usable n-gram!  Mark its tokens as masked and add it to return.
    masked_tokens[gram.begin:gram.end] = [True] * (gram.end-gram.begin)
    output_ngrams.append(gram)
  return output_ngrams


def _tokens_to_grams(tokens):
  """Reconstitue grams (words) from `tokens`.

  E.g.,
     tokens: ['[CLS]', 'That', 'lit', '##tle', 'blue', 'tru', '##ck', '[SEP]']
      grams: [          [1,2), [2,         4),  [4,5) , [5,       6)]

  Args:
    tokens: list of tokens (word pieces or sentence pieces).

  Returns:
    List of _Grams representing spans of whole words
    (without "[CLS]" and "[SEP]").
  """
  grams = []
  gram_start_pos = None
  for i, token in enumerate(tokens):
    if gram_start_pos is not None and token.startswith("##"):
      continue
    if gram_start_pos is not None:
      grams.append(_Gram(gram_start_pos, i))
    if token not in ["[CLS]", "[SEP]"]:
      gram_start_pos = i
    else:
      gram_start_pos = None
  if gram_start_pos is not None:
    grams.append(_Gram(gram_start_pos, len(tokens)))
  return grams


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 do_whole_word_mask,
                                 max_ngram_size=None):
  """Creates the predictions for the masked LM objective."""
  if do_whole_word_mask:
    grams = _tokens_to_grams(tokens)
  else:
    # Here we consider each token to be a word to allow for sub-word masking.
    if max_ngram_size:
      raise ValueError("cannot use ngram masking without whole word masking")
    grams = [_Gram(i, i+1) for i in range(0, len(tokens))
             if tokens[i] not in ["[CLS]", "[SEP]"]]

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))
  # Generate masks.  If `max_ngram_size` in [0, None] it means we're doing
  # whole word masking or token level masking.  Both of these can be treated
  # as the `max_ngram_size=1` case.
  masked_grams = _masking_ngrams(grams, max_ngram_size or 1,
                                 num_to_predict, rng)
  masked_lms = []
  output_tokens = list(tokens)
  for gram in masked_grams:
    # 80% of the time, replace all n-gram tokens with [MASK]
    if rng.random() < 0.8:
      replacement_action = lambda idx: "[MASK]"
    else:
      # 10% of the time, keep all the original n-gram tokens.
      if rng.random() < 0.5:
        replacement_action = lambda idx: tokens[idx]
      # 10% of the time, replace each n-gram token with a random word.
      else:
        replacement_action = lambda idx: rng.choice(vocab_words)

    for idx in range(gram.begin, gram.end):
      output_tokens[idx] = replacement_action(idx)
      masked_lms.append(MaskedLmInstance(index=idx, label=tokens[idx]))

  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def get_processor_text_fn(is_sentence_piece, do_lower_case):
  def processor_text_fn(text):
    text = tokenization.convert_to_unicode(text)
    if is_sentence_piece:
      # Additional preprocessing specific to the SentencePiece tokenizer.
      text = tokenization.preprocess_text(text, lower=do_lower_case)

    return text.strip()

  return processor_text_fn


def main(_):
  if FLAGS.tokenization == "WordPiece":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case
    )
    processor_text_fn = get_processor_text_fn(False, FLAGS.do_lower_case)
  else:
    assert FLAGS.tokenization == "SentencePiece"
    tokenizer = tokenization.FullSentencePieceTokenizer(FLAGS.sp_model_file)
    processor_text_fn = get_processor_text_fn(True, FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  logging.info("*** Reading from input files ***")
  for input_file in input_files:
    logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files,
      tokenizer,
      processor_text_fn,
      FLAGS.max_seq_length,
      FLAGS.dupe_factor,
      FLAGS.short_seq_prob,
      FLAGS.masked_lm_prob,
      FLAGS.max_predictions_per_seq,
      rng,
      FLAGS.do_whole_word_mask,
      FLAGS.max_ngram_size,
  )

  output_files = FLAGS.output_file.split(",")
  logging.info("*** Writing to output files ***")
  for output_file in output_files:
    logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files,
                                  FLAGS.gzip_compress,
                                  FLAGS.use_v2_feature_names)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  app.run(main)
