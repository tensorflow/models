# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Basic CTC+recoder decoder.

Decodes a sequence of class-ids into UTF-8 text.
For basic information on CTC See:
Alex Graves et al. Connectionist Temporal Classification: Labelling Unsegmented
Sequence Data with Recurrent Neural Networks.
http://www.cs.toronto.edu/~graves/icml_2006.pdf
"""
import collections
import re

import errorcounter as ec
import tensorflow as tf

# Named tuple Part describes a part of a multi (1 or more) part code that
# represents a utf-8 string. For example, Chinese character 'x' might be
# represented by 3 codes of which (utf8='x', index=1, num_codes3) would be the
# middle part. (The actual code is not stored in the tuple).
Part = collections.namedtuple('Part', 'utf8 index, num_codes')


# Class that decodes a sequence of class-ids into UTF-8 text.
class Decoder(object):
  """Basic CTC+recoder decoder."""

  def __init__(self, filename):
    r"""Constructs a Decoder.

    Reads the text file describing the encoding and build the encoder.
    The text file contains lines of the form:
    <code>[,<code>]*\t<string>
    Each line defines a mapping from a sequence of one or more integer codes to
    a corresponding utf-8 string.
    Args:
      filename:   Name of file defining the decoding sequences.
    """
    # self.decoder is a list of lists of Part(utf8, index, num_codes).
    # The index to the top-level list is a code. The list given by the code
    # index is a list of the parts represented by that code, Eg if the code 42
    # represents the 2nd (index 1) out of 3 part of Chinese character 'x', then
    # self.decoder[42] = [..., (utf8='x', index=1, num_codes3), ...] where ...
    # means all other uses of the code 42.
    self.decoder = []
    if filename:
      self._InitializeDecoder(filename)

  def SoftmaxEval(self, sess, model, num_steps):
    """Evaluate a model in softmax mode.

    Adds char, word recall and sequence error rate events to the sw summary
    writer, and returns them as well
    TODO(rays) Add LogisticEval.
    Args:
      sess:  A tensor flow Session.
      model: The model to run in the session. Requires a VGSLImageModel or any
        other class that has a using_ctc attribute and a RunAStep(sess) method
        that reurns a softmax result with corresponding labels.
      num_steps: Number of steps to evaluate for.
    Returns:
      ErrorRates named tuple.
    Raises:
      ValueError: If an unsupported number of dimensions is used.
    """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Run the requested number of evaluation steps, gathering the outputs of the
    # softmax and the true labels of the evaluation examples.
    total_label_counts = ec.ErrorCounts(0, 0, 0, 0)
    total_word_counts = ec.ErrorCounts(0, 0, 0, 0)
    sequence_errors = 0
    for _ in xrange(num_steps):
      softmax_result, labels = model.RunAStep(sess)
      # Collapse softmax to same shape as labels.
      predictions = softmax_result.argmax(axis=-1)
      # Exclude batch from num_dims.
      num_dims = len(predictions.shape) - 1
      batch_size = predictions.shape[0]
      null_label = softmax_result.shape[-1] - 1
      for b in xrange(batch_size):
        if num_dims == 2:
          # TODO(rays) Support 2-d data.
          raise ValueError('2-d label data not supported yet!')
        else:
          if num_dims == 1:
            pred_batch = predictions[b, :]
            labels_batch = labels[b, :]
          else:
            pred_batch = [predictions[b]]
            labels_batch = [labels[b]]
          text = self.StringFromCTC(pred_batch, model.using_ctc, null_label)
          truth = self.StringFromCTC(labels_batch, False, null_label)
          # Note that recall_errs is false negatives (fn) aka drops/deletions.
          # Actual recall would be 1-fn/truth_words.
          # Likewise precision_errs is false positives (fp) aka adds/insertions.
          # Actual precision would be 1-fp/ocr_words.
          total_word_counts = ec.AddErrors(total_word_counts,
                                           ec.CountWordErrors(text, truth))
          total_label_counts = ec.AddErrors(total_label_counts,
                                            ec.CountErrors(text, truth))
          if text != truth:
            sequence_errors += 1

    coord.request_stop()
    coord.join(threads)
    return ec.ComputeErrorRates(total_label_counts, total_word_counts,
                                sequence_errors, num_steps * batch_size)

  def StringFromCTC(self, ctc_labels, merge_dups, null_label):
    """Decodes CTC output to a string.

    Extracts only sequences of codes that are allowed by self.decoder.
    Labels that make illegal code sequences are dropped.
    Note that, by its nature of taking only top choices, this is much weaker
    than a full-blown beam search that considers all the softmax outputs.
    For languages without many multi-code sequences, this doesn't make much
    difference, but for complex scripts the accuracy will be much lower.
    Args:
      ctc_labels: List of class labels including null characters to remove.
      merge_dups: If True, Duplicate labels will be merged
      null_label: Label value to ignore.

    Returns:
      Labels decoded to a string.
    """
    # Run regular ctc on the labels, extracting a list of codes.
    codes = self._CodesFromCTC(ctc_labels, merge_dups, null_label)
    length = len(codes)
    if length == 0:
      return ''
    # strings and partials are both indexed by the same index as codes.
    # strings[i] is the best completed string upto position i, and
    # partials[i] is a list of partial code sequences at position i.
    # Warning: memory is squared-order in length.
    strings = []
    partials = []
    for pos in xrange(length):
      code = codes[pos]
      parts = self.decoder[code]
      partials.append([])
      strings.append('')
      # Iterate over the parts that this code can represent.
      for utf8, index, num_codes in parts:
        if index > pos:
          continue
        # We can use code if it is an initial code (index==0) or continues a
        # sequence in the partials list at the previous position.
        if index == 0 or partials[pos - 1].count(
            Part(utf8, index - 1, num_codes)) > 0:
          if index < num_codes - 1:
            # Save the partial sequence.
            partials[-1].append(Part(utf8, index, num_codes))
          elif not strings[-1]:
            # A code sequence is completed. Append to the best string that we
            # had where it started.
            if pos >= num_codes:
              strings[-1] = strings[pos - num_codes] + utf8
            else:
              strings[-1] = utf8
      if not strings[-1] and pos > 0:
        # We didn't get anything here so copy the previous best string, skipping
        # the current code, but it may just be a partial anyway.
        strings[-1] = strings[-2]
    return strings[-1]

  def _InitializeDecoder(self, filename):
    """Reads the decoder file and initializes self.decoder from it.

    Args:
      filename: Name of text file mapping codes to utf8 strings.
    Raises:
      ValueError: if the input file is not parsed correctly.
    """
    line_re = re.compile(r'(?P<codes>\d+(,\d+)*)\t(?P<utf8>.+)')
    with tf.gfile.GFile(filename) as f:
      for line in f:
        m = line_re.match(line)
        if m is None:
          raise ValueError('Unmatched line:', line)
        # codes is the sequence that maps to the string.
        str_codes = m.groupdict()['codes'].split(',')
        codes = []
        for code in str_codes:
          codes.append(int(code))
        utf8 = m.groupdict()['utf8']
        num_codes = len(codes)
        for index, code in enumerate(codes):
          while code >= len(self.decoder):
            self.decoder.append([])
          self.decoder[code].append(Part(utf8, index, num_codes))

  def _CodesFromCTC(self, ctc_labels, merge_dups, null_label):
    """Collapses CTC output to regular output.

    Args:
      ctc_labels: List of class labels including null characters to remove.
      merge_dups: If True, Duplicate labels will be merged.
      null_label: Label value to ignore.

    All trailing zeros are removed!!
    TODO(rays) This may become a problem with non-CTC models.
    If using charset, this should not be a problem as zero is always space.
    tf.pad can only append zero, so we have to be able to drop them, as a
    non-ctc will have learned to output trailing zeros instead of trailing
    nulls. This is awkward, as the stock ctc loss function requires that the
    null character be num_classes-1.
    Returns:
      (List of) Labels with null characters removed.
    """
    out_labels = []
    prev_label = -1
    zeros_needed = 0
    for label in ctc_labels:
      if label == null_label:
        prev_label = -1
      elif label != prev_label or not merge_dups:
        if label == 0:
          # Count zeros and only emit them when it is clear there is a non-zero
          # after, so as to truncate away all trailing zeros.
          zeros_needed += 1
        else:
          if merge_dups and zeros_needed > 0:
            out_labels.append(0)
          else:
            out_labels += [0] * zeros_needed
          zeros_needed = 0
          out_labels.append(label)
        prev_label = label
    return out_labels
