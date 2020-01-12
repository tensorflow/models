# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
"""Utility functions for RACE dataset."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import collections
import json
import os
import classifier_utils
import modeling
import optimization
import tokenization
import tensorflow.compat.v1 as tf
from tensorflow.contrib import tpu as contrib_tpu


class InputExample(object):
  """A single training/test example for the RACE dataset."""

  def __init__(self,
               example_id,
               context_sentence,
               start_ending,
               endings,
               label=None):
    self.example_id = example_id
    self.context_sentence = context_sentence
    self.start_ending = start_ending
    self.endings = endings
    self.label = label

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    l = [
        "id: {}".format(self.example_id),
        "context_sentence: {}".format(self.context_sentence),
        "start_ending: {}".format(self.start_ending),
        "ending_0: {}".format(self.endings[0]),
        "ending_1: {}".format(self.endings[1]),
        "ending_2: {}".format(self.endings[2]),
        "ending_3: {}".format(self.endings[3]),
    ]

    if self.label is not None:
      l.append("label: {}".format(self.label))

    return ", ".join(l)


class RaceProcessor(object):
  """Processor for the RACE data set."""

  def __init__(self, use_spm, do_lower_case, high_only, middle_only):
    super(RaceProcessor, self).__init__()
    self.use_spm = use_spm
    self.do_lower_case = do_lower_case
    self.high_only = high_only
    self.middle_only = middle_only

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    return self.read_examples(
        os.path.join(data_dir, "RACE", "train"))

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    return self.read_examples(
        os.path.join(data_dir, "RACE", "dev"))

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    return self.read_examples(
        os.path.join(data_dir, "RACE", "test"))

  def get_labels(self):
    """Gets the list of labels for this data set."""
    return ["A", "B", "C", "D"]

  def process_text(self, text):
    if self.use_spm:
      return tokenization.preprocess_text(text, lower=self.do_lower_case)
    else:
      return tokenization.convert_to_unicode(text)

  def read_examples(self, data_dir):
    """Read examples from RACE json files."""
    examples = []
    for level in ["middle", "high"]:
      if level == "middle" and self.high_only: continue
      if level == "high" and self.middle_only: continue
      cur_dir = os.path.join(data_dir, level)

      cur_path = os.path.join(cur_dir, "all.txt")
      with tf.gfile.Open(cur_path) as f:
        for line in f:
          cur_data = json.loads(line.strip())

          answers = cur_data["answers"]
          options = cur_data["options"]
          questions = cur_data["questions"]
          context = self.process_text(cur_data["article"])

          for i in range(len(answers)):
            label = ord(answers[i]) - ord("A")
            qa_list = []

            question = self.process_text(questions[i])
            for j in range(4):
              option = self.process_text(options[i][j])

              if "_" in question:
                qa_cat = question.replace("_", option)
              else:
                qa_cat = " ".join([question, option])

              qa_list.append(qa_cat)

            examples.append(
                InputExample(
                    example_id=cur_data["id"],
                    context_sentence=context,
                    start_ending=None,
                    endings=[qa_list[0], qa_list[1], qa_list[2], qa_list[3]],
                    label=label
                )
            )

    return examples


def convert_single_example(example_index, example, label_size, max_seq_length,
                           tokenizer, max_qa_length):
  """Loads a data file into a list of `InputBatch`s."""

  # RACE is a multiple choice task. To perform this task using AlBERT,
  # we will use the formatting proposed in "Improving Language
  # Understanding by Generative Pre-Training" and suggested by
  # @jacobdevlin-google in this issue
  # https://github.com/google-research/bert/issues/38.
  #
  # Each choice will correspond to a sample on which we run the
  # inference. For a given RACE example, we will create the 4
  # following inputs:
  # - [CLS] context [SEP] choice_1 [SEP]
  # - [CLS] context [SEP] choice_2 [SEP]
  # - [CLS] context [SEP] choice_3 [SEP]
  # - [CLS] context [SEP] choice_4 [SEP]
  # The model will output a single value for each input. To get the
  # final decision of the model, we will run a softmax over these 4
  # outputs.
  if isinstance(example, classifier_utils.PaddingInputExample):
    return classifier_utils.InputFeatures(
        example_id=0,
        input_ids=[[0] * max_seq_length] * label_size,
        input_mask=[[0] * max_seq_length] * label_size,
        segment_ids=[[0] * max_seq_length] * label_size,
        label_id=0,
        is_real_example=False)
  else:
    context_tokens = tokenizer.tokenize(example.context_sentence)
    if example.start_ending is not None:
      start_ending_tokens = tokenizer.tokenize(example.start_ending)

    all_input_tokens = []
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    for ending in example.endings:
      # We create a copy of the context tokens in order to be
      # able to shrink it according to ending_tokens
      context_tokens_choice = context_tokens[:]
      if example.start_ending is not None:
        ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
      else:
        ending_tokens = tokenizer.tokenize(ending)
      # Modifies `context_tokens_choice` and `ending_tokens` in
      # place so that the total length is less than the
      # specified length.  Account for [CLS], [SEP], [SEP] with
      # "- 3"
      ending_tokens = ending_tokens[- max_qa_length:]

      if len(context_tokens_choice) + len(ending_tokens) > max_seq_length - 3:
        context_tokens_choice = context_tokens_choice[: (
            max_seq_length - 3 - len(ending_tokens))]
      tokens = ["[CLS]"] + context_tokens_choice + (
          ["[SEP]"] + ending_tokens + ["[SEP]"])
      segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (
          len(ending_tokens) + 1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding = [0] * (max_seq_length - len(input_ids))
      input_ids += padding
      input_mask += padding
      segment_ids += padding

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      all_input_tokens.append(tokens)
      all_input_ids.append(input_ids)
      all_input_mask.append(input_mask)
      all_segment_ids.append(segment_ids)

    label = example.label
    if example_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("id: {}".format(example.example_id))
      for choice_idx, (tokens, input_ids, input_mask, segment_ids) in \
           enumerate(zip(all_input_tokens, all_input_ids, all_input_mask, all_segment_ids)):
        tf.logging.info("choice: {}".format(choice_idx))
        tf.logging.info("tokens: {}".format(" ".join(tokens)))
        tf.logging.info(
            "input_ids: {}".format(" ".join(map(str, input_ids))))
        tf.logging.info(
            "input_mask: {}".format(" ".join(map(str, input_mask))))
        tf.logging.info(
            "segment_ids: {}".format(" ".join(map(str, segment_ids))))
        tf.logging.info("label: {}".format(label))

    return classifier_utils.InputFeatures(
        example_id=example.example_id,
        input_ids=all_input_ids,
        input_mask=all_input_mask,
        segment_ids=all_segment_ids,
        label_id=label
    )


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer,
    output_file, max_qa_length):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, len(label_list),
                                     max_seq_length, tokenizer, max_qa_length)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(sum(feature.input_ids, []))
    features["input_mask"] = create_int_feature(sum(feature.input_mask, []))
    features["segment_ids"] = create_int_feature(sum(feature.segment_ids, []))
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def create_model(albert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, max_seq_length,
                 dropout_prob):
  """Creates a classification model."""
  bsz_per_core = tf.shape(input_ids)[0]

  model = modeling.AlbertModel(
      config=albert_config,
      is_training=is_training,
      input_ids=tf.reshape(input_ids, [bsz_per_core * num_labels,
                                       max_seq_length]),
      input_mask=tf.reshape(input_mask, [bsz_per_core * num_labels,
                                         max_seq_length]),
      token_type_ids=tf.reshape(segment_ids, [bsz_per_core * num_labels,
                                              max_seq_length]),
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [1],
      initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(
          output_layer, keep_prob=1 - dropout_prob)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [bsz_per_core, num_labels])
    probabilities = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(
        labels, depth=tf.cast(num_labels, dtype=tf.int32), dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, probabilities, logits, predictions)


def model_fn_builder(albert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, max_seq_length, dropout_prob):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, probabilities, logits, predictions) = \
        create_model(albert_config, is_training, input_ids, input_mask,
                     segment_ids, label_ids, num_labels,
                     use_one_hot_embeddings, max_seq_length, dropout_prob)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions,
            weights=is_real_example)
        loss = tf.metrics.mean(
            values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities,
                       "predictions": predictions},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

