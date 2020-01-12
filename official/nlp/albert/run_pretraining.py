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
# Lint as: python2, python3
"""Run masked LM/next sentence masked_lm pre-training for ALBERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import modeling
import optimization
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import tpu as contrib_tpu

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ALBERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 4096, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")

flags.DEFINE_enum("optimizer", "lamb", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 0.00176, "The initial learning rate.")

flags.DEFINE_float("poly_power", 1.0, "The power of poly decay.")

flags.DEFINE_integer("num_train_steps", 125000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 3125, "Number of warmup steps.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How many checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("init_from_group0", False, "Whether to initialize"
                  "parameters of other groups from group 0")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float(
    "masked_lm_budget", 0,
    "If >0, the ratio of masked ngrams to unmasked ngrams. Default 0,"
    "for offline masking")


def model_fn_builder(albert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer, poly_power,
                     start_warmup_step):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    # Note: We keep this feature name `next_sentence_labels` to be compatible
    # with the original data created by lanzhzh@. However, in the ALBERT case
    # it does represent sentence_order_labels.
    sentence_order_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.AlbertModel(
        config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss, masked_lm_example_loss,
     masked_lm_log_probs) = get_masked_lm_output(albert_config,
                                                 model.get_sequence_output(),
                                                 model.get_embedding_table(),
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights)

    (sentence_order_loss, sentence_order_example_loss,
     sentence_order_log_probs) = get_sentence_order_output(
         albert_config, model.get_pooled_output(), sentence_order_labels)

    total_loss = masked_lm_loss + sentence_order_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      tf.logging.info("number of hidden group %d to initialize",
                      albert_config.num_hidden_groups)
      num_of_initialize_group = 1
      if FLAGS.init_from_group0:
        num_of_initialize_group = albert_config.num_hidden_groups
        if albert_config.net_structure_type > 0:
          num_of_initialize_group = albert_config.num_hidden_layers
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(
              tvars, init_checkpoint, num_of_initialize_group)
      if use_tpu:

        def tpu_scaffold():
          for gid in range(num_of_initialize_group):
            tf.logging.info("initialize the %dth layer", gid)
            tf.logging.info(assignment_map[gid])
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map[gid])
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        for gid in range(num_of_initialize_group):
          tf.logging.info("initialize the %dth layer", gid)
          tf.logging.info(assignment_map[gid])
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map[gid])

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
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer, poly_power, start_warmup_step)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(*args):
        """Computes the loss and accuracy of the model."""
        (masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
         masked_lm_weights, sentence_order_example_loss,
         sentence_order_log_probs, sentence_order_labels) = args[:7]


        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        metrics = {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
        }

        sentence_order_log_probs = tf.reshape(
            sentence_order_log_probs, [-1, sentence_order_log_probs.shape[-1]])
        sentence_order_predictions = tf.argmax(
            sentence_order_log_probs, axis=-1, output_type=tf.int32)
        sentence_order_labels = tf.reshape(sentence_order_labels, [-1])
        sentence_order_accuracy = tf.metrics.accuracy(
            labels=sentence_order_labels,
            predictions=sentence_order_predictions)
        sentence_order_mean_loss = tf.metrics.mean(
            values=sentence_order_example_loss)
        metrics.update({
            "sentence_order_accuracy": sentence_order_accuracy,
            "sentence_order_loss": sentence_order_mean_loss
        })
        return metrics

      metric_values = [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, sentence_order_example_loss,
          sentence_order_log_probs, sentence_order_labels
      ]

      eval_metrics = (metric_fn, metric_values)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(albert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)


  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=albert_config.embedding_size,
          activation=modeling.get_activation(albert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[albert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=albert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_sentence_order_output(albert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, albert_config.hidden_size],
        initializer=modeling.create_initializer(
            albert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        # Note: We keep this feature name `next_sentence_labels` to be
        # compatible with the original data created by lanzhzh@. However, in
        # the ALBERT case it does represent sentence_order_labels.
        "next_sentence_labels": tf.FixedLenFeature([1], tf.int64),
    }

    if FLAGS.masked_lm_budget:
      name_to_features.update({
          "token_boundary":
              tf.FixedLenFeature([max_seq_length], tf.int64)})
    else:
      name_to_features.update({
          "masked_lm_positions":
              tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
          "masked_lm_ids":
              tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
          "masked_lm_weights":
              tf.FixedLenFeature([max_predictions_per_seq], tf.float32)})

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          contrib_data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch_with_legacy_function(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    tf.logging.info(d)
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      albert_config=albert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      optimizer=FLAGS.optimizer,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    global_step = -1
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    writer = tf.gfile.GFile(output_eval_file, "w")
    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)
    best_perf = 0
    key_name = "masked_lm_accuracy"
    while global_step < FLAGS.num_train_steps:
      if estimator.latest_checkpoint() is None:
        tf.logging.info("No checkpoint found yet. Sleeping.")
        time.sleep(1)
      else:
        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
        global_step = result["global_step"]
        tf.logging.info("***** Eval results *****")
        checkpoint_path = estimator.latest_checkpoint()
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))
          if result[key_name] > best_perf:
            best_perf = result[key_name]
            for ext in ["meta", "data-00000-of-00001", "index"]:
              src_ckpt = checkpoint_path + ".{}".format(ext)
              tgt_ckpt = checkpoint_path.rsplit(
                  "-", 1)[0] + "-best.{}".format(ext)
              tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
              tf.gfile.Copy(src_ckpt, tgt_ckpt, overwrite=True)
              writer.write("saved {} to {}\n".format(src_ckpt, tgt_ckpt))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("albert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
