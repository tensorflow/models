# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TriviaQA training script."""
import collections
import contextlib
import functools
import json
import operator
import os

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
import tensorflow_datasets as tfds

import sentencepiece as spm
from official.nlp import optimization as nlp_optimization
from official.nlp.configs import encoders
from official.nlp.projects.triviaqa import evaluation
from official.nlp.projects.triviaqa import inputs
from official.nlp.projects.triviaqa import modeling
from official.nlp.projects.triviaqa import prediction

flags.DEFINE_string('data_dir', None, 'Data directory for TensorFlow Datasets.')

flags.DEFINE_string(
    'validation_gold_path', None,
    'Path to golden validation. Usually, the wikipedia-dev.json file.')

flags.DEFINE_string('model_dir', None,
                    'Directory for checkpoints and summaries.')

flags.DEFINE_string('model_config_path', None,
                    'JSON file containing model coniguration.')

flags.DEFINE_string('sentencepiece_model_path', None,
                    'Path to sentence piece model.')

flags.DEFINE_enum('encoder', 'bigbird',
                  ['bert', 'bigbird', 'albert', 'mobilebert'],
                  'Which transformer encoder model to use.')

flags.DEFINE_integer('bigbird_block_size', 64,
                     'Size of blocks for sparse block attention.')

flags.DEFINE_string('init_checkpoint_path', None,
                    'Path from which to initialize weights.')

flags.DEFINE_integer('train_sequence_length', 4096,
                     'Maximum number of tokens for training.')

flags.DEFINE_integer('train_global_sequence_length', 320,
                     'Maximum number of global tokens for training.')

flags.DEFINE_integer('validation_sequence_length', 4096,
                     'Maximum number of tokens for validation.')

flags.DEFINE_integer('validation_global_sequence_length', 320,
                     'Maximum number of global tokens for validation.')

flags.DEFINE_integer('batch_size', 32, 'Size of batch.')

flags.DEFINE_string('master', '', 'Address of the TPU master.')

flags.DEFINE_integer('decode_top_k', 8,
                     'Maximum number of tokens to consider for begin/end.')

flags.DEFINE_integer('decode_max_size', 16,
                     'Maximum number of sentence pieces in an answer.')

flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate for hidden layers.')

flags.DEFINE_float('attention_dropout_rate', 0.3,
                   'Dropout rate for attention layers.')

flags.DEFINE_float('label_smoothing', 1e-1, 'Degree of label smoothing.')

flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files')

FLAGS = flags.FLAGS


@contextlib.contextmanager
def worker_context():
  if FLAGS.master:
    with tf.device('/job:worker') as d:
      yield d
  else:
    yield


def read_sentencepiece_model(path):
  with tf.io.gfile.GFile(path, 'rb') as file:
    processor = spm.SentencePieceProcessor()
    processor.LoadFromSerializedProto(file.read())
  return processor


# Rename old BERT v1 configuration parameters.
_MODEL_CONFIG_REPLACEMENTS = {
    'num_hidden_layers': 'num_layers',
    'attention_probs_dropout_prob': 'attention_dropout_rate',
    'hidden_dropout_prob': 'dropout_rate',
    'hidden_act': 'hidden_activation',
    'window_size': 'block_size',
}


def read_model_config(encoder,
                      path,
                      bigbird_block_size=None) -> encoders.EncoderConfig:
  """Merges the JSON configuration into the encoder configuration."""
  with tf.io.gfile.GFile(path) as f:
    model_config = json.load(f)
  for key, value in _MODEL_CONFIG_REPLACEMENTS.items():
    if key in model_config:
      model_config[value] = model_config.pop(key)
  model_config['attention_dropout_rate'] = FLAGS.attention_dropout_rate
  model_config['dropout_rate'] = FLAGS.dropout_rate
  model_config['block_size'] = bigbird_block_size
  encoder_config = encoders.EncoderConfig(type=encoder)
  # Override the default config with those loaded from the JSON file.
  encoder_config_keys = encoder_config.get().as_dict().keys()
  overrides = {}
  for key, value in model_config.items():
    if key in encoder_config_keys:
      overrides[key] = value
    else:
      logging.warning('Ignoring config parameter %s=%s', key, value)
  encoder_config.get().override(overrides)
  return encoder_config


@gin.configurable(blacklist=[
    'model',
    'strategy',
    'train_dataset',
    'model_dir',
    'init_checkpoint_path',
    'evaluate_fn',
])
def fit(model,
        strategy,
        train_dataset,
        model_dir,
        init_checkpoint_path=None,
        evaluate_fn=None,
        learning_rate=1e-5,
        learning_rate_polynomial_decay_rate=1.,
        weight_decay_rate=1e-1,
        num_warmup_steps=5000,
        num_decay_steps=51000,
        num_epochs=6):
  """Train and evaluate."""
  hparams = dict(
      learning_rate=learning_rate,
      num_decay_steps=num_decay_steps,
      num_warmup_steps=num_warmup_steps,
      num_epochs=num_epochs,
      weight_decay_rate=weight_decay_rate,
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      label_smoothing=FLAGS.label_smoothing)
  logging.info(hparams)
  learning_rate_schedule = nlp_optimization.WarmUp(
      learning_rate,
      tf.keras.optimizers.schedules.PolynomialDecay(
          learning_rate,
          num_decay_steps,
          end_learning_rate=0.,
          power=learning_rate_polynomial_decay_rate), num_warmup_steps)
  with strategy.scope():
    optimizer = nlp_optimization.AdamWeightDecay(
        learning_rate_schedule,
        weight_decay_rate=weight_decay_rate,
        epsilon=1e-6,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    model.compile(optimizer, loss=modeling.SpanOrCrossEntropyLoss())

  def init_fn(init_checkpoint_path):
    ckpt = tf.train.Checkpoint(encoder=model.encoder)
    ckpt.restore(init_checkpoint_path).assert_existing_objects_matched()

  with worker_context():
    ckpt_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(model=model, optimizer=optimizer),
        model_dir,
        max_to_keep=None,
        init_fn=(functools.partial(init_fn, init_checkpoint_path)
                 if init_checkpoint_path else None))
    with strategy.scope():
      ckpt_manager.restore_or_initialize()
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(model_dir, 'val'))
    best_exact_match = 0.
    for epoch in range(len(ckpt_manager.checkpoints), num_epochs):
      model.fit(
          train_dataset,
          callbacks=[
              tf.keras.callbacks.TensorBoard(model_dir, write_graph=False),
          ])
      ckpt_path = ckpt_manager.save()
      if evaluate_fn is None:
        continue
      metrics = evaluate_fn()
      logging.info('Epoch %d: %s', epoch + 1, metrics)
      if best_exact_match < metrics['exact_match']:
        best_exact_match = metrics['exact_match']
        model.save(os.path.join(model_dir, 'export'), include_optimizer=False)
        logging.info('Exporting %s as SavedModel.', ckpt_path)
      with val_summary_writer.as_default():
        for name, data in metrics.items():
          tf.summary.scalar(name, data, epoch + 1)


def evaluate(sp_processor, features_map_fn, labels_map_fn, logits_fn,
             decode_logits_fn, split_and_pad_fn, distribute_strategy,
             validation_dataset, ground_truth):
  """Run evaluation."""
  loss_metric = tf.keras.metrics.Mean()

  @tf.function
  def update_loss(y, logits):
    loss_fn = modeling.SpanOrCrossEntropyLoss(
        reduction=tf.keras.losses.Reduction.NONE)
    return loss_metric(loss_fn(y, logits))

  predictions = collections.defaultdict(list)
  for _, (features, labels) in validation_dataset.enumerate():
    token_ids = features['token_ids']
    y = labels_map_fn(token_ids, labels)
    x = split_and_pad_fn(features_map_fn(features))
    logits = tf.concat(
        distribute_strategy.experimental_local_results(logits_fn(x)), 0)
    logits = logits[:features['token_ids'].shape[0]]
    update_loss(y, logits)
    end_limit = token_ids.row_lengths() - 1  # inclusive
    begin, end, scores = decode_logits_fn(logits, end_limit)
    answers = prediction.decode_answer(features['context'], begin, end,
                                       features['token_offsets'],
                                       end_limit).numpy()
    for _, (qid, token_id, offset, score, answer) in enumerate(
        zip(features['qid'].numpy(),
            tf.gather(features['token_ids'], begin, batch_dims=1).numpy(),
            tf.gather(features['token_offsets'], begin, batch_dims=1).numpy(),
            scores, answers)):
      if not answer:
        continue
      if sp_processor.IdToPiece(int(token_id)).startswith('▁') and offset > 0:
        answer = answer[1:]
      predictions[qid.decode('utf-8')].append((score, answer.decode('utf-8')))
  predictions = {
      qid: evaluation.normalize_answer(
          sorted(answers, key=operator.itemgetter(0), reverse=True)[0][1])
      for qid, answers in predictions.items()
  }
  metrics = evaluation.evaluate_triviaqa(ground_truth, predictions, mute=True)
  metrics['loss'] = loss_metric.result().numpy()
  return metrics


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  gin.parse_config(FLAGS.gin_bindings)
  model_config = read_model_config(
      FLAGS.encoder,
      FLAGS.model_config_path,
      bigbird_block_size=FLAGS.bigbird_block_size)
  logging.info(model_config.get().as_dict())
  # Configure input processing.
  sp_processor = read_sentencepiece_model(FLAGS.sentencepiece_model_path)
  features_map_fn = functools.partial(
      inputs.features_map_fn,
      local_radius=FLAGS.bigbird_block_size,
      relative_pos_max_distance=24,
      use_hard_g2l_mask=True,
      padding_id=sp_processor.PieceToId('<pad>'),
      eos_id=sp_processor.PieceToId('</s>'),
      null_id=sp_processor.PieceToId('<empty>'),
      cls_id=sp_processor.PieceToId('<ans>'),
      sep_id=sp_processor.PieceToId('<sep_0>'))
  train_features_map_fn = tf.function(
      functools.partial(
          features_map_fn,
          sequence_length=FLAGS.train_sequence_length,
          global_sequence_length=FLAGS.train_global_sequence_length),
      autograph=False)
  train_labels_map_fn = tf.function(
      functools.partial(
          inputs.labels_map_fn, sequence_length=FLAGS.train_sequence_length))
  # Connect to TPU cluster.
  if FLAGS.master:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.master)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
  else:
    strategy = tf.distribute.MirroredStrategy()
  # Initialize datasets.
  with worker_context():
    _ = tf.random.get_global_generator()
    train_dataset = inputs.read_batches(
        FLAGS.data_dir,
        tfds.Split.TRAIN,
        FLAGS.batch_size,
        shuffle=True,
        drop_final_batch=True)
    validation_dataset = inputs.read_batches(FLAGS.data_dir,
                                             tfds.Split.VALIDATION,
                                             FLAGS.batch_size)

    def train_map_fn(x, y):
      features = train_features_map_fn(x)
      labels = modeling.smooth_labels(FLAGS.label_smoothing,
                                      train_labels_map_fn(x['token_ids'], y),
                                      features['question_lengths'],
                                      features['token_ids'])
      return features, labels

    train_dataset = train_dataset.map(train_map_fn, 16).prefetch(16)
  # Initialize model and compile.
  with strategy.scope():
    model = modeling.TriviaQaModel(model_config, FLAGS.train_sequence_length)
  logits_fn = tf.function(
      functools.partial(prediction.distributed_logits_fn, model))
  decode_logits_fn = tf.function(
      functools.partial(prediction.decode_logits, FLAGS.decode_top_k,
                        FLAGS.decode_max_size))
  split_and_pad_fn = tf.function(
      functools.partial(prediction.split_and_pad, strategy, FLAGS.batch_size))
  # Evaluation strategy.
  with tf.io.gfile.GFile(FLAGS.validation_gold_path) as f:
    ground_truth = {
        datum['QuestionId']: datum['Answer'] for datum in json.load(f)['Data']
    }
  validation_features_map_fn = tf.function(
      functools.partial(
          features_map_fn,
          sequence_length=FLAGS.validation_sequence_length,
          global_sequence_length=FLAGS.validation_global_sequence_length),
      autograph=False)
  validation_labels_map_fn = tf.function(
      functools.partial(
          inputs.labels_map_fn,
          sequence_length=FLAGS.validation_sequence_length))
  evaluate_fn = functools.partial(
      evaluate,
      sp_processor=sp_processor,
      features_map_fn=validation_features_map_fn,
      labels_map_fn=validation_labels_map_fn,
      logits_fn=logits_fn,
      decode_logits_fn=decode_logits_fn,
      split_and_pad_fn=split_and_pad_fn,
      distribute_strategy=strategy,
      validation_dataset=validation_dataset,
      ground_truth=ground_truth)
  logging.info('Model initialized. Beginning training fit loop.')
  fit(model, strategy, train_dataset, FLAGS.model_dir,
      FLAGS.init_checkpoint_path, evaluate_fn)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'model_config_path', 'model_dir', 'sentencepiece_model_path',
      'validation_gold_path'
  ])
  app.run(main)
