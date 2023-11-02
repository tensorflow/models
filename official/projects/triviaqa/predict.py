# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""TriviaQA script for inference."""
import collections
import contextlib
import functools
import json
import operator

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf, tf_keras
import tensorflow_datasets as tfds

import sentencepiece as spm
from official.nlp.configs import encoders  # pylint: disable=unused-import
from official.projects.triviaqa import evaluation
from official.projects.triviaqa import inputs
from official.projects.triviaqa import prediction

flags.DEFINE_string('data_dir', None, 'TensorFlow Datasets directory.')

flags.DEFINE_enum('split', None,
                  [tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST],
                  'For which split to generate predictions.')

flags.DEFINE_string('predictions_path', None, 'Output for predictions.')

flags.DEFINE_string('sentencepiece_model_path', None,
                    'Path to sentence piece model.')

flags.DEFINE_integer('bigbird_block_size', 64,
                     'Size of blocks for sparse block attention.')

flags.DEFINE_string('saved_model_dir', None,
                    'Path from which to initialize model and weights.')

flags.DEFINE_integer('sequence_length', 4096, 'Maximum number of tokens.')

flags.DEFINE_integer('global_sequence_length', 320,
                     'Maximum number of global tokens.')

flags.DEFINE_integer('batch_size', 32, 'Size of batch.')

flags.DEFINE_string('master', '', 'Address of the TPU master.')

flags.DEFINE_integer('decode_top_k', 8,
                     'Maximum number of tokens to consider for begin/end.')

flags.DEFINE_integer('decode_max_size', 16,
                     'Maximum number of sentence pieces in an answer.')

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


def predict(sp_processor, features_map_fn, logits_fn, decode_logits_fn,
            split_and_pad_fn, distribute_strategy, dataset):
  """Make predictions."""
  predictions = collections.defaultdict(list)
  for _, features in dataset.enumerate():
    token_ids = features['token_ids']
    x = split_and_pad_fn(features_map_fn(features))
    logits = tf.concat(
        distribute_strategy.experimental_local_results(logits_fn(x)), 0)
    logits = logits[:features['token_ids'].shape[0]]
    end_limit = token_ids.row_lengths() - 1  # inclusive
    begin, end, scores = decode_logits_fn(logits, end_limit)
    answers = prediction.decode_answer(features['context'], begin, end,
                                       features['token_offsets'],
                                       end_limit).numpy()
    for j, (qid, token_id, offset, score, answer) in enumerate(
        zip(features['qid'].numpy(),
            tf.gather(features['token_ids'], begin, batch_dims=1).numpy(),
            tf.gather(features['token_offsets'], begin, batch_dims=1).numpy(),
            scores, answers)):
      if not answer:
        logging.info('%s: %s | NO_ANSWER, %f',
                     features['id'][j].numpy().decode('utf-8'),
                     features['question'][j].numpy().decode('utf-8'), score)
        continue
      if sp_processor.IdToPiece(int(token_id)).startswith('▁') and offset > 0:
        answer = answer[1:]
      logging.info('%s: %s | %s, %f', features['id'][j].numpy().decode('utf-8'),
                   features['question'][j].numpy().decode('utf-8'),
                   answer.decode('utf-8'), score)
      predictions[qid.decode('utf-8')].append((score, answer.decode('utf-8')))
  predictions = {
      qid: evaluation.normalize_answer(
          sorted(answers, key=operator.itemgetter(0), reverse=True)[0][1])
      for qid, answers in predictions.items()
  }
  return predictions


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Configure input processing.
  sp_processor = read_sentencepiece_model(FLAGS.sentencepiece_model_path)
  features_map_fn = tf.function(
      functools.partial(
          inputs.features_map_fn,
          local_radius=FLAGS.bigbird_block_size,
          relative_pos_max_distance=24,
          use_hard_g2l_mask=True,
          sequence_length=FLAGS.sequence_length,
          global_sequence_length=FLAGS.global_sequence_length,
          padding_id=sp_processor.PieceToId('<pad>'),
          eos_id=sp_processor.PieceToId('</s>'),
          null_id=sp_processor.PieceToId('<empty>'),
          cls_id=sp_processor.PieceToId('<ans>'),
          sep_id=sp_processor.PieceToId('<sep_0>')),
      autograph=False)
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
    dataset = inputs.read_batches(
        FLAGS.data_dir, FLAGS.split, FLAGS.batch_size, include_answers=False)
  # Initialize model and compile.
  with strategy.scope():
    model = tf_keras.models.load_model(FLAGS.saved_model_dir, compile=False)
  logging.info('Model initialized. Beginning prediction loop.')
  logits_fn = tf.function(
      functools.partial(prediction.distributed_logits_fn, model))
  decode_logits_fn = tf.function(
      functools.partial(prediction.decode_logits, FLAGS.decode_top_k,
                        FLAGS.decode_max_size))
  split_and_pad_fn = tf.function(
      functools.partial(prediction.split_and_pad, strategy, FLAGS.batch_size))
  # Prediction strategy.
  predict_fn = functools.partial(
      predict,
      sp_processor=sp_processor,
      features_map_fn=features_map_fn,
      logits_fn=logits_fn,
      decode_logits_fn=decode_logits_fn,
      split_and_pad_fn=split_and_pad_fn,
      distribute_strategy=strategy,
      dataset=dataset)
  with worker_context():
    predictions = predict_fn()
  with tf.io.gfile.GFile(FLAGS.predictions_path, 'w') as f:
    json.dump(predictions, f)


if __name__ == '__main__':
  flags.mark_flags_as_required(['split', 'predictions_path', 'saved_model_dir'])
  app.run(main)
