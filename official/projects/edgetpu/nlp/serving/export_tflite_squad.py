# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=line-too-long
r"""Export tflite for MobileBERT-EdgeTPU with SQUAD head.

Example usage:

python3 export_tflite_squad.py \
--config_file=official/projects/edgetpu/nlp/experiments/mobilebert_edgetpu_xs.yaml \
--export_path=/tmp/ \
--quantization_method=full-integer
"""
# pylint: enable=line-too-long
import os
import tempfile
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import orbit
import tensorflow as tf

from official.common import flags as tfm_flags
from official.nlp.data import data_loader_factory
from official.nlp.data import question_answering_dataloader
from official.nlp.modeling import models
from official.projects.edgetpu.nlp.configs import params
from official.projects.edgetpu.nlp.modeling import model_builder
from official.projects.edgetpu.nlp.utils import utils


FLAGS = flags.FLAGS
SQUAD_TRAIN_SPLIT = 'gs://**/tp/bert/squad_v1.1/train.tf_record'

flags.DEFINE_string('export_path', '/tmp/',
                    'File path to store tflite model.')
flags.DEFINE_enum('quantization_method', 'float',
                  ['full-integer', 'hybrid', 'float'], 'Quantization method.')
flags.DEFINE_integer('batch_size', 1,
                     'Fixed batch size for exported TFLite model.')
flags.DEFINE_integer('sequence_length', 384,
                     'Fixed sequence length.')
flags.DEFINE_string('model_checkpoint', None,
                    'Checkpoint path for the model. Model will be initialized'
                    'with random weights if path is None.')


def build_model_for_serving(model: tf.keras.Model,
                            sequence_length: int = 384,
                            batch_size: int = 1) -> tf.keras.Model:
  """Builds MLPerf evaluation compatible models.

  To run the model on device, the model input/output datatype and node names
  need to match the MLPerf setup.

  Args:
    model: Input keras model.
    sequence_length: BERT model sequence length.
    batch_size: Inference batch size.
  Returns:
    Keras model with new input/output nodes.
  """
  word_ids = tf.keras.Input(shape=(sequence_length,),
                            batch_size=batch_size,
                            dtype=tf.int32,
                            name='input_word_ids')
  mask = tf.keras.Input(shape=(sequence_length,),
                        batch_size=batch_size,
                        dtype=tf.int32, name='input_mask')
  type_ids = tf.keras.Input(shape=(sequence_length,),
                            batch_size=batch_size,
                            dtype=tf.int32, name='input_type_ids')
  model_output = model([word_ids, type_ids, mask])

  # Use identity layers wrapped in lambdas to explicitly name the output
  # tensors.
  start_logits = tf.keras.layers.Lambda(
      tf.identity, name='start_positions')(
          model_output[0])
  end_logits = tf.keras.layers.Lambda(
      tf.identity, name='end_positions')(
          model_output[1])
  model = tf.keras.Model(
      inputs=[word_ids, type_ids, mask],
      outputs=[start_logits, end_logits])

  return model


def build_inputs(data_params, input_context=None):
  """Returns tf.data.Dataset for sentence_prediction task."""
  return data_loader_factory.get_data_loader(data_params).load(input_context)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Set up experiment params and load the configs from file/files.
  experiment_params = params.EdgeTPUBERTCustomParams()
  experiment_params = utils.config_override(experiment_params, FLAGS)

  # change the input mask type to tf.float32 to avoid additional casting op.
  experiment_params.student_model.encoder.mobilebert.input_mask_dtype = 'float32'

  # Experiments indicate using -120 as the mask value for Softmax is good enough
  # for both int8 and bfloat. So we set quantization_friendly to True for both
  # quant and float model.
  pretrainer_model = model_builder.build_bert_pretrainer(
      experiment_params.student_model,
      name='pretrainer',
      quantization_friendly=True)

  encoder_network = pretrainer_model.encoder_network
  model = models.BertSpanLabeler(
      network=encoder_network,
      initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))

  # Load model weights.
  if FLAGS.model_checkpoint is not None:
    checkpoint_dict = {'model': model}
    checkpoint = tf.train.Checkpoint(**checkpoint_dict)
    checkpoint.restore(FLAGS.model_checkpoint).assert_existing_objects_matched()

  model_for_serving = build_model_for_serving(model)
  model_for_serving.summary()

  # TODO(b/194449109): Need to save the model to file and then convert tflite
  # with 'tf.lite.TFLiteConverter.from_saved_model()' to get the expected
  # accuracy
  tmp_dir = tempfile.TemporaryDirectory().name
  model_for_serving.save(tmp_dir)

  def _representative_dataset():
    dataset_params = question_answering_dataloader.QADataConfig()
    dataset_params.input_path = SQUAD_TRAIN_SPLIT
    dataset_params.drop_remainder = False
    dataset_params.global_batch_size = 1
    dataset_params.is_training = True

    dataset = orbit.utils.make_distributed_dataset(tf.distribute.get_strategy(),
                                                   build_inputs, dataset_params)
    for example in dataset.take(100):
      inputs = example[0]
      input_word_ids = inputs['input_word_ids']
      input_mask = inputs['input_mask']
      input_type_ids = inputs['input_type_ids']
      yield [input_word_ids, input_mask, input_type_ids]

  converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
  if FLAGS.quantization_method in ['full-integer', 'hybrid']:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
  if FLAGS.quantization_method in ['full-integer']:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    converter.representative_dataset = _representative_dataset

  tflite_quant_model = converter.convert()
  export_model_path = os.path.join(FLAGS.export_path, 'model.tflite')
  with tf.io.gfile.GFile(export_model_path, 'wb') as f:
    f.write(tflite_quant_model)
  logging.info('Successfully save the tflite to %s', FLAGS.export_path)


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
