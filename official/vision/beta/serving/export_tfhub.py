# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""A script to export the image classification as a TF-Hub SavedModel."""

# Import libraries
from absl import app
from absl import flags

import tensorflow as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.modeling import factory


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'experiment', None, 'experiment type, e.g. resnet_imagenet')
flags.DEFINE_string(
    'checkpoint_path', None, 'Checkpoint path.')
flags.DEFINE_string(
    'export_path', None, 'The export directory.')
flags.DEFINE_multi_string(
    'config_file',
    None,
    'A YAML/JSON files which specifies overrides. The override order '
    'follows the order of args. Note that each file '
    'can be used as an override template to override the default parameters '
    'specified in Python. If the same parameter is specified in both '
    '`--config_file` and `--params_override`, `config_file` will be used '
    'first, followed by params_override.')
flags.DEFINE_string(
    'params_override', '',
    'The JSON/YAML file or string which specifies the parameter to be overriden'
    ' on top of `config_file` template.')
flags.DEFINE_integer(
    'batch_size', None, 'The batch size.')
flags.DEFINE_string(
    'input_image_size',
    '224,224',
    'The comma-separated string of two integers representing the height,width '
    'of the input to the model.')
flags.DEFINE_boolean(
    'skip_logits_layer',
    False,
    'Whether to skip the prediction layer and only output the feature vector.')


def export_model_to_tfhub(params,
                          batch_size,
                          input_image_size,
                          skip_logits_layer,
                          checkpoint_path,
                          export_path):
  """Export an image classification model to TF-Hub."""
  input_specs = tf.keras.layers.InputSpec(shape=[batch_size] +
                                          input_image_size + [3])

  model = factory.build_classification_model(
      input_specs=input_specs,
      model_config=params.task.model,
      l2_regularizer=None,
      skip_logits_layer=skip_logits_layer)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
  model.save(export_path, include_optimizer=False, save_format='tf')


def main(_):
  params = exp_factory.get_exp_config(FLAGS.experiment)
  for config_file in FLAGS.config_file or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)
  if FLAGS.params_override:
    params = hyperparams.override_params_dict(
        params, FLAGS.params_override, is_strict=True)
  params.validate()
  params.lock()

  export_model_to_tfhub(
      params=params,
      batch_size=FLAGS.batch_size,
      input_image_size=[int(x) for x in FLAGS.input_image_size.split(',')],
      skip_logits_layer=FLAGS.skip_logits_layer,
      checkpoint_path=FLAGS.checkpoint_path,
      export_path=FLAGS.export_path)


if __name__ == '__main__':
  app.run(main)
