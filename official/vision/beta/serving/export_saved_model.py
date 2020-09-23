# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Vision models export binary for serving/inference.

To export a trained checkpoint in saved_model format (shell script):

EXPERIMENT_TYPE = XX
CHECKPOINT_PATH = XX
EXPORT_DIR_PATH = XX
export_saved_model --experiment=${EXPERIMENT_TYPE} \
                   --export_dir=${EXPORT_DIR_PATH}/ \
                   --checkpoint_path=${CHECKPOINT_PATH} \
                   --batch_size=2 \
                   --input_image_size=224,224

To serve (python):

export_dir_path = XX
input_type = XX
input_images = XX
imported = tf.saved_model.load(export_dir_path)
model_fn = .signatures['serving_default']
output = model_fn(input_images)
"""

import os

from absl import app
from absl import flags
import tensorflow.compat.v2 as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.core import train_utils
from official.modeling import hyperparams
from official.vision.beta import configs
from official.vision.beta.serving import image_classification

FLAGS = flags.FLAGS


flags.DEFINE_string(
    'experiment', None, 'experiment type, e.g. retinanet_resnetfpn_coco')
flags.DEFINE_string('export_dir', None, 'The export directory.')
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path.')
flags.DEFINE_multi_string(
    'config_file',
    default=None,
    help='YAML/JSON files which specifies overrides. The override order '
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
    'input_type', 'image_tensor',
    'One of `image_tensor`, `image_bytes`, `tf_example`.')
flags.DEFINE_string(
    'input_image_size', '224,224',
    'The comma-separated string of two integers representing the height,width '
    'of the input to the model.')


def export_inference_graph(input_type, batch_size, input_image_size, params,
                           checkpoint_path, export_dir):
  """Exports inference graph for the model specified in the exp config.

  Saved model is stored at export_dir/saved_model, checkpoint is saved
  at export_dir/checkpoint, and params is saved at export_dir/params.yaml.

  Args:
    input_type: One of `image_tensor`, `image_bytes`, `tf_example`.
    batch_size: 'int', or None.
    input_image_size: List or Tuple of height and width.
    params: Experiment params.
    checkpoint_path: Trained checkpoint path or directory.
    export_dir: Export directory path.
  """

  output_checkpoint_directory = os.path.join(export_dir, 'checkpoint')
  output_saved_model_directory = os.path.join(export_dir, 'saved_model')

  if isinstance(params.task,
                configs.image_classification.ImageClassificationTask):
    export_module = image_classification.ClassificationModule(
        params=params,
        batch_size=batch_size,
        input_image_size=input_image_size)
  else:
    raise ValueError('Export module not implemented for {} task.'.format(
        type(params.task)))

  model = export_module.build_model()

  ckpt = tf.train.Checkpoint(model=model)

  ckpt_dir_or_file = checkpoint_path
  if tf.io.gfile.isdir(ckpt_dir_or_file):
    ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
  status = ckpt.restore(ckpt_dir_or_file).expect_partial()

  if input_type == 'image_tensor':
    input_signature = tf.TensorSpec(
        shape=[batch_size, input_image_size[0], input_image_size[1], 3],
        dtype=tf.uint8)
    signatures = {
        'serving_default':
            export_module.inference_from_image.get_concrete_function(
                input_signature)
    }
  elif input_type == 'image_bytes':
    input_signature = tf.TensorSpec(shape=[batch_size], dtype=tf.string)
    signatures = {
        'serving_default':
            export_module.inference_from_image_bytes.get_concrete_function(
                input_signature)
    }
  elif input_type == 'tf_example':
    input_signature = tf.TensorSpec(shape=[batch_size], dtype=tf.string)
    signatures = {
        'serving_default':
            export_module.inference_from_tf_example.get_concrete_function(
                input_signature)
    }
  else:
    raise ValueError('Unrecognized `input_type`')

  status.assert_existing_objects_matched()

  ckpt.save(os.path.join(output_checkpoint_directory, 'ckpt'))

  tf.saved_model.save(export_module,
                      output_saved_model_directory,
                      signatures=signatures)

  train_utils.serialize_config(params, export_dir)


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

  export_inference_graph(
      input_type=FLAGS.input_type,
      batch_size=FLAGS.batch_size,
      input_image_size=[int(x) for x in FLAGS.input_image_size.split(',')],
      params=params,
      checkpoint_path=FLAGS.checkpoint_path,
      export_dir=FLAGS.export_dir)


if __name__ == '__main__':
  app.run(main)
