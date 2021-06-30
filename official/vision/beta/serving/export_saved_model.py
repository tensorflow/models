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

# Lint as: python3
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
model_fn = imported.signatures['serving_default']
output = model_fn(input_images)
"""

from absl import app
from absl import flags

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.serving import export_saved_model_lib

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
flags.DEFINE_string('export_checkpoint_subdir', 'checkpoint',
                    'The subdirectory for checkpoints.')
flags.DEFINE_string('export_saved_model_subdir', 'saved_model',
                    'The subdirectory for saved model.')


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

  export_saved_model_lib.export_inference_graph(
      input_type=FLAGS.input_type,
      batch_size=FLAGS.batch_size,
      input_image_size=[int(x) for x in FLAGS.input_image_size.split(',')],
      params=params,
      checkpoint_path=FLAGS.checkpoint_path,
      export_dir=FLAGS.export_dir,
      export_checkpoint_subdir=FLAGS.export_checkpoint_subdir,
      export_saved_model_subdir=FLAGS.export_saved_model_subdir)


if __name__ == '__main__':
  app.run(main)
