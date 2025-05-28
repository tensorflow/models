# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

r"""Volumetric model export binary for serving/inference.

To export a trained checkpoint in saved_model format (shell script):

EXPERIMENT_TYPE = XX
CHECKPOINT_PATH = XX
EXPORT_DIR_PATH = XX
export_saved_model --experiment=${EXPERIMENT_TYPE} \
                   --export_dir=${EXPORT_DIR_PATH}/ \
                   --checkpoint_path=${CHECKPOINT_PATH} \
                   --batch_size=1 \
                   --input_image_size=128,128,128 \
                   --num_channels=1

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
from official.projects.volumetric_models.serving import semantic_segmentation_3d
from official.vision.serving import export_saved_model_lib

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
flags.DEFINE_list(
    'input_image_size', None,
    'The comma-separated string of three integers representing the '
    'height, width and depth of the input to the model.')
flags.DEFINE_integer('num_channels', 1,
                     'The number of channels of input image.')

flags.register_validator(
    'input_image_size',
    lambda value: value is not None and len(value) == 3,
    message='--input_image_size must be comma-separated string of three '
    'integers representing the height, width and depth of the input to '
    'the model.')


def main(_):
  flags.mark_flag_as_required('export_dir')
  flags.mark_flag_as_required('checkpoint_path')

  params = exp_factory.get_exp_config(FLAGS.experiment)
  for config_file in FLAGS.config_file or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)
  if FLAGS.params_override:
    params = hyperparams.override_params_dict(
        params, FLAGS.params_override, is_strict=True)

  params.validate()
  params.lock()

  input_image_size = FLAGS.input_image_size

  export_module = semantic_segmentation_3d.SegmentationModule(
      params=params,
      batch_size=1,
      input_image_size=input_image_size,
      num_channels=FLAGS.num_channels)

  export_saved_model_lib.export_inference_graph(
      input_type=FLAGS.input_type,
      batch_size=FLAGS.batch_size,
      input_image_size=input_image_size,
      params=params,
      checkpoint_path=FLAGS.checkpoint_path,
      export_dir=FLAGS.export_dir,
      num_channels=FLAGS.num_channels,
      export_module=export_module,
      export_checkpoint_subdir='checkpoint',
      export_saved_model_subdir='saved_model')


if __name__ == '__main__':
  app.run(main)
