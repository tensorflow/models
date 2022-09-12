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

r"""Panoptic MaskRCNN model export binary for serving/inference.

To export a trained checkpoint in saved_model format (shell script):

CHECKPOINT_PATH = XX
EXPORT_DIR_PATH = XX
CONFIG_FILE_PATH = XX
export_saved_model --export_dir=${EXPORT_DIR_PATH}/ \
                   --checkpoint_path=${CHECKPOINT_PATH} \
                   --config_file=${CONFIG_FILE_PATH} \
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
import tensorflow as tf

from official.core import exp_factory
from official.modeling import hyperparams
# pylint: disable=unused-import
from official.projects.panoptic.configs import panoptic_deeplab as panoptic_deeplab_cfg
from official.projects.panoptic.configs import panoptic_maskrcnn as panoptic_maskrcnn_cfg
# pylint: enable=unused-import
from official.projects.panoptic.modeling import factory
from official.projects.panoptic.serving import panoptic_deeplab
from official.projects.panoptic.serving import panoptic_maskrcnn
# pylint: disable=unused-import
from official.projects.panoptic.tasks import panoptic_deeplab as panoptic_deeplab_task
from official.projects.panoptic.tasks import panoptic_maskrcnn as panoptic_maskrcnn_task
# pylint: enable=unused-import
from official.vision.serving import export_saved_model_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'panoptic_maskrcnn',
                    'model type, one of panoptic_maskrcnn and panoptic_deeplab')
flags.DEFINE_string('experiment', 'panoptic_fpn_coco',
                    'experiment type, e.g. panoptic_fpn_coco')
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
flags.DEFINE_integer('batch_size', None, 'The batch size.')
flags.DEFINE_string('input_type', 'image_tensor',
                    'One of `image_tensor`, `image_bytes`, `tf_example`.')
flags.DEFINE_string(
    'input_image_size', '224,224',
    'The comma-separated string of two integers representing the height,width '
    'of the input to the model.')


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

  input_image_size = [int(x) for x in FLAGS.input_image_size.split(',')]
  input_specs = tf.keras.layers.InputSpec(
      shape=[FLAGS.batch_size, *input_image_size, 3])

  if FLAGS.model == 'panoptic_deeplab':
    build_model = factory.build_panoptic_deeplab
    panoptic_module = panoptic_deeplab.PanopticSegmentationModule
  elif FLAGS.model == 'panoptic_maskrcnn':
    build_model = factory.build_panoptic_maskrcnn
    panoptic_module = panoptic_maskrcnn.PanopticSegmentationModule
  else:
    raise ValueError('Unsupported model type: %s' % FLAGS.model)

  model = build_model(input_specs=input_specs, model_config=params.task.model)
  export_module = panoptic_module(
      params=params,
      model=model,
      batch_size=FLAGS.batch_size,
      input_image_size=[int(x) for x in FLAGS.input_image_size.split(',')],
      num_channels=3)
  export_saved_model_lib.export_inference_graph(
      input_type=FLAGS.input_type,
      batch_size=FLAGS.batch_size,
      input_image_size=input_image_size,
      params=params,
      checkpoint_path=FLAGS.checkpoint_path,
      export_dir=FLAGS.export_dir,
      export_module=export_module,
      export_checkpoint_subdir='checkpoint',
      export_saved_model_subdir='saved_model')

if __name__ == '__main__':
  app.run(main)
