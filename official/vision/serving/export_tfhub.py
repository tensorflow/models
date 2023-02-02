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

from official.core import exp_factory
from official.modeling import hyperparams
from official.vision import registry_imports  # pylint: disable=unused-import
from official.vision.serving import export_tfhub_lib

FLAGS = flags.FLAGS

_EXPERIMENT = flags.DEFINE_string(
    'experiment', None, 'experiment type, e.g. retinanet_resnetfpn_coco'
)
_EXPORT_DIR = flags.DEFINE_string('export_dir', None, 'The export directory.')
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path', None, 'Checkpoint path.'
)
_CONFIG_FILE = flags.DEFINE_multi_string(
    'config_file',
    default=None,
    help=(
        'YAML/JSON files which specifies overrides. The override order follows'
        ' the order of args. Note that each file can be used as an override'
        ' template to override the default parameters specified in Python. If'
        ' the same parameter is specified in both `--config_file` and'
        ' `--params_override`, `config_file` will be used first, followed by'
        ' params_override.'
    ),
)
_PARAMS_OVERRIDE = flags.DEFINE_string(
    'params_override',
    '',
    (
        'The JSON/YAML file or string which specifies the parameter to be'
        ' overriden on top of `config_file` template.'
    ),
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'The batch size.')
_INPUT_IMAGE_SIZE = flags.DEFINE_string(
    'input_image_size',
    '224,224',
    (
        'The comma-separated string of two integers representing the'
        ' height,width of the input to the model.'
    ),
)
_SKIP_LOGITS_LAYER = flags.DEFINE_boolean(
    'skip_logits_layer',
    False,
    'Whether to skip the prediction layer and only output the feature vector.',
)


def main(_):
  params = exp_factory.get_exp_config(_EXPERIMENT.value)
  for config_file in _CONFIG_FILE.value or []:
    try:
      params = hyperparams.override_params_dict(
          params, config_file, is_strict=True
      )
    except KeyError:
      params = hyperparams.override_params_dict(
          params, config_file, is_strict=False
      )
  if _PARAMS_OVERRIDE.value:
    try:
      params = hyperparams.override_params_dict(
          params, _PARAMS_OVERRIDE.value, is_strict=True
      )
    except KeyError:
      params = hyperparams.override_params_dict(
          params, _PARAMS_OVERRIDE.value, is_strict=False
      )
  params.validate()
  params.lock()

  export_tfhub_lib.export_model_to_tfhub(
      params=params,
      batch_size=_BATCH_SIZE.value,
      input_image_size=[int(x) for x in _INPUT_IMAGE_SIZE.value.split(',')],
      checkpoint_path=_CHECKPOINT_PATH.value,
      export_path=_EXPORT_DIR.value,
      num_channels=3,
      skip_logits_layer=_SKIP_LOGITS_LAYER.value,
  )


if __name__ == '__main__':
  app.run(main)
