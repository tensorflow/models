# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""A script to export PointPillars model."""

from absl import app
from absl import flags
from absl import logging

from official.core import exp_factory
from official.modeling import hyperparams
from official.projects.pointpillars import registry_imports  # pylint: disable=unused-import
from official.projects.pointpillars.utils import model_exporter

_EXPERIMENT = flags.DEFINE_string(
    'experiment', None, 'experiment type, e.g. retinanet_resnetfpn_coco')
_EXPORT_DIR = flags.DEFINE_string('export_dir', None, 'The export directory.')
_CHECKPOINT_PATH = flags.DEFINE_string('checkpoint_path', None,
                                       'Checkpoint path.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size.')
_CONFIG_FILE = flags.DEFINE_string(
    'config_file',
    default=None,
    help='YAML/JSON files which specifies overrides.')
_TEST_INFERENCE = flags.DEFINE_boolean(
    'test_inference',
    default=False,
    help='True if want to load saved model and run inference.')


def main(_):
  params = exp_factory.get_exp_config(_EXPERIMENT.value)
  if _CONFIG_FILE.value:
    params = hyperparams.override_params_dict(
        params, _CONFIG_FILE.value, is_strict=True)
  params.validate()
  params.lock()

  model_exporter.export_inference_graph(
      batch_size=_BATCH_SIZE.value,
      params=params,
      checkpoint_path=_CHECKPOINT_PATH.value,
      export_dir=_EXPORT_DIR.value)
  logging.info('Successfully exported model to %s', _EXPORT_DIR.value)

  if _TEST_INFERENCE.value:
    predict_fn = model_exporter.load_model_predict_fn(_EXPORT_DIR.value)
    pillars, indices = model_exporter.random_input_tensors(
        batch_size=_BATCH_SIZE.value, params=params,
    )
    _ = predict_fn(pillars=pillars, indices=indices)
    logging.info('Successfully test model inference')

if __name__ == '__main__':
  app.run(main)
