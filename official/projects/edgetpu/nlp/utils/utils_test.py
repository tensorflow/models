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

"""Tests for utils.py."""

from absl import flags
import tensorflow as tf
import yaml

from official.projects.edgetpu.nlp.configs import params
from official.projects.edgetpu.nlp.modeling import model_builder
from official.projects.edgetpu.nlp.utils import utils

FLAGS = flags.FLAGS


# Helper function to compare two nested Dicts.
# Note that this function only ensures all the fields in dict_a have definition
# and same value in dict_b. This function does not guarantee that
# dict_a == dict_b.
def nested_dict_compare(dict_a, dict_b):
  for k, v in sorted(dict_a.items()):
    if k not in dict_b:
      return False
    if isinstance(v, dict) and isinstance(dict_b[k], dict):
      if not nested_dict_compare(dict_a[k], dict_b[k]):
        return False
    else:
      # A caveat: When dict_a[k] = 1, dict_b[k] = True, the return is True.
      if dict_a[k] != dict_b[k]:
        return False
  return True


class UtilsTest(tf.test.TestCase):

  def test_config_override(self):
    # Define several dummy flags which are call by the utils.config_override
    # function.
    file_path = 'third_party/tensorflow_models/official/projects/edgetpu/nlp/experiments/mobilebert_edgetpu_m.yaml'
    flags.DEFINE_string('tpu', None, 'tpu_address.')
    flags.DEFINE_list('config_file', [file_path],
                      'A list of config files path.')
    flags.DEFINE_string('params_override', None, 'Override params.')
    flags.DEFINE_string('model_dir', '/tmp/', 'Model saving directory.')
    flags.DEFINE_list('mode', ['train'], 'Job mode.')
    flags.DEFINE_bool('use_vizier', False,
                      'Whether to enable vizier based hyperparameter search.')
    experiment_params = params.EdgeTPUBERTCustomParams()
    experiment_params = utils.config_override(experiment_params, FLAGS)
    experiment_params_dict = experiment_params.as_dict()

    with tf.io.gfile.GFile(file_path, 'r') as f:
      loaded_dict = yaml.load(f, Loader=yaml.FullLoader)

    # experiment_params contains all the configs but the loaded_dict might
    # only contains partial of the configs.
    self.assertTrue(nested_dict_compare(loaded_dict, experiment_params_dict))

  def test_load_checkpoint(self):
    """Test the pretrained model can be successfully loaded."""
    experiment_params = params.EdgeTPUBERTCustomParams()
    student_pretrainer = experiment_params.student_model
    student_pretrainer.encoder.type = 'mobilebert'
    pretrainer = model_builder.build_bert_pretrainer(
        pretrainer_cfg=student_pretrainer,
        name='test_model')
    # Makes sure the pretrainer variables are created.
    checkpoint_path = self.create_tempfile().full_path
    _ = pretrainer(pretrainer.inputs)
    pretrainer.save_weights(checkpoint_path)

    utils.load_checkpoint(pretrainer, checkpoint_path)


if __name__ == '__main__':
  tf.test.main()
