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

r"""Training driver.

To train:

CONFIG_FILE=official/projects/movinet/configs/yaml/movinet_a0_k600_8x8.yaml
python3 official/projects/movinet/train.py \
    --experiment=movinet_kinetics600 \
    --mode=train \
    --model_dir=/tmp/movinet/ \
    --config_file=${CONFIG_FILE} \
    --params_override="" \
    --gin_file="" \
    --gin_params="" \
    --tpu="" \
    --tf_data_service=""
"""

from absl import app
from absl import flags
import gin

from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance
# Import movinet libraries to register the backbone and model into tf.vision
# model garden factory.
# pylint: disable=unused-import
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.vision import registry_imports
# pylint: enable=unused-import

FLAGS = flags.FLAGS


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  if 'train_and_eval' in FLAGS.mode:
    assert (params.task.train_data.feature_shape ==
            params.task.validation_data.feature_shape), (
                f'train {params.task.train_data.feature_shape} != validate '
                f'{params.task.validation_data.feature_shape}')

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
