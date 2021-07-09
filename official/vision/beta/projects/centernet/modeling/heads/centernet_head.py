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

"""Contains the definitions of decoder for CenterNet."""
from typing import Any, Mapping, List

import tensorflow as tf

from official.vision.beta.projects.centernet.modeling.layers import nn_blocks


@tf.keras.utils.register_keras_serializable(package='centernet')
class CenterNetHead(tf.keras.Model):
  """
  CenterNet Decoder
  """
  
  def __init__(self,
               input_specs: List[tf.TensorShape],
               task_outputs: Mapping[str, int],
               heatmap_bias: float = -2.19,
               num_inputs: int = 2,
               **kwargs):
    """
    Args:
      input_specs: A `list` of input specifications.
      task_outputs: A `dict`, with key-value pairs denoting the names of the
        outputs and the desired channel depth of each output
      heatmap_bias: `float`, constant value to initialize the convolution layer
        bias vector if it is responsible for generating a heatmap (not for
        regressed predictions)
      num_inputs: `int`, indicates number of output branches from
        multiple backbone modules. For example, hourglass-104 has two
        hourglass-52 modules.

    call Returns:
      dictionary where the keys-value pairs denote the names of the output
      and the respective output tensor
    """
    self._input_specs = input_specs
    self._task_outputs = task_outputs
    self._heatmap_bias = heatmap_bias
    self._num_inputs = num_inputs
    
    inputs = [tf.keras.layers.Input(shape=value[1:])
              for value in self._input_specs]
    outputs = {}
    
    for key in self._task_outputs:
      outputs[key] = [
          nn_blocks.CenterNetDecoderConv(
              output_filters=self._task_outputs[key],
              bias_init=self._heatmap_bias if 'heatmaps' in key else 0,
              name=key + str(i),
          )(inputs[i])
          for i in range(self._num_inputs)
      ]
    
    self._output_specs = {
        key: [value[i].get_shape() for i in range(num_inputs)]
        for key, value in outputs.items()
    }
    
    super().__init__(inputs=inputs, outputs=outputs,
                     name='CenterNetDecoder', **kwargs)
  
  def get_config(self) -> Mapping[str, Any]:
    config = {
        'input_spec': self._input_specs,
        'task_outputs': self._task_outputs,
        'heatmap_bias': self._heatmap_bias,
        'num_inputs': self._num_inputs,
    }
    
    base_config = super(CenterNetHead, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  
  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
  
  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
