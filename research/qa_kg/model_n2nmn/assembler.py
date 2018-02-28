# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

import numpy as np

# the number of attention input to each module
_module_input_num = {
    '_key_find': 0,
    '_key_filter': 1,
    '_val_desc': 1}
_module_output_type = {
    '_key_find': 'att',
    '_key_filter': 'att',
    '_val_desc': 'ans'
}

INVALID_EXPR = 'INVALID_EXPR'


class Assembler:

  def __init__(self, config):
    # read the module list, and record the index of each module and <eos>
    self.module_names = config.module_names
    # find the index of <eos>
    for n_s in range(len(self.module_names)):
      if self.module_names[n_s] == '<eos>':
        self.EOS_idx = n_s
        break
    # build a dictionary from module name to token index
    self.name2idx_dict = {
        name: n_s
        for n_s, name in enumerate(self.module_names)
    }

  def module_list2tokens(self, module_list, max_len=None):
    layout_tokens = [self.name2idx_dict[name] for name in module_list]
    if max_len is not None:
      if len(module_list) >= max_len:
        raise ValueError('Not enough time steps to add <eos>')
      layout_tokens += [self.EOS_idx] * (max_len - len(module_list))
    return layout_tokens

  def _layout_tokens2str(self, layout_tokens):
    return ' '.join([self.module_names[idx] for idx in layout_tokens])

  def _invalid_expr(self, layout_tokens, error_str):
    return {
        'module': INVALID_EXPR,
        'expr_str': self._layout_tokens2str(layout_tokens),
        'error': error_str
    }

  def _assemble_layout_tokens(self, layout_tokens, batch_idx):
    # Every module takes a time_idx as the index from LSTM hidden states
    # (even if it doesn't need it, like _and), and different arity of
    # attention inputs. The output type can be either attention or answer
    #
    # The final assembled expression for each instance is as follows:
    # expr_type :=
    #    {'module': '_find',        'output_type': 'att', 'time_idx': idx}
    #  | {'module': '_relocate',   'output_type': 'att', 'time_idx': idx,
    #     'inputs_0': <expr_type>}
    #  | {'module': '_and',         'output_type': 'att', 'time_idx': idx,
    #     'inputs_0': <expr_type>,  'inputs_1': <expr_type>)}
    #  | {'module': '_describe',      'output_type': 'ans', 'time_idx': idx,
    #     'inputs_0': <expr_type>}
    #  | {'module': INVALID_EXPR, 'expr_str': '...', 'error': '...',
    #     'assembly_loss': <float32>} (for invalid expressions)
    #

    # A valid layout must contain <eos>. Assembly fails if it doesn't.
    if not np.any(layout_tokens == self.EOS_idx):
      return self._invalid_expr(layout_tokens, 'cannot find <eos>')

    # Decoding Reverse Polish Notation with a stack
    decoding_stack = []
    for t in range(len(layout_tokens)):
      # decode a module/operation
      module_idx = layout_tokens[t]
      if module_idx == self.EOS_idx:
        break
      module_name = self.module_names[module_idx]
      expr = {
          'module': module_name,
          'output_type': _module_output_type[module_name],
          'time_idx': t,
          'batch_idx': batch_idx
      }

      input_num = _module_input_num[module_name]
      # Check if there are enough input in the stack
      if len(decoding_stack) < input_num:
        # Invalid expression. Not enough input.
        return self._invalid_expr(layout_tokens,
                                  'not enough input for ' + module_name)

      # Get the input from stack
      for n_input in range(input_num - 1, -1, -1):
        stack_top = decoding_stack.pop()
        if stack_top['output_type'] != 'att':
          # Invalid expression. Input must be attention
          return self._invalid_expr(layout_tokens,
                                    'input incompatible for ' + module_name)
        expr['input_%d' % n_input] = stack_top

      decoding_stack.append(expr)

    # After decoding the reverse polish expression, there should be exactly
    # one expression in the stack
    if len(decoding_stack) != 1:
      return self._invalid_expr(
          layout_tokens,
          'final stack size not equal to 1 (%d remains)' % len(decoding_stack))

    result = decoding_stack[0]
    # The result type should be answer, not attention
    if result['output_type'] != 'ans':
      return self._invalid_expr(layout_tokens,
                                'result type must be ans, not att')
    return result

  def assemble(self, layout_tokens_batch):
    # layout_tokens_batch is a numpy array with shape [max_dec_len, batch_size],
    # containing module tokens and <eos>, in Reverse Polish Notation.
    _, batch_size = layout_tokens_batch.shape
    expr_list = [
        self._assemble_layout_tokens(layout_tokens_batch[:, batch_i], batch_i)
        for batch_i in range(batch_size)
    ]
    expr_validity = np.array(
        [expr['module'] != INVALID_EXPR for expr in expr_list], np.bool)
    return expr_list, expr_validity
