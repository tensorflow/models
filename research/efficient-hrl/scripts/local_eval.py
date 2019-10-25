# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Script to run run_eval.py locally.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from subprocess import call
import sys

CONFIGS_PATH = 'configs'
CONTEXT_CONFIGS_PATH = 'context/configs'

def main():
  bb = './'
  base_num_args = 6
  if len(sys.argv) < base_num_args:
    print(
        "usage: python %s <exp_name> <context_setting_gin> <context_gin> "
        "<agent_gin> <suite> [params...]"
        % sys.argv[0])
    sys.exit(0)
  exp = sys.argv[1]
  context_setting = sys.argv[2]
  context = sys.argv[3]
  agent = sys.argv[4]
  assert sys.argv[5] in ["suite"], "args[5] must be `suite'"
  suite = ""
  binary = "python {bb}/run_eval{suite}.py ".format(bb=bb, suite=suite)

  h = os.environ["HOME"]
  ucp = CONFIGS_PATH
  ccp = CONTEXT_CONFIGS_PATH
  extra = ''
  command_str = ("{binary} "
                 "--logtostderr "
                 "--checkpoint_dir={h}/tmp/{context_setting}/{context}/{agent}/{exp}/train "
                 "--eval_dir={h}/tmp/{context_setting}/{context}/{agent}/{exp}/eval "
                 "--config_file={ucp}/{agent}.gin "
                 "--config_file={ucp}/eval_{extra}uvf.gin "
                 "--config_file={ccp}/{context_setting}.gin "
                 "--config_file={ccp}/{context}.gin ").format(
                     h=h,
                     ucp=ucp,
                     ccp=ccp,
                     context_setting=context_setting,
                     context=context,
                     agent=agent,
                     extra=extra,
                     suite=suite,
                     exp=exp,
                     binary=binary)
  for extra_arg in sys.argv[base_num_args:]:
    command_str += "--params='%s' " % extra_arg

  print(command_str)
  call(command_str, shell=True)


if __name__ == "__main__":
  main()
