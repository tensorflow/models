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

"""Script to run run_train.py locally.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import random
from subprocess import call

CONFIGS_PATH = './configs'
CONTEXT_CONFIGS_PATH = './context/configs'

def main(args):
  h = os.environ["HOME"]
  ucp = CONFIGS_PATH
  ccp = CONTEXT_CONFIGS_PATH
  port = random.randint(2000, 8000)

  train_script = "python run_train_{}.py".format(args.train_script_version)
  command_str = (train_script + " "  # don't forget the little spaces after each argument
                 "--exp_dir={h}/tmp/{algo_context}/{env_context}/{agent_context}/{exp} "
                 "--config_file={ucp}/{agent_context}.gin "
                 "--config_file={ucp}/train_uvf.gin "
                 "--config_file={ccp}/{algo_context}.gin "
                 "--config_file={ccp}/{env_context}.gin "
                 "--summarize_gradients=False "
                 "--save_interval_secs={save_interval_secs} "
                 "--save_summaries_secs={save_summaries_secs} "  # this annoys debugging
                 "--s3_save_policy_path={s3_save_policy_path} "
                 "--master=local "
                 "--alsologtostderr ").format(h=h, ucp=ucp,
                                              algo_context=args.algo_context,
                                              env_context=args.env_context, ccp=ccp,
                                              agent_context=args.agent_context,
                                              exp=args.exp_name,
                                              port=port,
                                              save_interval_secs=(60000 if args.debug else 60),
                                              save_summaries_secs=(60000 if args.debug else 1),
                                              s3_save_policy_path=args.s3_save_policy_path)
  for extra_arg in args.gin_params:
    command_str += "--params='%s' " % extra_arg

  print(command_str)
  call(command_str, shell=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('train_script_version', type=str, choices=["", "v2"],
                      help='Version of run_train.py')
  parser.add_argument('exp_name', type=str, help='Exp name determines where data are stored to')
  parser.add_argument('algo_context', type=str,
                      choices=['hiro_xy', 'hiro_orig', 'hiro_repr'],
                      help='Name of the algorithm gin.')
  parser.add_argument('env_context', type=str, help='Name of the env gin.')
  parser.add_argument('agent_context', type=str, help='Name of the agent gin.')
  parser.add_argument('--s3_save_policy_path', type=str,
                      help='Which S3 path to upload the policy snapshots to', default='None')
  parser.add_argument('--debug', action='store_true', default=False)
  args, unknown_args = parser.parse_known_args()
  args.gin_params = unknown_args
  main(args)
