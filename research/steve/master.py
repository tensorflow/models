from builtins import str
from builtins import range
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

import multiprocessing
import os, sys, time

from config import config, log_config
import util

AGENT_COUNT = config["agent_config"]["count"]
EVALUATOR_COUNT = config["evaluator_config"]["count"]
MODEL_AUGMENTED = config["model_config"] is not False
if config["resume"]:
  ROOT_PATH = "output/" + config["env"]["name"] + "/" + config["name"]
else:
  ROOT_PATH = util.create_and_wipe_directory("output/" + config["env"]["name"] + "/" + config["name"])
log_config()
import learner, agent, valuerl_learner
if MODEL_AUGMENTED: import worldmodel_learner

if __name__ == '__main__':
  all_procs = set([])
  interaction_procs = set([])

  # lock
  policy_lock = multiprocessing.Lock()
  model_lock = multiprocessing.Lock() if MODEL_AUGMENTED else None

  # queue
  policy_replay_frame_queue = multiprocessing.Queue(1)
  model_replay_frame_queue = multiprocessing.Queue(1) if MODEL_AUGMENTED else None

  # interactors
  for interact_proc_i in range(AGENT_COUNT):
    interact_proc = multiprocessing.Process(target=agent.main, args=(interact_proc_i, False, policy_replay_frame_queue, model_replay_frame_queue, policy_lock, config))
    all_procs.add(interact_proc)
    interaction_procs.add(interact_proc)

  # evaluators
  for interact_proc_i in range(EVALUATOR_COUNT):
    interact_proc = multiprocessing.Process(target=agent.main, args=(interact_proc_i, True, policy_replay_frame_queue, model_replay_frame_queue, policy_lock, config))
    all_procs.add(interact_proc)
    interaction_procs.add(interact_proc)

  # policy training
  train_policy_proc = multiprocessing.Process(target=learner.run_learner, args=(valuerl_learner.ValueRLLearner, policy_replay_frame_queue, policy_lock, config, config["env"], config["policy_config"]), kwargs={"model_lock": model_lock})
  all_procs.add(train_policy_proc)

  # model training
  if MODEL_AUGMENTED:
    train_model_proc = multiprocessing.Process(target=learner.run_learner, args=(worldmodel_learner.WorldmodelLearner, model_replay_frame_queue, model_lock, config, config["env"], config["model_config"]))
    all_procs.add(train_model_proc)

  # start all policies
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  for i, proc in enumerate(interaction_procs):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    proc.start()

  os.environ['CUDA_VISIBLE_DEVICES'] = str(int(sys.argv[2]))
  train_policy_proc.start()

  if MODEL_AUGMENTED:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1+int(sys.argv[2]))
    train_model_proc.start()

  while True:
    try:
      pass
    except:
      for proc in all_procs: proc.join()
