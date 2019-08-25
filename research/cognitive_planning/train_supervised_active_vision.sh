#!/bin/bash
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

# blaze build -c opt train_supervised_active_vision
# bazel build -c opt --config=cuda --copt=-mavx train_supervised_active_vision && \
bazel-bin/research/cognitive_planning/train_supervised_active_vision \
  --mode='train' \
  --logdir=/usr/local/google/home/kosecka/local_avd_train/ \
  --modality_types='det' \
  --batch_size=8 \
  --train_iters=200000 \
  --lstm_cell_size=2048 \
  --policy_fc_size=2048 \
  --sequence_length=20 \
  --max_eval_episode_length=100 \
  --test_iters=194 \
  --gin_config=envs/configs/active_vision_config.gin \
  --gin_params='ActiveVisionDatasetEnv.dataset_root="/cns/jn-d/home/kosecka/AVD_Minimal/"' \
  --logtostderr
