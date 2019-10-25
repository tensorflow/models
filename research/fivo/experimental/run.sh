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


model="forward"
T=5
num_obs=1
var=0.1
n=4
lr=0.0001
bound="fivo-aux"
q_type="normal"
resampling_method="multinomial"
rgrad="true"
p_type="unimodal"
use_bs=false

LOGDIR=/tmp/fivo/model-$model-$bound-$resampling_method-resampling-rgrad-$rgrad-T-$T-var-$var-n-$n-lr-$lr-q-$q_type-p-$p_type

python train.py \
  --logdir=$LOGDIR \
  --model=$model \
  --bound=$bound \
  --q_type=$q_type \
  --p_type=$p_type \
  --variance=$var \
  --use_resampling_grads=$rgrad \
  --resampling=always \
  --resampling_method=$resampling_method \
  --batch_size=4 \
  --num_samples=$n \
  --num_timesteps=$T \
  --num_eval_samples=256 \
  --summarize_every=100 \
  --learning_rate=$lr  \
  --decay_steps=1000000 \
  --max_steps=1000000000 \
  --random_seed=1234 \
  --train_p=false \
  --use_bs=$use_bs \
  --alsologtostderr
