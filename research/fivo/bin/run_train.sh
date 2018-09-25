#!/bin/bash
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

# An example of running training.

PIANOROLL_DIR=$HOME/pianorolls

python run_fivo.py \
  --mode=train \
  --logdir=/tmp/fivo \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=4 \
  --num_samples=4 \
  --learning_rate=0.0001 \
  --dataset_path="$PIANOROLL_DIR/jsb.pkl" \
  --dataset_type="pianoroll"
