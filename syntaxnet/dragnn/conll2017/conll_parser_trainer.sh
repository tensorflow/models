#!/bin/sh
# Copyright 2016 Google Inc. All Rights Reserved.
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

# A script to train the CONLL2017 baseline.
set -e

language=English
output_dir=./trained-"$language"

training_corpus=$1
dev_corpus=$2

bazel build -c opt //dragnn/tools:trainer //dragnn/conll2017:make_parser_spec

mkdir -p $output_dir
bazel-bin/dragnn/conll2017/make_parser_spec \
  --spec_file="$output_dir/parser_spec.textproto"

bazel-bin/dragnn/tools/trainer \
  --logtostderr \
  --compute_lexicon \
  --dragnn_spec="$output_dir/parser_spec.textproto" \
  --resource_path="$output_dir/resources" \
  --training_corpus_path="$training_corpus" \
  --tune_corpus_path="$dev_corpus" \
  --tensorboard_dir="$output_dir/tensorboard" \
  --checkpoint_filename="$output_dir/checkpoint.model"
