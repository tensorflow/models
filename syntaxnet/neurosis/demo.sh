#!/bin/bash
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

# A script that runs a tokenizer, a part-of-speech tagger and a dependency
# parser on an English text file, with a sentence per line.
#
# Example usage:
#  bazel build -c opt neurosis:parser_eval
#  cat en-sentences.txt | neurosis/demo.sh > output.conll

PARSER_EVAL=bazel-bin/neurosis/parser_eval

$PARSER_EVAL \
  --task_context=neurosis/models/treebank_union/context \
  --hidden_layer_sizes=256 \
  --arg_prefix=brain_pos \
  --graph_builder=greedy \
  --model_path=neurosis/models/treebank_union/tagger_model \
  --batch_size=1024 \
  | \
  $PARSER_EVAL \
  --input=stdin-conll \
  --task_context=neurosis/models/treebank_union/context \
  --hidden_layer_sizes=512,512 \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --model_path=neurosis/models/treebank_union/parser_model \
  --batch_size=1024
