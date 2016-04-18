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
#  blaze build -c opt nlp/saft/components/dependencies/opensource:parser_eval
#  cat en-sentences.txt |
#    nlp/saft/components/dependencies/opensource/demo.sh > output.conll
#
# To run on a conll file, change --input=stdin below to --input=stdin-conll.
#

PARSER_EVAL=bazel-bin/neurosis/parser_eval
MODEL_DIR=neurosis/models/treebank_union

$PARSER_EVAL \
  --input=stdin-conll \
  --output=stdout-conll \
  --hidden_layer_sizes=64 \
  --arg_prefix=brain_tagger \
  --graph_builder=structured \
  --task_context=$MODEL_DIR/context.pbtxt \
  --model_path=$MODEL_DIR/tagger-params \
  --batch_size=1024 \
  --max_steps=300 \
  --alsologtostderr \
   | \
  $PARSER_EVAL \
  --input=stdin-conll \
  --output=stdout-conll \
  --hidden_layer_sizes=512,512 \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --task_context=$MODEL_DIR/context.pbtxt \
  --model_path=$MODEL_DIR/parser-params \
  --batch_size=1024 \
  --max_steps=300 \
  --alsologtostderr
