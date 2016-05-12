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

# This test trains a parser on a small dataset, then runs it in greedy mode and
# in structured mode with beam 1, and checks that the result is identical.




set -eux

BINDIR=$TEST_SRCDIR/syntaxnet
CONTEXT=$BINDIR/testdata/context.pbtxt
TMP_DIR=/tmp/syntaxnet-output

mkdir -p $TMP_DIR
sed "s=SRCDIR=$TEST_SRCDIR=" "$CONTEXT" | \
  sed "s=OUTPATH=$TMP_DIR=" > $TMP_DIR/context

PARAMS=128-0.08-3600-0.9-0

"$BINDIR/parser_trainer" \
  --arg_prefix=brain_parser \
  --batch_size=32 \
  --compute_lexicon \
  --decay_steps=3600 \
  --graph_builder=greedy \
  --hidden_layer_sizes=128 \
  --learning_rate=0.08 \
  --momentum=0.9 \
  --output_path=$TMP_DIR \
  --task_context=$TMP_DIR/context \
  --training_corpus=training-corpus \
  --tuning_corpus=tuning-corpus \
  --params=$PARAMS \
  --num_epochs=12 \
  --report_every=100 \
  --checkpoint_every=1000 \
  --logtostderr

"$BINDIR/parser_eval" \
  --task_context=$TMP_DIR/brain_parser/greedy/$PARAMS/context \
  --hidden_layer_sizes=128 \
  --input=tuning-corpus \
  --output=stdout \
  --arg_prefix=brain_parser \
  --graph_builder=greedy \
  --model_path=$TMP_DIR/brain_parser/greedy/$PARAMS/model \
  --logtostderr \
  > $TMP_DIR/greedy-out

"$BINDIR/parser_eval" \
  --task_context=$TMP_DIR/context \
  --hidden_layer_sizes=128 \
  --beam_size=1 \
  --input=tuning-corpus \
  --output=stdout \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --model_path=$TMP_DIR/brain_parser/greedy/$PARAMS/model \
  --logtostderr \
  > $TMP_DIR/struct-beam1-out

diff $TMP_DIR/greedy-out $TMP_DIR/struct-beam1-out

STRUCT_PARAMS=128-0.001-3600-0.9-0

"$BINDIR/parser_trainer" \
  --arg_prefix=brain_parser \
  --batch_size=8 \
  --compute_lexicon \
  --decay_steps=3600 \
  --graph_builder=structured \
  --hidden_layer_sizes=128 \
  --learning_rate=0.001 \
  --momentum=0.9 \
  --pretrained_params=$TMP_DIR/brain_parser/greedy/$PARAMS/model \
  --pretrained_params_names=\
embedding_matrix_0,embedding_matrix_1,embedding_matrix_2,bias_0,weights_0 \
  --output_path=$TMP_DIR \
  --task_context=$TMP_DIR/context \
  --training_corpus=training-corpus \
  --tuning_corpus=tuning-corpus \
  --params=$STRUCT_PARAMS \
  --num_epochs=20 \
  --report_every=25 \
  --checkpoint_every=200 \
  --logtostderr

"$BINDIR/parser_eval" \
  --task_context=$TMP_DIR/context \
  --hidden_layer_sizes=128 \
  --beam_size=8 \
  --input=tuning-corpus \
  --output=stdout \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --model_path=$TMP_DIR/brain_parser/structured/$STRUCT_PARAMS/model \
  --logtostderr \
  > $TMP_DIR/struct-beam8-out

echo "PASS"
