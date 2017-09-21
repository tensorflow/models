# A script that runs a tokenizer on a text file with one sentence per line.
#
# Example usage:
#  bazel build syntaxnet:parser_eval
#  cat untokenized-sentences.txt |
#    syntaxnet/models/parsey_universal/tokenize.sh \
#    $MODEL_DIRECTORY > output.conll
#
# Models can be downloaded from
#  http://download.tensorflow.org/models/parsey_universal/<language>.zip
# for the languages listed at
#  https://github.com/tensorflow/models/blob/master/syntaxnet/universal.md
#

PARSER_EVAL=bazel-bin/syntaxnet/parser_eval
CONTEXT=syntaxnet/models/parsey_universal/context.pbtxt
INPUT_FORMAT=stdin-untoken
MODEL_DIR=$1

$PARSER_EVAL \
  --input=$INPUT_FORMAT \
  --output=stdin-untoken \
  --hidden_layer_sizes=128,128 \
  --arg_prefix=brain_tokenizer \
  --graph_builder=greedy \
  --task_context=$CONTEXT \
  --resource_dir=$MODEL_DIR \
  --model_path=$MODEL_DIR/tokenizer-params \
  --batch_size=32 \
  --alsologtostderr \
  --slim_model
