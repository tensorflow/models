# A script that runs a traditional Chinese tokenizer on a text file with one
# sentence per line.
#
# Example usage:
#  bazel build syntaxnet:parser_eval
#  cat untokenized-sentences.txt |
#    syntaxnet/models/parsey_universal/tokenize_zh.sh \
#    $MODEL_DIRECTORY > output.conll
#
# The traditional Chinese model can be downloaded from
#  http://download.tensorflow.org/models/parsey_universal/Chinese.zip
#

PARSER_EVAL=bazel-bin/syntaxnet/parser_eval
CONTEXT=syntaxnet/models/parsey_universal/context-tokenize-zh.pbtxt
INPUT_FORMAT=stdin-untoken
MODEL_DIR=$1

$PARSER_EVAL \
  --input=$INPUT_FORMAT \
  --output=stdin-untoken \
  --hidden_layer_sizes=256,256 \
  --arg_prefix=brain_tokenizer_zh \
  --graph_builder=structured \
  --task_context=$CONTEXT \
  --resource_dir=$MODEL_DIR \
  --model_path=$MODEL_DIR/tokenizer-params \
  --batch_size=1024 \
  --alsologtostderr \
  --slim_model
