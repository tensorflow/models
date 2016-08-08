# A script that runs a tokenizer on a text file with one sentence per line.
#
# Example usage:
#  blaze build -c opt nlp/saft/components/dependencies/opensource:parser_eval
#  cat untokenized-sentences.txt |
#    nlp/saft/components/dependencies/opensource/models/parsey_universal/tokenize.sh \
#    $MODEL_DIRECTORY > output.conll
#
# Models can be downloaded from
#  http://download.tensorflow.org/models/parsey_universal/<language>.zip
# for the languages listed at
#  https://github.com/tensorflow/models/blob/master/syntaxnet/universal.md
#

PARSER_EVAL=blaze-bin/nlp/saft/components/dependencies/opensource/parser_eval
CONTEXT=nlp/saft/components/dependencies/opensource/models/parsey_universal/context.pbtxt
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
