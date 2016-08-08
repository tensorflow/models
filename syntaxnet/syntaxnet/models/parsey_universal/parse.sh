# A script that runs a morphological analyzer, a part-of-speech tagger and a
# dependency parser on a text file, with one sentence per line.
#
# Example usage:
#  blaze build -c opt nlp/saft/components/dependencies/opensource:parser_eval
#  cat sentences.txt |
#    nlp/saft/components/dependencies/opensource/models/parsey_universal/parse.sh \
#    $MODEL_DIRECTORY > output.conll
#
# To run on a conll formatted file, add the --conll command line argument:
#  cat sentences.conll |
#    nlp/saft/components/dependencies/opensource/models/parsey_universal/parse.sh \
#    --conll $MODEL_DIRECTORY > output.conll
#
# Models can be downloaded from
#  http://download.tensorflow.org/models/parsey_universal/<language>.zip
# for the languages listed at
#  https://github.com/tensorflow/models/blob/master/syntaxnet/universal.md
#

PARSER_EVAL=blaze-bin/nlp/saft/components/dependencies/opensource/parser_eval
CONTEXT=nlp/saft/components/dependencies/opensource/models/parsey_universal/context.pbtxt
if [[ "$1" == "--conll" ]]; then
  INPUT_FORMAT=stdin-conll
  shift
else
  INPUT_FORMAT=stdin
fi
MODEL_DIR=$1

$PARSER_EVAL \
  --input=$INPUT_FORMAT \
  --output=stdout-conll \
  --hidden_layer_sizes=64 \
  --arg_prefix=brain_morpher \
  --graph_builder=structured \
  --task_context=$CONTEXT \
  --resource_dir=$MODEL_DIR \
  --model_path=$MODEL_DIR/morpher-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr \
  | \
  $PARSER_EVAL \
  --input=stdin-conll \
  --output=stdout-conll \
  --hidden_layer_sizes=64 \
  --arg_prefix=brain_tagger \
  --graph_builder=structured \
  --task_context=$CONTEXT \
  --resource_dir=$MODEL_DIR \
  --model_path=$MODEL_DIR/tagger-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr \
  | \
  $PARSER_EVAL \
  --input=stdin-conll \
  --output=stdout-conll \
  --hidden_layer_sizes=512,512 \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --task_context=$CONTEXT \
  --resource_dir=$MODEL_DIR \
  --model_path=$MODEL_DIR/parser-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr
