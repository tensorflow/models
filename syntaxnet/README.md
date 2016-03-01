# Neurosis: Neural Models of Syntax.

[TOC]

## Installation

Running and training Neurosis models requires building this package from
source.  You'll need to install:

* bazel, following the instructions [here](http://bazel.io/docs/install.html),
* swig:
    * `apt-get install swig'`on Ubuntu,
* a version of protocol buffers supported by TensorFlow:
    * check your protobuf version with `pip freeze | grep protobuf1`,
    * upgrade to a supported version with `pip install -U protobuf==3.0.0b2`

Once you completed the above steps, you can build and test Neurosis with the
following commands:

```shell
  git clone --recursive rpc://team/saft/neurosis
  cd neurosis/tensorflow
  ./configure
  cd ..
  bazel test neurosis/... util/utf8/...
```

Bazel will work for approximately 5 to 10 minutes (depending on your
machine power) and the command should complete reporting all tests passed.

## Running a Neurosis model

Once you successfully built Neurosis, you can start parsing text (in conll
format) right away with one of our bundled models, located under
`neurosis/models`.

Edit the text file `neurosis/models/512x512/context` so that the file pattern
for the input `dev-corpus` points to your input conll data, and the one for
`parsed-dev-corpus` points to where you want the script to output parsed
data. The model can then be used for parsing by running:

```shell
bazel-bin/neurosis/parser_eval \
  --task_context neurosis/models/512x512/context \
  --hidden_layer_sizes=512,512 \
  --input=dev-corpus \
  --output=parsed-dev-corpus \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --model_path neurosis/models/512x512/model \
  --logtostderr
```

## Internal Google Usage

This is a set of instructions to train a POS tagger and a parser at Google using
blaze and borg. The necessary tools can be compiled with:

```shell
blaze build -c opt \
  nlp/saft/components/dependencies/opensource:parser_trainer.par \
  nlp/saft/components/dependencies/opensource:parser_eval
```

### POS Tagger

A POS tagger can then be trained running:

```shell
OUTPATH=/cns/lg-d/home/$USER/20160114_opensource/
borgcfg nlp/saft/components/dependencies/opensource/train.borg \
  reload neurosis-greedy \
  --vars=arg_prefix=brain_pos,compute_lexicon=true,output_path=$OUTPATH
```

This command will start a borg job with multiple tasks, each training a POS
tagger with different hyperparameters or initialization seeds. Periodically each
task will also evaluate the current performance on a tuning set. An evaluation
metric of about 97.3% should be achieved in about 15 minutes. This metric is POS
tagging accuracy on the tuning set.

The `arg_prefix` var controls which parameters should be read from the task
context file `context.pbtxt`. In this case `arg_prefix` is set to `brain_pos`,
so the paramters being used in this training run are
`brain_pos_transition_system`, `brain_pos_embedding_dims`, `brain_pos_features`
and, `brain_pos_embedding_names`. To train the dependency parser later
`arg_prefix` will be set to `brain_parser`.

The model trained by a task can be used to process data with `parser_eval.py`.
For example the model trained with hyperparamters `200x200-0.08-4400-0.9-4` can
be run over training, tuning, and dev set with the following command:

```shell
PARAMS=200x200-0.08-4000-0.9-2
for SET in training tuning dev; do
  blaze-bin/nlp/saft/components/dependencies/opensource/parser_eval \
    --task_context=$OUTPATH/brain_pos/greedy/$PARAMS/context \
    --hidden_layer_sizes=200,200 \
    --input=$SET-corpus \
    --output=tagged-$SET-corpus \
    --arg_prefix=brain_pos \
    --logtostderr \
    --graph_builder=greedy \
    --model_path=$OUTPATH/brain_pos/greedy/$PARAMS/model \
    --nocfs_log_all_errors
done
```

The above will write tagged versions of the training, tuning, and dev set to the
directory `$OUTPATH/brain_pos/greedy/$PARAMS/`. `parser_eval.py` also logs POS
tagging accuracy after the output tagged datasets have been written.

Tagging accuracy and speeds should be as follows:

Dataset  | Scored tokens | POS Accuracy | Tokens per second
-------- | ------------- | ------------ | ------------------
training | 973949        | 98.43%       | 9500
tuning   | 33692         | 97.34%       | 2800
dev      | 41219         | 97.22%       | 3000

Once the tagged datasets are available the greedy dependency parser can be
trained with the following command:

### Greedy Parser

```shell
borgcfg nlp/saft/components/dependencies/opensource/train.borg \
  reload neurosis-greedy \
  --vars=arg_prefix=brain_parser,projectivize=true,\
output_path=$OUTPATH,task_context=$OUTPATH/brain_pos/greedy/$PARAMS/context,\
training_corpus=tagged-training-corpus,tuning_corpus=tagged-tuning-corpus
```

Note that we point the trainer to the context corresponding to the POS tagger
that we picked previously. This allows the parser to reuse the lexicons and the
tagged datasets that were created in the previous steps.

Processing data can be done similarly to how tagging was done above. For example
if in this case we pick parameters `200x200-0.08-4000-0.85-4`, tuning and dev
set can be parsed with the following command:

```shell
PARAMS=200x200-0.08-4400-0.85-4
for SET in tuning dev; do
  blaze-bin/nlp/saft/components/dependencies/opensource/parser_eval \
    --task_context=$OUTPATH/brain_parser/greedy/$PARAMS/context \
    --hidden_layer_sizes=200,200 \
    --input=tagged-$SET-corpus \
    --output=parsed-$SET-corpus \
    --arg_prefix=brain_parser \
    --logtostderr \
    --graph_builder=greedy \
    --model_path=$OUTPATH/brain_parser/greedy/$PARAMS/model \
    --nocfs_log_all_errors
done
```

The evaluation metrics reported by this tool are as follows:

Dataset  | Scored tokens | UAS          | Tokens per second
-------- | ------------- | ------------ | ------------------
tuning   | 29978         | 91.91%       | 1900
dev      | 36613         | 92.23%       | 2000

### Beam Parser

A beam version of this parser can now be trained with the following command:

```shell
borgcfg nlp/saft/components/dependencies/opensource/train.borg \
  reload neurosis-beam \
  --vars=arg_prefix=brain_parser,output_path=$OUTPATH,\
task_context=$OUTPATH/brain_parser/greedy/$PARAMS/context,\
training_corpus=projectivized-training-corpus,tuning_corpus=tagged-tuning-corpus,\
pretrained_parameters=$OUTPATH/brain_parser/greedy/$PARAMS/model
```

Training in this case takes between 3 and 4 hours. Evaluation can again be done
with `parser_eval.py`. In this case we pick parameters `200x200-0.02-100-0.9-0`,
and we evaluate on tuning and dev set can with the following command:

```shell
PARAMS=200x200-0.02-100-0.9-0
for SET in tuning dev; do
  blaze-bin/nlp/saft/components/dependencies/opensource/parser_eval \
    --task_context=$OUTPATH/brain_parser/structured/$PARAMS/context \
    --hidden_layer_sizes=200,200 \
    --input=tagged-$SET-corpus \
    --output=beam-parsed-$SET-corpus \
    --arg_prefix=brain_parser \
    --logtostderr \
    --graph_builder=structured \
    --model_path=$OUTPATH/brain_parser/structured/$PARAMS/model \
    --nocfs_log_all_errors
done
```

The evaluation metrics reported in this case are:

Dataset  | Scored tokens | UAS          | Tokens per second
-------- | ------------- | ------------ | ------------------
tuning   | 29978         | 92.74%       | 450
dev      | 36613         | 93.04%       | 470

## Credits

Original authors of the code in this package include:

*  bohnetbd@google.com (Bernd Bohnet)
*  chrisalberti@google.com (Chris Alberti)
*  credo@google.com (Tim Credo)
*  danielandor@google.com (Daniel Andor)
*  djweiss@google.com (David Weiss)
*  epitler@google.com (Emily Pitler)
*  gcoppola@google.com (Greg Coppola)
*  golding@google.com (Andy Golding)
*  istefan@google.com (Stefan Istrate)
*  kbhall@google.com (Keith Hall)
*  kuzman@google.com (Kuzman Ganchev)
*  ringgaard@google.com (Michael Ringgaard)
*  ryanmcd@google.com (Ryan Mcdonald)
*  slav@google.com (Slav Petrov)
*  terrykoo@google.com (Terry Koo)
