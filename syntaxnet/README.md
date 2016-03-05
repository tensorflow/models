# Neurosis: Neural Models of Syntax.

[TOC]

A TensorFlow implementation of the models described in [[http://arxiv.org/?]].

## Installation

Running and training Neurosis models requires building this package from
source.  You'll need to install:

* bazel, following the instructions [here](http://bazel.io/docs/install.html),
* swig:
    * `apt-get install swig` on Ubuntu,
    * `brew install swig` on OSX,
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

Bazel should complete reporting all tests passed.

## Tutorial: Parsing Text with a Neurosis model

Once have you successfully built Neurosis, you can start parsing text (in conll
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

## Tutorial: Training a POS Tagger and a Dependency Parser with Neurosis

This is a set of instructions to train a POS tagger and a dependency parser
with Neurosis.

### POS Tagger

First edit `neurosis/context.pbtxt` so that the inputs `training-corpus`,
`tuning-corpus`, and `dev-corpus` point to the location of your data.
You can then train a part-of-speech tagger with:

```shell
bazel-bin/neurosis/parser_trainer \
  --arg_prefix=brain_pos \
  --batch_size=32 \
  --compute_lexicon \
  --decay_steps=3600 \
  --graph_builder=greedy \
  --hidden_layer_sizes=128 \
  --learning_rate=0.08 \
  --momentum=0.9 \
  --optimizer=momentum \
  --output_path=models \
  --noprojectivize_training_set \
  --task_context=neurosis/context.pbtxt \
  --seed=0 \
  --training_corpus=training-corpus \
  --tuning_corpus=tuning-corpus \
  --params=128-0.08-3600-0.9-0
```

For best results, you should repeat this command with at least 3 different
seeds, and possibly with a few different values for `--learning_rate` and
`--decay_steps`. Good values for `--learning_rate` are usually close to 0.1, and
you usually want `--decay_steps` to correspond to about one tenth of your
corpus. The `--params` flag is only a human readable identifier for the model
being trained, used to construct the full output path.

The `--arg_prefix` flag controls which parameters should be read from the task
context file `context.pbtxt`. In this case `arg_prefix` is set to `brain_pos`,
so the paramters being used in this training run are
`brain_pos_transition_system`, `brain_pos_embedding_dims`, `brain_pos_features`
and, `brain_pos_embedding_names`. To train the dependency parser later
`arg_prefix` will be set to `brain_parser`.

The model trained by a task can be used to process data with `parser_eval.py`.
For example the model `128-0.08-3600-0.9-0` trained above can
be run over training, tuning, and dev set with the following command:

```shell
PARAMS=128-0.08-3600-0.9-0
for SET in training tuning dev; do
  bazel-bin/neurosis/parser_eval \
    --task_context=models/brain_pos/greedy/$PARAMS/context \
    --hidden_layer_sizes=128 \
    --input=$SET-corpus \
    --output=tagged-$SET-corpus \
    --arg_prefix=brain_pos \
    --graph_builder=greedy \
    --model_path=models/brain_pos/greedy/$PARAMS/model
done
```

The above will write tagged versions of the training, tuning, and dev set to the
directory `models/brain_pos/greedy/$PARAMS/`. `parser_eval.py` also logs POS
tagging accuracy after the output tagged datasets have been written.

### Greedy Parser

Once the tagged datasets are available a greedy dependency parser can be
trained with the following command:

```shell
bazel-bin/neurosis/parser_trainer \
  --arg_prefix=brain_parser \
  --batch_size=32 \
  --projectivize_training_set \
  --decay_steps=4400 \
  --graph_builder=greedy \
  --hidden_layer_sizes=200,200 \
  --learning_rate=0.08 \
  --momentum=0.85 \
  --optimizer=momentum \
  --output_path=models \
  --task_context=models/brain_pos/greedy/$PARAMS/context \
  --seed=4 \
  --training_corpus=tagged-training-corpus \
  --tuning_corpus=tagged-tuning-corpus \
  --params=200x200-0.08-4400-0.85-4
```

Note that we point the trainer to the context corresponding to the POS tagger
that we picked previously. This allows the parser to reuse the lexicons and the
tagged datasets that were created in the previous steps.

Processing data can be done similarly to how tagging was done above. For example
if in this case we picked parameters `200x200-0.08-4400-0.85-4`, tuning and dev
set can be parsed with the following command:

```shell
PARAMS=200x200-0.08-4400-0.85-4
for SET in training tuning dev; do
  bazel-bin/neurosis/parser_eval \
    --task_context=models/brain_pos/greedy/$PARAMS/context \
    --hidden_layer_sizes=200,200 \
    --input=tagged-$SET-corpus \
    --output=parsed-$SET-corpus \
    --arg_prefix=brain_parser \
    --graph_builder=greedy \
    --model_path=models/brain_parser/greedy/$PARAMS/model
done
```

### Beam Parser

A beam version of this parser can now be trained with the following command:

```shell
bazel-bin/neurosis/parser_trainer \
  --arg_prefix=brain_parser \
  --batch_size=8 \
  --decay_steps=100 \
  --graph_builder=structured \
  --hidden_layer_sizes=200,200 \
  --learning_rate=0.02 \
  --momentum=0.9 \
  --optimizer=momentum \
  --output_path=models \
  --task_context=models/brain_parser/greedy/$PARAMS/context \
  --seed=0 \
  --training_corpus=projectivized-training-corpus \
  --tuning_corpus=tagged-tuning-corpus \
  --params=200x200-0.02-100-0.9-0 \
  --pretrained_params=models/brain_parser/greedy/$PARAMS/model \
  --pretrained_params_names=\
embedding_matrix_0,embedding_matrix_1,embedding_matrix_2\
bias_0,weights_0,bias_1,weights_1
```

Training a beam model with the structured builder will take a lot longer than
the greedy training runs above, perhaps 3 or 4 times longer. Evaluation can
again be done with `parser_eval.py`. In this case we use parameters
`200x200-0.02-100-0.9-0`, and we evaluate on tuning and dev set can with the
following command:

```shell
PARAMS=200x200-0.02-100-0.9-0
for SET in training tuning dev; do
  bazel-bin/neurosis/parser_eval \
    --task_context=models/brain_pos/greedy/$PARAMS/context \
    --hidden_layer_sizes=200,200 \
    --input=tagged-$SET-corpus \
    --output=beam-parsed-$SET-corpus \
    --arg_prefix=brain_parser \
    --graph_builder=structured \
    --model_path=models/brain_parser/structured/$PARAMS/model
done
```

## Contact

To ask questions or report issues please contact:
*  neurosis-oss@google.com

## Credits

Original authors of the code in this package include:

*  bohnetbd (Bernd Bohnet)
*  chrisalberti (Chris Alberti)
*  credo (Tim Credo)
*  danielandor (Daniel Andor)
*  djweiss (David Weiss)
*  epitler (Emily Pitler)
*  gcoppola (Greg Coppola)
*  golding (Andy Golding)
*  istefan (Stefan Istrate)
*  kbhall (Keith Hall)
*  kuzman (Kuzman Ganchev)
*  ringgaard (Michael Ringgaard)
*  ryanmcd (Ryan Mcdonald)
*  slav (Slav Petrov)
*  terrykoo (Terry Koo)
