# SyntaxNet: Neural Models of Syntax.

*A TensorFlow implementation of the models described in [Andor et al. (2016)]
(http://arxiv.org/abs/1603.06042).*

**Update**: Parsey models are now [available](universal.md) for 40 languages
trained on Universal Dependencies datasets, with support for text segmentation
and morphological analysis.

At Google, we spend a lot of time thinking about how computer systems can read
and understand human language in order to process it in intelligent ways. We are
excited to share the fruits of our research with the broader community by
releasing SyntaxNet, an open-source neural network framework for [TensorFlow]
(http://www.tensorflow.org) that provides a foundation for Natural Language
Understanding (NLU) systems. Our release includes all the code needed to train
new SyntaxNet models on your own data, as well as *Parsey McParseface*, an
English parser that we have trained for you, and that you can use to analyze
English text.

So, how accurate is Parsey McParseface? For this release, we tried to balance a
model that runs fast enough to be useful on a single machine (e.g. ~600
words/second on a modern desktop) and that is also the most accurate parser
available. Here's how Parsey McParseface compares to the academic literature on
several different English domains: (all numbers are % correct head assignments
in the tree, or unlabelled attachment score)

Model                                                                                                           | News  | Web   | Questions
--------------------------------------------------------------------------------------------------------------- | :---: | :---: | :-------:
[Martins et al. (2013)](http://www.cs.cmu.edu/~ark/TurboParser/)                                                | 93.10 | 88.23 | 94.21
[Zhang and McDonald (2014)](http://research.google.com/pubs/archive/38148.pdf)                                  | 93.32 | 88.65 | 93.37
[Weiss et al. (2015)](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43800.pdf) | 93.91 | 89.29 | 94.17
[Andor et al. (2016)](http://arxiv.org/abs/1603.06042)*                                                         | 94.44 | 90.17 | 95.40
Parsey McParseface                                                                                              | 94.15 | 89.08 | 94.77

We see that Parsey McParseface is state-of-the-art; more importantly, with
SyntaxNet you can train larger networks with more hidden units and bigger beam
sizes if you want to push the accuracy even further: [Andor et al. (2016)]
(http://arxiv.org/abs/1603.06042)* is simply a SyntaxNet model with a
larger beam and network. For futher information on the datasets, see that paper
under the section "Treebank Union".

Parsey McParseface is also state-of-the-art for part-of-speech (POS) tagging
(numbers below are per-token accuracy):

Model                                                                      | News  | Web   | Questions
-------------------------------------------------------------------------- | :---: | :---: | :-------:
[Ling et al. (2015)](http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf) | 97.44 | 94.03 | 96.18
[Andor et al. (2016)](http://arxiv.org/abs/1603.06042)*                    | 97.77 | 94.80 | 96.86
Parsey McParseface                                                         | 97.52 | 94.24 | 96.45

The first part of this tutorial describes how to install the necessary tools and
use the already trained models provided in this release. In the second part of
the tutorial we provide more background about the models, as well as
instructions for training models on other datasets.

## Contents
* [Installation](#installation)
* [Getting Started](#getting-started)
    * [Parsing from Standard Input](#parsing-from-standard-input)
    * [Annotating a Corpus](#annotating-a-corpus)
    * [Configuring the Python Scripts](#configuring-the-python-scripts)
    * [Next Steps](#next-steps)
* [Detailed Tutorial: Building an NLP Pipeline with SyntaxNet](#detailed-tutorial-building-an-nlp-pipeline-with-syntaxnet)
    * [Obtaining Data](#obtaining-data)
    * [Part-of-Speech Tagging](#part-of-speech-tagging)
    * [Training the SyntaxNet POS Tagger](#training-the-syntaxnet-pos-tagger)
    * [Preprocessing with the Tagger](#preprocessing-with-the-tagger)
    * [Dependency Parsing: Transition-Based Parsing](#dependency-parsing-transition-based-parsing)
    * [Training a Parser Step 1: Local Pretraining](#training-a-parser-step-1-local-pretraining)
    * [Training a Parser Step 2: Global Training](#training-a-parser-step-2-global-training)
* [Contact](#contact)
* [Credits](#credits)

## Installation

Running and training SyntaxNet models requires building this package from
source. You'll need to install:

*   python 2.7:
    * python 3 support is not available yet
*   bazel:
    *   **version 0.4.3**
    *   follow the instructions [here](http://bazel.build/docs/install.html)
    *   Alternately, Download bazel (0.4.3) <.deb> from
        [https://github.com/bazelbuild/bazel/releases]
        (https://github.com/bazelbuild/bazel/releases) for your system
        configuration.
    *   Install it using the command: sudo dpkg -i <.deb file>
    *   Check for the bazel version by typing: bazel version
*   swig:
    *   `apt-get install swig` on Ubuntu
    *   `brew install swig` on OSX
*   protocol buffers, with a version supported by TensorFlow:
    *   check your protobuf version with `pip freeze | grep protobuf`
    *   upgrade to a supported version with `pip install -U protobuf==3.0.0b2`
*   mock, the testing package:
    *   `pip install mock`
*   asciitree, to draw parse trees on the console for the demo:
    *   `pip install asciitree`
*   numpy, package for scientific computing:
    *   `pip install numpy`

Once you completed the above steps, you can build and test SyntaxNet with the
following commands:

```shell
  git clone --recursive https://github.com/tensorflow/models.git
  cd models/syntaxnet/tensorflow
  ./configure
  cd ..
  bazel test syntaxnet/... util/utf8/...
  # On Mac, run the following:
  bazel test --linkopt=-headerpad_max_install_names \
    syntaxnet/... util/utf8/...
```

Bazel should complete reporting all tests passed.

You can also compile SyntaxNet in a [Docker](https://www.docker.com/what-docker)
container using this [Dockerfile](Dockerfile).

To build SyntaxNet with GPU support please refer to the instructions in
[issues/248](https://github.com/tensorflow/models/issues/248).

**Note:** If you are running Docker on OSX, make sure that you have enough
memory allocated for your Docker VM.

## Getting Started

Once you have successfully built SyntaxNet, you can start parsing text right
away with Parsey McParseface, located under `syntaxnet/models`. The easiest
thing is to use or modify the included script `syntaxnet/demo.sh`, which shows a
basic setup to parse English taking plain text as input.

### Parsing from Standard Input

Simply pass one sentence per line of text into the script at
`syntaxnet/demo.sh`. The script will break the text into words, run the POS
tagger, run the parser, and then generate an ASCII version of the parse tree:

```shell
echo 'Bob brought the pizza to Alice.' | syntaxnet/demo.sh

Input: Bob brought the pizza to Alice .
Parse:
brought VBD ROOT
 +-- Bob NNP nsubj
 +-- pizza NN dobj
 |   +-- the DT det
 +-- to IN prep
 |   +-- Alice NNP pobj
 +-- . . punct
```

The ASCII tree shows the text organized as in the parse, not left-to-right as
visualized in our tutorial graphs. In this example, we see that the verb
"brought" is the root of the sentence, with the subject "Bob", the object
"pizza", and the prepositional phrase "to Alice".

If you want to feed in tokenized, CONLL-formatted text, you can run `demo.sh
--conll`.

### Annotating a Corpus

To change the pipeline to read and write to specific files (as opposed to piping
through stdin and stdout), we have to modify the `demo.sh` to point to the files
we want. The SyntaxNet models are configured via a combination of run-time flags
(which are easy to change) and a text format `TaskSpec` protocol buffer. The
spec file used in the demo is in
`syntaxnet/models/parsey_mcparseface/context.pbtxt`.

To use corpora instead of stdin/stdout, we have to:

1.  Create or modify an `input` field inside the `TaskSpec`, with the
    `file_pattern` specifying the location we want. If the input corpus is in
    CONLL format, make sure to put `record_format: 'conll-sentence'`.
1.  Change the `--input` and/or `--output` flag to use the name of the resource
    as the output, instead of `stdin` and `stdout`.

E.g., if we wanted to POS tag the CONLL corpus `./wsj.conll`, we would create
two entries, one for the input and one for the output:

```protosame
input {
  name: 'wsj-data'
  record_format: 'conll-sentence'
  Part {
    file_pattern: './wsj.conll'
  }
}
input {
  name: 'wsj-data-tagged'
  record_format: 'conll-sentence'
  Part {
    file_pattern: './wsj-tagged.conll'
  }
}
```

Then we can use `--input=wsj-data --output=wsj-data-tagged` on the command line
to specify reading and writing to these files.

### Configuring the Python Scripts

As mentioned above, the python scripts are configured in two ways:

1.  **Run-time flags** are used to point to the `TaskSpec` file, switch between
    inputs for reading and writing, and set various run-time model parameters.
    At training time, these flags are used to set the learning rate, hidden
    layer sizes, and other key parameters.
1.  The **`TaskSpec` proto** stores configuration about the transition system,
    the features, and a set of named static resources required by the parser. It
    is specified via the `--task_context` flag. A few key notes to remember:

    -   The `Parameter` settings in the `TaskSpec` have a prefix: either
        `brain_pos` (they apply to the tagger) or `brain_parser` (they apply to
        the parser). The `--prefix` run-time flag switches between reading from
        the two configurations.
    -   The resources will be created and/or modified during multiple stages of
        training. As described above, the resources can also be used at
        evaluation time to read or write to specific files. These resources are
        also separate from the model parameters, which are saved separately via
        calls to TensorFlow ops, and loaded via the `--model_path` flag.
    -   Because the `TaskSpec` contains file path, remember that copying around
        this file is not enough to relocate a trained model: you need up move
        and update all the paths as well.

Note that some run-time flags need to be consistent between training and testing
(e.g. the number of hidden units).

### Next Steps

There are many ways to extend this framework, e.g. adding new features, changing
the model structure, training on other languages, etc. We suggest reading the
detailed tutorial below to get a handle on the rest of the framework.

## Detailed Tutorial: Building an NLP Pipeline with SyntaxNet

In this tutorial, we'll go over how to train new models, and explain in a bit
more technical detail the NLP side of the models. Our goal here is to explain
the NLP pipeline produced by this package.

### Obtaining Data

The included English parser, Parsey McParseface, was trained on the the standard
corpora of the [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) and
[OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19), as well as the [English
Web Treebank](https://catalog.ldc.upenn.edu/LDC2012T13), but these are
unfortunately not freely available.

However, the [Universal Dependencies](http://universaldependencies.org/) project
provides freely available treebank data in a number of languages. SyntaxNet can
be trained and evaluated on any of these corpora.

### Part-of-Speech Tagging

Consider the following sentence, which exhibits several ambiguities that affect
its interpretation:

> I saw the man with glasses.

This sentence is composed of words: strings of characters that are segmented
into groups (e.g. "I", "saw", etc.) Each word in the sentence has a *grammatical
function* that can be useful for understanding the meaning of language. For
example, "saw" in this example is a past tense of the verb "to see". But any
given word might have different meanings in different contexts: "saw" could just
as well be a noun (e.g., a saw used for cutting) or a present tense verb (using
a saw to cut something).

A logical first step in understanding language is figuring out these roles for
each word in the sentence. This process is called *Part-of-Speech (POS)
Tagging*. The roles are called POS tags. Although a given word might have
multiple possible tags depending on the context, given any one interpretation of
a sentence each word will generally only have one tag.

One interesting challenge of POS tagging is that the problem of defining a
vocabulary of POS tags for a given language is quite involved. While the concept
of nouns and verbs is pretty common, it has been traditionally difficult to
agree on a standard set of roles across all languages. The [Universal
Dependencies](http://www.universaldependencies.org) project aims to solve this
problem.

### Training the SyntaxNet POS Tagger

In general, determining the correct POS tag requires understanding the entire
sentence and the context in which it is uttered. In practice, we can do very
well just by considering a small window of words around the word of interest.
For example, words that follow the word ‘the’ tend to be adjectives or nouns,
rather than verbs.

To predict POS tags, we use a simple setup. We process the sentences
left-to-right. For any given word, we extract features of that word and a window
around it, and use these as inputs to a feed-forward neural network classifier,
which predicts a probability distribution over POS tags. Because we make
decisions in left-to-right order, we also use prior decisions as features in
subsequent ones (e.g. "the previous predicted tag was a noun.").

All the models in this package use a flexible markup language to define
features. For example, the features in the POS tagger are found in the
`brain_pos_features` parameter in the `TaskSpec`, and look like this (modulo
spacing):

```
stack(3).word stack(2).word stack(1).word stack.word input.word input(1).word input(2).word input(3).word;
input.digit input.hyphen;
stack.suffix(length=2) input.suffix(length=2) input(1).suffix(length=2);
stack.prefix(length=2) input.prefix(length=2) input(1).prefix(length=2)
```

Note that `stack` here means "words we have already tagged." Thus, this feature
spec uses three types of features: words, suffixes, and prefixes. The features
are grouped into blocks that share an embedding matrix, concatenated together,
and fed into a chain of hidden layers. This structure is based upon the model
proposed by [Chen and Manning (2014)]
(http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf).

We show this layout in the schematic below: the state of the system (a stack and
a buffer, visualized below for both the POS and the dependency parsing task) is
used to extract sparse features, which are fed into the network in groups. We
show only a small subset of the features to simplify the presentation in the
schematic:

![Schematic](ff_nn_schematic.png "Feed-forward Network Structure")

In the configuration above, each block gets its own embedding matrix and the
blocks in the configuration above are delineated with a semi-colon. The
dimensions of each block are controlled in the `brain_pos_embedding_dims`
parameter. **Important note:** unlike many simple NLP models, this is *not* a
bag of words model. Remember that although certain features share embedding
matrices, the above features will be concatenated, so the interpretation of
`input.word` will be quite different from `input(1).word`. This also means that
adding features increases the dimension of the `concat` layer of the model as
well as the number of parameters for the first hidden layer.

To train the model, first edit `syntaxnet/context.pbtxt` so that the inputs
`training-corpus`, `tuning-corpus`, and `dev-corpus` point to the location of
your training data. You can then train a part-of-speech tagger with:

```shell
bazel-bin/syntaxnet/parser_trainer \
  --task_context=syntaxnet/context.pbtxt \
  --arg_prefix=brain_pos \  # read from POS configuration
  --compute_lexicon \       # required for first stage of pipeline
  --graph_builder=greedy \  # no beam search
  --training_corpus=training-corpus \  # names of training/tuning set
  --tuning_corpus=tuning-corpus \
  --output_path=models \  # where to save new resources
  --batch_size=32 \       # Hyper-parameters
  --decay_steps=3600 \
  --hidden_layer_sizes=128 \
  --learning_rate=0.08 \
  --momentum=0.9 \
  --seed=0 \
  --params=128-0.08-3600-0.9-0  # name for these parameters
```

This will read in the data, construct a lexicon, build a tensorflow graph for
the model with the specific hyperparameters, and train the model. Every so often
the model will be evaluated on the tuning set, and only the checkpoint with the
highest accuracy on this set will be saved. **Note that you should never use a
corpus you intend to test your model on as your tuning set, as you will inflate
your test set results.**

For best results, you should repeat this command with at least 3 different
seeds, and possibly with a few different values for `--learning_rate` and
`--decay_steps`. Good values for `--learning_rate` are usually close to 0.1, and
you usually want `--decay_steps` to correspond to about one tenth of your
corpus. The `--params` flag is only a human readable identifier for the model
being trained, used to construct the full output path, so that you don't need to
worry about clobbering old models by accident.

The `--arg_prefix` flag controls which parameters should be read from the task
context file `context.pbtxt`. In this case `arg_prefix` is set to `brain_pos`,
so the paramters being used in this training run are
`brain_pos_transition_system`, `brain_pos_embedding_dims`, `brain_pos_features`
and, `brain_pos_embedding_names`. To train the dependency parser later
`arg_prefix` will be set to `brain_parser`.

### Preprocessing with the Tagger

Now that we have a trained POS tagging model, we want to use the output of this
model as features in the parser. Thus the next step is to run the trained model
over our training, tuning, and dev (evaluation) sets. We can use the
parser_eval.py` script for this.

For example, the model `128-0.08-3600-0.9-0` trained above can be run over the
training, tuning, and dev sets with the following command:

```shell
PARAMS=128-0.08-3600-0.9-0
for SET in training tuning dev; do
  bazel-bin/syntaxnet/parser_eval \
    --task_context=models/brain_pos/greedy/$PARAMS/context \
    --hidden_layer_sizes=128 \
    --input=$SET-corpus \
    --output=tagged-$SET-corpus \
    --arg_prefix=brain_pos \
    --graph_builder=greedy \
    --model_path=models/brain_pos/greedy/$PARAMS/model
done
```

**Important note:** This command only works because we have created entries for
you in `context.pbtxt` that correspond to `tagged-training-corpus`,
`tagged-dev-corpus`, and `tagged-tuning-corpus`. From these default settings,
the above will write tagged versions of the training, tuning, and dev set to the
directory `models/brain_pos/greedy/$PARAMS/`. This location is chosen because
the `input` entries do not have `file_pattern` set: instead, they have `creator:
brain_pos/greedy`, which means that `parser_trainer.py` will construct *new*
files when called with `--arg_prefix=brain_pos --graph_builder=greedy` using the
`--model_path` flag to determine the location.

For convenience, `parser_eval.py` also logs POS tagging accuracy after the
output tagged datasets have been written.

### Dependency Parsing: Transition-Based Parsing

Now that we have a prediction for the grammatical role of the words, we want to
understand how the words in the sentence relate to each other. This parser is
built around the *head-modifier* construction: for each word, we choose a
*syntactic head* that it modifies according to some grammatical role.

An example for the above sentence is as follows:

![Figure](sawman.png)

Below each word in the sentence we see both a fine-grained part-of-speech
(*PRP*, *VBD*, *DT*, *NN* etc.), and a coarse-grained part-of-speech (*PRON*,
*VERB*, *DET*, *NOUN*, etc.). Coarse-grained POS tags encode basic grammatical
categories, while the fine-grained POS tags make further distinctions: for
example *NN* is a singular noun (as opposed, for example, to *NNS*, which is a
plural noun), and *VBD* is a past-tense verb. For more discussion see [Petrov et
al. (2012)](http://www.lrec-conf.org/proceedings/lrec2012/pdf/274_Paper.pdf).

Crucially, we also see directed arcs signifying grammatical relationships
between different words in the sentence. For example *I* is the subject of
*saw*, as signified by the directed arc labeled *nsubj* between these words;
*man* is the direct object (dobj) of *saw*; the preposition *with* modifies
*man* with a prep relation, signifiying modification by a prepositional phrase;
and so on. In addition the verb *saw* is identified as the *root* of the entire
sentence.

Whenever we have a directed arc between two words, we refer to the word at the
start of the arc as the *head*, and the word at the end of the arc as the
*modifier*. For example we have one arc where the head is *saw* and the modifier
is *I*, another where the head is *saw* and the modifier is *man*, and so on.

The grammatical relationships encoded in dependency structures are directly
related to the underlying meaning of the sentence in question. They allow us to
easily recover the answers to various questions, for example *whom did I see?*,
*who saw the man with glasses?*, and so on.

SyntaxNet is a **transition-based** dependency parser [Nivre (2007)]
(http://www.mitpressjournals.org/doi/pdfplus/10.1162/coli.07-056-R1-07-027) that
constructs a parse incrementally. Like the tagger, it processes words
left-to-right. The words all start as unprocessed input, called the *buffer*. As
words are encountered they are put onto a *stack*. At each step, the parser can
do one of three things:

1.  **SHIFT:** Push another word onto the top of the stack, i.e. shifting one
    token from the buffer to the stack.
1.  **LEFT_ARC:** Pop the top two words from the stack. Attach the second to the
    first, creating an arc pointing to the **left**. Push the **first** word
    back on the stack.
1.  **RIGHT_ARC:** Pop the top two words from the stack. Attach the second to
    the first, creating an arc point to the **right**. Push the **second** word
    back on the stack.

At each step, we call the combination of the stack and the buffer the
*configuration* of the parser. For the left and right actions, we also assign a
dependency relation label to that arc. This process is visualized in the
following animation for a short sentence:

![Animation](looping-parser.gif "Parsing in Action")

Note that this parser is following a sequence of actions, called a
**derivation**, to produce a "gold" tree labeled by a linguist. We can use this
sequence of decisions to learn a classifier that takes a configuration and
predicts the next action to take.

### Training a Parser Step 1: Local Pretraining

As described in our [paper](http://arxiv.org/abs/1603.06042), the first
step in training the model is to *pre-train* using *local* decisions. In this
phase, we use the gold dependency to guide the parser, and train a softmax layer
to predict the correct action given these gold dependencies. This can be
performed very efficiently, since the parser's decisions are all independent in
this setting.

Once the tagged datasets are available, a locally normalized dependency parsing
model can be trained with the following command:

```shell
bazel-bin/syntaxnet/parser_trainer \
  --arg_prefix=brain_parser \
  --batch_size=32 \
  --projectivize_training_set \
  --decay_steps=4400 \
  --graph_builder=greedy \
  --hidden_layer_sizes=200,200 \
  --learning_rate=0.08 \
  --momentum=0.85 \
  --output_path=models \
  --task_context=models/brain_pos/greedy/$PARAMS/context \
  --seed=4 \
  --training_corpus=tagged-training-corpus \
  --tuning_corpus=tagged-tuning-corpus \
  --params=200x200-0.08-4400-0.85-4
```

Note that we point the trainer to the context corresponding to the POS tagger
that we picked previously. This allows the parser to reuse the lexicons and the
tagged datasets that were created in the previous steps. Processing data can be
done similarly to how tagging was done above. For example if in this case we
picked parameters `200x200-0.08-4400-0.85-4`, the training, tuning and dev sets
can be parsed with the following command:

```shell
PARAMS=200x200-0.08-4400-0.85-4
for SET in training tuning dev; do
  bazel-bin/syntaxnet/parser_eval \
    --task_context=models/brain_parser/greedy/$PARAMS/context \
    --hidden_layer_sizes=200,200 \
    --input=tagged-$SET-corpus \
    --output=parsed-$SET-corpus \
    --arg_prefix=brain_parser \
    --graph_builder=greedy \
    --model_path=models/brain_parser/greedy/$PARAMS/model
done
```

### Training a Parser Step 2: Global Training

As we describe in the paper, there are several problems with the locally
normalized models we just trained. The most important is the *label-bias*
problem: the model doesn't learn what a good parse looks like, only what action
to take given a history of gold decisions. This is because the scores are
normalized *locally* using a softmax for each decision.

In the paper, we show how we can achieve much better results using a *globally*
normalized model: in this model, the softmax scores are summed in log space, and
the scores are not normalized until we reach a final decision. When the parser
stops, the scores of each hypothesis are normalized against a small set of
possible parses (in the case of this model, a beam size of 8). When training, we
force the parser to stop during parsing when the gold derivation falls off the
beam (a strategy known as early-updates).

We give a simplified view of how this training works for a [garden path
sentence](https://en.wikipedia.org/wiki/Garden_path_sentence), where it is
important to maintain multiple hypotheses. A single mistake early on in parsing
leads to a completely incorrect parse; after training, the model learns to
prefer the second (correct) parse.

![Beam search training](beam_search_training.png)

Parsey McParseface correctly parses this sentence. Even though the correct parse
is initially ranked 4th out of multiple hypotheses, when the end of the garden
path is reached, Parsey McParseface can recover due to the beam; using a larger
beam will get a more accurate model, but it will be slower (we used beam 32 for
the models in the paper).

Once you have the pre-trained locally normalized model, a globally normalized
parsing model can now be trained with the following command:

```shell
bazel-bin/syntaxnet/parser_trainer \
  --arg_prefix=brain_parser \
  --batch_size=8 \
  --decay_steps=100 \
  --graph_builder=structured \
  --hidden_layer_sizes=200,200 \
  --learning_rate=0.02 \
  --momentum=0.9 \
  --output_path=models \
  --task_context=models/brain_parser/greedy/$PARAMS/context \
  --seed=0 \
  --training_corpus=projectivized-training-corpus \
  --tuning_corpus=tagged-tuning-corpus \
  --params=200x200-0.02-100-0.9-0 \
  --pretrained_params=models/brain_parser/greedy/$PARAMS/model \
  --pretrained_params_names=\
embedding_matrix_0,embedding_matrix_1,embedding_matrix_2,\
bias_0,weights_0,bias_1,weights_1
```

Training a beam model with the structured builder will take a lot longer than
the greedy training runs above, perhaps 3 or 4 times longer. Note once again
that multiple restarts of training will yield the most reliable results.
Evaluation can again be done with `parser_eval.py`. In this case we use
parameters `200x200-0.02-100-0.9-0` to evaluate on the training, tuning and dev
sets with the following command:

```shell
PARAMS=200x200-0.02-100-0.9-0
for SET in training tuning dev; do
  bazel-bin/syntaxnet/parser_eval \
    --task_context=models/brain_parser/structured/$PARAMS/context \
    --hidden_layer_sizes=200,200 \
    --input=tagged-$SET-corpus \
    --output=beam-parsed-$SET-corpus \
    --arg_prefix=brain_parser \
    --graph_builder=structured \
    --model_path=models/brain_parser/structured/$PARAMS/model
done
```

Hooray! You now have your very own cousin of Parsey McParseface, ready to go out
and parse text in the wild.

## Contact

To ask questions or report issues please post on Stack Overflow with the tag
[syntaxnet](http://stackoverflow.com/questions/tagged/syntaxnet)
or open an issue on the tensorflow/models
[issues tracker](https://github.com/tensorflow/models/issues).
Please assign SyntaxNet issues to @calberti or @andorardo.

## Credits

Original authors of the code in this package include (in alphabetical order):

*   Alessandro Presta
*   Aliaksei Severyn
*   Andy Golding
*   Bernd Bohnet
*   Chris Alberti
*   Daniel Andor
*   David Weiss
*   Emily Pitler
*   Greg Coppola
*   Ji Ma
*   Keith Hall
*   Kuzman Ganchev
*   Michael Collins
*   Michael Ringgaard
*   Ryan McDonald
*   Slav Petrov
*   Stefan Istrate
*   Terry Koo
*   Tim Credo
