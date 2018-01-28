Sequence-to-Sequence with Attention Model for Text Summarization.

Authors:

Xin Pan (xpan@google.com, github:panyx0718),
Peter Liu (peterjliu@google.com, github:peterjliu)

<b>Introduction</b>

The core model is the traditional sequence-to-sequence model with attention.
It is customized (mostly inputs/outputs) for the text summarization task. The
model has been trained on Gigaword dataset and achieved state-of-the-art
results (as of June 2016).

The results described below are based on model trained on multi-gpu and
multi-machine settings. It has been simplified to run on only one machine
for open source purpose.

<b>Dataset</b>

We used the Gigaword dataset described in [Rush et al. A Neural Attention Model
for Sentence Summarization](https://arxiv.org/abs/1509.00685).

We cannot provide the dataset due to the license. See ExampleGen in data.py
about the data format. data/data contains a toy example. Also see data/vocab
for example vocabulary format. In <b>How To Run</b> below, users can use toy
data and vocab provided in the data/ directory to run the training by replacing
the data directory flag.

data_convert_example.py contains example of convert between binary and text.


<b>Experiment Result</b>

8000 examples from testset are sampled to generate summaries and rouge score is
calculated for the generated summaries. Here is the best rouge score on
Gigaword dataset:

ROUGE-1 Average_R: 0.38272 (95%-conf.int. 0.37774 - 0.38755)

ROUGE-1 Average_P: 0.50154 (95%-conf.int. 0.49509 - 0.50780)

ROUGE-1 Average_F: 0.42568 (95%-conf.int. 0.42016 - 0.43099)

ROUGE-2 Average_R: 0.20576 (95%-conf.int. 0.20060 - 0.21112)

ROUGE-2 Average_P: 0.27565 (95%-conf.int. 0.26851 - 0.28257)

ROUGE-2 Average_F: 0.23126 (95%-conf.int. 0.22539 - 0.23708)

<b>Configuration:</b>

Following is the configuration for the best trained model on Gigaword:

batch_size: 64

bidirectional encoding layer: 4

article length: first 2 sentences, total words within 120.

summary length: total words within 30.

word embedding size: 128

LSTM hidden units: 256

Sampled softmax: 4096

vocabulary size: Most frequent 200k words from dataset's article and summaries.

<b>How To Run</b>

Prerequisite: install TensorFlow and Bazel.

```shell
# cd to your workspace
# 1. Clone the textsum code to your workspace 'textsum' directory.
# 2. Create an empty 'WORKSPACE' file in your workspace.
# 3. Move the train/eval/test data to your workspace 'data' directory.
#    In the following example, I named the data training-*, test-*, etc.
#    If your data files have different names, update the --data_path.
#    If you don't have data but want to try out the model, copy the toy
#    data from the textsum/data/data to the data/ directory in the workspace.
$ ls -R
.:
data  textsum  WORKSPACE

./data:
vocab  test-0  training-0  training-1  validation-0 ...(omitted)

./textsum:
batch_reader.py       beam_search.py       BUILD    README.md                    seq2seq_attention_model.py  data
data.py  seq2seq_attention_decode.py  seq2seq_attention.py        seq2seq_lib.py

./textsum/data:
data  vocab

$ bazel build -c opt --config=cuda textsum/...

# Run the training.
$ bazel-bin/textsum/seq2seq_attention \
    --mode=train \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/training-* \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --train_dir=textsum/log_root/train

# Run the eval. Try to avoid running on the same machine as training.
$ bazel-bin/textsum/seq2seq_attention \
    --mode=eval \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/validation-* \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --eval_dir=textsum/log_root/eval

# Run the decode. Run it when the model is mostly converged.
$ bazel-bin/textsum/seq2seq_attention \
    --mode=decode \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/test-* \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --decode_dir=textsum/log_root/decode \
    --beam_size=8
```


<b>Examples:</b>

The following are some text summarization examples, including experiments
using dataset other than Gigaword.

article: novell inc. chief executive officer eric schmidt has been named chairman of the internet search-engine company google .

human: novell ceo named google chairman

machine:  novell chief executive named to head internet company

======================================

article: gulf newspapers voiced skepticism thursday over whether newly re - elected us president bill clinton could help revive the troubled middle east peace process but saw a glimmer of hope .

human: gulf skeptical about whether clinton will revive peace process

machine:  gulf press skeptical over clinton 's prospects for peace process

======================================

article:  the european court of justice ( ecj ) recently ruled in lock v british gas trading ltd that eu law requires a worker 's statutory holiday pay to take commission payments into account - it should not be based solely on basic salary . the case is not over yet , but its outcome could potentially be costly for employers with workers who are entitled to commission . mr lock , an energy salesman for british gas , was paid a basic salary and sales commission on a monthly basis . his sales commission made up around 60 % of his remuneration package . when he took two weeks ' annual leave in december 2012 , he was paid his basic salary and also received commission from previous sales that fell due during that period . lock obviously did not generate new sales while he was on holiday , which meant that in the following period he suffered a reduced income through lack of commission . he brought an employment tribunal claim asserting that this amounted to a breach of the working time regulations 1998 .....deleted rest for readability...

abstract: will british gas ecj ruling fuel holiday pay hike ?

decode: eu law requires worker 's statutory holiday pay

======================================

article:  the junior all whites have been eliminated from the fifa u - 20 world cup in colombia with results on the final day of pool play confirming their exit . sitting on two points , new zealand needed results in one of the final two groups to go their way to join the last 16 as one of the four best third place teams . but while spain helped the kiwis ' cause with a 5 - 1 thrashing of australia , a 3 - 0 win for ecuador over costa rica saw the south americans climb to second in group c with costa rica 's three points also good enough to progress in third place . that left the junior all whites hopes hanging on the group d encounter between croatia and honduras finishing in a draw . a stalemate - and a place in the knockout stages for new zealand - appeared on the cards until midfielder marvin ceballos netted an 81st minute winner that sent guatemala through to the second round and left the junior all whites packing their bags . new zealand finishes the 24 - nation tournament in 17th place , having claimed their first ever points at this level in just their second appearance at the finals .

abstract: junior all whites exit world cup

decoded:  junior all whites eliminated from u- 20 world cup

