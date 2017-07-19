<font size=4><b>Language Model on One Billion Word Benchmark</b></font>

<b>Authors:</b>

Oriol Vinyals (vinyals@google.com, github: OriolVinyals),
Xin Pan (xpan@google.com, github: panyx0718)

<b>Paper Authors:</b>

Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, Yonghui Wu

<b>TL;DR</b>

This is a pretrained model on One Billion Word Benchmark.
If you use this model in your publication, please cite the original paper:

@article{jozefowicz2016exploring,
  title={Exploring the Limits of Language Modeling},
  author={Jozefowicz, Rafal and Vinyals, Oriol and Schuster, Mike
          and Shazeer, Noam and Wu, Yonghui},
  journal={arXiv preprint arXiv:1602.02410},
  year={2016}
}

<b>Introduction</b>

In this release, we open source a model trained on the One Billion Word
Benchmark (http://arxiv.org/abs/1312.3005), a large language corpus in English
which was released in 2013. This dataset contains about one billion words, and
has a vocabulary size of about 800K words. It contains mostly news data. Since
sentences in the training set are shuffled, models can ignore the context and
focus on sentence level language modeling.

In the original release and subsequent work, people have used the same test set
to train models on this dataset as a standard benchmark for language modeling.
Recently, we wrote an article (http://arxiv.org/abs/1602.02410) describing a
model hybrid between character CNN, a large and deep LSTM, and a specific
Softmax architecture which allowed us to train the best model on this dataset
thus far, almost halving the best perplexity previously obtained by others.

<b>Code Release</b>

The open-sourced components include:

* TensorFlow GraphDef proto buffer text file.
* TensorFlow pre-trained checkpoint shards.
* Code used to evaluate the pre-trained model.
* Vocabulary file.
* Test set from LM-1B evaluation.

The code supports 4 evaluation modes:

* Given provided dataset, calculate the model's perplexity.
* Given a prefix sentence, predict the next words.
* Dump the softmax embedding, character-level CNN word embeddings.
* Give a sentence, dump the embedding from the LSTM state.

<b>Results</b>

Model | Test Perplexity | Number of Params [billions]
------|-----------------|----------------------------
Sigmoid-RNN-2048 [Blackout] | 68.3 | 4.1
Interpolated KN 5-gram, 1.1B n-grams [chelba2013one] | 67.6 | 1.76
Sparse Non-Negative Matrix LM [shazeer2015sparse] | 52.9 | 33
RNN-1024 + MaxEnt 9-gram features [chelba2013one] | 51.3 | 20
LSTM-512-512 | 54.1 | 0.82
LSTM-1024-512 | 48.2 | 0.82
LSTM-2048-512 | 43.7 | 0.83
LSTM-8192-2048 (No Dropout) | 37.9 | 3.3
LSTM-8192-2048 (50\% Dropout) | 32.2 | 3.3
2-Layer LSTM-8192-1024 (BIG LSTM) | 30.6 | 1.8
(THIS RELEASE) BIG LSTM+CNN Inputs | <b>30.0</b> | <b>1.04</b>

<b>How To Run</b>

Prerequisites:

* Install TensorFlow.
* Install Bazel.
* Download the data files:
  * Model GraphDef file:
  [link](http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt)
  * Model Checkpoint sharded file:
  [1](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base)
  [2](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding)
  [3](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm)
  [4](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0)
  [5](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1)
  [6](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2)
  [7](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3)
  [8](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4)
  [9](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5)
  [10](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6)
  [11](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7)
  [12](http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8)
  * Vocabulary file:
  [link](http://download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt)
  * test dataset: link
  [link](http://download.tensorflow.org/models/LM_LSTM_CNN/test/news.en.heldout-00000-of-00050)
* It is recommended to run on a modern desktop instead of a laptop.

```shell
# 1. Clone the code to your workspace.
# 2. Download the data to your workspace.
# 3. Create an empty WORKSPACE file in your workspace.
# 4. Create an empty output directory in your workspace.
# Example directory structure below:
$ ls -R
.:
data  lm_1b  output  WORKSPACE

./data:
ckpt-base            ckpt-lstm      ckpt-softmax1  ckpt-softmax3  ckpt-softmax5
ckpt-softmax7  graph-2016-09-10.pbtxt          vocab-2016-09-10.txt
ckpt-char-embedding  ckpt-softmax0  ckpt-softmax2  ckpt-softmax4  ckpt-softmax6
ckpt-softmax8  news.en.heldout-00000-of-00050

./lm_1b:
BUILD  data_utils.py  lm_1b_eval.py  README.md

./output:

# Build the codes.
$ bazel build -c opt lm_1b/...
# Run sample mode:
$ bazel-bin/lm_1b/lm_1b_eval --mode sample \
                             --prefix "I love that I" \
                             --pbtxt data/graph-2016-09-10.pbtxt \
                             --vocab_file data/vocab-2016-09-10.txt  \
                             --ckpt 'data/ckpt-*'
...(omitted some TensorFlow output)
I love
I love that
I love that I
I love that I find
I love that I find that
I love that I find that amazing
...(omitted)

# Run eval mode:
$ bazel-bin/lm_1b/lm_1b_eval --mode eval \
                             --pbtxt data/graph-2016-09-10.pbtxt \
                             --vocab_file data/vocab-2016-09-10.txt  \
                             --input_data data/news.en.heldout-00000-of-00050 \
                             --ckpt 'data/ckpt-*'
...(omitted some TensorFlow output)
Loaded step 14108582.
# perplexity is high initially because words without context are harder to
# predict.
Eval Step: 0, Average Perplexity: 2045.512297.
Eval Step: 1, Average Perplexity: 229.478699.
Eval Step: 2, Average Perplexity: 208.116787.
Eval Step: 3, Average Perplexity: 338.870601.
Eval Step: 4, Average Perplexity: 228.950107.
Eval Step: 5, Average Perplexity: 197.685857.
Eval Step: 6, Average Perplexity: 156.287063.
Eval Step: 7, Average Perplexity: 124.866189.
Eval Step: 8, Average Perplexity: 147.204975.
Eval Step: 9, Average Perplexity: 90.124864.
Eval Step: 10, Average Perplexity: 59.897914.
Eval Step: 11, Average Perplexity: 42.591137.
...(omitted)
Eval Step: 4529, Average Perplexity: 29.243668.
Eval Step: 4530, Average Perplexity: 29.302362.
Eval Step: 4531, Average Perplexity: 29.285674.
...(omitted. At convergence, it should be around 30.)

# Run dump_emb mode:
$ bazel-bin/lm_1b/lm_1b_eval --mode dump_emb \
                             --pbtxt data/graph-2016-09-10.pbtxt \
                             --vocab_file data/vocab-2016-09-10.txt  \
                             --ckpt 'data/ckpt-*' \
                             --save_dir output
...(omitted some TensorFlow output)
Finished softmax weights
Finished word embedding 0/793471
Finished word embedding 1/793471
Finished word embedding 2/793471
...(omitted)
$ ls output/
embeddings_softmax.npy ...

# Run dump_lstm_emb mode:
$ bazel-bin/lm_1b/lm_1b_eval --mode dump_lstm_emb \
                             --pbtxt data/graph-2016-09-10.pbtxt \
                             --vocab_file data/vocab-2016-09-10.txt \
                             --ckpt 'data/ckpt-*' \
                             --sentence "I love who I am ." \
                             --save_dir output
$ ls output/
lstm_emb_step_0.npy  lstm_emb_step_2.npy  lstm_emb_step_4.npy
lstm_emb_step_6.npy  lstm_emb_step_1.npy  lstm_emb_step_3.npy
lstm_emb_step_5.npy
```
