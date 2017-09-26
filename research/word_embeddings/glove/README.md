This directory contains a model for unsupervised training of word embeddings
using the model described in:

(Pennington, et. al.) [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf).

Assuming you have cloned the git repository, navigate into this directory. To download the example text and evaluation data:

```shell
curl http://mattmahoney.net/dc/text8.zip > text8.zip
unzip text8.zip
curl https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip > source-archive.zip
unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
rm text8.zip source-archive.zip
```

You will need to compile the ops as follows:

```shell
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared glove_ops.cc glove_kernels.cc -o glove_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```

(For an explanation of what this is doing, see the tutorial on [Adding a New Op to TensorFlow](https://www.tensorflow.org/how_tos/adding_an_op/#building_the_op_library). The flag `-D_GLIBCXX_USE_CXX11_ABI=0` is included to support newer versions of gcc. However, if you compiled TensorFlow from source using gcc 5 or later, you may need to exclude the flag.)
Then run using:

```shell
python glove.py \
  --train_data=text8 \
  --eval_data=questions-words.txt \
  --save_path=/tmp/
```

Here is a short overview of what is in this directory.

File | What's in it?
--- | ---
`glove.py` | A version of word2vec implemented using TensorFlow ops and minibatching.
`test_glove.py` | Test for the GloVe model, such as veryfing the Co-Ocurrence matrix.
`glove_kernels.cc` | Kernels for the custom input for the GloVe model.
`glove_ops.cc` | The declarations of the custom ops.
`test_data`    | Directory that contains test data to be used when testing the model
