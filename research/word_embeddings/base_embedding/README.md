This directory contains base files for implementing word embedding models. The
files are based on the implementations provided by the word2vec implementation
in tensorflow:

[word2vec](https://github.com/tensorflow/models/tree/master/tutorials/embedding).

Here is a short overview of what is in this directory.

File | What's in it?
--- | ---
`word_embedding.py` | Provides a base model for any word embedding model, by
                      providing standard ways to evaluate the model and train
                      it concurrently.
`word_embedding_op.cc` | Provides a base class to create a corpus that will be
                         used to feed the model. It possess method to create
                         a vocabulary for the model, a word frequency count and
                         a word to id dictionary.
