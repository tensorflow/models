# skip-thoughts

Sent2Vec encoder and training code from the paper [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726), reimplemented in tensorflow.

A keras implementation is [here](https://github.com/ryankiros/skip-thoughts)

## Idea in Brief

Skipthought is a way to encode sentences (or sequences of categories) by looking at their surrounding sentences. The fundamental idea is that you can predict the preceeding and following sentences by just knowing the middle sentence.

The model encodes the middle sentence as a vector and uses that vector to decode the preceding and following sentences.

## API

Skip-Thought is implemented in Tensorflow 0.8r and uses the numpy for some data processing.

### How to use

#### Downloading the data

You can download the training data from [here](http://www.cs.toronto.edu/~mbweb/books_large_p1.txt) and [here](http://www.cs.toronto.edu/~mbweb/books_large_p2.txt) or use the data utils function `maybe_download`. Generally speaking all of the below functions will seek to download the data if it is not seen in flagged directory.

That being said any tokenized newline separated sentence files with three or more sentences will work, as in below:

```
with clinicals looming to enable her to finish her nursing degree , she 'd known she would n't be able to work fulltime .
although she loved the freedom and independence of having her own apartment , there was no way she could afford it and daycare for mason .
so , she 'd packed up , tucked her tail between her legs , and moved back home to her parents ' finished basement .
it was n't all bad .
```

#### Training

To train the data, call the function `train()` in the `skipthought.py` file. You will need to specify the hyper-parameters described below as flags:

* learning_rate:  Learning rate.
learning_rate_decay_factor: the learning rate decays by this much after three iterations of non-decreasing loss
* max_gradient_norm: Clip gradients to this norm.
* encoder_cell_size: The number of GRUs in an encoder cell.
* decoder_cell_size: The number of GRUs in a decoder cell.
* word_embedding_size:  size of the word embeddings.
* max_epochs: The total number of training epochs.
* vocab_size:  vocabulary size.
* max_sentence_len:  max sentence length we will use (longer sentences will be truncated to this length).
* batch_size:  number of sentences in our batch.
* steps_per_checkpoint:  number of steps per checkpoint.
* model_dir: Training directory.
* data_dir: Data directory.
* train_data_name: Name of the train data file.
* summary_dir: Summary directory.

I have defaulted them to the hyperparameters used in the paper.

Upon training you will see the following output:

```
Creating vocabulary data/vocab200 from data data/books_tiny_p1.txt
Tokenizing data in data/books_tiny_p1.txt
Created model with fresh parameters.
global step 1 learning rate 0.0080 step-time 0.00 perplexity 1.01
Epoch ended with final loss at [1.0116370328031492]
...
```

These steps are creating the vocab for the model and a fresh model with those params. If your model dir already contains a model, then the model will be initialized with that model.

#### Generating Sentences

As a proof of concept I have given code to generate the forwards and backwards sentences given a model and a middle sentence. The command will look as follows:

```
>> generate(middle_sentence, forwards_sentence, backwards_sentence)
Generated Forwards Sentence
although she loved the freedom and independence of having her own apartment , there was no way she could afford it and daycare for mason . _EOS
Generated Backwards Sentence
_EOS . bad all n't was it
```

This is not the general purpose of Skip-Thought, however it is a nice proof of concept on toy examples or even with a fully trained model.

#### Converting Sequences

I have provided a general util for converting sequences with a trained model. This is the `convert_relatedness` function. This takes the additional parameters:

* relatedness_regression_factors: Name of the relatedness regression factors data file.
* relatedness_regression_targets: Name of the relatedness regression targets data file.

This will take a file named `SICK_train.txt` of the below format (tab separated):

```
sentence_A	sentence_B	relatedness_score
A group of kids is playing in a yard and an old man is standing in the background	A group of boys in a yard is playing and a man is standing in the background	4.5
...
```

If the file is not provided it will download the file from [this url](http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip) and unzip it.

It will then dump the converted sentences into the file specified by the `relatedness_regression_factors` flag.

#### Testing

Finally I have included functionality to test the performance of Skip-Thought. While Skip-Thought can be used for a variety of tasks (sentiment, QA, etc.) I have implemented a test on semantic entailment (how semantically related are the sentences).

This is located in the `relatedness.py` file. And is called with the `evaluate()` function. This file only takes three params:

* relatedness_regression_factors: Name of the relatedness regression factors data file.
* relatedness_regression_targets: Name of the relatedness regression targets data file.
* data_dir: the directory where data is stored

The output will look like this:

```
Train on 3375 samples, validate on 1125 samples
Epoch 1/10
0s - loss: 1.3655 - val_loss: 1.3610
Epoch 2/10
0s - loss: 1.3655 - val_loss: 1.3610
Epoch 3/10
0s - loss: 1.3655 - val_loss: 1.3610
Epoch 4/10
0s - loss: 1.3655 - val_loss: 1.3610
Epoch 5/10
0s - loss: 1.3655 - val_loss: 1.3610
...
```

## Contact Info

Feel free to reach out to me at knt(at google) or k.nathaniel.tucker(at gmail)
