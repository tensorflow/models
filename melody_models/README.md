# Melody RNN Models

This directory contains a number of RNN (recurrent neural network) models that can generate melodies. These models fall under the category of language models, in that they output the probability of the next note given a sequence of previous notes in a modely. Melodies can be sampled from these models by searching for the most likely sequence of notes (using beam search for example).

## Minimal RNN

minimal_lstm.py defines a minimal LSTM (long short-term memory) model for predicting notes in a melody. It is minimal in that it only uses a stack of LSTM cells. This model also serves as an example of recurrent graphs in TensorFlow, giving two ways to build one: with tf.nn.dynamic_rnn or tf.nn.state_saving_rnn.

*Disclaimer*

This is a work in progress and more commits are comming. Things may not work in the meantime.