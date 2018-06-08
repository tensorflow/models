# Sentiment Analysis
## Overview
This is an implementation of the Sentiment Analysis model as described in the [this](https://arxiv.org/abs/1412.1058) paper. The implementation is with the reference to [paddle version](https://github.com/mlperf/reference/tree/master/sentiment_analysis/paddle).

The model makes use of concatenation of two CNN layers with different kernel sizes. Dropout and batch normalization layers are used to prevent over fitting.

## Dataset
The [keras](https://keras.io)'s [IMDB Movie reviews sentiment classification](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification) dataset is used.

## Running Code
### Train and evaluate model
To train and evaluate the model, issue the following command:
```
python sentiment_main.py
```
Arguments:
  * `--vocabulary_size`: The number of words included in the dataset. The most frequent words are chosen. The deault is 6000.
  * `--sentence_length`: The length of the sentence
  * `--dataset`: The dataset name to be downloaded and preprocessed. By default, it is `imdb`.

There are other arguments about models and training process. Use the `--help` or `-h` flag to get a full list of possible arguments with detailed descriptions.

## Benchmarks (TBA)