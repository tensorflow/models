import data.imdb as imdb

DATASET_IMDB = "imdb"


def construct_input_fns(dataset, batch_size, vocabulary_size,
                        sentence_length, repeat=1):
  """Returns training and evaluation input functions.

  Args:
    dataset: Dataset to be trained and evaluated.
      Currently only imdb is supported.
    batch_size: Number of data in each batch.
    vocabulary_size: The number of the most frequent tokens
      to be used from the corpus.
    sentence_length: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
    repeat: The number of epoch.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    A tuple of training and evaluation input function.
  """
  if dataset == DATASET_IMDB:
    train_input_fn, eval_input_fn = imdb.construct_input_fns(
        vocabulary_size, sentence_length, batch_size, repeat=repeat)
    return train_input_fn, eval_input_fn
  else:
    raise ValueError("unsupported dataset: " + dataset)


def get_num_class(dataset):
  """Returns an integer for the number of label classes.

  Args:
    dataset: Dataset to be trained and evaluated.
      Currently only imdb is supported.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    str: The dataset name.
  """
  if dataset == DATASET_IMDB:
    return imdb.NUM_CLASS
  else:
    raise ValueError("unsupported dataset: " + dataset)
