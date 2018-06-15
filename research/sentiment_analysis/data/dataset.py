import data.imdb as imdb

DATASET_IMDB = 'imdb'

def input_fn(dataset, if_training, batch_size, vocabulary_size, sentence_length, repeat=1):
    """Initialize CNN model.
    Args:
      dataset: Dataset to be trained and evaluated. Currently only imdb is supported.
      batch_size: Number of data in each batch.
      vocabulary_size: The number of the most frequent tokens to be used from the corpus.
      sentence_length: The number of words in each sentence. Longer sentences get cut, shorter ones padded.
      repeat: The number of epoch.
    Raises:
      ValueError: if the dataset value is not valid.
    """
    if dataset == DATASET_IMDB:
        return imdb.input_fn(if_training, vocabulary_size, sentence_length, batch_size, repeat=repeat)
    else:
        raise ValueError('unsupported dataset: ' + dataset)