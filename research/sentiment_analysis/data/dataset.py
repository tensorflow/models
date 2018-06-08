import data.imdb as imdb

DATASET_IMDB = 'imdb'

def input_fn(dataset, if_training, batch_size, vocabulary_size, sentence_length, repeat=1):
    if dataset == DATASET_IMDB:
        return imdb.input_fn(if_training, vocabulary_size, sentence_length, batch_size, repeat=repeat)
    else:
        pass
