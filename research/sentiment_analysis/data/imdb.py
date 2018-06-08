import tensorflow as tf, numpy as np
from data.util import to_dataset, pad_sentence, START_CHAR, OOV_CHAR

NUM_CLASS = 2
_x_train, _y_train, _x_test, _y_test = None, None, None, None

def input_fn(if_training, vocabulary_size, sentence_length, batch_size, repeat=1):
    global _x_train, _y_train, _x_test, _y_test

    if _x_train is None:
        (_x_train, _y_train), (_x_test, _y_test) = tf.keras.datasets.imdb.load_data(path="imdb.npz",
                                                          num_words=vocabulary_size,
                                                          skip_top=0,
                                                          maxlen=None,
                                                          seed=113,
                                                          start_char=START_CHAR,
                                                          oov_char=OOV_CHAR,
                                                          index_from=OOV_CHAR+1)

    if if_training:
        return to_dataset(np.array([pad_sentence(s, sentence_length) for s in _x_train]), np.eye(NUM_CLASS)[_y_train], batch_size, repeat)
    else:
        return to_dataset(np.array([pad_sentence(s, sentence_length) for s in _x_test]), np.eye(NUM_CLASS)[_y_test], batch_size, repeat)
