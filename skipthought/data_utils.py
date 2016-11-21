'''
Implements all of the data input and output functions used by the SkipThoughtModel.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from cStringIO import StringIO
import zipfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _UNK, _EOS]

PAD_ID = 0
GO_ID = 1
UNK_ID = 2
EOS_ID = 3


_DIGIT_RE = re.compile(br"\d")

_BOOKCORPUS_DATASET_P1_URL = "http://www.cs.toronto.edu/~mbweb/books_large_p1.txt"
_BOOKCORPUS_DATASET_P2_URL = "http://www.cs.toronto.edu/~mbweb/books_large_p2.txt"

_SICK_DATASET_URL = "http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip"

_SICK_DATASET_NAME = "SICK_train.txt"


def maybe_download(directory, filename, url, is_zip=False):
    '''
    Download filename from url unless it's already in directory.
    '''
    if not tf.gfile.Exists(directory):
        print("Creating directory %s" % directory)
        tf.gfile.MakeDirs(directory)
    filepath = os.path.join(directory, filename)
    if not tf.gfile.Exists(filepath):
        if not is_zip:
            print("Downloading %s to %s" % (url, filepath))
            filepath, _ = urllib.request.urlretrieve(url, filepath)
            statinfo = tf.gfile.Stat(filepath)
            print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
        else:
            file_zip_path = filepath + ".zip"
            print("Downloading %s to %s" % (url, file_zip_path))
            file_zip_path, _ = urllib.request.urlretrieve(url, file_zip_path)
            statinfo = tf.gfile.Stat(file_zip_path)
            print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
            zip_ref = zipfile.ZipFile(file_zip_path, 'r')
            zip_ref.extract(filename, directory)
            zip_ref.close()
    return filepath


def basic_tokenizer(sentence):
    '''
    Very basic tokenizer: split the sentence into a list of tokens.
    '''
    return sentence.strip().split(" ")


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    '''
    Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    '''
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" %
              (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = tokenizer(
                    line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, b"0",
                                  w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + \
                sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with tf.gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    '''
    Initialize vocabulary from file.
    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].
    Args:
      vocabulary_path: path to the file containing the vocabulary.
    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).
    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    '''
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    '''
    Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    Returns:
      a list of integers, the token-ids for the sentence.
    '''

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [GO_ID] + [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words] + [EOS_ID]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    '''
    Tokenize data file and turn into token-ids using given vocabulary file.
    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.
    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    '''
    if not tf.gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with tf.gfile.GFile(data_path, mode="rb") as data_file:
            with tf.gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    tokens_file.write(
                        " ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_skip_thought_data(data_dir, data_file, vocab_size, tokenizer=None):
    '''
    Given a data file and a vocab size convert the data into a fully tokenized
    data input file.
    Args:
      data_dir: path to the data file in one-sentence-per-line format.
      data_file: data file in one-sentence-per-line format.
      vocab_size: size of vocab you want to use.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
    '''
    train_path = maybe_download(
        data_dir, data_file, _BOOKCORPUS_DATASET_P1_URL)

    vocab_path = os.path.join(data_dir, "vocab{}".format(vocab_size))
    create_vocabulary(vocab_path, train_path, vocab_size, tokenizer)

    train_ids_path = train_path + ".ids"
    data_to_token_ids(train_path, train_ids_path, vocab_path, tokenizer)

    return train_path, vocab_path, train_ids_path


def prepare_relatedness_data(data_dir, vocab_path, tokenizer=None):
    '''
    Given a SICK-like file, convert that file into a useable format given a
    vocab path.
    Args:
      data_dir: path to the data file in one-sentence-per-line format.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
    '''
    data_path = maybe_download(data_dir, _SICK_DATASET_NAME, _SICK_DATASET_URL, is_zip=True)
    vocab, _ = initialize_vocabulary(vocab_path)

    data = []
    with tf.gfile.GFile(data_path, mode="rb") as f:
        data.extend(f.readlines())
    sentence_A_ids = []
    sentence_B_ids = []
    relatedness_scores = []
    first_row = True
    for datum in data:
        if first_row:
            first_row = False
            continue
        datum = datum.split('\t')
        sentence_A_ids.append(sentence_to_token_ids(datum[1], vocab))
        sentence_B_ids.append(sentence_to_token_ids(datum[2], vocab))
        relatedness_scores.append(datum[3])

    return sentence_A_ids, sentence_B_ids, relatedness_scores


def sentence_iterator(sentence_ids_path, batch_size, max_size=None):
    '''
    Given the converted data and a batch size make a sentence generator
    Args:
      sentence_ids_path: path to the data file in final format.
      batch_size: Size of the batch.
      max_size: The max size of the epoch;
        if None, will run over the entire file.
    '''
    with tf.gfile.GFile(sentence_ids_path, mode="r") as sentence_ids_file:
        prev_sentence = sentence_ids_file.readline()
        middle_sentence = sentence_ids_file.readline()
        next_sentence = sentence_ids_file.readline()

        counter = 1
        backwards_decoder_inputs = []
        forwards_decoder_inputs = []
        encoder_inputs = []
        while next_sentence and (not max_size or counter <= max_size):
            counter += 1

            backwards_decoder_inputs.append([
                int(w) for w in reversed(prev_sentence.split())])
            forwards_decoder_inputs.append([int(w) for w in next_sentence.split()])
            encoder_inputs.append([int(w) for w in middle_sentence.split()])

            prev_sentence = sentence_ids_file.readline()
            middle_sentence = sentence_ids_file.readline()
            next_sentence = sentence_ids_file.readline()

            if counter % batch_size == 0:
                yield (backwards_decoder_inputs, encoder_inputs, forwards_decoder_inputs)
                backwards_decoder_inputs = []
                forwards_decoder_inputs = []
                encoder_inputs = []


def save_np_array(file_dir, arr):
    s = StringIO()
    np.savetxt(s, arr, fmt='%.5f')
    f = tf.gfile.Open(file_dir, 'w')
    f.write(s.getvalue())
    f.close()
