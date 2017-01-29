# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for downloading data from WMT, tokenizing, vocabularies."""

import gzip
import os
import re
import tarfile

from six.moves import urllib
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_CHAR_UNK"
_SPACE = b"_SPACE"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _SPACE]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
SPACE_ID = 4

# Regular expressions used to tokenize.
_CHAR_MARKER = "_CHAR_"
_CHAR_MARKER_LEN = len(_CHAR_MARKER)
_SPEC_CHARS = "" + chr(226) + chr(153) + chr(128)
_PUNCTUATION = "][.,!?\"':;%$#@&*+}{|><=/^~)(_`,0123456789" + _SPEC_CHARS + "-"
_WORD_SPLIT = re.compile(b"([" + _PUNCTUATION + "])")
_OLD_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not tf.gfile.Exists(directory):
    print "Creating directory %s" % directory
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not tf.gfile.Exists(filepath):
    print "Downloading %s to %s" % (url, filepath)
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print "Succesfully downloaded", filename, statinfo.st_size, "bytes"
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print "Unpacking %s to %s" % (gz_path, new_path)
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "giga-fren.release2.fixed")
  if not (tf.gfile.Exists(train_path +".fr") and
          tf.gfile.Exists(train_path +".en")):
    corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                 _WMT_ENFR_TRAIN_URL)
    print "Extracting tar file %s" % corpus_file
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(directory)
    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
    gunzip_file(train_path + ".en.gz", train_path + ".en")
  return train_path


def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "newstest2013"
  dev_path = os.path.join(directory, dev_name)
  if not (tf.gfile.Exists(dev_path + ".fr") and
          tf.gfile.Exists(dev_path + ".en")):
    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
    print "Extracting tgz file %s" % dev_file
    with tarfile.open(dev_file, "r:gz") as dev_tar:
      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
      en_dev_file.name = dev_name + ".en"
      dev_tar.extract(fr_dev_file, directory)
      dev_tar.extract(en_dev_file, directory)
  return dev_path


def is_char(token):
  if len(token) > _CHAR_MARKER_LEN:
    if token[:_CHAR_MARKER_LEN] == _CHAR_MARKER:
      return True
  return False


def basic_detokenizer(tokens):
  """Reverse the process of the basic tokenizer below."""
  result = []
  previous_nospace = True
  for t in tokens:
    if is_char(t):
      result.append(t[_CHAR_MARKER_LEN:])
      previous_nospace = True
    elif t == _SPACE:
      result.append(" ")
      previous_nospace = True
    elif previous_nospace:
      result.append(t)
      previous_nospace = False
    else:
      result.extend([" ", t])
      previous_nospace = False
  return "".join(result)


old_style = False


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  if old_style:
    for space_separated_fragment in sentence.strip().split():
      words.extend(re.split(_OLD_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]
  for space_separated_fragment in sentence.strip().split():
    tokens = [t for t in re.split(_WORD_SPLIT, space_separated_fragment) if t]
    first_is_char = False
    for i, t in enumerate(tokens):
      if len(t) == 1 and t in _PUNCTUATION:
        tokens[i] = _CHAR_MARKER + t
        if i == 0:
          first_is_char = True
    if words and words[-1] != _SPACE and (first_is_char or is_char(words[-1])):
      tokens = [_SPACE] + tokens
    spaced_tokens = []
    for i, tok in enumerate(tokens):
      spaced_tokens.append(tokens[i])
      if i < len(tokens) - 1:
        if tok != _SPACE and not (is_char(tok) or is_char(tokens[i+1])):
          spaced_tokens.append(_SPACE)
    words.extend(spaced_tokens)
  return words


def space_tokenizer(sentence):
  return sentence.strip().split()


def is_pos_tag(token):
  """Check if token is a part-of-speech tag."""
  return(token in ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR",
                   "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
                   "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO",
                   "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP",
                   "WP$", "WRB", ".", ",", ":", ")", "-LRB-", "(", "-RRB-",
                   "HYPH", "$", "``", "''", "ADD", "AFX", "QTR", "BES", "-DFL-",
                   "GW", "HVS", "NFP"])


def parse_constraints(inpt, res):
  ntags = len(res)
  nwords = len(inpt)
  npostags = len([x for x in res if is_pos_tag(x)])
  nclose = len([x for x in res if x[0] == "/"])
  nopen = ntags - nclose - npostags
  return (abs(npostags - nwords), abs(nclose - nopen))


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.

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
  """
  if not tf.gfile.Exists(vocabulary_path):
    print "Creating vocabulary %s from data %s" % (vocabulary_path, data_path)
    vocab, chars = {}, {}
    for c in _PUNCTUATION:
      chars[c] = 1

    # Read French file.
    with tf.gfile.GFile(data_path + ".fr", mode="rb") as f:
      counter = 0
      for line_in in f:
        line = " ".join(line_in.split())
        counter += 1
        if counter % 100000 == 0:
          print "  processing fr line %d" % counter
        for c in line:
          if c in chars:
            chars[c] += 1
          else:
            chars[c] = 1
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        tokens = [t for t in tokens if not is_char(t) and t != _SPACE]
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1000000000  # We want target words first.
          else:
            vocab[word] = 1000000000

    # Read English file.
    with tf.gfile.GFile(data_path + ".en", mode="rb") as f:
      counter = 0
      for line_in in f:
        line = " ".join(line_in.split())
        counter += 1
        if counter % 100000 == 0:
          print "  processing en line %d" % counter
        for c in line:
          if c in chars:
            chars[c] += 1
          else:
            chars[c] = 1
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        tokens = [t for t in tokens if not is_char(t) and t != _SPACE]
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1

      sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
      sorted_chars = sorted(chars, key=vocab.get, reverse=True)
      sorted_chars = [_CHAR_MARKER + c for c in sorted_chars]
      vocab_list = _START_VOCAB + sorted_chars + sorted_vocab
      if tokenizer:
        vocab_list = _START_VOCAB + sorted_vocab
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with tf.gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

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
  """
  if tf.gfile.Exists(vocabulary_path):
    rev_vocab = []
    with tf.gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids_raw(sentence, vocabulary,
                              tokenizer=None, normalize_digits=old_style):
  """Convert a string to list of integers representing token-ids.

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
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  result = []
  for w in words:
    if normalize_digits:
      w = re.sub(_DIGIT_RE, b"0", w)
    if w in vocabulary:
      result.append(vocabulary[w])
    else:
      if tokenizer:
        result.append(UNK_ID)
      else:
        result.append(SPACE_ID)
        for c in w:
          result.append(vocabulary.get(_CHAR_MARKER + c, UNK_ID))
        result.append(SPACE_ID)
  while result and result[0] == SPACE_ID:
    result = result[1:]
  while result and result[-1] == SPACE_ID:
    result = result[:-1]
  return result


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=old_style):
  """Convert a string to list of integers representing token-ids, tab=0."""
  tab_parts = sentence.strip().split("\t")
  toks = [sentence_to_token_ids_raw(t, vocabulary, tokenizer, normalize_digits)
          for t in tab_parts]
  res = []
  for t in toks:
    res.extend(t)
    res.append(0)
  return res[:-1]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

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
  """
  if not tf.gfile.Exists(target_path):
    print "Tokenizing data in %s" % data_path
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with tf.gfile.GFile(data_path, mode="rb") as data_file:
      with tf.gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print "  tokenizing line %d" % counter
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, vocabulary_size,
                     tokenizer=None, normalize_digits=False):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    vocabulary_size: size of the joint vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the vocabulary file,
      (6) path to the vocabulary file (for compatibility with non-joint vocab).
  """
  # Get wmt data to the specified directory.
  train_path = get_wmt_enfr_train_set(data_dir)
  dev_path = get_wmt_enfr_dev_set(data_dir)

  # Create vocabularies of the appropriate sizes.
  vocab_path = os.path.join(data_dir, "vocab%d.txt" % vocabulary_size)
  create_vocabulary(vocab_path, train_path, vocabulary_size,
                    tokenizer=tokenizer, normalize_digits=normalize_digits)

  # Create token ids for the training data.
  fr_train_ids_path = train_path + (".ids%d.fr" % vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.en" % vocabulary_size)
  data_to_token_ids(train_path + ".fr", fr_train_ids_path, vocab_path,
                    tokenizer=tokenizer, normalize_digits=normalize_digits)
  data_to_token_ids(train_path + ".en", en_train_ids_path, vocab_path,
                    tokenizer=tokenizer, normalize_digits=normalize_digits)

  # Create token ids for the development data.
  fr_dev_ids_path = dev_path + (".ids%d.fr" % vocabulary_size)
  en_dev_ids_path = dev_path + (".ids%d.en" % vocabulary_size)
  data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, vocab_path,
                    tokenizer=tokenizer, normalize_digits=normalize_digits)
  data_to_token_ids(dev_path + ".en", en_dev_ids_path, vocab_path,
                    tokenizer=tokenizer, normalize_digits=normalize_digits)

  return (en_train_ids_path, fr_train_ids_path,
          en_dev_ids_path, fr_dev_ids_path,
          vocab_path, vocab_path)
