# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Loads the WikiQuestions dataset.

An example consists of question, table. Additionally, we store the processed
columns which store the entries after performing number, date and other
preprocessing as done in the baseline.
columns, column names and processed columns are split into word and number
columns.
lookup answer (or matrix) is also split into number and word lookup matrix
Author: aneelakantan (Arvind Neelakantan)
"""
import math
import os
import re
import numpy as np
import unicodedata as ud
import tensorflow as tf

bad_number = -200000.0  #number that is added to a corrupted table entry in a number column

def is_nan_or_inf(number):
  return math.isnan(number) or math.isinf(number)

def strip_accents(s):
  u = unicode(s, "utf-8")
  u_new = ''.join(c for c in ud.normalize('NFKD', u) if ud.category(c) != 'Mn')
  return u_new.encode("utf-8")


def correct_unicode(string):
  string = strip_accents(string)
  string = re.sub("\xc2\xa0", " ", string).strip()
  string = re.sub("\xe2\x80\x93", "-", string).strip()
  #string = re.sub(ur'[\u0300-\u036F]', "", string)
  string = re.sub("â€š", ",", string)
  string = re.sub("â€¦", "...", string)
  #string = re.sub("[Â·ãƒ»]", ".", string)
  string = re.sub("Ë†", "^", string)
  string = re.sub("Ëœ", "~", string)
  string = re.sub("â€¹", "<", string)
  string = re.sub("â€º", ">", string)
  #string = re.sub("[â€˜â€™Â´`]", "'", string)
  #string = re.sub("[â€œâ€Â«Â»]", "\"", string)
  #string = re.sub("[â€¢â€ â€¡]", "", string)
  #string = re.sub("[â€â€‘â€“â€”]", "-", string)
  string = re.sub(ur'[\u2E00-\uFFFF]', "", string)
  string = re.sub("\\s+", " ", string).strip()
  return string


def simple_normalize(string):
  string = correct_unicode(string)
  # Citations
  string = re.sub("\[(nb ?)?\d+\]", "", string)
  string = re.sub("\*+$", "", string)
  # Year in parenthesis
  string = re.sub("\(\d* ?-? ?\d*\)", "", string)
  string = re.sub("^\"(.*)\"$", "", string)
  return string


def full_normalize(string):
  #print "an: ", string
  string = simple_normalize(string)
  # Remove trailing info in brackets
  string = re.sub("\[[^\]]*\]", "", string)
  # Remove most unicode characters in other languages
  string = re.sub(ur'[\u007F-\uFFFF]', "", string.strip())
  # Remove trailing info in parenthesis
  string = re.sub("\([^)]*\)$", "", string.strip())
  string = final_normalize(string)
  # Get rid of question marks
  string = re.sub("\?", "", string).strip()
  # Get rid of trailing colons (usually occur in column titles)
  string = re.sub("\:$", " ", string).strip()
  # Get rid of slashes
  string = re.sub(r"/", " ", string).strip()
  string = re.sub(r"\\", " ", string).strip()
  # Replace colon, slash, and dash with space
  # Note: need better replacement for this when parsing time
  string = re.sub(r"\:", " ", string).strip()
  string = re.sub("/", " ", string).strip()
  string = re.sub("-", " ", string).strip()
  # Convert empty strings to UNK
  # Important to do this last or near last
  if not string:
    string = "UNK"
  return string

def final_normalize(string):
  # Remove leading and trailing whitespace
  string = re.sub("\\s+", " ", string).strip()
  # Convert entirely to lowercase
  string = string.lower()
  # Get rid of strangely escaped newline characters
  string = re.sub("\\\\n", " ", string).strip()
  # Get rid of quotation marks
  string = re.sub(r"\"", "", string).strip()
  string = re.sub(r"\'", "", string).strip()
  string = re.sub(r"`", "", string).strip()
  # Get rid of *
  string = re.sub("\*", "", string).strip()
  return string

def is_number(x):
  try:
    f = float(x)
    return not is_nan_or_inf(f)
  except ValueError:
    return False
  except TypeError:
    return False


class WikiExample(object):

  def __init__(self, id, question, answer, table_key):
    self.question_id = id
    self.question = question
    self.answer = answer
    self.table_key = table_key
    self.lookup_matrix = []
    self.is_bad_example = False
    self.is_word_lookup = False
    self.is_ambiguous_word_lookup = False
    self.is_number_lookup = False
    self.is_number_calc = False
    self.is_unknown_answer = False


class TableInfo(object):

  def __init__(self, word_columns, word_column_names, word_column_indices,
               number_columns, number_column_names, number_column_indices,
               processed_word_columns, processed_number_columns, orig_columns):
    self.word_columns = word_columns
    self.word_column_names = word_column_names
    self.word_column_indices = word_column_indices
    self.number_columns = number_columns
    self.number_column_names = number_column_names
    self.number_column_indices = number_column_indices
    self.processed_word_columns = processed_word_columns
    self.processed_number_columns = processed_number_columns
    self.orig_columns = orig_columns


class WikiQuestionLoader(object):

  def __init__(self, data_name, root_folder):
    self.root_folder = root_folder
    self.data_folder = os.path.join(self.root_folder, "data")
    self.examples = []
    self.data_name = data_name

  def num_questions(self):
    return len(self.examples)

  def load_qa(self):
    data_source = os.path.join(self.data_folder, self.data_name)
    f = tf.gfile.GFile(data_source, "r")
    id_regex = re.compile("\(id ([^\)]*)\)")
    for line in f:
      id_match = id_regex.search(line)
      id = id_match.group(1)
      self.examples.append(id)

  def load(self):
    self.load_qa()


def is_date(word):
  if (not (bool(re.search("[a-z0-9]", word, re.IGNORECASE)))):
    return False
  if (len(word) != 10):
    return False
  if (word[4] != "-"):
    return False
  if (word[7] != "-"):
    return False
  for i in range(len(word)):
    if (not (word[i] == "X" or word[i] == "x" or word[i] == "-" or re.search(
        "[0-9]", word[i]))):
      return False
  return True


class WikiQuestionGenerator(object):

  def __init__(self, train_name, dev_name, test_name, root_folder):
    self.train_name = train_name
    self.dev_name = dev_name
    self.test_name = test_name
    self.train_loader = WikiQuestionLoader(train_name, root_folder)
    self.dev_loader = WikiQuestionLoader(dev_name, root_folder)
    self.test_loader = WikiQuestionLoader(test_name, root_folder)
    self.bad_examples = 0
    self.root_folder = root_folder   
    self.data_folder = os.path.join(self.root_folder, "annotated/data")
    self.annotated_examples = {}
    self.annotated_tables = {}
    self.annotated_word_reject = {}
    self.annotated_word_reject["-lrb-"] = 1
    self.annotated_word_reject["-rrb-"] = 1
    self.annotated_word_reject["UNK"] = 1

  def is_money(self, word):
    if (not (bool(re.search("[a-z0-9]", word, re.IGNORECASE)))):
      return False
    for i in range(len(word)):
      if (not (word[i] == "E" or word[i] == "." or re.search("[0-9]",
                                                             word[i]))):
        return False
    return True

  def remove_consecutive(self, ner_tags, ner_values):
    for i in range(len(ner_tags)):
      if ((ner_tags[i] == "NUMBER" or ner_tags[i] == "MONEY" or
           ner_tags[i] == "PERCENT" or ner_tags[i] == "DATE") and
          i + 1 < len(ner_tags) and ner_tags[i] == ner_tags[i + 1] and
          ner_values[i] == ner_values[i + 1] and ner_values[i] != ""):
        word = ner_values[i]
        word = word.replace(">", "").replace("<", "").replace("=", "").replace(
            "%", "").replace("~", "").replace("$", "").replace("£", "").replace(
                "€", "")
        if (re.search("[A-Z]", word) and not (is_date(word)) and not (
            self.is_money(word))):
          ner_values[i] = "A"
        else:
          ner_values[i] = ","
    return ner_tags, ner_values

  def pre_process_sentence(self, tokens, ner_tags, ner_values):
    sentence = []
    tokens = tokens.split("|")
    ner_tags = ner_tags.split("|")
    ner_values = ner_values.split("|")
    ner_tags, ner_values = self.remove_consecutive(ner_tags, ner_values)
    #print "old: ", tokens
    for i in range(len(tokens)):
      word = tokens[i]
      if (ner_values[i] != "" and
          (ner_tags[i] == "NUMBER" or ner_tags[i] == "MONEY" or
           ner_tags[i] == "PERCENT" or ner_tags[i] == "DATE")):
        word = ner_values[i]
        word = word.replace(">", "").replace("<", "").replace("=", "").replace(
            "%", "").replace("~", "").replace("$", "").replace("£", "").replace(
                "€", "")
        if (re.search("[A-Z]", word) and not (is_date(word)) and not (
            self.is_money(word))):
          word = tokens[i]
        if (is_number(ner_values[i])):
          word = float(ner_values[i])
        elif (is_number(word)):
          word = float(word)
        if (tokens[i] == "score"):
          word = "score"
      if (is_number(word)):
        word = float(word)
      if (not (self.annotated_word_reject.has_key(word))):
        if (is_number(word) or is_date(word) or self.is_money(word)):
          sentence.append(word)
        else:
          word = full_normalize(word)
          if (not (self.annotated_word_reject.has_key(word)) and
              bool(re.search("[a-z0-9]", word, re.IGNORECASE))):
            m = re.search(",", word)
            sentence.append(word.replace(",", ""))
    if (len(sentence) == 0):
      sentence.append("UNK")
    return sentence

  def load_annotated_data(self, in_file):
    self.annotated_examples = {}
    self.annotated_tables = {}
    f = tf.gfile.GFile(in_file, "r")
    counter = 0
    for line in f:
      if (counter > 0):
        line = line.strip()
        (question_id, utterance, context, target_value, tokens, lemma_tokens,
         pos_tags, ner_tags, ner_values, target_canon) = line.split("\t")
        question = self.pre_process_sentence(tokens, ner_tags, ner_values)
        target_canon = target_canon.split("|")
        self.annotated_examples[question_id] = WikiExample(
            question_id, question, target_canon, context)
        self.annotated_tables[context] = []
      counter += 1
    print "Annotated examples loaded ", len(self.annotated_examples)
    f.close()

  def is_number_column(self, a):
    for w in a:
      if (len(w) != 1):
        return False
      if (not (is_number(w[0]))):
        return False
    return True

  def convert_table(self, table):
    answer = []
    for i in range(len(table)):
      temp = []
      for j in range(len(table[i])):
        temp.append(" ".join([str(w) for w in table[i][j]]))
      answer.append(temp)
    return answer

  def load_annotated_tables(self):
    for table in self.annotated_tables.keys():
      annotated_table = table.replace("csv", "annotated")
      orig_columns = []
      processed_columns = []
      f = tf.gfile.GFile(os.path.join(self.root_folder, annotated_table), "r")
      counter = 0
      for line in f:
        if (counter > 0):
          line = line.strip()
          line = line + "\t" * (13 - len(line.split("\t")))
          (row, col, read_id, content, tokens, lemma_tokens, pos_tags, ner_tags,
           ner_values, number, date, num2, read_list) = line.split("\t")
        counter += 1
      f.close()
      max_row = int(row)
      max_col = int(col)
      for i in range(max_col + 1):
        orig_columns.append([])
        processed_columns.append([])
        for j in range(max_row + 1):
          orig_columns[i].append(bad_number)
          processed_columns[i].append(bad_number)
      #print orig_columns
      f = tf.gfile.GFile(os.path.join(self.root_folder, annotated_table), "r")
      counter = 0
      column_names = []
      for line in f:
        if (counter > 0):
          line = line.strip()
          line = line + "\t" * (13 - len(line.split("\t")))
          (row, col, read_id, content, tokens, lemma_tokens, pos_tags, ner_tags,
           ner_values, number, date, num2, read_list) = line.split("\t")
          entry = self.pre_process_sentence(tokens, ner_tags, ner_values)
          if (row == "-1"):
            column_names.append(entry)
          else:
            orig_columns[int(col)][int(row)] = entry
            if (len(entry) == 1 and is_number(entry[0])):
              processed_columns[int(col)][int(row)] = float(entry[0])
            else:
              for single_entry in entry:
                if (is_number(single_entry)):
                  processed_columns[int(col)][int(row)] = float(single_entry)
                  break
              nt = ner_tags.split("|")
              nv = ner_values.split("|")
              for i_entry in range(len(tokens.split("|"))):
                if (nt[i_entry] == "DATE" and
                    is_number(nv[i_entry].replace("-", "").replace("X", ""))):
                  processed_columns[int(col)][int(row)] = float(nv[
                      i_entry].replace("-", "").replace("X", ""))
                  #processed_columns[int(col)][int(row)] =  float(nv[i_entry])
            if (len(entry) == 1 and (is_number(entry[0]) or is_date(entry[0]) or
                                     self.is_money(entry[0]))):
              if (len(entry) == 1 and not (is_number(entry[0])) and
                  is_date(entry[0])):
                entry[0] = entry[0].replace("X", "x")
        counter += 1
      word_columns = []
      processed_word_columns = []
      word_column_names = []
      word_column_indices = []
      number_columns = []
      processed_number_columns = []
      number_column_names = []
      number_column_indices = []
      for i in range(max_col + 1):
        if (self.is_number_column(orig_columns[i])):
          number_column_indices.append(i)
          number_column_names.append(column_names[i])
          temp = []
          for w in orig_columns[i]:
            if (is_number(w[0])):
              temp.append(w[0])
          number_columns.append(temp)
          processed_number_columns.append(processed_columns[i])
        else:
          word_column_indices.append(i)
          word_column_names.append(column_names[i])
          word_columns.append(orig_columns[i])
          processed_word_columns.append(processed_columns[i])
      table_info = TableInfo(
          word_columns, word_column_names, word_column_indices, number_columns,
          number_column_names, number_column_indices, processed_word_columns,
          processed_number_columns, orig_columns)
      self.annotated_tables[table] = table_info
      f.close()

  def answer_classification(self):
    lookup_questions = 0
    number_lookup_questions = 0
    word_lookup_questions = 0
    ambiguous_lookup_questions = 0
    number_questions = 0
    bad_questions = 0
    ice_bad_questions = 0
    tot = 0
    got = 0
    ice = {}
    with tf.gfile.GFile(
        self.root_folder + "/arvind-with-norms-2.tsv", mode="r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if (not (self.annotated_examples.has_key(line.split("\t")[0]))):
          continue
        if (len(line.split("\t")) == 4):
          line = line + "\t" * (5 - len(line.split("\t")))
          if (not (is_number(line.split("\t")[2]))):
            ice_bad_questions += 1
        (example_id, ans_index, ans_raw, process_answer,
         matched_cells) = line.split("\t")
        if (ice.has_key(example_id)):
          ice[example_id].append(line.split("\t"))
        else:
          ice[example_id] = [line.split("\t")]
    for q_id in self.annotated_examples.keys():
      tot += 1
      example = self.annotated_examples[q_id]
      table_info = self.annotated_tables[example.table_key]
      # Figure out if the answer is numerical or lookup
      n_cols = len(table_info.orig_columns)
      n_rows = len(table_info.orig_columns[0])
      example.lookup_matrix = np.zeros((n_rows, n_cols))
      exact_matches = {}
      for (example_id, ans_index, ans_raw, process_answer,
           matched_cells) in ice[q_id]:
        for match_cell in matched_cells.split("|"):
          if (len(match_cell.split(",")) == 2):
            (row, col) = match_cell.split(",")
            row = int(row)
            col = int(col)
            if (row >= 0):
              exact_matches[ans_index] = 1
      answer_is_in_table = len(exact_matches) == len(example.answer)
      if (answer_is_in_table):
        for (example_id, ans_index, ans_raw, process_answer,
             matched_cells) in ice[q_id]:
          for match_cell in matched_cells.split("|"):
            if (len(match_cell.split(",")) == 2):
              (row, col) = match_cell.split(",")
              row = int(row)
              col = int(col)
              example.lookup_matrix[row, col] = float(ans_index) + 1.0
      example.lookup_number_answer = 0.0
      if (answer_is_in_table):
        lookup_questions += 1
        if len(example.answer) == 1 and is_number(example.answer[0]):
          example.number_answer = float(example.answer[0])
          number_lookup_questions += 1
          example.is_number_lookup = True
        else:
          #print "word lookup"
          example.calc_answer = example.number_answer = 0.0
          word_lookup_questions += 1
          example.is_word_lookup = True
      else:
        if (len(example.answer) == 1 and is_number(example.answer[0])):
          example.number_answer = example.answer[0]
          example.is_number_calc = True
        else:
          bad_questions += 1
          example.is_bad_example = True
          example.is_unknown_answer = True
      example.is_lookup = example.is_word_lookup or example.is_number_lookup
      if not example.is_word_lookup and not example.is_bad_example:
        number_questions += 1
        example.calc_answer = example.answer[0]
        example.lookup_number_answer = example.calc_answer
      # Split up the lookup matrix into word part and number part
      number_column_indices = table_info.number_column_indices
      word_column_indices = table_info.word_column_indices
      example.word_columns = table_info.word_columns
      example.number_columns = table_info.number_columns
      example.word_column_names = table_info.word_column_names
      example.processed_number_columns = table_info.processed_number_columns
      example.processed_word_columns = table_info.processed_word_columns
      example.number_column_names = table_info.number_column_names
      example.number_lookup_matrix = example.lookup_matrix[:,
                                                           number_column_indices]
      example.word_lookup_matrix = example.lookup_matrix[:, word_column_indices]

  def load(self):
    train_data = []
    dev_data = []
    test_data = []
    self.load_annotated_data(
        os.path.join(self.data_folder, "training.annotated"))
    self.load_annotated_tables()
    self.answer_classification()
    self.train_loader.load()
    self.dev_loader.load()
    for i in range(self.train_loader.num_questions()):
      example = self.train_loader.examples[i]
      example = self.annotated_examples[example]
      train_data.append(example)
    for i in range(self.dev_loader.num_questions()):
      example = self.dev_loader.examples[i]
      dev_data.append(self.annotated_examples[example])

    self.load_annotated_data(
        os.path.join(self.data_folder, "pristine-unseen-tables.annotated"))
    self.load_annotated_tables()
    self.answer_classification()
    self.test_loader.load()
    for i in range(self.test_loader.num_questions()):
      example = self.test_loader.examples[i]
      test_data.append(self.annotated_examples[example])
    return train_data, dev_data, test_data
