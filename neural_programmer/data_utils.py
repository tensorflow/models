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
"""Functions for constructing vocabulary, converting the examples to integer format and building the required masks for batch computation Author: aneelakantan (Arvind Neelakantan)
"""

import copy
import numbers
import numpy as np
import wiki_data


def return_index(a):
  for i in range(len(a)):
    if (a[i] == 1.0):
      return i


def construct_vocab(data, utility, add_word=False):
  ans = []
  for example in data:
    sent = ""
    for word in example.question:
      if (not (isinstance(word, numbers.Number))):
        sent += word + " "
    example.original_nc = copy.deepcopy(example.number_columns)
    example.original_wc = copy.deepcopy(example.word_columns)
    example.original_nc_names = copy.deepcopy(example.number_column_names)
    example.original_wc_names = copy.deepcopy(example.word_column_names)
    if (add_word):
      continue
    number_found = 0
    if (not (example.is_bad_example)):
      for word in example.question:
        if (isinstance(word, numbers.Number)):
          number_found += 1
        else:
          if (not (utility.word_ids.has_key(word))):
            utility.words.append(word)
            utility.word_count[word] = 1
            utility.word_ids[word] = len(utility.word_ids)
            utility.reverse_word_ids[utility.word_ids[word]] = word
          else:
            utility.word_count[word] += 1
      for col_name in example.word_column_names:
        for word in col_name:
          if (isinstance(word, numbers.Number)):
            number_found += 1
          else:
            if (not (utility.word_ids.has_key(word))):
              utility.words.append(word)
              utility.word_count[word] = 1
              utility.word_ids[word] = len(utility.word_ids)
              utility.reverse_word_ids[utility.word_ids[word]] = word
            else:
              utility.word_count[word] += 1
      for col_name in example.number_column_names:
        for word in col_name:
          if (isinstance(word, numbers.Number)):
            number_found += 1
          else:
            if (not (utility.word_ids.has_key(word))):
              utility.words.append(word)
              utility.word_count[word] = 1
              utility.word_ids[word] = len(utility.word_ids)
              utility.reverse_word_ids[utility.word_ids[word]] = word
            else:
              utility.word_count[word] += 1


def word_lookup(word, utility):
  if (utility.word_ids.has_key(word)):
    return word
  else:
    return utility.unk_token


def convert_to_int_2d_and_pad(a, utility):
  ans = []
  #print a
  for b in a:
    temp = []
    if (len(b) > utility.FLAGS.max_entry_length):
      b = b[0:utility.FLAGS.max_entry_length]
    for remaining in range(len(b), utility.FLAGS.max_entry_length):
      b.append(utility.dummy_token)
    assert len(b) == utility.FLAGS.max_entry_length
    for word in b:
      temp.append(utility.word_ids[word_lookup(word, utility)])
    ans.append(temp)
  #print ans
  return ans


def convert_to_bool_and_pad(a, utility):
  a = a.tolist()
  for i in range(len(a)):
    for j in range(len(a[i])):
      if (a[i][j] < 1):
        a[i][j] = False
      else:
        a[i][j] = True
    a[i] = a[i] + [False] * (utility.FLAGS.max_elements - len(a[i]))
  return a


seen_tables = {}


def partial_match(question, table, number):
  answer = []
  match = {}
  for i in range(len(table)):
    temp = []
    for j in range(len(table[i])):
      temp.append(0)
    answer.append(temp)
  for i in range(len(table)):
    for j in range(len(table[i])):
      for word in question:
        if (number):
          if (word == table[i][j]):
            answer[i][j] = 1.0
            match[i] = 1.0
        else:
          if (word in table[i][j]):
            answer[i][j] = 1.0
            match[i] = 1.0
  return answer, match


def exact_match(question, table, number):
  #performs exact match operation
  answer = []
  match = {}
  matched_indices = []
  for i in range(len(table)):
    temp = []
    for j in range(len(table[i])):
      temp.append(0)
    answer.append(temp)
  for i in range(len(table)):
    for j in range(len(table[i])):
      if (number):
        for word in question:
          if (word == table[i][j]):
            match[i] = 1.0
            answer[i][j] = 1.0
      else:
        table_entry = table[i][j]
        for k in range(len(question)):
          if (k + len(table_entry) <= len(question)):
            if (table_entry == question[k:(k + len(table_entry))]):
              #if(len(table_entry) == 1):
              #print "match: ", table_entry, question
              match[i] = 1.0
              answer[i][j] = 1.0
              matched_indices.append((k, len(table_entry)))
  return answer, match, matched_indices


def partial_column_match(question, table, number):
  answer = []
  for i in range(len(table)):
    answer.append(0)
  for i in range(len(table)):
    for word in question:
      if (word in table[i]):
        answer[i] = 1.0
  return answer


def exact_column_match(question, table, number):
  #performs exact match on column names
  answer = []
  matched_indices = []
  for i in range(len(table)):
    answer.append(0)
  for i in range(len(table)):
    table_entry = table[i]
    for k in range(len(question)):
      if (k + len(table_entry) <= len(question)):
        if (table_entry == question[k:(k + len(table_entry))]):
          answer[i] = 1.0
          matched_indices.append((k, len(table_entry)))
  return answer, matched_indices


def get_max_entry(a):
  e = {}
  for w in a:
    if (w != "UNK, "):
      if (e.has_key(w)):
        e[w] += 1
      else:
        e[w] = 1
  if (len(e) > 0):
    (key, val) = sorted(e.items(), key=lambda x: -1 * x[1])[0]
    if (val > 1):
      return key
    else:
      return -1.0
  else:
    return -1.0


def list_join(a):
  ans = ""
  for w in a:
    ans += str(w) + ", "
  return ans


def group_by_max(table, number):
  #computes the most frequently occurring entry in a column
  answer = []
  for i in range(len(table)):
    temp = []
    for j in range(len(table[i])):
      temp.append(0)
    answer.append(temp)
  for i in range(len(table)):
    if (number):
      curr = table[i]
    else:
      curr = [list_join(w) for w in table[i]]
    max_entry = get_max_entry(curr)
    #print i, max_entry
    for j in range(len(curr)):
      if (max_entry == curr[j]):
        answer[i][j] = 1.0
      else:
        answer[i][j] = 0.0
  return answer


def pick_one(a):
  for i in range(len(a)):
    if (1.0 in a[i]):
      return True
  return False


def check_processed_cols(col, utility):
  return True in [
      True for y in col
      if (y != utility.FLAGS.pad_int and y !=
          utility.FLAGS.bad_number_pre_process)
  ]


def complete_wiki_processing(data, utility, train=True):
  #convert to integers and padding
  processed_data = []
  num_bad_examples = 0
  for example in data:
    number_found = 0
    if (example.is_bad_example):
      num_bad_examples += 1
    if (not (example.is_bad_example)):
      example.string_question = example.question[:]
      #entry match
      example.processed_number_columns = example.processed_number_columns[:]
      example.processed_word_columns = example.processed_word_columns[:]
      example.word_exact_match, word_match, matched_indices = exact_match(
          example.string_question, example.original_wc, number=False)
      example.number_exact_match, number_match, _ = exact_match(
          example.string_question, example.original_nc, number=True)
      if (not (pick_one(example.word_exact_match)) and not (
          pick_one(example.number_exact_match))):
        assert len(word_match) == 0
        assert len(number_match) == 0
        example.word_exact_match, word_match = partial_match(
            example.string_question, example.original_wc, number=False)
      #group by max
      example.word_group_by_max = group_by_max(example.original_wc, False)
      example.number_group_by_max = group_by_max(example.original_nc, True)
      #column name match
      example.word_column_exact_match, wcol_matched_indices = exact_column_match(
          example.string_question, example.original_wc_names, number=False)
      example.number_column_exact_match, ncol_matched_indices = exact_column_match(
          example.string_question, example.original_nc_names, number=False)
      if (not (1.0 in example.word_column_exact_match) and not (
          1.0 in example.number_column_exact_match)):
        example.word_column_exact_match = partial_column_match(
            example.string_question, example.original_wc_names, number=False)
        example.number_column_exact_match = partial_column_match(
            example.string_question, example.original_nc_names, number=False)
      if (len(word_match) > 0 or len(number_match) > 0):
        example.question.append(utility.entry_match_token)
      if (1.0 in example.word_column_exact_match or
          1.0 in example.number_column_exact_match):
        example.question.append(utility.column_match_token)
      example.string_question = example.question[:]
      example.number_lookup_matrix = np.transpose(
          example.number_lookup_matrix)[:]
      example.word_lookup_matrix = np.transpose(example.word_lookup_matrix)[:]
      example.columns = example.number_columns[:]
      example.word_columns = example.word_columns[:]
      example.len_total_cols = len(example.word_column_names) + len(
          example.number_column_names)
      example.column_names = example.number_column_names[:]
      example.word_column_names = example.word_column_names[:]
      example.string_column_names = example.number_column_names[:]
      example.string_word_column_names = example.word_column_names[:]
      example.sorted_number_index = []
      example.sorted_word_index = []
      example.column_mask = []
      example.word_column_mask = []
      example.processed_column_mask = []
      example.processed_word_column_mask = []
      example.word_column_entry_mask = []
      example.question_attention_mask = []
      example.question_number = example.question_number_1 = -1
      example.question_attention_mask = []
      example.ordinal_question = []
      example.ordinal_question_one = []
      new_question = []
      if (len(example.number_columns) > 0):
        example.len_col = len(example.number_columns[0])
      else:
        example.len_col = len(example.word_columns[0])
      for (start, length) in matched_indices:
        for j in range(length):
          example.question[start + j] = utility.unk_token
      #print example.question
      for word in example.question:
        if (isinstance(word, numbers.Number) or wiki_data.is_date(word)):
          if (not (isinstance(word, numbers.Number)) and
              wiki_data.is_date(word)):
            word = word.replace("X", "").replace("-", "")
          number_found += 1
          if (number_found == 1):
            example.question_number = word
            if (len(example.ordinal_question) > 0):
              example.ordinal_question[len(example.ordinal_question) - 1] = 1.0
            else:
              example.ordinal_question.append(1.0)
          elif (number_found == 2):
            example.question_number_1 = word
            if (len(example.ordinal_question_one) > 0):
              example.ordinal_question_one[len(example.ordinal_question_one) -
                                           1] = 1.0
            else:
              example.ordinal_question_one.append(1.0)
        else:
          new_question.append(word)
          example.ordinal_question.append(0.0)
          example.ordinal_question_one.append(0.0)
      example.question = [
          utility.word_ids[word_lookup(w, utility)] for w in new_question
      ]
      example.question_attention_mask = [0.0] * len(example.question)
      #when the first question number occurs before a word
      example.ordinal_question = example.ordinal_question[0:len(
          example.question)]
      example.ordinal_question_one = example.ordinal_question_one[0:len(
          example.question)]
      #question-padding
      example.question = [utility.word_ids[utility.dummy_token]] * (
          utility.FLAGS.question_length - len(example.question)
      ) + example.question
      example.question_attention_mask = [-10000.0] * (
          utility.FLAGS.question_length - len(example.question_attention_mask)
      ) + example.question_attention_mask
      example.ordinal_question = [0.0] * (utility.FLAGS.question_length -
                                          len(example.ordinal_question)
                                         ) + example.ordinal_question
      example.ordinal_question_one = [0.0] * (utility.FLAGS.question_length -
                                              len(example.ordinal_question_one)
                                             ) + example.ordinal_question_one
      if (True):
        #number columns and related-padding
        num_cols = len(example.columns)
        start = 0
        for column in example.number_columns:
          if (check_processed_cols(example.processed_number_columns[start],
                                   utility)):
            example.processed_column_mask.append(0.0)
          sorted_index = sorted(
              range(len(example.processed_number_columns[start])),
              key=lambda k: example.processed_number_columns[start][k],
              reverse=True)
          sorted_index = sorted_index + [utility.FLAGS.pad_int] * (
              utility.FLAGS.max_elements - len(sorted_index))
          example.sorted_number_index.append(sorted_index)
          example.columns[start] = column + [utility.FLAGS.pad_int] * (
              utility.FLAGS.max_elements - len(column))
          example.processed_number_columns[start] += [utility.FLAGS.pad_int] * (
              utility.FLAGS.max_elements -
              len(example.processed_number_columns[start]))
          start += 1
          example.column_mask.append(0.0)
        for remaining in range(num_cols, utility.FLAGS.max_number_cols):
          example.sorted_number_index.append([utility.FLAGS.pad_int] *
                                             (utility.FLAGS.max_elements))
          example.columns.append([utility.FLAGS.pad_int] *
                                 (utility.FLAGS.max_elements))
          example.processed_number_columns.append([utility.FLAGS.pad_int] *
                                                  (utility.FLAGS.max_elements))
          example.number_exact_match.append([0.0] *
                                            (utility.FLAGS.max_elements))
          example.number_group_by_max.append([0.0] *
                                             (utility.FLAGS.max_elements))
          example.column_mask.append(-100000000.0)
          example.processed_column_mask.append(-100000000.0)
          example.number_column_exact_match.append(0.0)
          example.column_names.append([utility.dummy_token])
        #word column  and related-padding
        start = 0
        word_num_cols = len(example.word_columns)
        for column in example.word_columns:
          if (check_processed_cols(example.processed_word_columns[start],
                                   utility)):
            example.processed_word_column_mask.append(0.0)
          sorted_index = sorted(
              range(len(example.processed_word_columns[start])),
              key=lambda k: example.processed_word_columns[start][k],
              reverse=True)
          sorted_index = sorted_index + [utility.FLAGS.pad_int] * (
              utility.FLAGS.max_elements - len(sorted_index))
          example.sorted_word_index.append(sorted_index)
          column = convert_to_int_2d_and_pad(column, utility)
          example.word_columns[start] = column + [[
              utility.word_ids[utility.dummy_token]
          ] * utility.FLAGS.max_entry_length] * (utility.FLAGS.max_elements -
                                                 len(column))
          example.processed_word_columns[start] += [utility.FLAGS.pad_int] * (
              utility.FLAGS.max_elements -
              len(example.processed_word_columns[start]))
          example.word_column_entry_mask.append([0] * len(column) + [
              utility.word_ids[utility.dummy_token]
          ] * (utility.FLAGS.max_elements - len(column)))
          start += 1
          example.word_column_mask.append(0.0)
        for remaining in range(word_num_cols, utility.FLAGS.max_word_cols):
          example.sorted_word_index.append([utility.FLAGS.pad_int] *
                                           (utility.FLAGS.max_elements))
          example.word_columns.append([[utility.word_ids[utility.dummy_token]] *
                                       utility.FLAGS.max_entry_length] *
                                      (utility.FLAGS.max_elements))
          example.word_column_entry_mask.append(
              [utility.word_ids[utility.dummy_token]] *
              (utility.FLAGS.max_elements))
          example.word_exact_match.append([0.0] * (utility.FLAGS.max_elements))
          example.word_group_by_max.append([0.0] * (utility.FLAGS.max_elements))
          example.processed_word_columns.append([utility.FLAGS.pad_int] *
                                                (utility.FLAGS.max_elements))
          example.word_column_mask.append(-100000000.0)
          example.processed_word_column_mask.append(-100000000.0)
          example.word_column_exact_match.append(0.0)
          example.word_column_names.append([utility.dummy_token] *
                                           utility.FLAGS.max_entry_length)
        seen_tables[example.table_key] = 1
      #convert column and word column names to integers
      example.column_ids = convert_to_int_2d_and_pad(example.column_names,
                                                     utility)
      example.word_column_ids = convert_to_int_2d_and_pad(
          example.word_column_names, utility)
      for i_em in range(len(example.number_exact_match)):
        example.number_exact_match[i_em] = example.number_exact_match[
            i_em] + [0.0] * (utility.FLAGS.max_elements -
                             len(example.number_exact_match[i_em]))
        example.number_group_by_max[i_em] = example.number_group_by_max[
            i_em] + [0.0] * (utility.FLAGS.max_elements -
                             len(example.number_group_by_max[i_em]))
      for i_em in range(len(example.word_exact_match)):
        example.word_exact_match[i_em] = example.word_exact_match[
            i_em] + [0.0] * (utility.FLAGS.max_elements -
                             len(example.word_exact_match[i_em]))
        example.word_group_by_max[i_em] = example.word_group_by_max[
            i_em] + [0.0] * (utility.FLAGS.max_elements -
                             len(example.word_group_by_max[i_em]))
      example.exact_match = example.number_exact_match + example.word_exact_match
      example.group_by_max = example.number_group_by_max + example.word_group_by_max
      example.exact_column_match = example.number_column_exact_match + example.word_column_exact_match
      #answer and related mask, padding
      if (example.is_lookup):
        example.answer = example.calc_answer
        example.number_print_answer = example.number_lookup_matrix.tolist()
        example.word_print_answer = example.word_lookup_matrix.tolist()
        for i_answer in range(len(example.number_print_answer)):
          example.number_print_answer[i_answer] = example.number_print_answer[
              i_answer] + [0.0] * (utility.FLAGS.max_elements -
                                   len(example.number_print_answer[i_answer]))
        for i_answer in range(len(example.word_print_answer)):
          example.word_print_answer[i_answer] = example.word_print_answer[
              i_answer] + [0.0] * (utility.FLAGS.max_elements -
                                   len(example.word_print_answer[i_answer]))
        example.number_lookup_matrix = convert_to_bool_and_pad(
            example.number_lookup_matrix, utility)
        example.word_lookup_matrix = convert_to_bool_and_pad(
            example.word_lookup_matrix, utility)
        for remaining in range(num_cols, utility.FLAGS.max_number_cols):
          example.number_lookup_matrix.append([False] *
                                              utility.FLAGS.max_elements)
          example.number_print_answer.append([0.0] * utility.FLAGS.max_elements)
        for remaining in range(word_num_cols, utility.FLAGS.max_word_cols):
          example.word_lookup_matrix.append([False] *
                                            utility.FLAGS.max_elements)
          example.word_print_answer.append([0.0] * utility.FLAGS.max_elements)
        example.print_answer = example.number_print_answer + example.word_print_answer
      else:
        example.answer = example.calc_answer
        example.print_answer = [[0.0] * (utility.FLAGS.max_elements)] * (
            utility.FLAGS.max_number_cols + utility.FLAGS.max_word_cols)
      #question_number masks
      if (example.question_number == -1):
        example.question_number_mask = np.zeros([utility.FLAGS.max_elements])
      else:
        example.question_number_mask = np.ones([utility.FLAGS.max_elements])
      if (example.question_number_1 == -1):
        example.question_number_one_mask = -10000.0
      else:
        example.question_number_one_mask = np.float64(0.0)
      if (example.len_col > utility.FLAGS.max_elements):
        continue
      processed_data.append(example)
  return processed_data


def add_special_words(utility):
  utility.words.append(utility.entry_match_token)
  utility.word_ids[utility.entry_match_token] = len(utility.word_ids)
  utility.reverse_word_ids[utility.word_ids[
      utility.entry_match_token]] = utility.entry_match_token
  utility.entry_match_token_id = utility.word_ids[utility.entry_match_token]
  print "entry match token: ", utility.word_ids[
      utility.entry_match_token], utility.entry_match_token_id
  utility.words.append(utility.column_match_token)
  utility.word_ids[utility.column_match_token] = len(utility.word_ids)
  utility.reverse_word_ids[utility.word_ids[
      utility.column_match_token]] = utility.column_match_token
  utility.column_match_token_id = utility.word_ids[utility.column_match_token]
  print "entry match token: ", utility.word_ids[
      utility.column_match_token], utility.column_match_token_id
  utility.words.append(utility.dummy_token)
  utility.word_ids[utility.dummy_token] = len(utility.word_ids)
  utility.reverse_word_ids[utility.word_ids[
      utility.dummy_token]] = utility.dummy_token
  utility.dummy_token_id = utility.word_ids[utility.dummy_token]
  utility.words.append(utility.unk_token)
  utility.word_ids[utility.unk_token] = len(utility.word_ids)
  utility.reverse_word_ids[utility.word_ids[
      utility.unk_token]] = utility.unk_token


def perform_word_cutoff(utility):
  if (utility.FLAGS.word_cutoff > 0):
    for word in utility.word_ids.keys():
      if (utility.word_count.has_key(word) and utility.word_count[word] <
          utility.FLAGS.word_cutoff and word != utility.unk_token and
          word != utility.dummy_token and word != utility.entry_match_token and
          word != utility.column_match_token):
        utility.word_ids.pop(word)
        utility.words.remove(word)


def word_dropout(question, utility):
  if (utility.FLAGS.word_dropout_prob > 0.0):
    new_question = []
    for i in range(len(question)):
      if (question[i] != utility.dummy_token_id and
          utility.random.random() > utility.FLAGS.word_dropout_prob):
        new_question.append(utility.word_ids[utility.unk_token])
      else:
        new_question.append(question[i])
    return new_question
  else:
    return question


def generate_feed_dict(data, curr, batch_size, gr, train=False, utility=None):
  #prepare feed dict dictionary
  feed_dict = {}
  feed_examples = []
  for j in range(batch_size):
    feed_examples.append(data[curr + j])
  if (train):
    feed_dict[gr.batch_question] = [
        word_dropout(feed_examples[j].question, utility)
        for j in range(batch_size)
    ]
  else:
    feed_dict[gr.batch_question] = [
        feed_examples[j].question for j in range(batch_size)
    ]
  feed_dict[gr.batch_question_attention_mask] = [
      feed_examples[j].question_attention_mask for j in range(batch_size)
  ]
  feed_dict[
      gr.batch_answer] = [feed_examples[j].answer for j in range(batch_size)]
  feed_dict[gr.batch_number_column] = [
      feed_examples[j].columns for j in range(batch_size)
  ]
  feed_dict[gr.batch_processed_number_column] = [
      feed_examples[j].processed_number_columns for j in range(batch_size)
  ]
  feed_dict[gr.batch_processed_sorted_index_number_column] = [
      feed_examples[j].sorted_number_index for j in range(batch_size)
  ]
  feed_dict[gr.batch_processed_sorted_index_word_column] = [
      feed_examples[j].sorted_word_index for j in range(batch_size)
  ]
  feed_dict[gr.batch_question_number] = np.array(
      [feed_examples[j].question_number for j in range(batch_size)]).reshape(
          (batch_size, 1))
  feed_dict[gr.batch_question_number_one] = np.array(
      [feed_examples[j].question_number_1 for j in range(batch_size)]).reshape(
          (batch_size, 1))
  feed_dict[gr.batch_question_number_mask] = [
      feed_examples[j].question_number_mask for j in range(batch_size)
  ]
  feed_dict[gr.batch_question_number_one_mask] = np.array(
      [feed_examples[j].question_number_one_mask for j in range(batch_size)
      ]).reshape((batch_size, 1))
  feed_dict[gr.batch_print_answer] = [
      feed_examples[j].print_answer for j in range(batch_size)
  ]
  feed_dict[gr.batch_exact_match] = [
      feed_examples[j].exact_match for j in range(batch_size)
  ]
  feed_dict[gr.batch_group_by_max] = [
      feed_examples[j].group_by_max for j in range(batch_size)
  ]
  feed_dict[gr.batch_column_exact_match] = [
      feed_examples[j].exact_column_match for j in range(batch_size)
  ]
  feed_dict[gr.batch_ordinal_question] = [
      feed_examples[j].ordinal_question for j in range(batch_size)
  ]
  feed_dict[gr.batch_ordinal_question_one] = [
      feed_examples[j].ordinal_question_one for j in range(batch_size)
  ]
  feed_dict[gr.batch_number_column_mask] = [
      feed_examples[j].column_mask for j in range(batch_size)
  ]
  feed_dict[gr.batch_number_column_names] = [
      feed_examples[j].column_ids for j in range(batch_size)
  ]
  feed_dict[gr.batch_processed_word_column] = [
      feed_examples[j].processed_word_columns for j in range(batch_size)
  ]
  feed_dict[gr.batch_word_column_mask] = [
      feed_examples[j].word_column_mask for j in range(batch_size)
  ]
  feed_dict[gr.batch_word_column_names] = [
      feed_examples[j].word_column_ids for j in range(batch_size)
  ]
  feed_dict[gr.batch_word_column_entry_mask] = [
      feed_examples[j].word_column_entry_mask for j in range(batch_size)
  ]
  return feed_dict
