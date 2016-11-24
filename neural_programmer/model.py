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
"""Author: aneelakantan (Arvind Neelakantan)
"""

import numpy as np
import tensorflow as tf
import nn_utils


class Graph():

  def __init__(self, utility, batch_size, max_passes, mode="train"):
    self.utility = utility
    self.data_type = self.utility.tf_data_type[self.utility.FLAGS.data_type]
    self.max_elements = self.utility.FLAGS.max_elements
    max_elements = self.utility.FLAGS.max_elements
    self.num_cols = self.utility.FLAGS.max_number_cols
    self.num_word_cols = self.utility.FLAGS.max_word_cols
    self.question_length = self.utility.FLAGS.question_length
    self.batch_size = batch_size
    self.max_passes = max_passes
    self.mode = mode
    self.embedding_dims = self.utility.FLAGS.embedding_dims
    #input question and a mask
    self.batch_question = tf.placeholder(tf.int32,
                                         [batch_size, self.question_length])
    self.batch_question_attention_mask = tf.placeholder(
        self.data_type, [batch_size, self.question_length])
    #ground truth scalar answer and lookup answer
    self.batch_answer = tf.placeholder(self.data_type, [batch_size])
    self.batch_print_answer = tf.placeholder(
        self.data_type,
        [batch_size, self.num_cols + self.num_word_cols, max_elements])
    #number columns and its processed version
    self.batch_number_column = tf.placeholder(
        self.data_type, [batch_size, self.num_cols, max_elements
                        ])  #columns with numeric entries
    self.batch_processed_number_column = tf.placeholder(
        self.data_type, [batch_size, self.num_cols, max_elements])
    self.batch_processed_sorted_index_number_column = tf.placeholder(
        tf.int32, [batch_size, self.num_cols, max_elements])
    #word columns and its processed version
    self.batch_processed_word_column = tf.placeholder(
        self.data_type, [batch_size, self.num_word_cols, max_elements])
    self.batch_processed_sorted_index_word_column = tf.placeholder(
        tf.int32, [batch_size, self.num_word_cols, max_elements])
    self.batch_word_column_entry_mask = tf.placeholder(
        tf.int32, [batch_size, self.num_word_cols, max_elements])
    #names of word and number columns along with their mask
    self.batch_word_column_names = tf.placeholder(
        tf.int32,
        [batch_size, self.num_word_cols, self.utility.FLAGS.max_entry_length])
    self.batch_word_column_mask = tf.placeholder(
        self.data_type, [batch_size, self.num_word_cols])
    self.batch_number_column_names = tf.placeholder(
        tf.int32,
        [batch_size, self.num_cols, self.utility.FLAGS.max_entry_length])
    self.batch_number_column_mask = tf.placeholder(self.data_type,
                                                   [batch_size, self.num_cols])
    #exact match and group by max operation
    self.batch_exact_match = tf.placeholder(
        self.data_type,
        [batch_size, self.num_cols + self.num_word_cols, max_elements])
    self.batch_column_exact_match = tf.placeholder(
        self.data_type, [batch_size, self.num_cols + self.num_word_cols])
    self.batch_group_by_max = tf.placeholder(
        self.data_type,
        [batch_size, self.num_cols + self.num_word_cols, max_elements])
    #numbers in the question along with their position. This is used to compute arguments to the comparison operations
    self.batch_question_number = tf.placeholder(self.data_type, [batch_size, 1])
    self.batch_question_number_one = tf.placeholder(self.data_type,
                                                    [batch_size, 1])
    self.batch_question_number_mask = tf.placeholder(
        self.data_type, [batch_size, max_elements])
    self.batch_question_number_one_mask = tf.placeholder(self.data_type,
                                                         [batch_size, 1])
    self.batch_ordinal_question = tf.placeholder(
        self.data_type, [batch_size, self.question_length])
    self.batch_ordinal_question_one = tf.placeholder(
        self.data_type, [batch_size, self.question_length])

  def LSTM_question_embedding(self, sentence, sentence_length):
    #LSTM processes the input question
    lstm_params = "question_lstm"
    hidden_vectors = []
    sentence = self.batch_question
    question_hidden = tf.zeros(
        [self.batch_size, self.utility.FLAGS.embedding_dims], self.data_type)
    question_c_hidden = tf.zeros(
        [self.batch_size, self.utility.FLAGS.embedding_dims], self.data_type)
    if (self.utility.FLAGS.rnn_dropout > 0.0):
      if (self.mode == "train"):
        rnn_dropout_mask = tf.cast(
            tf.random_uniform(
                tf.shape(question_hidden), minval=0.0, maxval=1.0) <
            self.utility.FLAGS.rnn_dropout,
            self.data_type) / self.utility.FLAGS.rnn_dropout
      else:
        rnn_dropout_mask = tf.ones_like(question_hidden)
    for question_iterator in range(self.question_length):
      curr_word = sentence[:, question_iterator]
      question_vector = nn_utils.apply_dropout(
          nn_utils.get_embedding(curr_word, self.utility, self.params),
          self.utility.FLAGS.dropout, self.mode)
      question_hidden, question_c_hidden = nn_utils.LSTMCell(
          question_vector, question_hidden, question_c_hidden, lstm_params,
          self.params)
      if (self.utility.FLAGS.rnn_dropout > 0.0):
        question_hidden = question_hidden * rnn_dropout_mask
      hidden_vectors.append(tf.expand_dims(question_hidden, 0))
    hidden_vectors = tf.concat(0, hidden_vectors)
    return question_hidden, hidden_vectors

  def history_recurrent_step(self, curr_hprev, hprev):
    #A single RNN step for controller or history RNN
    return tf.tanh(
        tf.matmul(
            tf.concat(1, [hprev, curr_hprev]), self.params[
                "history_recurrent"])) + self.params["history_recurrent_bias"]

  def question_number_softmax(self, hidden_vectors):
    #Attention on quetsion to decide the question number to passed to comparison ops
    def compute_ans(op_embedding, comparison):
      op_embedding = tf.expand_dims(op_embedding, 0)
      #dot product of operation embedding with hidden state to the left of the number occurence
      first = tf.transpose(
          tf.matmul(op_embedding,
                    tf.transpose(
                        tf.reduce_sum(hidden_vectors * tf.tile(
                            tf.expand_dims(
                                tf.transpose(self.batch_ordinal_question), 2),
                            [1, 1, self.utility.FLAGS.embedding_dims]), 0))))
      second = self.batch_question_number_one_mask + tf.transpose(
          tf.matmul(op_embedding,
                    tf.transpose(
                        tf.reduce_sum(hidden_vectors * tf.tile(
                            tf.expand_dims(
                                tf.transpose(self.batch_ordinal_question_one), 2
                            ), [1, 1, self.utility.FLAGS.embedding_dims]), 0))))
      question_number_softmax = tf.nn.softmax(tf.concat(1, [first, second]))
      if (self.mode == "test"):
        cond = tf.equal(question_number_softmax,
                        tf.reshape(
                            tf.reduce_max(question_number_softmax, 1),
                            [self.batch_size, 1]))
        question_number_softmax = tf.select(
            cond,
            tf.fill(tf.shape(question_number_softmax), 1.0),
            tf.fill(tf.shape(question_number_softmax), 0.0))
        question_number_softmax = tf.cast(question_number_softmax,
                                          self.data_type)
      ans = tf.reshape(
          tf.reduce_sum(question_number_softmax * tf.concat(
              1, [self.batch_question_number, self.batch_question_number_one]),
                        1), [self.batch_size, 1])
      return ans

    def compute_op_position(op_name):
      for i in range(len(self.utility.operations_set)):
        if (op_name == self.utility.operations_set[i]):
          return i

    def compute_question_number(op_name):
      op_embedding = tf.nn.embedding_lookup(self.params_unit,
                                            compute_op_position(op_name))
      return compute_ans(op_embedding, op_name)

    curr_greater_question_number = compute_question_number("greater")
    curr_lesser_question_number = compute_question_number("lesser")
    curr_geq_question_number = compute_question_number("geq")
    curr_leq_question_number = compute_question_number("leq")
    return curr_greater_question_number, curr_lesser_question_number, curr_geq_question_number, curr_leq_question_number

  def perform_attention(self, context_vector, hidden_vectors, length, mask):
    #Performs attention on hiddent_vectors using context vector
    context_vector = tf.tile(
        tf.expand_dims(context_vector, 0), [length, 1, 1])  #time * bs * d
    attention_softmax = tf.nn.softmax(
        tf.transpose(tf.reduce_sum(context_vector * hidden_vectors, 2)) +
        mask)  #batch_size * time
    attention_softmax = tf.tile(
        tf.expand_dims(tf.transpose(attention_softmax), 2),
        [1, 1, self.embedding_dims])
    ans_vector = tf.reduce_sum(attention_softmax * hidden_vectors, 0)
    return ans_vector

  #computes embeddings for column names using parameters of question module
  def get_column_hidden_vectors(self):
    #vector representations for the column names
    self.column_hidden_vectors = tf.reduce_sum(
        nn_utils.get_embedding(self.batch_number_column_names, self.utility,
                               self.params), 2)
    self.word_column_hidden_vectors = tf.reduce_sum(
        nn_utils.get_embedding(self.batch_word_column_names, self.utility,
                               self.params), 2)

  def create_summary_embeddings(self):
    #embeddings for each text entry in the table using parameters of the question module
    self.summary_text_entry_embeddings = tf.reduce_sum(
        tf.expand_dims(self.batch_exact_match, 3) * tf.expand_dims(
            tf.expand_dims(
                tf.expand_dims(
                    nn_utils.get_embedding(self.utility.entry_match_token_id,
                                           self.utility, self.params), 0), 1),
            2), 2)

  def compute_column_softmax(self, column_controller_vector, time_step):
    #compute softmax over all the columns using column controller vector
    column_controller_vector = tf.tile(
        tf.expand_dims(column_controller_vector, 1),
        [1, self.num_cols + self.num_word_cols, 1])  #max_cols * bs * d
    column_controller_vector = nn_utils.apply_dropout(
        column_controller_vector, self.utility.FLAGS.dropout, self.mode)
    self.full_column_hidden_vectors = tf.concat(
        1, [self.column_hidden_vectors, self.word_column_hidden_vectors])
    self.full_column_hidden_vectors += self.summary_text_entry_embeddings
    self.full_column_hidden_vectors = nn_utils.apply_dropout(
        self.full_column_hidden_vectors, self.utility.FLAGS.dropout, self.mode)
    column_logits = tf.reduce_sum(
        column_controller_vector * self.full_column_hidden_vectors, 2) + (
            self.params["word_match_feature_column_name"] *
            self.batch_column_exact_match) + self.full_column_mask
    column_softmax = tf.nn.softmax(column_logits)  #batch_size * max_cols
    return column_softmax

  def compute_first_or_last(self, select, first=True):
    #perform first ot last operation on row select with probabilistic row selection
    answer = tf.zeros_like(select)
    running_sum = tf.zeros([self.batch_size, 1], self.data_type)
    for i in range(self.max_elements):
      if (first):
        current = tf.slice(select, [0, i], [self.batch_size, 1])
      else:
        current = tf.slice(select, [0, self.max_elements - 1 - i],
                           [self.batch_size, 1])
      curr_prob = current * (1 - running_sum)
      curr_prob = curr_prob * tf.cast(curr_prob >= 0.0, self.data_type)
      running_sum += curr_prob
      temp_ans = []
      curr_prob = tf.expand_dims(tf.reshape(curr_prob, [self.batch_size]), 0)
      for i_ans in range(self.max_elements):
        if (not (first) and i_ans == self.max_elements - 1 - i):
          temp_ans.append(curr_prob)
        elif (first and i_ans == i):
          temp_ans.append(curr_prob)
        else:
          temp_ans.append(tf.zeros_like(curr_prob))
      temp_ans = tf.transpose(tf.concat(0, temp_ans))
      answer += temp_ans
    return answer

  def make_hard_softmax(self, softmax):
    #converts soft selection to hard selection. used at test time
    cond = tf.equal(
        softmax, tf.reshape(tf.reduce_max(softmax, 1), [self.batch_size, 1]))
    softmax = tf.select(
        cond, tf.fill(tf.shape(softmax), 1.0), tf.fill(tf.shape(softmax), 0.0))
    softmax = tf.cast(softmax, self.data_type)
    return softmax

  def compute_max_or_min(self, select, maxi=True):
    #computes the argmax and argmin of a column with probabilistic row selection
    answer = tf.zeros([
        self.batch_size, self.num_cols + self.num_word_cols, self.max_elements
    ], self.data_type)
    sum_prob = tf.zeros([self.batch_size, self.num_cols + self.num_word_cols],
                        self.data_type)
    for j in range(self.max_elements):
      if (maxi):
        curr_pos = j
      else:
        curr_pos = self.max_elements - 1 - j
      select_index = tf.slice(self.full_processed_sorted_index_column,
                              [0, 0, curr_pos], [self.batch_size, -1, 1])
      select_mask = tf.equal(
          tf.tile(
              tf.expand_dims(
                  tf.tile(
                      tf.expand_dims(tf.range(self.max_elements), 0),
                      [self.batch_size, 1]), 1),
              [1, self.num_cols + self.num_word_cols, 1]), select_index)
      curr_prob = tf.expand_dims(select, 1) * tf.cast(
          select_mask, self.data_type) * self.select_bad_number_mask
      curr_prob = curr_prob * tf.expand_dims((1 - sum_prob), 2)
      curr_prob = curr_prob * tf.expand_dims(
          tf.cast((1 - sum_prob) > 0.0, self.data_type), 2)
      answer = tf.select(select_mask, curr_prob, answer)
      sum_prob += tf.reduce_sum(curr_prob, 2)
    return answer

  def perform_operations(self, softmax, full_column_softmax, select,
                         prev_select_1, curr_pass):
    #performs all the 15 operations. computes scalar output, lookup answer and row selector
    column_softmax = tf.slice(full_column_softmax, [0, 0],
                              [self.batch_size, self.num_cols])
    word_column_softmax = tf.slice(full_column_softmax, [0, self.num_cols],
                                   [self.batch_size, self.num_word_cols])
    init_max = self.compute_max_or_min(select, maxi=True)
    init_min = self.compute_max_or_min(select, maxi=False)
    #operations that are column  independent
    count = tf.reshape(tf.reduce_sum(select, 1), [self.batch_size, 1])
    select_full_column_softmax = tf.tile(
        tf.expand_dims(full_column_softmax, 2),
        [1, 1, self.max_elements
        ])  #BS * (max_cols + max_word_cols) * max_elements
    select_word_column_softmax = tf.tile(
        tf.expand_dims(word_column_softmax, 2),
        [1, 1, self.max_elements])  #BS * max_word_cols * max_elements
    select_greater = tf.reduce_sum(
        self.init_select_greater * select_full_column_softmax,
        1) * self.batch_question_number_mask  #BS * max_elements
    select_lesser = tf.reduce_sum(
        self.init_select_lesser * select_full_column_softmax,
        1) * self.batch_question_number_mask  #BS * max_elements
    select_geq = tf.reduce_sum(
        self.init_select_geq * select_full_column_softmax,
        1) * self.batch_question_number_mask  #BS * max_elements
    select_leq = tf.reduce_sum(
        self.init_select_leq * select_full_column_softmax,
        1) * self.batch_question_number_mask  #BS * max_elements
    select_max = tf.reduce_sum(init_max * select_full_column_softmax,
                               1)  #BS * max_elements
    select_min = tf.reduce_sum(init_min * select_full_column_softmax,
                               1)  #BS * max_elements
    select_prev = tf.concat(1, [
        tf.slice(select, [0, 1], [self.batch_size, self.max_elements - 1]),
        tf.cast(tf.zeros([self.batch_size, 1]), self.data_type)
    ])
    select_next = tf.concat(1, [
        tf.cast(tf.zeros([self.batch_size, 1]), self.data_type), tf.slice(
            select, [0, 0], [self.batch_size, self.max_elements - 1])
    ])
    select_last_rs = self.compute_first_or_last(select, False)
    select_first_rs = self.compute_first_or_last(select, True)
    select_word_match = tf.reduce_sum(self.batch_exact_match *
                                      select_full_column_softmax, 1)
    select_group_by_max = tf.reduce_sum(self.batch_group_by_max *
                                        select_full_column_softmax, 1)
    length_content = 1
    length_select = 13
    length_print = 1
    values = tf.concat(1, [count])
    softmax_content = tf.slice(softmax, [0, 0],
                               [self.batch_size, length_content])
    #compute scalar output
    output = tf.reduce_sum(tf.mul(softmax_content, values), 1)
    #compute lookup answer
    softmax_print = tf.slice(softmax, [0, length_content + length_select],
                             [self.batch_size, length_print])
    curr_print = select_full_column_softmax * tf.tile(
        tf.expand_dims(select, 1),
        [1, self.num_cols + self.num_word_cols, 1
        ])  #BS * max_cols * max_elements (conisders only column)
    self.batch_lookup_answer = curr_print * tf.tile(
        tf.expand_dims(softmax_print, 2),
        [1, self.num_cols + self.num_word_cols, self.max_elements
        ])  #BS * max_cols * max_elements
    self.batch_lookup_answer = self.batch_lookup_answer * self.select_full_mask
    #compute row select
    softmax_select = tf.slice(softmax, [0, length_content],
                              [self.batch_size, length_select])
    select_lists = [
        tf.expand_dims(select_prev, 1), tf.expand_dims(select_next, 1),
        tf.expand_dims(select_first_rs, 1), tf.expand_dims(select_last_rs, 1),
        tf.expand_dims(select_group_by_max, 1),
        tf.expand_dims(select_greater, 1), tf.expand_dims(select_lesser, 1),
        tf.expand_dims(select_geq, 1), tf.expand_dims(select_leq, 1),
        tf.expand_dims(select_max, 1), tf.expand_dims(select_min, 1),
        tf.expand_dims(select_word_match, 1),
        tf.expand_dims(self.reset_select, 1)
    ]
    select = tf.reduce_sum(
        tf.tile(tf.expand_dims(softmax_select, 2), [1, 1, self.max_elements]) *
        tf.concat(1, select_lists), 1)
    select = select * self.select_whole_mask
    return output, select

  def one_pass(self, select, question_embedding, hidden_vectors, hprev,
               prev_select_1, curr_pass):
    #Performs one timestep which involves selecting an operation and a column
    attention_vector = self.perform_attention(
        hprev, hidden_vectors, self.question_length,
        self.batch_question_attention_mask)  #batch_size * embedding_dims
    controller_vector = tf.nn.relu(
        tf.matmul(hprev, self.params["controller_prev"]) + tf.matmul(
            tf.concat(1, [question_embedding, attention_vector]), self.params[
                "controller"]))
    column_controller_vector = tf.nn.relu(
        tf.matmul(hprev, self.params["column_controller_prev"]) + tf.matmul(
            tf.concat(1, [question_embedding, attention_vector]), self.params[
                "column_controller"]))
    controller_vector = nn_utils.apply_dropout(
        controller_vector, self.utility.FLAGS.dropout, self.mode)
    self.operation_logits = tf.matmul(controller_vector,
                                      tf.transpose(self.params_unit))
    softmax = tf.nn.softmax(self.operation_logits)
    soft_softmax = softmax
    #compute column softmax: bs * max_columns
    weighted_op_representation = tf.transpose(
        tf.matmul(tf.transpose(self.params_unit), tf.transpose(softmax)))
    column_controller_vector = tf.nn.relu(
        tf.matmul(
            tf.concat(1, [
                column_controller_vector, weighted_op_representation
            ]), self.params["break_conditional"]))
    full_column_softmax = self.compute_column_softmax(column_controller_vector,
                                                      curr_pass)
    soft_column_softmax = full_column_softmax
    if (self.mode == "test"):
      full_column_softmax = self.make_hard_softmax(full_column_softmax)
      softmax = self.make_hard_softmax(softmax)
    output, select = self.perform_operations(softmax, full_column_softmax,
                                             select, prev_select_1, curr_pass)
    return output, select, softmax, soft_softmax, full_column_softmax, soft_column_softmax

  def compute_lookup_error(self, val):
    #computes lookup error.
    cond = tf.equal(self.batch_print_answer, val)
    inter = tf.select(
        cond, self.init_print_error,
        tf.tile(
            tf.reshape(tf.constant(1e10, self.data_type), [1, 1, 1]), [
                self.batch_size, self.utility.FLAGS.max_word_cols +
                self.utility.FLAGS.max_number_cols,
                self.utility.FLAGS.max_elements
            ]))
    return tf.reduce_min(tf.reduce_min(inter, 1), 1) * tf.cast(
        tf.greater(
            tf.reduce_sum(tf.reduce_sum(tf.cast(cond, self.data_type), 1), 1),
            0.0), self.data_type)

  def soft_min(self, x, y):
    return tf.maximum(-1.0 * (1 / (
        self.utility.FLAGS.soft_min_value + 0.0)) * tf.log(
            tf.exp(-self.utility.FLAGS.soft_min_value * x) + tf.exp(
                -self.utility.FLAGS.soft_min_value * y)), tf.zeros_like(x))

  def error_computation(self):
    #computes the error of each example in a batch
    math_error = 0.5 * tf.square(tf.sub(self.scalar_output, self.batch_answer))
    #scale math error
    math_error = math_error / self.rows
    math_error = tf.minimum(math_error, self.utility.FLAGS.max_math_error *
                            tf.ones(tf.shape(math_error), self.data_type))
    self.init_print_error = tf.select(
        self.batch_gold_select, -1 * tf.log(self.batch_lookup_answer + 1e-300 +
                                            self.invert_select_full_mask), -1 *
        tf.log(1 - self.batch_lookup_answer)) * self.select_full_mask
    print_error_1 = self.init_print_error * tf.cast(
        tf.equal(self.batch_print_answer, 0.0), self.data_type)
    print_error = tf.reduce_sum(tf.reduce_sum((print_error_1), 1), 1)
    for val in range(1, 58):
      print_error += self.compute_lookup_error(val + 0.0)
    print_error = print_error * self.utility.FLAGS.print_cost / self.num_entries
    if (self.mode == "train"):
      error = tf.select(
          tf.logical_and(
              tf.not_equal(self.batch_answer, 0.0),
              tf.not_equal(
                  tf.reduce_sum(tf.reduce_sum(self.batch_print_answer, 1), 1),
                  0.0)),
          self.soft_min(math_error, print_error),
          tf.select(
              tf.not_equal(self.batch_answer, 0.0), math_error, print_error))
    else:
      error = tf.select(
          tf.logical_and(
              tf.equal(self.scalar_output, 0.0),
              tf.equal(
                  tf.reduce_sum(tf.reduce_sum(self.batch_lookup_answer, 1), 1),
                  0.0)),
          tf.ones_like(math_error),
          tf.select(
              tf.equal(self.scalar_output, 0.0), print_error, math_error))
    return error

  def batch_process(self):
    #Computes loss and fraction of correct examples in a batch.
    self.params_unit = nn_utils.apply_dropout(
        self.params["unit"], self.utility.FLAGS.dropout, self.mode)
    batch_size = self.batch_size
    max_passes = self.max_passes
    num_timesteps = 1
    max_elements = self.max_elements
    select = tf.cast(
        tf.fill([self.batch_size, max_elements], 1.0), self.data_type)
    hprev = tf.cast(
        tf.fill([self.batch_size, self.embedding_dims], 0.0),
        self.data_type)  #running sum of the hidden states of the model
    output = tf.cast(tf.fill([self.batch_size, 1], 0.0),
                     self.data_type)  #output of the model
    correct = tf.cast(
        tf.fill([1], 0.0), self.data_type
    )  #to compute accuracy, returns number of correct examples for this batch
    total_error = 0.0
    prev_select_1 = tf.zeros_like(select)
    self.create_summary_embeddings()
    self.get_column_hidden_vectors()
    #get question embedding
    question_embedding, hidden_vectors = self.LSTM_question_embedding(
        self.batch_question, self.question_length)
    #compute arguments for comparison operation
    greater_question_number, lesser_question_number, geq_question_number, leq_question_number = self.question_number_softmax(
        hidden_vectors)
    self.init_select_greater = tf.cast(
        tf.greater(self.full_processed_column,
                   tf.expand_dims(greater_question_number, 2)), self.
        data_type) * self.select_bad_number_mask  #bs * max_cols * max_elements
    self.init_select_lesser = tf.cast(
        tf.less(self.full_processed_column,
                tf.expand_dims(lesser_question_number, 2)), self.
        data_type) * self.select_bad_number_mask  #bs * max_cols * max_elements
    self.init_select_geq = tf.cast(
        tf.greater_equal(self.full_processed_column,
                         tf.expand_dims(geq_question_number, 2)), self.
        data_type) * self.select_bad_number_mask  #bs * max_cols * max_elements
    self.init_select_leq = tf.cast(
        tf.less_equal(self.full_processed_column,
                      tf.expand_dims(leq_question_number, 2)), self.
        data_type) * self.select_bad_number_mask  #bs * max_cols * max_elements
    self.init_select_word_match = 0
    if (self.utility.FLAGS.rnn_dropout > 0.0):
      if (self.mode == "train"):
        history_rnn_dropout_mask = tf.cast(
            tf.random_uniform(
                tf.shape(hprev), minval=0.0, maxval=1.0) <
            self.utility.FLAGS.rnn_dropout,
            self.data_type) / self.utility.FLAGS.rnn_dropout
      else:
        history_rnn_dropout_mask = tf.ones_like(hprev)
    select = select * self.select_whole_mask
    self.batch_log_prob = tf.zeros([self.batch_size], dtype=self.data_type)
    #Perform max_passes and at each  pass select operation and column
    for curr_pass in range(max_passes):
      print "step: ", curr_pass
      output, select, softmax, soft_softmax, column_softmax, soft_column_softmax = self.one_pass(
          select, question_embedding, hidden_vectors, hprev, prev_select_1,
          curr_pass)
      prev_select_1 = select
      #compute input to history RNN
      input_op = tf.transpose(
          tf.matmul(
              tf.transpose(self.params_unit), tf.transpose(
                  soft_softmax)))  #weighted average of emebdding of operations
      input_col = tf.reduce_sum(
          tf.expand_dims(soft_column_softmax, 2) *
          self.full_column_hidden_vectors, 1)
      history_input = tf.concat(1, [input_op, input_col])
      history_input = nn_utils.apply_dropout(
          history_input, self.utility.FLAGS.dropout, self.mode)
      hprev = self.history_recurrent_step(history_input, hprev)
      if (self.utility.FLAGS.rnn_dropout > 0.0):
        hprev = hprev * history_rnn_dropout_mask
    self.scalar_output = output
    error = self.error_computation()
    cond = tf.less(error, 0.0001, name="cond")
    correct_add = tf.select(
        cond, tf.fill(tf.shape(cond), 1.0), tf.fill(tf.shape(cond), 0.0))
    correct = tf.reduce_sum(correct_add)
    error = error / batch_size
    total_error = tf.reduce_sum(error)
    total_correct = correct / batch_size
    return total_error, total_correct

  def compute_error(self):
    #Sets mask variables and performs batch processing
    self.batch_gold_select = self.batch_print_answer > 0.0
    self.full_column_mask = tf.concat(
        1, [self.batch_number_column_mask, self.batch_word_column_mask])
    self.full_processed_column = tf.concat(
        1,
        [self.batch_processed_number_column, self.batch_processed_word_column])
    self.full_processed_sorted_index_column = tf.concat(1, [
        self.batch_processed_sorted_index_number_column,
        self.batch_processed_sorted_index_word_column
    ])
    self.select_bad_number_mask = tf.cast(
        tf.logical_and(
            tf.not_equal(self.full_processed_column,
                         self.utility.FLAGS.pad_int),
            tf.not_equal(self.full_processed_column,
                         self.utility.FLAGS.bad_number_pre_process)),
        self.data_type)
    self.select_mask = tf.cast(
        tf.logical_not(
            tf.equal(self.batch_number_column, self.utility.FLAGS.pad_int)),
        self.data_type)
    self.select_word_mask = tf.cast(
        tf.logical_not(
            tf.equal(self.batch_word_column_entry_mask,
                     self.utility.dummy_token_id)), self.data_type)
    self.select_full_mask = tf.concat(
        1, [self.select_mask, self.select_word_mask])
    self.select_whole_mask = tf.maximum(
        tf.reshape(
            tf.slice(self.select_mask, [0, 0, 0],
                     [self.batch_size, 1, self.max_elements]),
            [self.batch_size, self.max_elements]),
        tf.reshape(
            tf.slice(self.select_word_mask, [0, 0, 0],
                     [self.batch_size, 1, self.max_elements]),
            [self.batch_size, self.max_elements]))
    self.invert_select_full_mask = tf.cast(
        tf.concat(1, [
            tf.equal(self.batch_number_column, self.utility.FLAGS.pad_int),
            tf.equal(self.batch_word_column_entry_mask,
                     self.utility.dummy_token_id)
        ]), self.data_type)
    self.batch_lookup_answer = tf.zeros(tf.shape(self.batch_gold_select))
    self.reset_select = self.select_whole_mask
    self.rows = tf.reduce_sum(self.select_whole_mask, 1)
    self.num_entries = tf.reshape(
        tf.reduce_sum(tf.reduce_sum(self.select_full_mask, 1), 1),
        [self.batch_size])
    self.final_error, self.final_correct = self.batch_process()
    return self.final_error

  def create_graph(self, params, global_step):
    #Creates the graph to compute error, gradient computation and updates parameters
    self.params = params
    batch_size = self.batch_size
    learning_rate = tf.cast(self.utility.FLAGS.learning_rate, self.data_type)
    self.total_cost = self.compute_error() 
    optimize_params = self.params.values()
    optimize_names = self.params.keys()
    print "optimize params ", optimize_names
    if (self.utility.FLAGS.l2_regularizer > 0.0):
      reg_cost = 0.0
      for ind_param in self.params.keys():
        reg_cost += tf.nn.l2_loss(self.params[ind_param])
      self.total_cost += self.utility.FLAGS.l2_regularizer * reg_cost
    grads = tf.gradients(self.total_cost, optimize_params, name="gradients")
    grad_norm = 0.0
    for p, name in zip(grads, optimize_names):
      print "grads: ", p, name
      if isinstance(p, tf.IndexedSlices):
        grad_norm += tf.reduce_sum(p.values * p.values)
      elif not (p == None):
        grad_norm += tf.reduce_sum(p * p)
    grad_norm = tf.sqrt(grad_norm)
    max_grad_norm = np.float32(self.utility.FLAGS.clip_gradients).astype(
        self.utility.np_data_type[self.utility.FLAGS.data_type])
    grad_scale = tf.minimum(
        tf.cast(1.0, self.data_type), max_grad_norm / grad_norm)
    clipped_grads = list()
    for p in grads:
      if isinstance(p, tf.IndexedSlices):
        tmp = p.values * grad_scale
        clipped_grads.append(tf.IndexedSlices(tmp, p.indices))
      elif not (p == None):
        clipped_grads.append(p * grad_scale)
      else:
        clipped_grads.append(p)
    grads = clipped_grads
    self.global_step = global_step
    params_list = self.params.values()
    params_list.append(self.global_step)
    adam = tf.train.AdamOptimizer(
        learning_rate,
        epsilon=tf.cast(self.utility.FLAGS.eps, self.data_type),
        use_locking=True)
    self.step = adam.apply_gradients(zip(grads, optimize_params), 
					global_step=self.global_step)
    self.init_op = tf.initialize_all_variables()

