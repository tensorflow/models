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
"""Implementation of the Neural Programmer model described in https://openreview.net/pdf?id=ry2YOrcge

This file calls functions to load & pre-process data, construct the TF graph
and performs training or evaluation as specified by the flag evaluator_job
Author: aneelakantan (Arvind Neelakantan)
"""
import time
from random import Random
import numpy as np
import tensorflow as tf
import model
import wiki_data
import parameters
import data_utils

tf.flags.DEFINE_integer("train_steps", 100001, "Number of steps to train")
tf.flags.DEFINE_integer("eval_cycle", 500,
                        "Evaluate model at every eval_cycle steps")
tf.flags.DEFINE_integer("max_elements", 100,
                        "maximum rows that are  considered for processing")
tf.flags.DEFINE_integer(
    "max_number_cols", 15,
    "maximum number columns that are considered for processing")
tf.flags.DEFINE_integer(
    "max_word_cols", 25,
    "maximum number columns that are considered for processing")
tf.flags.DEFINE_integer("question_length", 62, "maximum question length")
tf.flags.DEFINE_integer("max_entry_length", 1, "")
tf.flags.DEFINE_integer("max_passes", 4, "number of operation passes")
tf.flags.DEFINE_integer("embedding_dims", 256, "")
tf.flags.DEFINE_integer("batch_size", 20, "")
tf.flags.DEFINE_float("clip_gradients", 1.0, "")
tf.flags.DEFINE_float("eps", 1e-6, "")
tf.flags.DEFINE_float("param_init", 0.1, "")
tf.flags.DEFINE_float("learning_rate", 0.001, "")
tf.flags.DEFINE_float("l2_regularizer", 0.0001, "")
tf.flags.DEFINE_float("print_cost", 50.0,
                      "weighting factor in the objective function")
tf.flags.DEFINE_string("job_id", "temp", """job id""")
tf.flags.DEFINE_string("output_dir", "../model/",
                       """output_dir""")
tf.flags.DEFINE_string("data_dir", "../data/",
                       """data_dir""")
tf.flags.DEFINE_integer("write_every", 500, "wrtie every N")
tf.flags.DEFINE_integer("param_seed", 150, "")
tf.flags.DEFINE_integer("python_seed", 200, "")
tf.flags.DEFINE_float("dropout", 0.8, "dropout keep probability")
tf.flags.DEFINE_float("rnn_dropout", 0.9,
                      "dropout keep probability for rnn connections")
tf.flags.DEFINE_float("pad_int", -20000.0,
                      "number columns are padded with pad_int")
tf.flags.DEFINE_string("data_type", "double", "float or double")
tf.flags.DEFINE_float("word_dropout_prob", 0.9, "word dropout keep prob")
tf.flags.DEFINE_integer("word_cutoff", 10, "")
tf.flags.DEFINE_integer("vocab_size", 10800, "")
tf.flags.DEFINE_boolean("evaluator_job", False,
                        "wehther to run as trainer/evaluator")
tf.flags.DEFINE_float(
    "bad_number_pre_process", -200000.0,
    "number that is added to a corrupted table entry in a number column")
tf.flags.DEFINE_float("max_math_error", 3.0,
                      "max square loss error that is considered")
tf.flags.DEFINE_float("soft_min_value", 5.0, "")
FLAGS = tf.flags.FLAGS


class Utility:
  #holds FLAGS and other variables that are used in different files
  def __init__(self):
    global FLAGS
    self.FLAGS = FLAGS
    self.unk_token = "UNK"
    self.entry_match_token = "entry_match"
    self.column_match_token = "column_match"
    self.dummy_token = "dummy_token"
    self.tf_data_type = {}
    self.tf_data_type["double"] = tf.float64
    self.tf_data_type["float"] = tf.float32
    self.np_data_type = {}
    self.np_data_type["double"] = np.float64
    self.np_data_type["float"] = np.float32
    self.operations_set = ["count"] + [
        "prev", "next", "first_rs", "last_rs", "group_by_max", "greater",
        "lesser", "geq", "leq", "max", "min", "word-match"
    ] + ["reset_select"] + ["print"]
    self.word_ids = {}
    self.reverse_word_ids = {}
    self.word_count = {}
    self.random = Random(FLAGS.python_seed)


def evaluate(sess, data, batch_size, graph, i):
  #computes accuracy
  num_examples = 0.0
  gc = 0.0
  for j in range(0, len(data) - batch_size + 1, batch_size):
    [ct] = sess.run([graph.final_correct],
                    feed_dict=data_utils.generate_feed_dict(data, j, batch_size,
                                                            graph))
    gc += ct * batch_size
    num_examples += batch_size
  print "dev set accuracy   after ", i, " : ", gc / num_examples
  print num_examples, len(data)
  print "--------"


def Train(graph, utility, batch_size, train_data, sess, model_dir,
          saver):
  #performs training
  curr = 0
  train_set_loss = 0.0
  utility.random.shuffle(train_data)
  start = time.time()
  for i in range(utility.FLAGS.train_steps):
    curr_step = i
    if (i > 0 and i % FLAGS.write_every == 0):
      model_file = model_dir + "/model_" + str(i)
      saver.save(sess, model_file)
    if curr + batch_size >= len(train_data):
      curr = 0
      utility.random.shuffle(train_data)
    step, cost_value = sess.run(
        [graph.step, graph.total_cost],
        feed_dict=data_utils.generate_feed_dict(
            train_data, curr, batch_size, graph, train=True, utility=utility))
    curr = curr + batch_size
    train_set_loss += cost_value
    if (i > 0 and i % FLAGS.eval_cycle == 0):
      end = time.time()
      time_taken = end - start
      print "step ", i, " ", time_taken, " seconds "
      start = end
      print " printing train set loss: ", train_set_loss / utility.FLAGS.eval_cycle
      train_set_loss = 0.0


def master(train_data, dev_data, utility):
  #creates TF graph and calls trainer or evaluator
  batch_size = utility.FLAGS.batch_size 
  model_dir = utility.FLAGS.output_dir + "/model" + utility.FLAGS.job_id + "/"
  #create all paramters of the model
  param_class = parameters.Parameters(utility)
  params, global_step, init = param_class.parameters(utility)
  key = "test" if (FLAGS.evaluator_job) else "train"
  graph = model.Graph(utility, batch_size, utility.FLAGS.max_passes, mode=key)
  graph.create_graph(params, global_step)
  prev_dev_error = 0.0
  final_loss = 0.0
  final_accuracy = 0.0
  #start session
  with tf.Session() as sess:
    sess.run(init.name)
    sess.run(graph.init_op.name)
    to_save = params.copy()
    saver = tf.train.Saver(to_save, max_to_keep=500)
    if (FLAGS.evaluator_job):
      while True:
        selected_models = {}
        file_list = tf.gfile.ListDirectory(model_dir)
        for model_file in file_list:
          if ("checkpoint" in model_file or "index" in model_file or
              "meta" in model_file):
            continue
          if ("data" in model_file):
            model_file = model_file.split(".")[0]
          model_step = int(
              model_file.split("_")[len(model_file.split("_")) - 1])
          selected_models[model_step] = model_file
        file_list = sorted(selected_models.items(), key=lambda x: x[0])
        if (len(file_list) > 0):
          file_list = file_list[0:len(file_list) - 1]
	print "list of models: ", file_list
        for model_file in file_list:
          model_file = model_file[1]
          print "restoring: ", model_file
          saver.restore(sess, model_dir + "/" + model_file)
          model_step = int(
              model_file.split("_")[len(model_file.split("_")) - 1])
          print "evaluating on dev ", model_file, model_step
          evaluate(sess, dev_data, batch_size, graph, model_step)
    else:
      ckpt = tf.train.get_checkpoint_state(model_dir)
      print "model dir: ", model_dir
      if (not (tf.gfile.IsDirectory(utility.FLAGS.output_dir))):
        print "create dir: ", utility.FLAGS.output_dir
        tf.gfile.MkDir(utility.FLAGS.output_dir)
      if (not (tf.gfile.IsDirectory(model_dir))):
        print "create dir: ", model_dir
        tf.gfile.MkDir(model_dir)
      Train(graph, utility, batch_size, train_data, sess, model_dir,
            saver)

def main(args):
  utility = Utility()
  train_name = "random-split-1-train.examples"
  dev_name = "random-split-1-dev.examples"
  test_name = "pristine-unseen-tables.examples"
  #load data
  dat = wiki_data.WikiQuestionGenerator(train_name, dev_name, test_name, FLAGS.data_dir)
  train_data, dev_data, test_data = dat.load()
  utility.words = []
  utility.word_ids = {}
  utility.reverse_word_ids = {}
  #construct vocabulary
  data_utils.construct_vocab(train_data, utility)
  data_utils.construct_vocab(dev_data, utility, True)
  data_utils.construct_vocab(test_data, utility, True)
  data_utils.add_special_words(utility)
  data_utils.perform_word_cutoff(utility)
  #convert data to int format and pad the inputs
  train_data = data_utils.complete_wiki_processing(train_data, utility, True)
  dev_data = data_utils.complete_wiki_processing(dev_data, utility, False)
  test_data = data_utils.complete_wiki_processing(test_data, utility, False)
  print "# train examples ", len(train_data)
  print "# dev examples ", len(dev_data)
  print "# test examples ", len(test_data)
  print "running open source"
  #construct TF graph and train or evaluate
  master(train_data, dev_data, utility)


if __name__ == "__main__":
  tf.app.run()
