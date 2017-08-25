# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from im2txt import configuration
from im2txt import show_and_tell_model
from im2txt import utils
import time


FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("input_file_pattern", "/mnt/raid/data/ni/dnn/zlian/mscoco/train-?????-of-?????",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_boolean("train_inception", True,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("log_every_n_steps", 100,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("train_dir",
                       # TODO: the behaviour of ckpt files are weird =_=
                       # "/mnt/raid/data/ni/dnn/zlian/ckpt-1-milli",
                       # "/mnt/raid/data/ni/dnn/zlian/checkpoint_1_million/",
                        "/mnt/raid/data/ni/dnn/zlian/ckpt-3milli/",
                       "Directory for saving and loading model checkpoints.")

vocab_file = "/mnt/raid/data/ni/dnn/zlian/ckpt-1-milli/word_counts_copy.txt"
flag_file = "/mnt/raid/data/ni/dnn/zlian/Google_image/flag.txt"

def train(number_of_steps):
  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = FLAGS.input_file_pattern
  training_config = configuration.TrainingConfig()
  # model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model=show_and_tell_model.ShowAndTellModel(
        model_config, mode="train", train_inception=FLAGS.train_inception)
    model.build()

    # Set up the learning rate.
    learning_rate_decay_fn = None
    if FLAGS.train_inception:
      learning_rate = tf.constant(training_config.train_inception_learning_rate)
    # if 0:
    #     print ('This should not be called :DDDDDDDD')
    else:
      learning_rate = tf.constant(training_config.initial_learning_rate)
      if training_config.learning_rate_decay_factor > 0:
        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
          return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=training_config.learning_rate_decay_factor,
              staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    # This was used when trying to calculate other kinds of losses somehow
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.25)
    # saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

  # Run training.
  tf.contrib.slim.learning.train(
      train_op,
      train_dir,
      log_every_n_steps=FLAGS.log_every_n_steps,
      graph=g,
      global_step=model.global_step,
      number_of_steps=number_of_steps,
      init_fn=model.init_fn,
      saver=saver)

def main(unused_argv):
    print ('Why OOM? :D')
    tf.logging.set_verbosity(tf.logging.INFO)
    # steps_per_epoch = 18323
    steps_per_epoch = 1000
    min_step=5042000

    # Train a few steps with only coco data
    # flag=1, crawler waits till training ends.
    # Crawler starts predicting, flag=1. Crawler starts crawling, flag=0.
    utils.writeflag(path=flag_file, flag=1, info='Train for a few steps')
    train(number_of_steps=min_step+100)
    utils.writeflag(path=flag_file, flag=0, info='Stop train for a few steps')
    time.sleep(600)

    min_step +=steps_per_epoch*1
    steps=[min_step]
    n_loops = 100
    i = 0
    while(i<n_loops):
        steps.append(steps[i]+steps_per_epoch)
        i+=1
    print (steps[:10])

    for step in steps:
        while True:
            flag = utils.readflag(path=flag_file)
            if not flag:
                utils.writeflag(path=flag_file, flag=1, info='start training')
                print ("Train until step %d" %step)
                train(number_of_steps=step)
                utils.writeflag(path=flag_file, flag=0, info='finish training, wait for the crawler')
                break
            else: time.sleep(600)
        while True:
            flag = utils.readflag(path=flag_file)
            if flag==2:
                utils.writeflag(path=flag_file, flag=0, info='Crawling is done. Move to the next step :D')
                # Allow some time for prediction :D
                time.sleep(600)
                break
            else:   time.sleep(300)
    print ("Eventually it's down! Yeah!")


if __name__ == "__main__":
  tf.app.run()
