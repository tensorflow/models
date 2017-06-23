from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np
import glob
import os
import logging
from im2txt import configuration
from im2txt import show_and_tell_model
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from icrawler.builtin import GoogleImageCrawler


input_file_pattern='/mnt/raid/data/ni/dnn/zlian/Google_image/train-00007-of-00008'
image_feature = "image/data"
caption_feature = "image/caption_ids"

# Reader for the input data.
reader = tf.TFRecordReader()
def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.
  Args:
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue
# Prefetch serialized SequenceExample protos.
input_queue = prefetch_input_data(
    reader,
    input_file_pattern,
    is_training=True,
    batch_size=32,
    values_per_shard=2300,
    input_queue_capacity_factor=16,
    num_reader_threads=2)




def parse_sequence_example(serialized, image_feature, caption_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_feature: tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  encoded_image = context[image_feature]
  caption = sequence[caption_feature]
  return encoded_image, caption


# serialized_sequence_example = input_queue.dequeue()
# encoded_image, caption = parse_sequence_example(
#     serialized_sequence_example,
#     image_feature=image_feature_name,
#     caption_feature=caption_feature_name)
#
# # Create the graph, etc.
# init_op = tf.global_variables_initializer()
# # Create a session for running operations in the Graph.
# sess = tf.Session()
# # Initialize the variables (like the epoch counter).
# sess.run(init_op)
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# try:
#     while not coord.should_stop():
#         print ("What's going on :")
#         # Run training steps or whatever
#         sess.run(caption)
# except tf.errors.OutOfRangeError:
#     print('Done training -- epoch limit reached')
# finally:
#     # When done, ask the threads to stop.
#     coord.request_stop()
# # Wait for threads to finish.
# coord.join(threads)
# sess.close()

# filenames = ['/mnt/raid/data/ni/dnn/zlian/mscoco/train-00000-of-00001']
# filenames = ['/mnt/raid/data/ni/dnn/zlian/mscoco/test-00000-of-00008']
filenames = ['/mnt/raid/data/ni/dnn/zlian/Google_image/train-00000-of-00008']
print ('Open TF records in file ---')
print (filenames)

filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
key, value = reader.read(filename_queue)
context, sequence = tf.parse_single_sequence_example(
    value,
    context_features={
        image_feature: tf.FixedLenFeature([], dtype=tf.string)
    },
    sequence_features={
        caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
    })

encoded_image = context[image_feature]
caption = sequence[caption_feature]
# example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(20):
        print caption.eval()
    coord.request_stop()
    coord.join(threads)


# print ("See some predictions")
# images_rand = [train_filenames[i] for i in np.random.randint(len(train_filenames), size=50)]
# predict_seqs = predict_images(filenames=images_rand, vocab = vocab, n_sentence=n_sentences)



