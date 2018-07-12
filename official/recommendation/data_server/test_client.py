
import grpc
import timeit
import numpy as np
import tensorflow as tf

from official.datasets import movielens
from official.recommendation.data_server import pipeline

from official.recommendation.data_server import server_command_pb2
from official.recommendation.data_server import server_command_pb2_grpc


def proto_test():
  users = np.array(list(range(10)), dtype=np.int32)
  items = np.array(list(range(0 + 30, 10 + 30)), dtype=np.uint16)
  np.random.seed(234)
  labels = np.random.randint(0, 2, (10,)).astype("int8")

  print(users)
  print(items)
  print(labels)

  batch = server_command_pb2.Batch(
      users=bytes(memoryview(users)),
      items=bytes(memoryview(items)),
      labels=bytes(memoryview(labels))
  )
  batch_bytes = batch.SerializeToString()
  print(batch)
  print(batch_bytes)
  print(type(batch_bytes))

  _, batch_tensor = tf.contrib.proto.decode_proto(
      batch_bytes,
      message_type="Batch",
      field_names=["users", "items", "labels"],
      output_types=[tf.string, tf.string, tf.string]
  )
  # for i in dir(batch):
  #   try:
  #     print(i.ljust(30), getattr(batch, i))
  #   except AttributeError:
  #     continue

  channel = grpc.insecure_channel("localhost:{}".format(46293))
  x = server_command_pb2.Batch
  for i in dir(x):
    print(i)


  # print(server_command_pb2.Batch.FromString(batch_bytes))
  # assert False
  # return
  # print(batch_tensor)
  # with tf.Session().as_default() as sess:
  #   print(sess.run(batch_tensor))


def main():
  ncf_dataset = pipeline.initialize(data_dir="/tmp/movielens_test", dataset="ml-1m", num_neg=4)

  input_fn = pipeline.get_input_fn(True, ncf_dataset, 16384, 1)
  with tf.Session().as_default() as sess:
    dataset = input_fn()  # type: tf.data.Dataset
    batch = dataset.make_one_shot_iterator().get_next()
    st = timeit.default_timer()
    for i in range(1, 100000000):
      try:
        result = sess.run(batch)
      except tf.errors.OutOfRangeError:
        break

      if i % 25 == 0:
        print(i / (timeit.default_timer() - st))

      # if i % 1000 == 0:
      #   print(result[0][movielens.USER_COLUMN][:, 0])
      #   print(result[0][movielens.ITEM_COLUMN][:, 0])
      #   print(result[1][:, 0])
      #   print()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  # main()
  proto_test()