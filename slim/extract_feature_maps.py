import numpy as np
import tensorflow as tf
from scipy.misc import imread

from nets import nets_factory

slim = tf.contrib.slim

reconstructed_images = []
tfrecords_filename = '/home/sina/datasets/lip_read_features/lipread_train_00000-of-00002.tfrecord'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

###########################


mouth_placeholder = tf.placeholder(tf.float32, (None, 47, 73, 9))
speech_placeholder = tf.placeholder(tf.float32, (None, 13, 15, 1))

MODEL_NAME = 'lipread_mouth'
network_fn = nets_factory.get_network_fn(
    'lipread_mouth',
    num_classes=2,
    is_training=False)
logits_mouth, endpoints_mouth = network_fn(mouth_placeholder)

network_fn = nets_factory.get_network_fn(
    'lipread_speech',
    num_classes=2,
    is_training=False)
logits_speech, endpoints_speech = network_fn(speech_placeholder)
distance_vector = tf.subtract(logits_speech, logits_mouth, name=None)
distance_weighted = slim.fully_connected(distance_vector, 1, scope='fc_weighted')

variables_to_restore = slim.get_variables_to_restore()

saver = tf.train.Saver(slim.get_variables_to_restore())

sess = tf.Session()
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

count = 0
for string_record in record_iterator:

    count += 1
    # speech = tf.image.decode_jpeg(string_record, channels=3)


    example = tf.train.Example()
    example.ParseFromString(string_record)

    mouth_string = (example.features.feature['pair/mouth']
                    .bytes_list
                    .value[0])
    speech_string = (example.features.feature['pair/speech']
                     .bytes_list
                     .value[0])

    speech_1d = np.fromstring(speech_string, dtype=np.float32)
    speech = speech_1d.reshape((1, 13, 15, -1))

    mouth_1d = np.fromstring(mouth_string, dtype=np.float32)
    mouth = mouth_1d.reshape((1, 47, 73, -1))

    distance = sess.run(distance_vector,
                        feed_dict={speech_placeholder: speech, mouth_placeholder: mouth})

    weighted = sess.run(distance_weighted,
                        feed_dict={speech_placeholder: speech, mouth_placeholder: mouth})

    print(distance)

    # print(reconstructed_img.shape)
    if count % 1000 == 0:
        print count



