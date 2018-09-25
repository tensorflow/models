"""A script to export attention ocr checkpoint to SavedModel format

Usage:
python export_saved_model.py \
  --checkpoint /tmp/attention_ocr/train/model.ckpt-123456 \
  --saved_dir ./saved/1
"""

import tensorflow as tf
from tensorflow.python.platform import flags

import common_flags
import data_provider
import datasets

FLAGS = flags.FLAGS
common_flags.define()


flags.DEFINE_string('saved_dir', './saved/1',
                    'Directory with version contains saved model')


def get_dataset_image_shape(dataset_name):
    ds_module = getattr(datasets, dataset_name)
    height, width, channel = ds_module.DEFAULT_CONFIG['image_shape']
  return height, width, channel


def create_model(dataset_name):
    height, width, channel = get_dataset_image_shape(dataset_name)
    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)

    model = common_flags.create_model(
        num_char_classes=dataset.num_char_classes,
        seq_length=dataset.max_sequence_length,
        num_views=dataset.num_of_views,
        null_code=dataset.null_code,
        charset=dataset.charset)

    raw_images = tf.placeholder(tf.uint8, shape=[1, height, width, channel])
    images = tf.map_fn(data_provider.preprocess_image, raw_images, dtype=tf.float32)
    endpoints = model.create_base(images, labels_one_hot=None)
    return raw_images, endpoints


def main():
    images_placeholder, endpoints = create_model(FLAGS.dataset_name)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, FLAGS.checkpoint)
        inputs = {'input': tf.saved_model.utils.build_tensor_info(images_placeholder)}
        out_classes = sess.graph.get_tensor_by_name('AttentionOcr_v1/ReduceJoin:0')
        outputs = {'output':  tf.saved_model.utils.build_tensor_info(out_classes)}

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        # Save out the SavedModel.
        builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.saved_dir)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
            },
            legacy_init_op=legacy_init_op)
        builder.save()

    print('Model is saved to', FLAGS.saved_dir)


if __name__ == '__main__':
    main()
