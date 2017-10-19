import tensorflow as tf

from object_detection.utils import dataset_util
import os
from PIL import Image
from object_detection.utils import label_map_util
import numpy as np

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet \
        --output_path=/home/user/pet/output
"""
flags = tf.app.flags
flags.DEFINE_string('output_path', r'E:\data_mining\temp\train\train.record', 'Path to output TFRecord')
flags.DEFINE_string('input_path', r'E:\data_mining\temp\train', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/logo_label_map.pbtxt', 'Path to label map proto')
FLAGS = flags.FLAGS


def create_tf_example(dir, filename, label_map_dict):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        Pascal dataset directory holding the actual image data.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    class_name = os.path.basename(dir)
    img_path = os.path.join(dir, filename)

    img = Image.open(img_path)
    height = img.height
    width = img.width
    img = img.resize((width, height))
    img = img.tobytes()  # 将图片转化为二进制格式
    # img = np.array(img)
    # img_raw = img.tostring()
    # print("width = ",img.width,",height = ",img.height)


    xmins = [100 / width]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [100 / width]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [100 / height]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [100 / height]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(img),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tfrecord():
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    for class_dir_name in os.listdir(FLAGS.input_path):
        class_dir_path = os.path.join(FLAGS.input_path, class_dir_name)
        if os.path.isdir(class_dir_path):
            print("class_dir_path = " + class_dir_path)
            for img_name in os.listdir(class_dir_path):
                tf_example = create_tf_example(class_dir_path, img_name, label_map_dict)
                writer.write(tf_example.SerializeToString())

    writer.close()


def read_and_decode(filename_queue):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 从文件中读出一个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例
    features = tf.parse_single_example(serialized_example, features={
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
        'image/object/class/text': tf.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.FixedLenFeature([], tf.int64),
    })
    # 将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    width = tf.cast(features['image/width'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    label = tf.cast(features['image/object/class/text'], tf.string)

    image = tf.reshape(image, [height, width, 3])
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # ????

    return image, label


# # 用于获取一个batch_size的图像和label
# def inputs(data_set, batch_size, num_epochs):
#     if not num_epochs:
#         num_epochs = None
#     if data_set == 'train':
#         file = FLAGS.output_path
#     # else:
#     #     file = VALIDATION_FILE
#
#     filename_queue = tf.train.string_input_producer(["E:\\data_mining\\temp\\dog_train.tfrecords"])  # 读入流中
#     image, label = read_and_decode(filename_queue)
#     with tf.Session() as sess:  # 开始一个会话
#         sess.run(tf.initialize_all_variables())
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         for i in range(20):
#             example, l = sess.run([image, label])  # 在会话中取出image和label
#             img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
#             save_path = os.path.join(cwd, str(i) + '_''Label_' + str(l) + '.jpg')
#             img.save(save_path)  # 存下图片
#             print(example, l)
#         coord.request_stop()
#         coord.join(threads)
#
#     return images, labels

def extract():
    file = r"E:\data_mining\temp\train\train.record"
    with tf.name_scope('input') as scope:
        filename_queue = tf.train.string_input_producer([file])
        image, label = read_and_decode(filename_queue)

    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            example, l = sess.run([image, label])  # 在会话中取出image和label
            img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
            save_path = os.path.join(FLAGS.input_path, str(i) + '_Label_' + str(l) + '.jpg')
            img.save(save_path)  # 存下图片
            print(example, l)
        coord.request_stop()
        coord.join(threads)


def main(_):
    # create_tfrecord()
    extract()


if __name__ == '__main__':
    tf.app.run()
