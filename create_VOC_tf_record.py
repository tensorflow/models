# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
        --train_proportion 0.7
"""

import hashlib
import io
import logging
import os, sys
import random
import re
import errno
from lxml import etree
import PIL.Image
import tensorflow as tf
import numpy as np
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import uuid
flags = tf.app.flags
flags.DEFINE_string('source', '', 'Folder containing the VOC dataset.')
flags.DEFINE_string('destination', '', 'Folder to save the tf_records')
flags.DEFINE_string(
    'label_map', '', 'Path to label map pbtxt file')
flags.DEFINE_float('train_proportion', 0.8, 'Proportion of training data')
FLAGS = flags.FLAGS

logging.basicConfig(level=logging.INFO)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
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
    extentions = ["", ".jpg",".JPG",".JPEG"]
    for ext in extentions:
      img_path = os.path.join(image_subdirectory, data['filename'] + ext)
      if os.path.exists(img_path):
          break
    assert os.path.exists(img_path), 'Path does not exist: {}'.format(img_path)

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = data['size']['width']
    heigh = data['size']['height']
    width = int(float(data['size']['width']))
    height = int(float(data['size']['height']))

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    newFileName = ''
    if "object" in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            if obj['name'] == 'ignore':
                mask_xmin = int(max(0.0,float(obj['bndbox']['xmin'])))
                mask_ymin = int(max(0.0,float(obj['bndbox']['ymin'])))
                mask_xmax = int(min(float(width),float(obj['bndbox']['xmax'])))
                mask_ymax = int(min(float(height),float(obj['bndbox']['ymax'])))

                mask = np.random.randint(255, size=(mask_ymax-mask_ymin, mask_xmax-mask_xmin, 3))

                image.paste(PIL.Image.fromarray(np.uint8(mask)), box=(mask_xmin, mask_ymin))
                newFileName = str(uuid.uuid4()) + '.jpg'
                continue
                
            
                
            difficult_obj.append(int(difficult))

            xmin.append(max(0.0,float(obj['bndbox']['xmin'])) / width)
            ymin.append(max(0.0,float(obj['bndbox']['ymin'])) / height)
            xmax.append(min(float(width),float(obj['bndbox']['xmax'])) / width)
            ymax.append(min(float(height),float(obj['bndbox']['ymax'])) / height)
            class_name = obj['name']
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))
    if not newFileName == '':
        
        image.save(newFileName)
        with tf.gfile.GFile(newFileName, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            os.path.basename(img_path).encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            os.path.basename(img_path).encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    
    if not newFileName == '':
        os.remove(newFileName)

    return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
    """Creates a TFRecord file from examples.
label_map_dict
    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary.
      annotations_dir: Directory where annotation files are stored.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        path = os.path.join(annotations_dir, example + '.xml')

        if not os.path.exists(path):
            logging.warning('Could not find %s, ignoring example.', path)
            continue
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()


def get_dirs(data_dir):
    images_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')
    return images_dir, annotations_dir


def get_examples(images_dir, annotations_dir):
    examples_list = []

    def get_filenames(dir):
        """Returns list of files contained in directory without extension"""
        files = os.listdir(dir)
        return [os.path.splitext(file)[0] for file in files]

    image_files = get_filenames(images_dir)
    annotations_files = get_filenames(annotations_dir)

    for img in image_files:
        if img in annotations_files:
            examples_list.append(img)
        else:
            print("Image {img} has no annotation file; image ignored".format(img=img))

    return examples_list


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


# TODO: Add test for pet/PASCAL main files.

def main(_):
    data_dir = FLAGS.source
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map)

    logging.info('Reading from dataset.')

    # TODO: Add case where the data_dir is as following 'train', 'val', 'JPEGImages' 'Annotations' ('ImageSets')
    # TODO: where 'train' and 'val' folders contain pertinent images
    image_sets_path = os.path.join(data_dir, 'ImageSets')
    traintxt_path = os.path.join(image_sets_path, 'train.txt')
    valtxt_path = os.path.join(image_sets_path, 'val.txt')
    
    if os.path.exists(traintxt_path) and os.path.exists(valtxt_path):
        create_from_txt = query_yes_no("Do you want to create records based on existing train.txt and val.txt ?")    
    else:
        create_from_txt = False
    
    if create_from_txt:
        logging.info("Creating records from train.txt and val.txt. Please ensure those are valid and that "
                     "annotations and files exist")
        train_examples = dataset_util.read_examples_list(traintxt_path)
        val_examples = dataset_util.read_examples_list(valtxt_path)
        train_image_dir, train_annotations_dir = get_dirs(data_dir)
        val_image_dir, val_annotations_dir = get_dirs(data_dir)

    else:
        train_image_dir, train_annotations_dir = get_dirs(data_dir)
        val_image_dir, val_annotations_dir = get_dirs(data_dir)
        examples_list = get_examples(train_image_dir, train_annotations_dir)

        # Test images are not included in the downloaded data set, so we shall perform
        # our own split.
        random.seed(42)
        random.shuffle(examples_list)
        num_examples = len(examples_list)
        if not 0 < FLAGS.train_proportion <= 1:
            print("Incorrect train proportion value")
            exit()
        num_train = int(FLAGS.train_proportion * num_examples)
        train_examples = examples_list[:num_train]
        val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))
    make_sure_path_exists(FLAGS.destination)
    train_output_path = os.path.join(FLAGS.destination, 'train.record')
    val_output_path = os.path.join(FLAGS.destination, 'val.record')
    create_tf_record(train_output_path, label_map_dict,
                     train_annotations_dir, train_image_dir, train_examples)
    create_tf_record(val_output_path, label_map_dict,
                     val_annotations_dir, val_image_dir, val_examples)

if __name__ == '__main__':
    tf.app.run()
    # this is a test
