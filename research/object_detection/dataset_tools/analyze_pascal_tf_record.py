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

"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

import numpy as np
import pandas as pd
from numpy.random import RandomState


import cv2                
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  

                              



flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'object_detection/data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], data['filename']+'.jpg')
  #os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  #print("full_path", full_path)


  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))
      brands[obj['name']]=brands[obj['name']]+1
      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      if (obj['name'] in label_map_dict.keys()):
        classes.append(label_map_dict[obj['name']])
      else:
          print("WARNING",full_path)
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))
  examples_list_number_classes_text[os.path.splitext(full_path)[0]]=len(classes_text)
  examples_list_number_classes[os.path.splitext(full_path)[0]]=len(classes)
  
  logging.info(xmin,ymin,xmax,ymax,classes_text,classes,poses,data['folder'], data['filename'])
  #print(xmin,ymin,xmax,ymax,classes_text,classes,poses,data['folder'], data['filename'])
  #print(xmin,ymin,xmax,ymax)
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['folder']+'/'+data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['folder']+'/'+data['filename'].encode('utf8')),
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
  if (examples_list_number_classes_text[os.path.splitext(full_path)[0]]!=examples_list_number_classes[os.path.splitext(full_path)[0]]):
      print(full_path,example)
  if ((data['folder']+'/'+data['filename']=='nike/img000105') or (len(classes_text) ==0)):
    #logging.info(example)
    #print(full_path,example)
    print(full_path,examples_list_number_classes_text[os.path.splitext(full_path)[0]],examples_list_number_classes[os.path.splitext(full_path)[0]])
    # extract pre-trained face detector
    #face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    # load color (BGR) image
    img = cv2.imread(full_path)
    # convert BGR image to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## find faces in image
    #faces = face_cascade.detectMultiScale(gray)

    # print number of faces detected in the image
    #print('Number of faces detected:', len(faces))

    ## get bounding box for each detected face
    #for (x,y,w,h) in (xmin,ymin,xmax,ymax):
    #    # add bounding box to color image
    #cv2.rectangle(img,(int(xmin[0]),int(ymin[0])),(int(xmax[0]),int(ymax[0])),(255,0,0),2)
    cv2.rectangle(img,(int(xmin[0]*width),int(ymin[0])),(int(xmax[0]),int(ymax[0])),(255,0,0),2)
    
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imshow("Image",cv_rgb)
    # display the image, along with bounding box
    #plt.imshow(cv_rgb)
    #plt.show()
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))

data_dir = FLAGS.data_dir
#years = ['VOC2007', 'VOC2012']
#if FLAGS.year != 'merged':
#  years = [FLAGS.year]

output_train_path = os.path.join(FLAGS.output_path,"pascal3_train.record")
output_val_path = os.path.join(FLAGS.output_path,"pascal3_val.record")
#writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
writer_train = tf.python_io.TFRecordWriter(output_train_path)
writer_val = tf.python_io.TFRecordWriter(output_val_path)

label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
brands= {}.fromkeys(label_map_dict.keys(),0)

print('Reading from PASCAL dataset.')
examples_path = os.path.join(data_dir)#(data_dir, year, 'ImageSets', 'Main','aeroplane_' + FLAGS.set + '.txt')

examples_list=[os.path.splitext(os.path.join(root, name))[0]
           for root, dirs, files in os.walk(examples_path)
           for name in files
           if name.endswith(("jpg"))]
print(len(examples_list),examples_list)

indices = list(range(len(examples_list)))
num_training_instances = int(0.8 * len(examples_list))
random_state=1234567890
rs = np.random.RandomState(random_state)
rs.shuffle(indices)
train_indices = indices[:num_training_instances]
val_indices = indices[num_training_instances:]
#print("indices",indices)

# split the actual data

examples_list_train, examples_list_val = list(pd.DataFrame(examples_list).iloc[train_indices].values.flatten()), list(pd.DataFrame(examples_list).iloc[val_indices].values.flatten())

examples_list_number_classes= {}.fromkeys(examples_list,0)
examples_list_number_classes_text= {}.fromkeys(examples_list,0)

#examples_list_train_number_classes= {}.fromkeys(examples_list_train,0)
#examples_list_train_number_classes_text= {}.fromkeys(examples_list_train,0)

#examples_list_val_number_classes= {}.fromkeys(examples_list_val,0)
#examples_list_val_number_classes_text= {}.fromkeys(examples_list_val,0)

annotations_dir = os.path.join(data_dir)#(data_dir, year, FLAGS.annotations_dir)
#examples_list = dataset_util.read_examples_list(examples_path)
#print("examples_list", examples_list)
print('#traing',len(examples_list_train))
print('#val',len(examples_list_val))


num_shards=10
#  output_filebase=FLAGS.output_path

with contextlib2.ExitStack() as tf_record_close_stack:
  output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_train_path, num_shards)


  for idx, example in enumerate(examples_list_train):
   if idx % 100 == 0:
     print('On image %d of %d', idx, len(examples_list_train))
   path = os.path.join(annotations_dir, example + '.xml')
   with tf.gfile.GFile(path, 'r') as fid:
     xml_str = fid.read()
   xml = etree.fromstring(xml_str)
   data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
   

   #print("FLAGS.data_dir",FLAGS.data_dir)
   tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                   FLAGS.ignore_difficult_instances)
   #print("tf_example",tf_example)
   #print("tf_example.SerializeToString())",tf_example.SerializeToString())
   #writer.write(tf_example.SerializeToString())
   output_shard_index = idx % num_shards
   #output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
  writer_train.close()

  with contextlib2.ExitStack() as tf_record_close_stack:
   output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_val_path, num_shards)



   for idx, example in enumerate(examples_list_val):
     if idx % 100 == 0:
       print('On image %d of %d', idx, len(examples_list_val))
     path = os.path.join(annotations_dir, example + '.xml')
     with tf.gfile.GFile(path, 'r') as fid:
       xml_str = fid.read()
     xml = etree.fromstring(xml_str)
     data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
     #print("example",example)
     #print("data",data)
     #print("FLAGS.data_dir",FLAGS.data_dir)
     tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
     #print("tf_example",tf_example)
     #print("tf_example.SerializeToString())",tf_example.SerializeToString())
     #writer.write(tf_example.SerializeToString())
     output_shard_index = idx % num_shards
     #output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
  writer_val.close()
  
  print("BRANDS",brands)
  print("BRANDS_len",len(brands.keys()))
  key_max = max(brands.keys(), key=(lambda k: brands[k]))
  key_min = min(brands.keys(), key=(lambda k: brands[k]))

  print('Maximum Value: ',brands[key_max],'at key',key_max)
  print('Minimum Value: ',brands[key_min],'at key',key_min)
  #print(examples_list_train_number_classes)

  key_max = max(examples_list_number_classes_text.keys(), key=(lambda k: examples_list_number_classes_text[k]))
  key_min = min(examples_list_number_classes_text.keys(), key=(lambda k: examples_list_number_classes_text[k]))

  print('Maximum Value no classes text: ',examples_list_number_classes_text[key_max],'at key',key_max)
  print('Minimum Value: ',examples_list_number_classes_text[key_min],'at key',key_min)
  key_max = max(examples_list_number_classes.keys(), key=(lambda k: examples_list_number_classes[k]))
  key_min = min(examples_list_number_classes.keys(), key=(lambda k: examples_list_number_classes[k]))

  print('Maximum Value no classes: ',examples_list_number_classes[key_max],'at key',key_max)
  print('Minimum Value: ',examples_list_number_classes[key_min],'at key',key_min)

if __name__ == '__main__':
  tf.app.run()
