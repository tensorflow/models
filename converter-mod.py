#!/usr/bin/env python
#
# SPDX-License-Identifier: MIT
# coding: utf-8
# -*- coding: utf-8 -*-
"""
Given a CVAT XML and a directory with the image dataset, this script reads the
CVAT XML and writes the annotations in tfrecords into a given
directory.

This implementation supports annotated images only.
"""
from __future__ import unicode_literals
import xml.etree.ElementTree as ET
import tensorflow as tf
from object_detection.utils import dataset_util
from collections import Counter
import codecs
import hashlib
from pathlib import Path
import argparse
import os
import string
import glob
import traceback

# we need it to filter out non-ASCII characters otherwise
# trainning will crash
printable = set(string.printable)

def parse_args():
    """Parse arguments of command line"""
    parser = argparse.ArgumentParser(
        description='Convert CVAT XML annotations to tfrecords format'
    )

    parser.add_argument(
        '--cvat-xml', metavar='FILE', required=False,
        help='input file with CVAT annotation in xml format'
    )

    parser.add_argument(
        '--cvat-dir', metavar='DIRECTORY', required=False,
        help='input directory with CVAT annotation in xml format'
    )

    parser.add_argument(
        '--image-dir', metavar='DIRECTORY', required=True,
        help='directory which contains original images'
    )

    parser.add_argument(
        '--output-dir', metavar='DIRECTORY', required=True,
        help='directory for output annotations in tfrecords format'
    )

    parser.add_argument(
        '--train-percentage', metavar='PERCENTAGE', required=False, default=90, type=int,
        help='the percentage of training data to total data (default: 90)'
    )

    parser.add_argument(
        '--min-train', metavar='NUM', required=False, default=10, type=int,
        help='The minimum number of images above which the label is considered (default: 10)'
    )

    parser.add_argument(
        '--attribute', metavar='NAME', required=False, default="",
        type=str,
        help='The attribute name based on which the object can identified'
    )

    parser.add_argument(
        '--index', metavar='NUM', required=False, default='0', type=str,
        help='The minimum number of images above which the label is considered (default: 10)'
    )


    return parser.parse_args()


def process_cvat_xml(args):
  """Transforms a single XML in CVAT format to tfrecords.
  """

  # print("inside process_cvat_xml "+args.cvat_xml)

  train_percentage = int(args.train_percentage)
  assert (train_percentage<=100 and train_percentage>=0)

  cvat_xml = ET.parse(args.cvat_xml).getroot()

  output_dir = Path(args.output_dir)
  if not output_dir.exists():
    print("Creating the output directory because it doesn't exist")
    output_dir.mkdir()

  cvat_name, output_dir, min_train = \
          args.attribute, output_dir.absolute(), args.min_train



  # extract the object names
  object_names = []
  num_imgs = 0
  for img in cvat_xml.findall('image'):
        num_imgs += 1
        for box in img:
            if cvat_name == "" :
                obj_name = ''.join(filter(lambda x: x in printable,
                    box.attrib['label']))
                object_names.append(obj_name)
            else :
                for attribute in box :
                    if attribute.attrib['name'] == cvat_name :
                        obj_name = ''.join(filter(lambda x: x in printable,
                            attribute.text.lower()))
                        object_names.append(obj_name)

  try:
    labels, values = zip(*Counter(object_names).items())
  except Exception as e:
    traceback.print_exc()
    print("An exception occurred for file",args.cvat_xml)
    print(e)
    return None

  # Open the tfrecord files for writing
  writer_train = tf.io.TFRecordWriter(
      os.path.join(output_dir.absolute(), 'train_' + args.index + '.tfrecord'))
  writer_eval  = tf.io.TFRecordWriter(
      os.path.join(output_dir.absolute(), 'eval_' + args.index + '.tfrecord'))

  # Create the label map file
  saved_dict = dict()
  reverse_dict = dict()
  with codecs.open(os.path.join(output_dir,'label_map.pbtxt'),
            'w', encoding='utf8') as f:
        counter = 1
        for iii, label in enumerate(labels):
            if values[iii] < min_train :
                continue
            saved_dict[label] = counter
            reverse_dict[counter] = label
            f.write(u'item {\n')
            f.write(u'\tid: {}\n'.format(counter))
            f.write(u"\tname: '{}'\n".format(label))
            f.write(u'}\n\n')
            counter+=1

  
  if os.path.exists(os.path.join(output_dir,'label_map.pbtxt')):
    os.remove(os.path.join(output_dir,'label_map.pbtxt'))
  num_iter = num_imgs
  eval_num = num_iter * (100 - train_percentage) // 100
  train_num = num_iter - eval_num


  for counter,example in enumerate(cvat_xml.findall('image')):
    tf_example = create_tf_example(example, args.attribute, saved_dict,  args.image_dir)
    if tf_example is None:
        continue
    if(counter < train_num):
        writer_train.write(tf_example.SerializeToString())
    else :
        writer_eval.write(tf_example.SerializeToString())

  writer_train.close()
  writer_eval.close()


  return saved_dict, num_imgs


# Defining the main conversion function
def create_tf_example(example, cvat_name, saved_dict, img_dir):
  # Process one image data per run
  height = int(example.attrib['height']) # Image height
  width = int(example.attrib['width']) # Image width
  filename = os.path.join(img_dir, example.attrib['name'])
  _, ext = os.path.splitext(example.attrib['name'])
  filename = filename.encode('utf8')
  try:
   with tf.io.gfile.GFile(filename,'rb') as fid:
       encoded_jpg = fid.read()
  except Exception:
      #traceback.print_exc()
      print(filename, 'not found')
      with open('logs.txt',"a") as f:
          f.write(filename)
      return None

  key = hashlib.sha256(encoded_jpg).hexdigest()

  if ext.lower() in ['.jpg','.jpeg'] :
    image_format = 'jpeg'.encode('utf8')
  elif ext.lower() == '.png' :
    image_format = 'png'.encode('utf8')
  else:
    print('File Format not supported, Skipping')
    return None

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  # Loop oer the boxes and fill the above fields
  counter = 0

  for box in example:
    list_of_labels = [
        "poster_posm_1"
        ]
    if box.attrib['label'] in list_of_labels:
        counter += 1

        box_name = ''
        if cvat_name == "" :
            box_name = box.attrib['label']
        else :
            for attr in box:
                if attr.attrib['name'] == cvat_name:
                    box_name = attr.text.lower()

        # filter out non-ASCII characters
        box_name = ''.join(filter(lambda x: x in printable, box_name))


        if 'points' in box.attrib:
            points=box.attrib['points']

            points = [x.split(",")  for x in points.split(";")]

            x_val = [float(r[0]) for r in points]
            y_val = [float(r[1]) for r in points]


            xmins.append(float(min(x_val)) / width)
            xmaxs.append(float(max(x_val)) / width)
            ymins.append(float(min(y_val)) / height)
            ymaxs.append(float(max(y_val)) / height)
            classes_text.append(box_name.encode('utf8'))
            classes.append(saved_dict[box_name])

            # print("Converted a polygon")


        elif box_name in saved_dict.keys():
            xmin = float(box.attrib['xtl'])
            xmax = float(box.attrib['xbr'])
            ymin = float(box.attrib['ytl'])
            ymax = float(box.attrib['ybr'])

            error = False

            if xmin > width:
                error = True
                # print('XMIN > width for file', filename)

            if xmin <= 0:
                error = True
                # print('XMIN < 0 for file', filename)

            if xmax > width:
                error = True
                # print('XMAX > width for file', filename)

            if ymin > height:
                error = True
                # print('YMIN > height for file', filename)

            if ymin <= 0:
                error = True
                # print('YMIN < 0 for file', filename)

            if ymax > height:
                error = True
                # print('YMAX > height for file', filename)

            if xmin >= xmax:
                error = True
                # print('xmin >= xmax for file', filename)

            if ymin >= ymax:
                error = True
                # print('ymin >= ymax for file', filename)

            # if error == True:
                # print('Error for file: %s' % filename)
                # print()
            if  error==False :
                xmins.append(float(box.attrib['xtl']) / width)
                xmaxs.append(float(box.attrib['xbr']) / width)
                ymins.append(float(box.attrib['ytl']) / height)
                ymaxs.append(float(box.attrib['ybr']) / height)
                classes_text.append(box_name.encode('utf8'))
                classes.append(saved_dict[box_name])

                # print("Converted a box")
    else:
        continue

  if counter == 0:
      return None

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def main():
  args = parse_args()
  xml_list = [item for i in [glob.glob(args.cvat_dir+'/*.%s' % ext) for ext in ["xml"]] for item in i]

  index = int(args.index)
  for i in range(index, index + len(xml_list)):
    args.cvat_xml=xml_list[i-index]
    args.index=str(i).zfill(6)
    process_cvat_xml(args)

if __name__== '__main__' :
  main()


