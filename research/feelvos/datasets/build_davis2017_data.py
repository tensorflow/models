# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Converts DAVIS 2017 data to TFRecord file format with SequenceExample protos.
"""

import io
import math
import os
from StringIO import StringIO
import numpy as np
import PIL
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_folder', 'DAVIS2017/',
                           'Folder containing the DAVIS 2017 data')

tf.app.flags.DEFINE_string('imageset', 'val',
                           'Which subset to use, either train or val')

tf.app.flags.DEFINE_string(
    'output_dir', './tfrecord',
    'Path to save converted TFRecords of TensorFlow examples.')

_NUM_SHARDS_TRAIN = 10
_NUM_SHARDS_VAL = 1


def read_image(path):
  with open(path) as fid:
    image_str = fid.read()
    image = PIL.Image.open(io.BytesIO(image_str))
    w, h = image.size
  return image_str, (h, w)


def read_annotation(path):
  """Reads a single image annotation from a png image.

  Args:
    path: Path to the png image.

  Returns:
    png_string: The png encoded as string.
    size: Tuple of (height, width).
  """
  with open(path) as fid:
    x = np.array(PIL.Image.open(fid))
    h, w = x.shape
    im = PIL.Image.fromarray(x)

  output = StringIO()
  im.save(output, format='png')
  png_string = output.getvalue()
  output.close()

  return png_string, (h, w)


def process_video(key, input_dir, anno_dir):
  """Creates a SequenceExample for the video.

  Args:
    key: Name of the video.
    input_dir: Directory which contains the image files.
    anno_dir: Directory which contains the annotation files.

  Returns:
    The created SequenceExample.
  """
  frame_names = sorted(tf.gfile.ListDirectory(input_dir))
  anno_files = sorted(tf.gfile.ListDirectory(anno_dir))
  assert len(frame_names) == len(anno_files)

  sequence = tf.train.SequenceExample()
  context = sequence.context.feature
  features = sequence.feature_lists.feature_list

  for i, name in enumerate(frame_names):
    image_str, image_shape = read_image(
        os.path.join(input_dir, name))
    anno_str, anno_shape = read_annotation(
        os.path.join(anno_dir, name[:-4] + '.png'))
    image_encoded = features['image/encoded'].feature.add()
    image_encoded.bytes_list.value.append(image_str)
    segmentation_encoded = features['segmentation/object/encoded'].feature.add()
    segmentation_encoded.bytes_list.value.append(anno_str)

    np.testing.assert_array_equal(np.array(image_shape), np.array(anno_shape))

    if i == 0:
      first_shape = np.array(image_shape)
    else:
      np.testing.assert_array_equal(np.array(image_shape), first_shape)

  context['video_id'].bytes_list.value.append(key.encode('ascii'))
  context['clip/frames'].int64_list.value.append(len(frame_names))
  context['image/format'].bytes_list.value.append('JPEG')
  context['image/channels'].int64_list.value.append(3)
  context['image/height'].int64_list.value.append(first_shape[0])
  context['image/width'].int64_list.value.append(first_shape[1])
  context['segmentation/object/format'].bytes_list.value.append('PNG')
  context['segmentation/object/height'].int64_list.value.append(first_shape[0])
  context['segmentation/object/width'].int64_list.value.append(first_shape[1])

  return sequence


def convert(data_folder, imageset, output_dir, num_shards):
  """Converts the specified subset of DAVIS 2017 to TFRecord format.

  Args:
    data_folder: The path to the DAVIS 2017 data.
    imageset: The subset to use, either train or val.
    output_dir: Where to store the TFRecords.
    num_shards: The number of shards used for storing the data.
  """
  sets_file = os.path.join(data_folder, 'ImageSets', '2017', imageset + '.txt')
  vids = [x.strip() for x in open(sets_file).readlines()]
  num_vids = len(vids)
  num_vids_per_shard = int(math.ceil(num_vids) / float(num_shards))
  for shard_id in range(num_shards):
    output_filename = os.path.join(
        output_dir,
        '%s-%05d-of-%05d.tfrecord' % (imageset, shard_id, num_shards))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_vids_per_shard
      end_idx = min((shard_id + 1) * num_vids_per_shard, num_vids)
      for i in range(start_idx, end_idx):
        print('Converting video %d/%d shard %d video %s' % (
            i + 1, num_vids, shard_id, vids[i]))
        img_dir = os.path.join(data_folder, 'JPEGImages', '480p', vids[i])
        anno_dir = os.path.join(data_folder, 'Annotations', '480p', vids[i])
        example = process_video(vids[i], img_dir, anno_dir)
        tfrecord_writer.write(example.SerializeToString())


def main(unused_argv):
  imageset = FLAGS.imageset
  assert imageset in ('train', 'val')
  if imageset == 'train':
    num_shards = _NUM_SHARDS_TRAIN
  else:
    num_shards = _NUM_SHARDS_VAL
  convert(FLAGS.data_folder, FLAGS.imageset, FLAGS.output_dir, num_shards)


if __name__ == '__main__':
  tf.app.run()
