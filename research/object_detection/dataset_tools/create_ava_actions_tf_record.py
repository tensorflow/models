# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

r"""Code to download and parse the AVA Actions dataset for TensorFlow models.

The [AVA Actions data set](
https://research.google.com/ava/index.html)
is a dataset for human action recognition.

This script downloads the annotations and prepares data from similar annotations
if local video files are available. The video files can be downloaded
from the following website:
https://github.com/cvdfoundation/ava-dataset

Prior to running this script, please run download_and_preprocess_ava.sh to
download input videos.

Running this code as a module generates the data set on disk. First, the
required files are downloaded (_download_data) which enables constructing the
label map. Then (in generate_examples), for each split in the data set, the
metadata and image frames are generated from the annotations for each sequence
example (_generate_examples). The data set is written to disk as a set of
numbered TFRecord files.

Generating the data on disk can take considerable time and disk space.
(Image compression quality is the primary determiner of disk usage.

If using the Tensorflow Object Detection API, set the input_type field
in the input_reader to TF_SEQUENCE_EXAMPLE. If using this script to generate
data for Context R-CNN scripts, the --examples_for_context flag should be
set to true, so that properly-formatted tf.example objects are written to disk.

This data is structured for per-clip action classification where images is
the sequence of images and labels are a one-hot encoded value. See
as_dataset() for more details.

Note that the number of videos changes in the data set over time, so it will
likely be necessary to change the expected number of examples.

The argument video_path_format_string expects a value as such:
  '/path/to/videos/{0}'

"""
import collections
import contextlib
import csv
import glob
import hashlib
import os
import random
import sys
import zipfile

from absl import app
from absl import flags
from absl import logging
import cv2
from six.moves import range
from six.moves import urllib
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import seq_example_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


POSSIBLE_TIMESTAMPS = range(902, 1798)
ANNOTATION_URL = 'https://research.google.com/ava/download/ava_v2.2.zip'
SECONDS_TO_MILLI = 1000
FILEPATTERN = 'ava_actions_%s_1fps_rgb'
SPLITS = {
    'train': {
        'shards': 1000,
        'examples': 862663,
        'csv': '',
        'excluded-csv': ''
    },
    'val': {
        'shards': 100,
        'examples': 243029,
        'csv': '',
        'excluded-csv': ''
    },
    # Test doesn't have ground truth, so TF Records can't be created
    'test': {
        'shards': 100,
        'examples': 0,
        'csv': '',
        'excluded-csv': ''
    }
}

NUM_CLASSES = 80


def feature_list_feature(value):
  return tf.train.FeatureList(feature=value)


class Ava(object):
  """Generates and loads the AVA Actions 2.2 data set."""

  def __init__(self, path_to_output_dir, path_to_data_download):
    if not path_to_output_dir:
      raise ValueError('You must supply the path to the data directory.')
    self.path_to_data_download = path_to_data_download
    self.path_to_output_dir = path_to_output_dir

  def generate_and_write_records(self,
                                 splits_to_process='train,val,test',
                                 video_path_format_string=None,
                                 seconds_per_sequence=10,
                                 hop_between_sequences=10,
                                 examples_for_context=False):
    """Downloads data and generates sharded TFRecords.

    Downloads the data files, generates metadata, and processes the metadata
    with MediaPipe to produce tf.SequenceExamples for training. The resulting
    files can be read with as_dataset(). After running this function the
    original data files can be deleted.

    Args:
      splits_to_process: csv string of which splits to process. Allows
        providing a custom CSV with the CSV flag. The original data is still
        downloaded to generate the label_map.
      video_path_format_string: The format string for the path to local files.
      seconds_per_sequence: The length of each sequence, in seconds.
      hop_between_sequences: The gap between the centers of
        successive sequences.
      examples_for_context: Whether to generate sequence examples with context
        for context R-CNN.
    """
    example_function = self._generate_sequence_examples
    if examples_for_context:
      example_function = self._generate_examples

    logging.info('Downloading data.')
    download_output = self._download_data()
    for key in splits_to_process.split(','):
      logging.info('Generating examples for split: %s', key)
      all_metadata = list(example_function(
          download_output[0][key][0], download_output[0][key][1],
          download_output[1], seconds_per_sequence, hop_between_sequences,
          video_path_format_string))
      logging.info('An example of the metadata: ')
      logging.info(all_metadata[0])
      random.seed(47)
      random.shuffle(all_metadata)
      shards = SPLITS[key]['shards']
      shard_names = [os.path.join(
          self.path_to_output_dir, FILEPATTERN % key + '-%05d-of-%05d' % (
              i, shards)) for i in range(shards)]
      writers = [tf.io.TFRecordWriter(shard) for shard in shard_names]
      with _close_on_exit(writers) as writers:
        for i, seq_ex in enumerate(all_metadata):
          writers[i % len(writers)].write(seq_ex.SerializeToString())
    logging.info('Data extraction complete.')

  def _generate_sequence_examples(self, annotation_file, excluded_file,
                                  label_map, seconds_per_sequence,
                                  hop_between_sequences,
                                  video_path_format_string):
    """For each row in the annotation CSV, generates corresponding examples.

    When iterating through frames for a single sequence example, skips over
    excluded frames. When moving to the next sequence example, also skips over
    excluded frames as if they don't exist. Generates equal-length sequence
    examples, each with length seconds_per_sequence (1 fps) and gaps of
    hop_between_sequences frames (and seconds) between them, possible greater
    due to excluded frames.

    Args:
      annotation_file: path to the file of AVA CSV annotations.
      excluded_file: path to a CSV file of excluded timestamps for each video.
      label_map: an {int: string} label map.
      seconds_per_sequence: The number of seconds per example in each example.
      hop_between_sequences: The hop between sequences. If less than
          seconds_per_sequence, will overlap.
      video_path_format_string: File path format to glob video files.

    Yields:
      Each prepared tf.SequenceExample of metadata also containing video frames
    """
    fieldnames = ['id', 'timestamp_seconds', 'xmin', 'ymin', 'xmax', 'ymax',
                  'action_label']
    frame_excluded = {}
    # create a sparse, nested map of videos and frame indices.
    with open(excluded_file, 'r') as excluded:
      reader = csv.reader(excluded)
      for row in reader:
        frame_excluded[(row[0], int(float(row[1])))] = True
    with open(annotation_file, 'r') as annotations:
      reader = csv.DictReader(annotations, fieldnames)
      frame_annotations = collections.defaultdict(list)
      ids = set()
      # aggreggate by video and timestamp:
      for row in reader:
        ids.add(row['id'])
        key = (row['id'], int(float(row['timestamp_seconds'])))
        frame_annotations[key].append(row)
      # for each video, find aggregates near each sampled frame.:
      logging.info('Generating metadata...')
      media_num = 1
      for media_id in ids:
        logging.info('%d/%d, ignore warnings.\n', media_num, len(ids))
        media_num += 1

        filepath = glob.glob(
            video_path_format_string.format(media_id) + '*')[0]
        cur_vid = cv2.VideoCapture(filepath)
        width = cur_vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cur_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        middle_frame_time = POSSIBLE_TIMESTAMPS[0]
        while middle_frame_time < POSSIBLE_TIMESTAMPS[-1]:
          start_time = middle_frame_time - seconds_per_sequence // 2 - (
              0 if seconds_per_sequence % 2 == 0 else 1)
          end_time = middle_frame_time + (seconds_per_sequence // 2)

          total_boxes = []
          total_labels = []
          total_label_strings = []
          total_images = []
          total_source_ids = []
          total_confidences = []
          total_is_annotated = []
          windowed_timestamp = start_time

          while windowed_timestamp < end_time:
            if (media_id, windowed_timestamp) in frame_excluded:
              end_time += 1
              windowed_timestamp += 1
              logging.info('Ignoring and skipping excluded frame.')
              continue

            cur_vid.set(cv2.CAP_PROP_POS_MSEC,
                        (windowed_timestamp) * SECONDS_TO_MILLI)
            _, image = cur_vid.read()
            _, buffer = cv2.imencode('.jpg', image)

            bufstring = buffer.tostring()
            total_images.append(bufstring)
            source_id = str(windowed_timestamp) + '_' + media_id
            total_source_ids.append(source_id)
            total_is_annotated.append(1)

            boxes = []
            labels = []
            label_strings = []
            confidences = []
            for row in frame_annotations[(media_id, windowed_timestamp)]:
              if len(row) > 2 and int(row['action_label']) in label_map:
                boxes.append([float(row['ymin']), float(row['xmin']),
                              float(row['ymax']), float(row['xmax'])])
                labels.append(int(row['action_label']))
                label_strings.append(label_map[int(row['action_label'])])
                confidences.append(1)
              else:
                logging.warning('Unknown label: %s', row['action_label'])

            total_boxes.append(boxes)
            total_labels.append(labels)
            total_label_strings.append(label_strings)
            total_confidences.append(confidences)
            windowed_timestamp += 1

          if total_boxes:
            yield seq_example_util.make_sequence_example(
                'AVA', media_id, total_images, int(height), int(width), 'jpeg',
                total_source_ids, None, total_is_annotated, total_boxes,
                total_label_strings, use_strs_for_source_id=True)

          # Move middle_time_frame, skipping excluded frames
          frames_mv = 0
          frames_excluded_count = 0
          while (frames_mv < hop_between_sequences + frames_excluded_count
                 and middle_frame_time + frames_mv < POSSIBLE_TIMESTAMPS[-1]):
            frames_mv += 1
            if (media_id, windowed_timestamp + frames_mv) in frame_excluded:
              frames_excluded_count += 1
          middle_frame_time += frames_mv

        cur_vid.release()

  def _generate_examples(self, annotation_file, excluded_file, label_map,
                         seconds_per_sequence, hop_between_sequences,
                         video_path_format_string):
    """For each row in the annotation CSV, generates examples.

    When iterating through frames for a single example, skips
    over excluded frames. Generates equal-length sequence examples, each with
    length seconds_per_sequence (1 fps) and gaps of hop_between_sequences
    frames (and seconds) between them, possible greater due to excluded frames.

    Args:
      annotation_file: path to the file of AVA CSV annotations.
      excluded_file: path to a CSV file of excluded timestamps for each video.
      label_map: an {int: string} label map.
      seconds_per_sequence: The number of seconds per example in each example.
      hop_between_sequences: The hop between sequences. If less than
          seconds_per_sequence, will overlap.
      video_path_format_string: File path format to glob video files.

    Yields:
      Each prepared tf.Example of metadata also containing video frames
    """
    del seconds_per_sequence
    del hop_between_sequences
    fieldnames = ['id', 'timestamp_seconds', 'xmin', 'ymin', 'xmax', 'ymax',
                  'action_label']
    frame_excluded = {}
    # create a sparse, nested map of videos and frame indices.
    with open(excluded_file, 'r') as excluded:
      reader = csv.reader(excluded)
      for row in reader:
        frame_excluded[(row[0], int(float(row[1])))] = True
    with open(annotation_file, 'r') as annotations:
      reader = csv.DictReader(annotations, fieldnames)
      frame_annotations = collections.defaultdict(list)
      ids = set()
      # aggreggate by video and timestamp:
      for row in reader:
        ids.add(row['id'])
        key = (row['id'], int(float(row['timestamp_seconds'])))
        frame_annotations[key].append(row)
      # for each video, find aggreggates near each sampled frame.:
      logging.info('Generating metadata...')
      media_num = 1
      for media_id in ids:
        logging.info('%d/%d, ignore warnings.\n', media_num, len(ids))
        media_num += 1

        filepath = glob.glob(
            video_path_format_string.format(media_id) + '*')[0]
        cur_vid = cv2.VideoCapture(filepath)
        width = cur_vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cur_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        middle_frame_time = POSSIBLE_TIMESTAMPS[0]
        total_non_excluded = 0
        while middle_frame_time < POSSIBLE_TIMESTAMPS[-1]:
          if (media_id, middle_frame_time) not in frame_excluded:
            total_non_excluded += 1
          middle_frame_time += 1

        middle_frame_time = POSSIBLE_TIMESTAMPS[0]
        cur_frame_num = 0
        while middle_frame_time < POSSIBLE_TIMESTAMPS[-1]:
          cur_vid.set(cv2.CAP_PROP_POS_MSEC,
                      middle_frame_time * SECONDS_TO_MILLI)
          _, image = cur_vid.read()
          _, buffer = cv2.imencode('.jpg', image)

          bufstring = buffer.tostring()

          if (media_id, middle_frame_time) in frame_excluded:
            middle_frame_time += 1
            logging.info('Ignoring and skipping excluded frame.')
            continue

          cur_frame_num += 1
          source_id = str(middle_frame_time) + '_' + media_id

          xmins = []
          xmaxs = []
          ymins = []
          ymaxs = []
          areas = []
          labels = []
          label_strings = []
          confidences = []
          for row in frame_annotations[(media_id, middle_frame_time)]:
            if len(row) > 2 and int(row['action_label']) in label_map:
              xmins.append(float(row['xmin']))
              xmaxs.append(float(row['xmax']))
              ymins.append(float(row['ymin']))
              ymaxs.append(float(row['ymax']))
              areas.append(float((xmaxs[-1] - xmins[-1]) *
                                 (ymaxs[-1] - ymins[-1])) / 2)
              labels.append(int(row['action_label']))
              label_strings.append(label_map[int(row['action_label'])])
              confidences.append(1)
            else:
              logging.warning('Unknown label: %s', row['action_label'])

          middle_frame_time += 1/3
          if abs(middle_frame_time - round(middle_frame_time) < 0.0001):
            middle_frame_time = round(middle_frame_time)

          key = hashlib.sha256(bufstring).hexdigest()
          date_captured_feature = (
              '2020-06-17 00:%02d:%02d' % ((middle_frame_time - 900)*3 // 60,
                                           (middle_frame_time - 900)*3 % 60))
          context_feature_dict = {
              'image/height':
                  dataset_util.int64_feature(int(height)),
              'image/width':
                  dataset_util.int64_feature(int(width)),
              'image/format':
                  dataset_util.bytes_feature('jpeg'.encode('utf8')),
              'image/source_id':
                  dataset_util.bytes_feature(source_id.encode('utf8')),
              'image/filename':
                  dataset_util.bytes_feature(source_id.encode('utf8')),
              'image/encoded':
                  dataset_util.bytes_feature(bufstring),
              'image/key/sha256':
                  dataset_util.bytes_feature(key.encode('utf8')),
              'image/object/bbox/xmin':
                  dataset_util.float_list_feature(xmins),
              'image/object/bbox/xmax':
                  dataset_util.float_list_feature(xmaxs),
              'image/object/bbox/ymin':
                  dataset_util.float_list_feature(ymins),
              'image/object/bbox/ymax':
                  dataset_util.float_list_feature(ymaxs),
              'image/object/area':
                  dataset_util.float_list_feature(areas),
              'image/object/class/label':
                  dataset_util.int64_list_feature(labels),
              'image/object/class/text':
                  dataset_util.bytes_list_feature(label_strings),
              'image/location':
                  dataset_util.bytes_feature(media_id.encode('utf8')),
              'image/date_captured':
                  dataset_util.bytes_feature(
                      date_captured_feature.encode('utf8')),
              'image/seq_num_frames':
                  dataset_util.int64_feature(total_non_excluded),
              'image/seq_frame_num':
                  dataset_util.int64_feature(cur_frame_num),
              'image/seq_id':
                  dataset_util.bytes_feature(media_id.encode('utf8')),
          }

          yield tf.train.Example(
              features=tf.train.Features(feature=context_feature_dict))

        cur_vid.release()

  def _download_data(self):
    """Downloads and extracts data if not already available."""
    if sys.version_info >= (3, 0):
      urlretrieve = urllib.request.urlretrieve
    else:
      urlretrieve = urllib.request.urlretrieve
    logging.info('Creating data directory.')
    tf.io.gfile.makedirs(self.path_to_data_download)
    logging.info('Downloading annotations.')
    paths = {}

    zip_path = os.path.join(self.path_to_data_download,
                            ANNOTATION_URL.split('/')[-1])
    urlretrieve(ANNOTATION_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(self.path_to_data_download)
    for split in ['train', 'test', 'val']:
      csv_path = os.path.join(self.path_to_data_download,
                              'ava_%s_v2.2.csv' % split)
      excl_name = 'ava_%s_excluded_timestamps_v2.2.csv' % split
      excluded_csv_path = os.path.join(self.path_to_data_download, excl_name)
      SPLITS[split]['csv'] = csv_path
      SPLITS[split]['excluded-csv'] = excluded_csv_path
      paths[split] = (csv_path, excluded_csv_path)

    label_map = self.get_label_map(os.path.join(
        self.path_to_data_download,
        'ava_action_list_v2.2_for_activitynet_2019.pbtxt'))
    return paths, label_map

  def get_label_map(self, path):
    """Parses a label map into {integer:string} format."""
    label_map_dict = label_map_util.get_label_map_dict(path)
    label_map_dict = {v: bytes(k, 'utf8') for k, v in label_map_dict.items()}
    logging.info(label_map_dict)
    return label_map_dict


@contextlib.contextmanager
def _close_on_exit(writers):
  """Call close on all writers on exit."""
  try:
    yield writers
  finally:
    for writer in writers:
      writer.close()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  Ava(flags.FLAGS.path_to_output_dir,
      flags.FLAGS.path_to_download_data).generate_and_write_records(
          flags.FLAGS.splits_to_process,
          flags.FLAGS.video_path_format_string,
          flags.FLAGS.seconds_per_sequence,
          flags.FLAGS.hop_between_sequences,
          flags.FLAGS.examples_for_context)

if __name__ == '__main__':
  flags.DEFINE_string('path_to_download_data',
                      '',
                      'Path to directory to download data to.')
  flags.DEFINE_string('path_to_output_dir',
                      '',
                      'Path to directory to write data to.')
  flags.DEFINE_string('splits_to_process',
                      'train,val',
                      'Process these splits. Useful for custom data splits.')
  flags.DEFINE_string('video_path_format_string',
                      None,
                      'The format string for the path to local video files. '
                      'Uses the Python string.format() syntax with possible '
                      'arguments of {video}, {start}, {end}, {label_name}, and '
                      '{split}, corresponding to columns of the data csvs.')
  flags.DEFINE_integer('seconds_per_sequence',
                       10,
                       'The number of seconds per example in each example.'
                       'Always 1 when examples_for_context is True.')
  flags.DEFINE_integer('hop_between_sequences',
                       10,
                       'The hop between sequences. If less than '
                       'seconds_per_sequence, will overlap. Always 1 when '
                       'examples_for_context is True.')
  flags.DEFINE_boolean('examples_for_context',
                       False,
                       'Whether to generate examples instead of sequence '
                       'examples. If true, will generate tf.Example objects '
                       'for use in Context R-CNN.')
  app.run(main)
