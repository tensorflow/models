# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

r"""Converts videos to training, validation, test, and debug tfrecords on cns.

Example usage:

# From phone videos.
x=learning/brain/research/tcn/videos_to_tfrecords && \
blaze build -c opt $x && \
set=tmp && videos=~/data/tcn/datasets/$set/ && \
blaze-bin/$x --logtostderr --output_dir /cns/oi-d/home/$USER/tcn_data/$set \
--input_dir $videos/train
--debug $dataset/debug --rotate 90 --max_per_shard 400

# From webcam videos.
mode=train
x=learning/brain/research/tcn/videos_to_tfrecords && \
blaze build -c opt $x && \
set=tmp && videos=/tmp/tcn/videos/$set/ && \
blaze-bin/$x --logtostderr \
--output_dir /cns/oi-d/home/$USER/tcn_data/$set/$mode \
--input_dir $videos/$mode --max_per_shard 400

"""
import glob
import math
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
from random import shuffle
import re
from StringIO import StringIO
import cv2
from PIL import Image
from PIL import ImageFile
from preprocessing import cv2resizeminedge
from preprocessing import cv2rotateimage
from preprocessing import shapestring
from utils.progress import Progress
import tensorflow.google as tf
tf.logging.set_verbosity(tf.logging.INFO)


tf.app.flags.DEFINE_string('view_pattern', '_view[_]*[0]+[.].*',
                           'view regexp pattern for first view')
tf.app.flags.DEFINE_string('input_dir', '', '''input data path''')
tf.app.flags.DEFINE_integer('resize_min_edge', 0,
                            '''resize the smallest edge to this size.''')
tf.app.flags.DEFINE_integer('rotate', 0, '''rotate the image in degrees.''')
tf.app.flags.DEFINE_string('rotate_if_matching', None,
                           'rotate only if video path matches regexp.')
tf.app.flags.DEFINE_string('output_dir', '', 'output directory for the dataset')
tf.app.flags.DEFINE_integer(
    'max_per_shard', -1, 'max # of frames per data chunk')
tf.app.flags.DEFINE_integer('expected_views', 2, 'expected number of views')
tf.app.flags.DEFINE_integer('log_frequency', 50, 'frequency of logging')
tf.app.flags.DEFINE_integer(
    'max_views_discrepancy', 100,
    'Maximum length difference (in frames) allowed between views')
tf.app.flags.DEFINE_boolean('overwrite', False, 'overwrite output files')
FLAGS = tf.app.flags.FLAGS

feature = tf.train.Feature
bytes_feature = lambda v: feature(bytes_list=tf.train.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
float_feature = lambda v: feature(float_list=tf.train.FloatList(value=v))


def FindPatternFiles(path, view_pattern, errors):
  """Recursively find all files matching a certain pattern."""
  if not path:
    return None
  tf.logging.info(
      'Recursively searching for files matching pattern \'%s\' in %s' %
      (view_pattern, path))
  view_patt = re.compile('.*' + view_pattern)
  sequences = []
  for root, _, filenames in os.walk(path, followlinks=True):
    path_root = root[:len(path)]
    assert path_root == path

    for filename in filenames:
      if view_patt.match(filename):
        fullpath = os.path.join(root, re.sub(view_pattern, '', filename))
        shortpath = re.sub(path, '', fullpath).lstrip('/')

        # Determine if this sequence should be sharded or not.
        shard = False
        if FLAGS.max_per_shard > 0:
          shard = True

        # Retrieve number of frames for this sequence.
        num_views, length, view_paths, num_frames = GetViewInfo(
            fullpath + view_pattern[0] + '*')
        if num_views != FLAGS.expected_views:
          tf.logging.info('Expected %d views but found: %s' %
                          (FLAGS.expected_views, str(view_paths)))
        assert num_views == FLAGS.expected_views
        assert length > 0
        # Drop sequences if view lengths differ too much.
        if max(num_frames) - min(num_frames) > FLAGS.max_views_discrepancy:
          error_msg = (
              'Error: ignoring sequence with views with length difference > %d:'
              '%s in %s') % (FLAGS.max_views_discrepancy, str(num_frames),
                             fullpath)
          errors.append(error_msg)
          tf.logging.error(error_msg)
        else:
          # Append sequence info.
          sequences.append({'full': fullpath, 'name': shortpath, 'len': length,
                            'start': 0, 'end': length, 'num_views': num_views,
                            'shard': shard})
  return sorted(sequences, key=lambda k: k['name'])


def ShardSequences(sequences, max_per_shard):
  """Find all sequences, shard and randomize them."""
  total_shards_len = 0
  total_shards = 0
  assert max_per_shard > 0
  for sequence in sequences:
    if sequence['shard']:
      sequence['shard'] = False  # Reset shard flag.
      length = sequence['len']
      start = sequence['start']
      end = sequence['end']
      name = sequence['name']
      assert end - start == length
      if length > max_per_shard:
        # Dividing sequence into smaller shards.
        num_shards = int(math.floor(length / max_per_shard)) + 1
        size = int(math.ceil(length / num_shards))
        tf.logging.info(
            'splitting sequence of length %d into %d shards of size %d' %
            (length, num_shards, size))
        last_end = 0
        for i in range(num_shards):
          shard_start = last_end
          shard_end = min(length, shard_start + size)
          if i == num_shards - 1:
            shard_end = length
          shard_len = shard_end - shard_start
          total_shards_len += shard_len
          shard_name = name + '_shard%02d' % i
          last_end = shard_end

          # Enqueuing shard.
          if i == 0:  # Replace current sequence.
            sequence['len'] = shard_len
            sequence['start'] = shard_start
            sequence['end'] = shard_end
            sequence['name'] = shard_name
          else:  # Enqueue new sequence.
            sequences.append(
                {'full': sequence['full'], 'name': shard_name,
                 'len': shard_len, 'start': shard_start, 'end': shard_end,
                 'num_views': sequence['num_views'], 'shard': False})

        total_shards += num_shards
        assert last_end == length

  # Print resulting sharding.
  if total_shards > 0:
    tf.logging.info('%d shards of average length %d' %
                    (total_shards, total_shards_len / total_shards))
  return sorted(sequences, key=lambda k: k['name'])


def RandomizeSets(sets):
  """Randomize each set."""
  for _, sequences in sorted(sets.iteritems()):
    if sequences:
      # Randomize order.
      shuffle(sequences)


def GetSpecificFrame(vid_path, frame_index):
  """Gets a frame at a specified index in a video."""
  cap = cv2.VideoCapture(vid_path)
  cap.set(1, frame_index)
  _, bgr = cap.read()
  cap.release()
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  return rgb


def JpegString(image, jpeg_quality=90):
  """Returns given PIL.Image instance as jpeg string.

  Args:
    image: A PIL image.
    jpeg_quality: The image quality, on a scale from 1 (worst) to 95 (best).

  Returns:
    a jpeg_string.
  """
  # This fix to PIL makes sure that we don't get an error when saving large
  # jpeg files. This is a workaround for a bug in PIL. The value should be
  # substantially larger than the size of the image being saved.
  ImageFile.MAXBLOCK = 640 * 512 * 64

  output_jpeg = StringIO()
  image.save(output_jpeg, 'jpeg', quality=jpeg_quality, optimize=True)
  return output_jpeg.getvalue()


def ParallelPreprocessing(args):
  """Parallel preprocessing: rotation, resize and jpeg encoding to string."""
  (vid_path, timestep, num_timesteps, view) = args
  try:
    image = GetSpecificFrame(vid_path, timestep)

    # Resizing.
    resize_str = ''
    if FLAGS.resize_min_edge > 0:
      resize_str += ', resize ' + shapestring(image)
      image = cv2resizeminedge(image, FLAGS.resize_min_edge)
      resize_str += ' => ' + shapestring(image)

    # Rotating.
    rotate = None
    if FLAGS.rotate:
      rotate = FLAGS.rotate
      if FLAGS.rotate_if_matching is not None:
        rotate = None
        patt = re.compile(FLAGS.rotate_if_matching)
        if patt.match(vid_path) is not None:
          rotate = FLAGS.rotate
      if rotate is not None:
        image = cv2rotateimage(image, FLAGS.rotate)

    # Jpeg encoding.
    image = Image.fromarray(image)
    im_string = bytes_feature([JpegString(image)])

    if timestep % FLAGS.log_frequency == 0:
      tf.logging.info('Loaded frame %d / %d for %s (rotation %s%s) from %s' %
                      (timestep, num_timesteps, view, str(rotate), resize_str,
                       vid_path))
    return im_string
  except cv2.error as e:
    tf.logging.error('Error while loading frame %d of %s: %s' %
                     (timestep, vid_path, str(e)))
    return None


def GetNumFrames(vid_path):
  """Gets the number of frames in a video."""
  cap = cv2.VideoCapture(vid_path)
  total_frames = cap.get(7)
  cap.release()
  return int(total_frames)


def GetViewInfo(views_fullname):
  """Return information about a group of views."""
  view_paths = sorted(glob.glob(views_fullname))
  num_frames = [GetNumFrames(i) for i in view_paths]
  min_num_frames = min(num_frames)
  num_views = len(view_paths)
  return num_views, min_num_frames, view_paths, num_frames


def AddSequence(sequence, writer, progress, errors):
  """Converts a sequence to a SequenceExample.

  Sequences have multiple viewpoint videos. Extract all frames from all
  viewpoint videos in parallel, build a single SequenceExample containing
  all viewpoint images for every timestep.

  Args:
    sequence: a dict with information on a sequence.
    writer: A TFRecordWriter.
    progress: A Progress object to report processing progress.
    errors: a list of string to append to in case of errors.
  """
  fullname = sequence['full']
  shortname = sequence['name']
  start = sequence['start']
  end = sequence['end']
  num_timesteps = sequence['len']

  # Build a list of all view paths for this fullname.
  path = fullname + FLAGS.view_pattern[0] + '*'
  tf.logging.info('Loading sequence from ' + path)
  view_paths = sorted(glob.glob(path))
  # Extract all images for all views
  num_frames = [GetNumFrames(i) for i in view_paths]
  tf.logging.info('Loading %s with [%d, %d[ (%d frames) from: %s %s' %
                  (shortname, start, end, num_timesteps,
                   str(num_frames), str(view_paths)))
  num_views = len(view_paths)
  total_timesteps = int(min(num_frames))
  assert num_views == FLAGS.expected_views
  assert num_views == sequence['num_views']

  # Create a worker pool to parallelize loading/rotating
  worker_pool = ThreadPool(multiprocessing.cpu_count())

  # Collect all images for each view.
  view_to_feature_list = {}
  view_images = []
  for view_idx, view in enumerate(
      ['view'+str(i) for i in range(num_views)]):
    # Flatten list to process in parallel
    work = []
    for i in range(start, end):
      work.append((view_paths[view_idx], i, total_timesteps, view))
    # Load and rotate images in parallel
    view_images.append(worker_pool.map(ParallelPreprocessing, work))
    # Report progress.
    progress.Add(len(view_images[view_idx]))
    tf.logging.info('%s' % str(progress))

  # Remove error frames from all views
  i = start
  num_errors = 0
  while i < len(view_images[0]):
    remove_frame = False
    # Check if one or more views have an error for this frame.
    for view_idx in range(num_views):
      if view_images[view_idx][i] is None:
        remove_frame = True
        error_msg = 'Removing frame %d for all views for %s ' % (i, fullname)
        errors.append(error_msg)
        tf.logging.error(error_msg)
    # Remove faulty frames.
    if remove_frame:
      num_errors += 1
      for view_idx in range(num_views):
        del view_images[view_idx][i]
    else:
      i += 1

  # Ignore sequences that have errors.
  if num_errors > 0:
    error_msg = 'Dropping sequence because of frame errors for %s' % fullname
    errors.append(error_msg)
    tf.logging.error(error_msg)
  else:
    # Build FeatureList objects for each view.
    for view_idx, view in enumerate(
        ['view'+str(i) for i in range(num_views)]):
      # Construct FeatureList from repeated feature.
      view_to_feature_list[view] = tf.train.FeatureList(
          feature=view_images[view_idx])

    context_features = tf.train.Features(feature={
        'task': bytes_feature([shortname]),
        'len': int64_feature([num_timesteps])
    })
    feature_lists = tf.train.FeatureLists(feature_list=view_to_feature_list)
    ex = tf.train.SequenceExample(
        context=context_features, feature_lists=feature_lists)
    writer.write(ex.SerializeToString())
    tf.logging.info('Done adding %s with %d timesteps'
                    % (fullname, num_timesteps))


def PrintSequencesInfo(sequences, prefix):
  """Print information about sequences and return the total number of frames."""
  tf.logging.info('')
  tf.logging.info(prefix)
  num_frames = 0
  for sequence in sequences:
    shard_str = ''
    if sequence['shard']:
      shard_str = ' (sharding)'
    tf.logging.info('frames [%d, %d[\t(%d frames * %d views)%s\t%s' % (
        sequence['start'], sequence['end'], sequence['len'],
        sequence['num_views'], shard_str, sequence['name']))
    num_frames += sequence['len'] * sequence['num_views']
  tf.logging.info(('%d frames (all views), %d sequences, average sequence'
                   ' length (all views): %d') %
                  (num_frames, len(sequences), num_frames / len(sequences)))
  tf.logging.info('')
  return num_frames


def CheckRecord(filename, sequence):
  """Check that an existing tfrecord corresponds to the expected sequence."""
  num_sequences = 0
  total_frames = 0
  for serialized_example in tf.python_io.tf_record_iterator(filename):
    num_sequences += 1
    example = tf.train.SequenceExample()
    example.ParseFromString(serialized_example)
    length = example.context.feature['len'].int64_list.value[0]
    name = example.context.feature['task'].bytes_list.value[0]
    total_frames += len(example.feature_lists.feature_list) * length
    if sequence['name'] != name or sequence['len'] != length:
      return False, total_frames
  if num_sequences == 0:
    return False, total_frames
  return True, total_frames


def AddSequences():
  """Creates one training, validation."""
  errors = []

  # Generate datasets file lists.
  sequences = FindPatternFiles(FLAGS.input_dir, FLAGS.view_pattern, errors)
  num_frames = PrintSequencesInfo(sequences,
                                  'Found the following datasets and files:')

  # Sharding and randomizing sets.
  if FLAGS.max_per_shard > 0:
    sequences = ShardSequences(sequences, FLAGS.max_per_shard)
    num_frames = PrintSequencesInfo(sequences, 'After sharding:')
    tf.logging.info('')

  # Process sets.
  progress = Progress(num_frames)
  output_list = []
  for sequence in sequences:
    record_name = os.path.join(
        FLAGS.output_dir, '%s.tfrecord' % sequence['name'])
    if tf.gfile.Exists(record_name) and not FLAGS.overwrite:
      ok, num_frames = CheckRecord(record_name, sequence)
      if ok:
        progress.Add(num_frames)
        tf.logging.info('Skipping existing output file: %s' % record_name)
        continue
      else:
        tf.logging.info('File does not match sequence, reprocessing...')
    output_dir = os.path.dirname(record_name)
    if not tf.gfile.Exists(output_dir):
      tf.logging.info('Creating output directory: %s' % output_dir)
      tf.gfile.MakeDirs(output_dir)
    output_list.append(record_name)
    tf.logging.info('Writing to ' + record_name)
    writer = tf.python_io.TFRecordWriter(record_name)
    AddSequence(sequence, writer, progress, errors)
    writer.close()
  tf.logging.info('Wrote dataset files: ' + str(output_list))
  tf.logging.info('All errors (%d): %s' % (len(errors), str(errors)))


def main(_):
  AddSequences()


if __name__ == '__main__':
  tf.app.run()
