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

r"""Collect images from multiple simultaneous webcams.

Usage:

1. Define some environment variables that describe what you're collecting.
dataset=your_dataset_name
mode=train
num_views=2
viddir=/tmp/tcn/videos
tmp_imagedir=/tmp/tcn/tmp_images
debug_vids=1

2. Run the script.
export DISPLAY=:0.0 && \
root=learning/brain/research/tcn && \
bazel build -c opt --copt=-mavx tcn/webcam && \
bazel-bin/tcn/webcam \
--dataset $dataset \
--mode $mode \
--num_views $num_views \
--tmp_imagedir $tmp_imagedir \
--viddir $viddir \
--debug_vids 1 \
--logtostderr

3. Hit Ctrl-C when done collecting, upon which the script will compile videos
for each view and optionally a debug video concatenating multiple
simultaneous views.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
from multiprocessing import Process
import os
import subprocess
import sys
import time
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import animation  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


tf.flags.DEFINE_string('dataset', '', 'Name of the dataset we`re collecting.')
tf.flags.DEFINE_string('mode', '',
                       'What type of data we`re collecting. E.g.:'
                       '`train`,`valid`,`test`, or `demo`')
tf.flags.DEFINE_string('seqname', '',
                       'Name of this sequence. If empty, the script will use'
                       'the name seq_N+1 where seq_N is the latest'
                       'integer-named sequence in the videos directory.')
tf.flags.DEFINE_integer('num_views', 2,
                        'Number of webcams.')
tf.flags.DEFINE_string('tmp_imagedir', '/tmp/tcn/data',
                       'Temporary outdir to write images.')
tf.flags.DEFINE_string('viddir', '/tmp/tcn/videos',
                       'Base directory to write debug videos.')
tf.flags.DEFINE_boolean('debug_vids', True,
                        'Whether to generate debug vids with multiple'
                        'concatenated views.')
tf.flags.DEFINE_string('debug_lhs_view', '0',
                       'Which viewpoint to use for the lhs video.')
tf.flags.DEFINE_string('debug_rhs_view', '1',
                       'Which viewpoint to use for the rhs video.')
tf.flags.DEFINE_integer('height', 1080, 'Raw input height.')
tf.flags.DEFINE_integer('width', 1920, 'Raw input width.')
tf.flags.DEFINE_string('webcam_ports', None,
                       'Comma-separated list of each webcam usb port.')
FLAGS = tf.app.flags.FLAGS


class ImageQueue(object):
  """An image queue holding each stream's most recent image.

  Basically implements a process-safe collections.deque(maxlen=1).
  """

  def __init__(self):
    self.lock = multiprocessing.Lock()
    self._queue = multiprocessing.Queue(maxsize=1)

  def append(self, data):
    with self.lock:
      if self._queue.full():
        # Pop the first element.
        _ = self._queue.get()
      self._queue.put(data)

  def get(self):
    with self.lock:
      return self._queue.get()

  def empty(self):
    return self._queue.empty()

  def close(self):
    return self._queue.close()


class WebcamViewer(object):
  """A class which displays a live stream from the webcams."""

  def __init__(self, display_queues):
    """Create a WebcamViewer instance."""
    self.height = FLAGS.height
    self.width = FLAGS.width
    self.queues = display_queues

  def _get_next_images(self):
    """Gets the next image to display."""
    # Wait for one image per view.
    not_found = True
    while not_found:
      if True in [q.empty() for q in self.queues]:
        # At least one image queue is empty; wait.
        continue
      else:
        # Retrieve the images.
        latest = [q.get() for q in self.queues]
        combined = np.concatenate(latest, axis=1)
      not_found = False
    return combined

  def run(self):
    """Displays the Kcam live stream in a window.

    This function blocks until the window is closed.
    """
    fig, rgb_axis = plt.subplots()

    image_rows = self.height
    image_cols = self.width * FLAGS.num_views
    initial_image = np.zeros((image_rows, image_cols, 3))
    rgb_image = rgb_axis.imshow(initial_image, interpolation='nearest')

    def update_figure(frame_index):
      """Animation function for matplotlib FuncAnimation. Updates the image.

      Args:
        frame_index: The frame number.
      Returns:
        An iterable of matplotlib drawables to clear.
      """
      _ = frame_index
      images = self._get_next_images()
      images = images[..., [2, 1, 0]]
      rgb_image.set_array(images)
      return rgb_image,

    # We must keep a reference to this animation in order for it to work.
    unused_animation = animation.FuncAnimation(
        fig, update_figure, interval=50, blit=True)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def reconcile(queues, write_queue):
  """Gets a list of concurrent images from each view queue.

  This waits for latest images to be available in all view queues,
  then continuously:
  - Creates a list of current images for each view.
  - Writes the list to a queue of image lists to write to disk.
  Args:
    queues: A list of `ImageQueues`, holding the latest image from each webcam.
    write_queue: A multiprocessing.Queue holding lists of concurrent images.
  """
  # Loop forever.
  while True:
    # Wait till all queues have an image.
    if True in [q.empty() for q in queues]:
      continue
    else:
      # Retrieve all views' images.
      latest = [q.get() for q in queues]
      # Copy the list of all concurrent images to the write queue.
      write_queue.put(latest)


def persist(write_queue, view_dirs):
  """Pulls lists of concurrent images off a write queue, writes them to disk.

  Args:
    write_queue: A multiprocessing.Queue holding lists of concurrent images;
      one image per view.
    view_dirs: A list of strings, holding the output image directories for each
      view.
  """
  timestep = 0
  while True:
    # Wait till there is work in the queue.
    if write_queue.empty():
      continue
    # Get a list of concurrent images to write to disk.
    view_ims = write_queue.get()
    for view_idx, image in enumerate(view_ims):
      view_base = view_dirs[view_idx]
      # Assign all concurrent view images the same sequence timestep.
      fname = os.path.join(view_base, '%s.png' % str(timestep).zfill(10))
      cv2.imwrite(fname, image)
    # Move to the next timestep.
    timestep += 1


def get_image(camera):
  """Captures a single image from the camera and returns it in PIL format."""
  data = camera.read()
  _, im = data
  return im


def capture_webcam(camera, display_queue, reconcile_queue):
  """Captures images from simultaneous webcams, writes them to queues.

  Args:
    camera: A cv2.VideoCapture object representing an open webcam stream.
    display_queue: An ImageQueue.
    reconcile_queue: An ImageQueue.
  """
  # Take some ramp images to allow cams to adjust for brightness etc.
  for i in range(60):
    tf.logging.info('Taking ramp image %d.' % i)
    get_image(camera)

  cnt = 0
  start = time.time()
  while True:
    # Get images for all cameras.
    im = get_image(camera)
    # Replace the current image in the display and reconcile queues.
    display_queue.append(im)
    reconcile_queue.append(im)
    cnt += 1
    current = time.time()
    if cnt % 100 == 0:
      tf.logging.info('Collected %s of video, %d frames at ~%.2f fps.' % (
          timer(start, current), cnt, cnt/(current-start)))


def timer(start, end):
  """Returns a formatted time elapsed."""
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)


def display_webcams(display_queues):
  """Builds an WebcamViewer to animate incoming images, runs it."""
  viewer = WebcamViewer(display_queues)
  viewer.run()


def create_vids(view_dirs, seqname):
  """Creates one video per view per sequence."""
  vidbase = os.path.join(FLAGS.viddir, FLAGS.dataset, FLAGS.mode)
  if not os.path.exists(vidbase):
    os.makedirs(vidbase)
  vidpaths = []
  for idx, view_dir in enumerate(view_dirs):
    vidname = os.path.join(vidbase, '%s_view%d.mp4' % (seqname, idx))
    encode_vid_cmd = r'mencoder mf://%s/*.png \
    -mf fps=29:type=png \
    -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell \
    -oac copy -o %s' % (view_dir, vidname)
    os.system(encode_vid_cmd)
    vidpaths.append(vidname)

  debugpath = None
  if FLAGS.debug_vids:
    lhs = vidpaths[FLAGS.debug_lhs_view]
    rhs = vidpaths[FLAGS.debug_rhs_view]
    debug_base = os.path.join('%s_debug' % FLAGS.viddir, FLAGS.dataset,
                              FLAGS.mode)
    if not os.path.exists(debug_base):
      os.makedirs(debug_base)
    debugpath = '%s/%s.mp4' % (debug_base, seqname)
    os.system(r"avconv \
      -i %s \
      -i %s \
      -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
      -map [vid] \
      -c:v libx264 \
      -crf 23 \
      -preset veryfast \
      %s" % (lhs, rhs, debugpath))

  return vidpaths, debugpath


def setup_paths():
  """Sets up the necessary paths to collect videos."""
  assert FLAGS.dataset
  assert FLAGS.mode
  assert FLAGS.num_views

  # Setup directory for final images used to create videos for this sequence.
  tmp_imagedir = os.path.join(FLAGS.tmp_imagedir, FLAGS.dataset, FLAGS.mode)
  if not os.path.exists(tmp_imagedir):
    os.makedirs(tmp_imagedir)

  # Create a base directory to hold all sequence videos if it doesn't exist.
  vidbase = os.path.join(FLAGS.viddir, FLAGS.dataset, FLAGS.mode)
  if not os.path.exists(vidbase):
    os.makedirs(vidbase)

  # Get one directory per concurrent view and a sequence name.
  view_dirs, seqname = get_view_dirs(vidbase, tmp_imagedir)

  # Get an output path to each view's video.
  vid_paths = []
  for idx, _ in enumerate(view_dirs):
    vid_path = os.path.join(vidbase, '%s_view%d.mp4' % (seqname, idx))
    vid_paths.append(vid_path)

  # Optionally build paths to debug_videos.
  debug_path = None
  if FLAGS.debug_vids:
    debug_base = os.path.join('%s_debug' % FLAGS.viddir, FLAGS.dataset,
                              FLAGS.mode)
    if not os.path.exists(debug_base):
      os.makedirs(debug_base)
    debug_path = '%s/%s.mp4' % (debug_base, seqname)

  return view_dirs, vid_paths, debug_path


def get_view_dirs(vidbase, tmp_imagedir):
  """Creates and returns one view directory per webcam."""
  # Create and append a sequence name.
  if FLAGS.seqname:
    seqname = FLAGS.seqname
  else:
    # If there's no video directory, this is the first sequence.
    if not os.listdir(vidbase):
      seqname = '0'
    else:
      # Otherwise, get the latest sequence name and increment it.
      seq_names = [i.split('_')[0] for i in os.listdir(vidbase)]
      latest_seq = sorted(map(int, seq_names), reverse=True)[0]
      seqname = str(latest_seq+1)
    tf.logging.info('No seqname specified, using: %s' % seqname)
  view_dirs = [os.path.join(
      tmp_imagedir, '%s_view%d' % (seqname, v)) for v in range(FLAGS.num_views)]
  for d in view_dirs:
    if not os.path.exists(d):
      os.makedirs(d)
  return view_dirs, seqname


def get_cameras():
  """Opens cameras using cv2, ensures they can take images."""
  # Try to get free webcam ports.
  if FLAGS.webcam_ports:
    ports = map(int, FLAGS.webcam_ports.split(','))
  else:
    ports = range(FLAGS.num_views)
  cameras = [cv2.VideoCapture(i) for i in ports]

  if not all([i.isOpened() for i in cameras]):
    try:
      # Try to find and kill hanging cv2 process_ids.
      output = subprocess.check_output(['lsof -t /dev/video*'], shell=True)
      tf.logging.info('Found hanging cv2 process_ids: \n')
      tf.logging.info(output)
      tf.logging.info('Killing hanging processes...')
      for process_id in output.split('\n')[:-1]:
        subprocess.call(['kill %s' % process_id], shell=True)
      time.sleep(3)
      # Recapture webcams.
      cameras = [cv2.VideoCapture(i) for i in ports]
    except subprocess.CalledProcessError:
      raise ValueError(
          'Cannot connect to cameras. Try running: \n'
          'ls -ltrh /dev/video* \n '
          'to see which ports your webcams are connected to. Then hand those '
          'ports as a comma-separated list to --webcam_ports, e.g. '
          '--webcam_ports 0,1')

  # Verify each camera is able to capture images.
  ims = map(get_image, cameras)
  assert False not in [i is not None for i in ims]
  return cameras


def launch_images_to_videos(view_dirs, vid_paths, debug_path):
  """Launch job in separate process to convert images to videos."""

  f = 'learning/brain/research/tcn/dataset/images_to_videos.py'
  cmd = ['python %s ' % f]
  cmd += ['--view_dirs %s ' % ','.join(i for i in view_dirs)]
  cmd += ['--vid_paths %s ' % ','.join(i for i in vid_paths)]
  cmd += ['--debug_path %s ' % debug_path]
  cmd += ['--debug_lhs_view %s ' % FLAGS.debug_lhs_view]
  cmd += ['--debug_rhs_view %s ' % FLAGS.debug_rhs_view]
  cmd += [' & ']
  cmd = ''.join(i for i in cmd)

  # Call images_to_videos asynchronously.
  fnull = open(os.devnull, 'w')
  subprocess.Popen([cmd], stdout=fnull, stderr=subprocess.STDOUT, shell=True)

  for p in vid_paths:
    tf.logging.info('Writing final video to: %s' % p)
  if debug_path:
    tf.logging.info('Writing debug video to: %s' % debug_path)


def main(_):
  # Initialize the camera capture objects.
  cameras = get_cameras()
  # Get one output directory per view.
  view_dirs, vid_paths, debug_path = setup_paths()
  try:
    # Wait for user input.
    try:
      tf.logging.info('About to write to:')
      for v in view_dirs:
        tf.logging.info(v)
      raw_input('Press Enter to continue...')
    except SyntaxError:
      pass

    # Create a queue per view for displaying and saving images.
    display_queues = [ImageQueue() for _ in range(FLAGS.num_views)]
    reconcile_queues = [ImageQueue() for _ in range(FLAGS.num_views)]

    # Create a queue for collecting all tuples of multi-view images to write to
    # disk.
    write_queue = multiprocessing.Queue()

    processes = []
    # Create a process to display collected images in real time.
    processes.append(Process(target=display_webcams, args=(display_queues,)))
    # Create a process to collect the latest simultaneous images from each view.
    processes.append(Process(
        target=reconcile, args=(reconcile_queues, write_queue,)))
    # Create a process to collect the latest simultaneous images from each view.
    processes.append(Process(
        target=persist, args=(write_queue, view_dirs,)))

    for (cam, dq, rq) in zip(cameras, display_queues, reconcile_queues):
      processes.append(Process(
          target=capture_webcam, args=(cam, dq, rq,)))

    for p in processes:
      p.start()
    for p in processes:
      p.join()

  except KeyboardInterrupt:
    # Close the queues.
    for q in display_queues + reconcile_queues:
      q.close()
    # Release the cameras.
    for cam in cameras:
      cam.release()

    # Launch images_to_videos script asynchronously.
    launch_images_to_videos(view_dirs, vid_paths, debug_path)

    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)  # pylint: disable=protected-access


if __name__ == '__main__':
  tf.app.run()
