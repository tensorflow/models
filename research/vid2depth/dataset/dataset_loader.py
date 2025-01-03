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

"""Classes to load KITTI and Cityscapes data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import re
from absl import logging
import numpy as np
import scipy.misc

CITYSCAPES_CROP_BOTTOM = True  # Crop bottom 25% to remove the car hood.
CITYSCAPES_CROP_PCT = 0.75
CITYSCAPES_SAMPLE_EVERY = 2  # Sample every 2 frames to match KITTI frame rate.
BIKE_SAMPLE_EVERY = 6  # 5fps, since the bike's motion is slower.


class Bike(object):
  """Load bike video frames."""

  def __init__(self,
               dataset_dir,
               img_height=128,
               img_width=416,
               seq_length=3,
               sample_every=BIKE_SAMPLE_EVERY):
    self.dataset_dir = dataset_dir
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.sample_every = sample_every
    self.frames = self.collect_frames()
    self.num_frames = len(self.frames)
    self.num_train = self.num_frames
    logging.info('Total frames collected: %d', self.num_frames)

  def collect_frames(self):
    """Create a list of unique ids for available frames."""
    video_list = os.listdir(self.dataset_dir)
    logging.info('video_list: %s', video_list)
    frames = []
    for video in video_list:
      im_files = glob.glob(os.path.join(self.dataset_dir, video, '*.jpg'))
      im_files = sorted(im_files, key=natural_keys)
      # Adding 3 crops of the video.
      frames.extend(['A' + video + '/' + os.path.basename(f) for f in im_files])
      frames.extend(['B' + video + '/' + os.path.basename(f) for f in im_files])
      frames.extend(['C' + video + '/' + os.path.basename(f) for f in im_files])
    return frames

  def get_example_with_index(self, target_index):
    if not self.is_valid_sample(target_index):
      return False
    example = self.load_example(target_index)
    return example

  def load_intrinsics(self, unused_frame_idx, cy):
    """Load intrinsics."""
    # https://www.wired.com/2013/05/calculating-the-angular-view-of-an-iphone/
    # https://codeyarns.com/2015/09/08/how-to-compute-intrinsic-camera-matrix-for-a-camera/
    # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
    # # iPhone: These numbers are for images with resolution 720 x 1280.
    # Assuming FOV = 50.9 => fx = (1280 // 2) / math.tan(fov / 2) = 1344.8
    intrinsics = np.array([[1344.8, 0, 1280 // 2],
                           [0, 1344.8, cy],
                           [0, 0, 1.0]])
    return intrinsics

  def is_valid_sample(self, target_index):
    """Checks whether we can find a valid sequence around this frame."""
    target_video, _ = self.frames[target_index].split('/')
    start_index, end_index = get_seq_start_end(target_index,
                                               self.seq_length,
                                               self.sample_every)
    if start_index < 0 or end_index >= self.num_frames:
      return False
    start_video, _ = self.frames[start_index].split('/')
    end_video, _ = self.frames[end_index].split('/')
    if target_video == start_video and target_video == end_video:
      return True
    return False

  def load_image_raw(self, frame_id):
    """Reads the image and crops it according to first letter of frame_id."""
    crop_type = frame_id[0]
    img_file = os.path.join(self.dataset_dir, frame_id[1:])
    img = scipy.misc.imread(img_file)
    allowed_height = int(img.shape[1] * self.img_height / self.img_width)
    # Starting height for the middle crop.
    mid_crop_top = int(img.shape[0] / 2 - allowed_height / 2)
    # How much to go up or down to get the other two crops.
    height_var = int(mid_crop_top / 3)
    if crop_type == 'A':
      crop_top = mid_crop_top - height_var
      cy = allowed_height / 2 + height_var
    elif crop_type == 'B':
      crop_top = mid_crop_top
      cy = allowed_height / 2
    elif crop_type == 'C':
      crop_top = mid_crop_top + height_var
      cy = allowed_height / 2 - height_var
    else:
      raise ValueError('Unknown crop_type: %s' % crop_type)
    crop_bottom = crop_top + allowed_height + 1
    return img[crop_top:crop_bottom, :, :], cy

  def load_image_sequence(self, target_index):
    """Returns a list of images around target index."""
    start_index, end_index = get_seq_start_end(target_index,
                                               self.seq_length,
                                               self.sample_every)
    image_seq = []
    for idx in range(start_index, end_index + 1, self.sample_every):
      frame_id = self.frames[idx]
      img, cy = self.load_image_raw(frame_id)
      if idx == target_index:
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
      img = scipy.misc.imresize(img, (self.img_height, self.img_width))
      image_seq.append(img)
    return image_seq, zoom_x, zoom_y, cy

  def load_example(self, target_index):
    """Returns a sequence with requested target frame."""
    image_seq, zoom_x, zoom_y, cy = self.load_image_sequence(target_index)
    target_video, target_filename = self.frames[target_index].split('/')
    # Put A, B, C at the end for better shuffling.
    target_video = target_video[1:] + target_video[0]
    intrinsics = self.load_intrinsics(target_index, cy)
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
    example = {}
    example['intrinsics'] = intrinsics
    example['image_seq'] = image_seq
    example['folder_name'] = target_video
    example['file_name'] = target_filename.split('.')[0]
    return example

  def scale_intrinsics(self, mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


class KittiRaw(object):
  """Reads KITTI raw data files."""

  def __init__(self,
               dataset_dir,
               split,
               load_pose=False,
               img_height=128,
               img_width=416,
               seq_length=3):
    static_frames_file = 'dataset/kitti/static_frames.txt'
    test_scene_file = 'dataset/kitti/test_scenes_' + split + '.txt'
    with open(get_resource_path(test_scene_file), 'r') as f:
      test_scenes = f.readlines()
    self.test_scenes = [t[:-1] for t in test_scenes]
    self.dataset_dir = dataset_dir
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.load_pose = load_pose
    self.cam_ids = ['02', '03']
    self.date_list = [
        '2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03'
    ]
    self.collect_static_frames(static_frames_file)
    self.collect_train_frames()

  def collect_static_frames(self, static_frames_file):
    with open(get_resource_path(static_frames_file), 'r') as f:
      frames = f.readlines()
    self.static_frames = []
    for fr in frames:
      if fr == '\n':
        continue
      unused_date, drive, frame_id = fr.split(' ')
      fid = '%.10d' % (np.int(frame_id[:-1]))
      for cam_id in self.cam_ids:
        self.static_frames.append(drive + ' ' + cam_id + ' ' + fid)

  def collect_train_frames(self):
    """Creates a list of training frames."""
    all_frames = []
    for date in self.date_list:
      date_dir = os.path.join(self.dataset_dir, date)
      drive_set = os.listdir(date_dir)
      for dr in drive_set:
        drive_dir = os.path.join(date_dir, dr)
        if os.path.isdir(drive_dir):
          if dr[:-5] in self.test_scenes:
            continue
          for cam in self.cam_ids:
            img_dir = os.path.join(drive_dir, 'image_' + cam, 'data')
            num_frames = len(glob.glob(img_dir + '/*[0-9].png'))
            for i in range(num_frames):
              frame_id = '%.10d' % i
              all_frames.append(dr + ' ' + cam + ' ' + frame_id)

    for s in self.static_frames:
      try:
        all_frames.remove(s)
      except ValueError:
        pass

    self.train_frames = all_frames
    self.num_train = len(self.train_frames)

  def is_valid_sample(self, frames, target_index):
    """Checks whether we can find a valid sequence around this frame."""
    num_frames = len(frames)
    target_drive, cam_id, _ = frames[target_index].split(' ')
    start_index, end_index = get_seq_start_end(target_index, self.seq_length)
    if start_index < 0 or end_index >= num_frames:
      return False
    start_drive, start_cam_id, _ = frames[start_index].split(' ')
    end_drive, end_cam_id, _ = frames[end_index].split(' ')
    if (target_drive == start_drive and target_drive == end_drive and
        cam_id == start_cam_id and cam_id == end_cam_id):
      return True
    return False

  def get_example_with_index(self, target_index):
    if not self.is_valid_sample(self.train_frames, target_index):
      return False
    example = self.load_example(self.train_frames, target_index)
    return example

  def load_image_sequence(self, frames, target_index):
    """Returns a sequence with requested target frame."""
    start_index, end_index = get_seq_start_end(target_index, self.seq_length)
    image_seq = []
    for index in range(start_index, end_index + 1):
      drive, cam_id, frame_id = frames[index].split(' ')
      img = self.load_image_raw(drive, cam_id, frame_id)
      if index == target_index:
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
      img = scipy.misc.imresize(img, (self.img_height, self.img_width))
      image_seq.append(img)
    return image_seq, zoom_x, zoom_y

  def load_pose_sequence(self, frames, target_index):
    """Returns a sequence of pose vectors for frames around the target frame."""
    target_drive, _, target_frame_id = frames[target_index].split(' ')
    target_pose = self.load_pose_raw(target_drive, target_frame_id)
    start_index, end_index = get_seq_start_end(target_frame_id, self.seq_length)
    pose_seq = []
    for index in range(start_index, end_index + 1):
      if index == target_frame_id:
        continue
      drive, _, frame_id = frames[index].split(' ')
      pose = self.load_pose_raw(drive, frame_id)
      # From target to index.
      pose = np.dot(np.linalg.inv(pose), target_pose)
      pose_seq.append(pose)
    return pose_seq

  def load_example(self, frames, target_index):
    """Returns a sequence with requested target frame."""
    image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, target_index)
    target_drive, target_cam_id, target_frame_id = (
        frames[target_index].split(' '))
    intrinsics = self.load_intrinsics_raw(target_drive, target_cam_id)
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
    example = {}
    example['intrinsics'] = intrinsics
    example['image_seq'] = image_seq
    example['folder_name'] = target_drive + '_' + target_cam_id + '/'
    example['file_name'] = target_frame_id
    if self.load_pose:
      pose_seq = self.load_pose_sequence(frames, target_index)
      example['pose_seq'] = pose_seq
    return example

  def load_pose_raw(self, drive, frame_id):
    date = drive[:10]
    pose_file = os.path.join(self.dataset_dir, date, drive, 'poses',
                             frame_id + '.txt')
    with open(pose_file, 'r') as f:
      pose = f.readline()
    pose = np.array(pose.split(' ')).astype(np.float32).reshape(3, 4)
    pose = np.vstack((pose, np.array([0, 0, 0, 1]).reshape((1, 4))))
    return pose

  def load_image_raw(self, drive, cam_id, frame_id):
    date = drive[:10]
    img_file = os.path.join(self.dataset_dir, date, drive, 'image_' + cam_id,
                            'data', frame_id + '.png')
    img = scipy.misc.imread(img_file)
    return img

  def load_intrinsics_raw(self, drive, cam_id):
    date = drive[:10]
    calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')
    filedata = self.read_raw_calib_file(calib_file)
    p_rect = np.reshape(filedata['P_rect_' + cam_id], (3, 4))
    intrinsics = p_rect[:3, :3]
    return intrinsics

  # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
  def read_raw_calib_file(self, filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
      for line in f:
        key, value = line.split(':', 1)
        # The only non-float values in these files are dates, which we don't
        # care about.
        try:
          data[key] = np.array([float(x) for x in value.split()])
        except ValueError:
          pass
    return data

  def scale_intrinsics(self, mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


class KittiOdom(object):
  """Reads KITTI odometry data files."""

  def __init__(self, dataset_dir, img_height=128, img_width=416, seq_length=3):
    self.dataset_dir = dataset_dir
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    self.test_seqs = [9, 10]

    self.collect_test_frames()
    self.collect_train_frames()

  def collect_test_frames(self):
    self.test_frames = []
    for seq in self.test_seqs:
      seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
      img_dir = os.path.join(seq_dir, 'image_2')
      num_frames = len(glob.glob(os.path.join(img_dir, '*.png')))
      for n in range(num_frames):
        self.test_frames.append('%.2d %.6d' % (seq, n))
    self.num_test = len(self.test_frames)

  def collect_train_frames(self):
    self.train_frames = []
    for seq in self.train_seqs:
      seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
      img_dir = os.path.join(seq_dir, 'image_2')
      num_frames = len(glob.glob(img_dir + '/*.png'))
      for n in range(num_frames):
        self.train_frames.append('%.2d %.6d' % (seq, n))
    self.num_train = len(self.train_frames)

  def is_valid_sample(self, frames, target_frame_index):
    """Checks whether we can find a valid sequence around this frame."""
    num_frames = len(frames)
    target_frame_drive, _ = frames[target_frame_index].split(' ')
    start_index, end_index = get_seq_start_end(target_frame_index,
                                               self.seq_length)
    if start_index < 0 or end_index >= num_frames:
      return False
    start_drive, _ = frames[start_index].split(' ')
    end_drive, _ = frames[end_index].split(' ')
    if target_frame_drive == start_drive and target_frame_drive == end_drive:
      return True
    return False

  def load_image_sequence(self, frames, target_frame_index):
    """Returns a sequence with requested target frame."""
    start_index, end_index = get_seq_start_end(target_frame_index,
                                               self.seq_length)
    image_seq = []
    for index in range(start_index, end_index + 1):
      drive, frame_id = frames[index].split(' ')
      img = self.load_image(drive, frame_id)
      if index == target_frame_index:
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
      img = scipy.misc.imresize(img, (self.img_height, self.img_width))
      image_seq.append(img)
    return image_seq, zoom_x, zoom_y

  def load_example(self, frames, target_frame_index):
    """Returns a sequence with requested target frame."""
    image_seq, zoom_x, zoom_y = self.load_image_sequence(frames,
                                                         target_frame_index)
    target_frame_drive, target_frame_id = frames[target_frame_index].split(' ')
    intrinsics = self.load_intrinsics(target_frame_drive, target_frame_id)
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
    example = {}
    example['intrinsics'] = intrinsics
    example['image_seq'] = image_seq
    example['folder_name'] = target_frame_drive
    example['file_name'] = target_frame_id
    return example

  def get_example_with_index(self, target_frame_index):
    if not self.is_valid_sample(self.train_frames, target_frame_index):
      return False
    example = self.load_example(self.train_frames, target_frame_index)
    return example

  def load_image(self, drive, frame_id):
    img_file = os.path.join(self.dataset_dir, 'sequences',
                            '%s/image_2/%s.png' % (drive, frame_id))
    img = scipy.misc.imread(img_file)
    return img

  def load_intrinsics(self, drive, unused_frame_id):
    calib_file = os.path.join(self.dataset_dir, 'sequences',
                              '%s/calib.txt' % drive)
    proj_c2p, _ = self.read_calib_file(calib_file)
    intrinsics = proj_c2p[:3, :3]
    return intrinsics

  def read_calib_file(self, filepath, cam_id=2):
    """Read in a calibration file and parse into a dictionary."""

    def parse_line(line, shape):
      data = line.split()
      data = np.array(data[1:]).reshape(shape).astype(np.float32)
      return data

    with open(filepath, 'r') as f:
      mat = f.readlines()
    proj_c2p = parse_line(mat[cam_id], shape=(3, 4))
    proj_v2c = parse_line(mat[-1], shape=(3, 4))
    filler = np.array([0, 0, 0, 1]).reshape((1, 4))
    proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
    return proj_c2p, proj_v2c

  def scale_intrinsics(self, mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


class Cityscapes(object):
  """Reads Cityscapes data files."""

  def __init__(self,
               dataset_dir,
               split='train',
               crop_bottom=CITYSCAPES_CROP_BOTTOM,  # Crop the car logo.
               crop_pct=CITYSCAPES_CROP_PCT,
               sample_every=CITYSCAPES_SAMPLE_EVERY,
               img_height=128,
               img_width=416,
               seq_length=3):
    self.dataset_dir = dataset_dir
    self.split = split
    self.crop_bottom = crop_bottom
    self.crop_pct = crop_pct
    self.sample_every = sample_every
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.frames = self.collect_frames(split)
    self.num_frames = len(self.frames)
    if split == 'train':
      self.num_train = self.num_frames
    else:
      self.num_test = self.num_frames
    logging.info('Total frames collected: %d', self.num_frames)

  def collect_frames(self, split):
    img_dir = os.path.join(self.dataset_dir, 'leftImg8bit_sequence', split)
    city_list = os.listdir(img_dir)
    frames = []
    for city in city_list:
      img_files = glob.glob(os.path.join(img_dir, city, '*.png'))
      for f in img_files:
        frame_id = os.path.basename(f).split('leftImg8bit')[0]
        frames.append(frame_id)
    return frames

  def get_example_with_index(self, target_index):
    target_frame_id = self.frames[target_index]
    if not self.is_valid_example(target_frame_id):
      return False
    example = self.load_example(self.frames[target_index])
    return example

  def load_intrinsics(self, frame_id, split):
    """Read intrinsics data for frame."""
    city, seq, _, _ = frame_id.split('_')
    camera_file = os.path.join(self.dataset_dir, 'camera', split, city,
                               city + '_' + seq + '_*_camera.json')
    camera_file = glob.glob(camera_file)[0]
    with open(camera_file, 'r') as f:
      camera = json.load(f)
    fx = camera['intrinsic']['fx']
    fy = camera['intrinsic']['fy']
    u0 = camera['intrinsic']['u0']
    v0 = camera['intrinsic']['v0']
    # Cropping the bottom of the image and then resizing it to the same
    # (height, width) amounts to stretching the image's height.
    if self.crop_bottom:
      fy *= 1.0 / self.crop_pct
    intrinsics = np.array([[fx, 0, u0],
                           [0, fy, v0],
                           [0, 0, 1]])
    return intrinsics

  def is_valid_example(self, target_frame_id):
    """Checks whether we can find a valid sequence around this frame."""
    city, snippet_id, target_local_frame_id, _ = target_frame_id.split('_')
    start_index, end_index = get_seq_start_end(
        int(target_local_frame_id), self.seq_length, self.sample_every)
    for index in range(start_index, end_index + 1, self.sample_every):
      local_frame_id = '%.6d' % index
      frame_id = '%s_%s_%s_' % (city, snippet_id, local_frame_id)
      image_filepath = os.path.join(self.dataset_dir, 'leftImg8bit_sequence',
                                    self.split, city,
                                    frame_id + 'leftImg8bit.png')
      if not os.path.exists(image_filepath):
        return False
    return True

  def load_image_sequence(self, target_frame_id):
    """Returns a sequence with requested target frame."""
    city, snippet_id, target_local_frame_id, _ = target_frame_id.split('_')
    start_index, end_index = get_seq_start_end(
        int(target_local_frame_id), self.seq_length, self.sample_every)
    image_seq = []
    for index in range(start_index, end_index + 1, self.sample_every):
      local_frame_id = '%.6d' % index
      frame_id = '%s_%s_%s_' % (city, snippet_id, local_frame_id)
      image_filepath = os.path.join(self.dataset_dir, 'leftImg8bit_sequence',
                                    self.split, city,
                                    frame_id + 'leftImg8bit.png')
      img = scipy.misc.imread(image_filepath)
      if self.crop_bottom:
        ymax = int(img.shape[0] * self.crop_pct)
        img = img[:ymax]
      raw_shape = img.shape
      if index == int(target_local_frame_id):
        zoom_y = self.img_height / raw_shape[0]
        zoom_x = self.img_width / raw_shape[1]
      img = scipy.misc.imresize(img, (self.img_height, self.img_width))
      image_seq.append(img)
    return image_seq, zoom_x, zoom_y

  def load_example(self, target_frame_id):
    """Returns a sequence with requested target frame."""
    image_seq, zoom_x, zoom_y = self.load_image_sequence(target_frame_id)
    intrinsics = self.load_intrinsics(target_frame_id, self.split)
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
    example = {}
    example['intrinsics'] = intrinsics
    example['image_seq'] = image_seq
    example['folder_name'] = target_frame_id.split('_')[0]
    example['file_name'] = target_frame_id[:-1]
    return example

  def scale_intrinsics(self, mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def get_resource_path(relative_path):
  return relative_path


def get_seq_start_end(target_index, seq_length, sample_every=1):
  """Returns absolute seq start and end indices for a given target frame."""
  half_offset = int((seq_length - 1) / 2) * sample_every
  end_index = target_index + half_offset
  start_index = end_index - (seq_length - 1) * sample_every
  return start_index, end_index


def atoi(text):
  return int(text) if text.isdigit() else text


def natural_keys(text):
  return [atoi(c) for c in re.split(r'(\d+)', text)]
