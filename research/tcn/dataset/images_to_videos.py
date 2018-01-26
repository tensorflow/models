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

"""Converts temp directories of images to videos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
# pylint: disable=invalid-name

parser = argparse.ArgumentParser()
parser.add_argument(
    '--view_dirs', type=str, default='',
    help='Comma-separated list of temp view image directories.')
parser.add_argument(
    '--vid_paths', type=str, default='',
    help='Comma-separated list of video output paths.')
parser.add_argument(
    '--debug_path', type=str, default='',
    help='Output path to debug video.')

parser.add_argument(
    '--debug_lhs_view', type=str, default='',
    help='Output path to debug video.')
parser.add_argument(
    '--debug_rhs_view', type=str, default='',
    help='Output path to debug video.')


def create_vids(view_dirs, vid_paths, debug_path=None,
                debug_lhs_view=0, debug_rhs_view=1):
  """Creates one video per view per sequence."""

  # Create the view videos.
  for (view_dir, vidpath) in zip(view_dirs, vid_paths):
    encode_vid_cmd = r'mencoder mf://%s/*.png \
    -mf fps=29:type=png \
    -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell \
    -oac copy -o %s' % (view_dir, vidpath)
    os.system(encode_vid_cmd)

  # Optionally create a debug side-by-side video.
  if debug_path:
    lhs = vid_paths[int(debug_lhs_view)]
    rhs = vid_paths[int(debug_rhs_view)]
    os.system(r"avconv \
      -i %s \
      -i %s \
      -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
      -map [vid] \
      -c:v libx264 \
      -crf 23 \
      -preset veryfast \
      %s" % (lhs, rhs, debug_path))


def main():
  FLAGS, _ = parser.parse_known_args()
  assert FLAGS.view_dirs
  assert FLAGS.vid_paths
  view_dirs = FLAGS.view_dirs.split(',')
  vid_paths = FLAGS.vid_paths.split(',')
  create_vids(view_dirs, vid_paths, FLAGS.debug_path,
              FLAGS.debug_lhs_view, FLAGS.debug_rhs_view)

  # Cleanup temp image dirs.
  for i in view_dirs:
    shutil.rmtree(i)

if __name__ == '__main__':
  main()
