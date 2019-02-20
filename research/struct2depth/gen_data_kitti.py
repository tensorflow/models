
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

""" Offline data generation for the KITTI dataset."""

import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import cv2
import os, glob

import alignment
from alignment import compute_overlap
from alignment import align


SEQ_LENGTH = 3
WIDTH = 416
HEIGHT = 128
STEPSIZE = 1
INPUT_DIR = '/usr/local/google/home/anelia/struct2depth/KITTI_FULL/kitti-raw-uncompressed'
OUTPUT_DIR = '/usr/local/google/home/anelia/struct2depth/KITTI_procesed/'


def get_line(file, start):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    ret = None
    for line in lines:
        nline = line.split(': ')
        if nline[0]==start:
            ret = nline[1].split(' ')
            ret = np.array([float(r) for r in ret], dtype=float)
            ret = ret.reshape((3,4))[0:3, 0:3]
            break
    file.close()
    return ret


def crop(img, segimg, fx, fy, cx, cy):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1-middle_perc
    half = left/2
    a = img[int(img.shape[0]*(half)):int(img.shape[0]*(1-half)), :]
    aseg = segimg[int(segimg.shape[0]*(half)):int(segimg.shape[0]*(1-half)), :]
    cy /= (1/middle_perc)

    # Resize to match target height while preserving aspect ratio.
    wdt = int((128*a.shape[1]/a.shape[0]))
    x_scaling = float(wdt)/a.shape[1]
    y_scaling = 128.0/a.shape[0]
    b = cv2.resize(a, (wdt, 128))
    bseg = cv2.resize(aseg, (wdt, 128))

    # Adjust intrinsics.
    fx*=x_scaling
    fy*=y_scaling
    cx*=x_scaling
    cy*=y_scaling

    # Perform center cropping horizontally.
    remain = b.shape[1] - 416
    cx /= (b.shape[1]/416)
    c = b[:, int(remain/2):b.shape[1]-int(remain/2)]
    cseg = bseg[:, int(remain/2):b.shape[1]-int(remain/2)]

    return c, cseg, fx, fy, cx, cy


def run_all():
  ct = 0
if not OUTPUT_DIR.endswith('/'):
    OUTPUT_DIR = OUTPUT_DIR + '/'

for d in glob.glob(INPUT_DIR + '/*/'):
    date = d.split('/')[-2]
    file_calibration = d + 'calib_cam_to_cam.txt'
    calib_raw = [get_line(file_calibration, 'P_rect_02'), get_line(file_calibration, 'P_rect_03')]

    for d2 in glob.glob(d + '*/'):
        seqname = d2.split('/')[-2]
        print('Processing sequence', seqname)
        for subfolder in ['image_02/data', 'image_03/data']:
            ct = 1
            seqname = d2.split('/')[-2] + subfolder.replace('image', '').replace('/data', '')
            if not os.path.exists(OUTPUT_DIR + seqname):
                os.mkdir(OUTPUT_DIR + seqname)

            calib_camera = calib_raw[0] if subfolder=='image_02/data' else calib_raw[1]
            folder = d2 + subfolder
            files = glob.glob(folder + '/*.png')
            files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
            files = sorted(files)
            for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
                imgnum = str(ct).zfill(10)
                if os.path.exists(OUTPUT_DIR + seqname + '/' + imgnum + '.png'):
                    ct+=1
                    continue
                big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
                wct = 0

                for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                    img = cv2.imread(files[j])
                    ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

                    zoom_x = WIDTH/ORIGINAL_WIDTH
                    zoom_y = HEIGHT/ORIGINAL_HEIGHT

                    # Adjust intrinsics.
                    calib_current = calib_camera.copy()
                    calib_current[0, 0] *= zoom_x
                    calib_current[0, 2] *= zoom_x
                    calib_current[1, 1] *= zoom_y
                    calib_current[1, 2] *= zoom_y

                    calib_representation = ','.join([str(c) for c in calib_current.flatten()])

                    img = cv2.resize(img, (WIDTH, HEIGHT))
                    big_img[:,wct*WIDTH:(wct+1)*WIDTH] = img
                    wct+=1
                cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '.png', big_img)
                f = open(OUTPUT_DIR + seqname + '/' + imgnum + '_cam.txt', 'w')
                f.write(calib_representation)
                f.close()
                ct+=1

def main(_):
  run_all()


if __name__ == '__main__':
  app.run(main)
