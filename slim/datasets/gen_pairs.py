"""
The audio windows are 20ms and the video frame rate is 30 f/s.
it means: 5 audio windows correspond to 3 video frames.

Shift_unit: The appropriate shifts on the cube for creating pairs.(5 per audio and 3 for mouth)
            ex:
            pair1: speech - indexes [0,15) & mouth - indexes [0,9)
            pair2: speech - indexes [5,15) & mouth - indexes [3,12)

            The above means shifting by ONE UNIT.

            No overlapping frames means: Shift_unit == 3!

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import numpy as np
import pickle

import tensorflow as tf

from datasets import dataset_utils

# Define the shift unit
Shift_unit = 1

# Desired time interval for considering pairs
time_interval = 0.3

# Number of videos per the desired time
num_video_frame_per_pair = 9

# Number of audios per the desired time
num_audio_frame_per_pair = 15


def gen_pairs(speech_data, mouth_data):
    "Generate pair"

    # Necessary calculations
    num_mouth_frames = mouth_data.shape[2]
    mouth_height = mouth_data.shape[0]
    mouth_width = mouth_data.shape[1]
    max_num_pairs_mouth = int(np.floor((num_mouth_frames - num_video_frame_per_pair)/(Shift_unit * 3.0)) + 1)
    print('max_num_pairs_mouth',max_num_pairs_mouth)

    num_speech_frames = speech_data.shape[1]
    num_speech_features = speech_data.shape[0]
    max_num_pairs_speech = int(np.floor((num_speech_frames - num_audio_frame_per_pair) / (Shift_unit * 5.0)) + 1)
    print('max_num_pairs_speech',max_num_pairs_speech)

    # Calculate number of frames
    num_pairs = min(max_num_pairs_mouth,max_num_pairs_speech)

    # Numpy vector
    mouth_pair_vector = np.zeros((num_pairs, mouth_height, mouth_width, num_video_frame_per_pair), dtype=np.float32)
    speech_pair_vector = np.zeros((num_pairs, num_speech_features, num_audio_frame_per_pair, 1), dtype=np.float32)

    for i in range(num_pairs):
        mouth_pair_vector[i, :, :, :] = mouth_data[:, :, i * (Shift_unit * 3): i * (Shift_unit * 3) + 9]
        speech_pair_vector[i, :, :, 0] = speech_data[:, i * (Shift_unit * 5): i * (Shift_unit * 5) + 15]

    return mouth_pair_vector, speech_pair_vector


