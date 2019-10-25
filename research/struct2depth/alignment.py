
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

"""Common utilities for data pre-processing, e.g. matching moving object across frames."""

import numpy as np

def compute_overlap(mask1, mask2):
    # Use IoU here.
    return np.sum(mask1 & mask2)/np.sum(mask1 | mask2)

def align(seg_img1, seg_img2, seg_img3, threshold_same=0.3):
    res_img1 = np.zeros_like(seg_img1)
    res_img2 = np.zeros_like(seg_img2)
    res_img3 = np.zeros_like(seg_img3)
    remaining_objects2 = list(np.unique(seg_img2.flatten()))
    remaining_objects3 = list(np.unique(seg_img3.flatten()))
    for seg_id in np.unique(seg_img1):
        # See if we can find correspondences to seg_id in seg_img2.
        max_overlap2 = float('-inf')
        max_segid2 = -1
        for seg_id2 in remaining_objects2:
            overlap = compute_overlap(seg_img1==seg_id, seg_img2==seg_id2)
            if overlap>max_overlap2:
                max_overlap2 = overlap
                max_segid2 = seg_id2
        if max_overlap2 > threshold_same:
            max_overlap3 = float('-inf')
            max_segid3 = -1
            for seg_id3 in remaining_objects3:
                overlap = compute_overlap(seg_img2==max_segid2, seg_img3==seg_id3)
                if overlap>max_overlap3:
                    max_overlap3 = overlap
                    max_segid3 = seg_id3
            if max_overlap3 > threshold_same:
                res_img1[seg_img1==seg_id] = seg_id
                res_img2[seg_img2==max_segid2] = seg_id
                res_img3[seg_img3==max_segid3] = seg_id
                remaining_objects2.remove(max_segid2)
                remaining_objects3.remove(max_segid3)
    return res_img1, res_img2, res_img3
