# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import argh
import os
import itertools
import re
import numpy as np
from tqdm import tqdm

def crawl(sgf_directory='sgf', print_summary=True): 
    max_w_upset = {'value': 0}
    max_b_upset = {'value': 0}

    worst_qs = []
    tot_files = 0
    num_resign_disabled = 0
    bad_resigns = 0
    bad_resign_files = []
    other_thresh = 0.9
    sgfs = lambda root,fils: [os.path.join(root, f) for f in fils if f.endswith('.sgf')]
    fs = [ i for sublist in [sgfs(root, files) for root, _, files in os.walk(sgf_directory)] for i in sublist]
    for filename in tqdm(fs):

          data = open(filename).read()
          result = re.search("RE\[([BWbw])\+", data)
          if not result:
              print("No result string found in sgf: ", filename)
              continue
          else:
              result = result.group(1)

          threshold = re.search("Resign Threshold: -(\d.\d*)", data)
          if not threshold:
              print("No threshold found for ", filename)
          else:
              threshold = float(threshold.group(1))
              if threshold == 1.0:
                  num_resign_disabled += 1

          tot_files += 1
          q_values = list(map(float, re.findall("C\[(-?\d.\d*)", data)))
          if result == "B":
              look_for = min
          else:
              look_for = max

          #print("%s:%s+:%s" % (filename, result, min(q_values)))
          worst_qs.append(look_for(q_values))

          if threshold == 1.0 and abs(look_for(q_values)) > other_thresh:
              bad_resigns += 1
              bad_resign_files.append(filename)

          if look_for == min and min(q_values) < max_b_upset['value']:
              max_b_upset = {"filename": filename,
                             "value": look_for(q_values)}
          elif look_for == max and max(q_values) > max_w_upset['value']:
              max_w_upset = {"filename": filename,
                             "value": max(q_values)}


    if print_summary:
        b_upsets = np.array([q for q in worst_qs if q < 0])
        w_upsets = np.array([q for q in worst_qs if q > 0])
        both = np.array(list(map(abs, worst_qs)))
        print("Biggest w upset:", max_w_upset)
        print("Biggest b upset:", max_b_upset)
        print ("99th percentiles (both/w/b)")
        print(np.percentile(both, 99))
        print(np.percentile(b_upsets, 1))
        print(np.percentile(w_upsets, 99))
        print ("Bad resigns: {} / {} ({:.2f}%) ".format(bad_resigns, num_resign_disabled, (bad_resigns / (num_resign_disabled+1)) * 100.0))
        print ("Total files:", tot_files)
        print (bad_resign_files)

if __name__ == '__main__':
    argh.dispatch_command(crawl)
