# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A simple smoke test that runs these examples for 1 training iteraton."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from six.moves import StringIO

import automobile_data

import dnn_regression
import linear_regression
import linear_regression_categorical
import custom_regression

# pylint: disable=line-too-long
FOUR_LINES = "\n".join([
    "1,?,alfa-romero,gas,std,two,hatchback,rwd,front,94.50,171.20,65.50,52.40,2823,ohcv,six,152,mpfi,2.68,3.47,9.00,154,5000,19,26,16500",
    "2,164,audi,gas,std,four,sedan,fwd,front,99.80,176.60,66.20,54.30,2337,ohc,four,109,mpfi,3.19,3.40,10.00,102,5500,24,30,13950",
    "2,164,audi,gas,std,four,sedan,4wd,front,99.40,176.60,66.40,54.30,2824,ohc,five,136,mpfi,3.19,3.40,8.00,115,5500,18,22,17450",
    "2,?,audi,gas,std,two,sedan,fwd,front,99.80,177.30,66.30,53.10,2507,ohc,five,136,mpfi,3.19,3.40,8.50,110,5500,19,25,15250",])
# pylint: enable=line-too-long

mock = tf.test.mock

def four_lines_dataframe():
  text = StringIO(FOUR_LINES)
  return pd.read_csv(text, names=automobile_data.COLUMN_TYPES.keys(),
                     dtype=automobile_data.COLUMN_TYPES, na_values="?")


def four_lines_dataset(*args, **kwargs):
  del args, kwargs
  return tf.data.Dataset.from_tensor_slices(FOUR_LINES.split("\n"))


class RegressionTest(tf.test.TestCase):
  """Test the regression examples in this directory."""

  @mock.patch.dict(automobile_data.__dict__, {"raw_dataframe": four_lines_dataframe})
  def test_linear_regression(self):
    linear_regression.main([None, "--train_steps=1"])

  @mock.patch.dict(automobile_data.__dict__, {"raw_dataframe": four_lines_dataframe})
  def test_linear_regression_categorical(self):
    linear_regression_categorical.main([None, "--train_steps=1"])

  @mock.patch.dict(automobile_data.__dict__, {"raw_dataframe": four_lines_dataframe})
  def test_dnn_regression(self):
    dnn_regression.main([None, "--train_steps=1"])

  @mock.patch.dict(automobile_data.__dict__, {"raw_dataframe": four_lines_dataframe})
  def test_custom_regression(self):
    custom_regression.main([None, "--train_steps=1"])

if __name__ == "__main__":
  tf.test.main()
