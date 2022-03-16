# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.tools.build_docs."""

import os
import shutil

import tensorflow as tf

from official.utils.docs import build_all_api_docs


class BuildDocsTest(tf.test.TestCase):

  def setUp(self):
    super(BuildDocsTest, self).setUp()
    self.workdir = self.get_temp_dir()
    if os.path.exists(self.workdir):
      shutil.rmtree(self.workdir)
    os.makedirs(self.workdir)

  def test_api_gen(self):
    build_all_api_docs.gen_api_docs(
        code_url_prefix="https://github.com/tensorflow/models/blob/master/tensorflow_models",
        site_path="tf_modeling/api_docs/python",
        output_dir=self.workdir,
        project_short_name="tfm",
        project_full_name="TensorFlow Modeling",
        search_hints=True)

    # Check that the "defined in" section is working
    with open(os.path.join(self.workdir, "tfm.md")) as f:
      content = f.read()
    self.assertIn("__init__.py", content)


if __name__ == "__main__":
  tf.test.main()
