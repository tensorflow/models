# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for official.tools.build_docs."""

import os
import shutil

import tensorflow as tf

from official.utils.docs import build_docs


class BuildDocsTest(tf.test.TestCase):

  def setUp(self):
    super(BuildDocsTest, self).setUp()
    self.workdir = self.get_temp_dir()
    if os.path.exists(self.workdir):
      shutil.rmtree(self.workdir)
    os.makedirs(self.workdir)

  def test_api_gen(self):
    build_docs.gen_api_docs(
        code_url_prefix="http://official/nlp/modeling/",
        site_path="tf_nlp_modeling/api_docs/python",
        output_dir=self.workdir,
        gen_report=False,
        project_short_name="tf_nlp_modeling",
        project_full_name="TensorFlow Modeling - NLP Library",
        search_hints=True)

    # Check that the "defined in" section is working
    with open(os.path.join(self.workdir, "tf_nlp_modeling.md")) as f:
      content = f.read()
    self.assertIn("__init__.py", content)


if __name__ == "__main__":
  tf.test.main()
