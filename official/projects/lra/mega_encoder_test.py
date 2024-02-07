# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.nlp.projects.lra.mega_encoder."""

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.lra import mega_encoder


class MegaEncoderTest(tf.test.TestCase):

  def test_encoder(self):
    sequence_length = 1024
    batch_size = 2
    vocab_size = 1024
    network = mega_encoder.MegaEncoder(
        num_layers=1,
        vocab_size=1024,
        max_sequence_length=4096,
    )
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length)
    )
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(2, size=(batch_size, sequence_length))
    outputs = network({
        "input_word_ids": word_id_data,
        "input_mask": mask_data,
        "input_type_ids": type_id_data,
    })
    self.assertEqual(
        outputs["sequence_output"].shape,
        (batch_size, sequence_length, 128),
    )


if __name__ == "__main__":
  tf.test.main()
