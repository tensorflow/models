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

"""Tests for mobilebert_edgetpu_trainer.py."""

import tensorflow as tf, tf_keras

from official.projects.edgetpu.nlp import mobilebert_edgetpu_trainer
from official.projects.edgetpu.nlp.configs import params
from official.projects.edgetpu.nlp.modeling import model_builder


# Helper function to create dummy dataset
def _dummy_dataset():
  def dummy_data(_):
    dummy_ids = tf.zeros((1, 64), dtype=tf.int32)
    dummy_lm = tf.zeros((1, 64), dtype=tf.int32)
    return dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids,
        masked_lm_positions=dummy_lm,
        masked_lm_ids=dummy_lm,
        masked_lm_weights=tf.cast(dummy_lm, dtype=tf.float32),
        next_sentence_labels=tf.zeros((1, 1), dtype=tf.int32))
  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class EdgetpuBertTrainerTest(tf.test.TestCase):

  def setUp(self):
    super(EdgetpuBertTrainerTest, self).setUp()
    self.experiment_params = params.EdgeTPUBERTCustomParams()
    self.strategy = tf.distribute.get_strategy()
    self.experiment_params.train_datasest.input_path = 'dummy'
    self.experiment_params.eval_dataset.input_path = 'dummy'

  def test_train_model_locally(self):
    """Tests training a model locally with one step."""
    teacher_model = model_builder.build_bert_pretrainer(
        pretrainer_cfg=self.experiment_params.teacher_model,
        name='teacher')
    _ = teacher_model(teacher_model.inputs)
    student_model = model_builder.build_bert_pretrainer(
        pretrainer_cfg=self.experiment_params.student_model,
        name='student')
    _ = student_model(student_model.inputs)
    trainer = mobilebert_edgetpu_trainer.MobileBERTEdgeTPUDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        strategy=self.strategy,
        experiment_params=self.experiment_params)

    # Rebuild dummy dataset since loading real dataset will cause timeout error.
    trainer.train_dataset = _dummy_dataset()
    trainer.eval_dataset = _dummy_dataset()
    train_dataset_iter = iter(trainer.train_dataset)
    eval_dataset_iter = iter(trainer.eval_dataset)
    trainer.train_loop_begin()

    trainer.train_step(train_dataset_iter)
    trainer.eval_step(eval_dataset_iter)


if __name__ == '__main__':
  tf.test.main()
